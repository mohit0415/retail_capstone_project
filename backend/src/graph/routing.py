"""Conditional-edge functions for the policy graph (v4 workflow).

Each function reads the state a node just wrote and names the edge to take. They are pure
decisions; the only side effect is a log line, so the path a request took through the graph can be
read straight out of the log (``edge <from> -> <to>``) without opening the trace.

The evidence stages always run in one fixed order - rag_path, then nl2sql_path, then
multi_agent_panel - and every route ends at compliance_validation. The route the planner settled on
only decides which of the stages a request needs (``evidence_stages``):

    rag              rag_path                                      -> compliance_validation
    nl2sql           nl2sql_path                                   -> compliance_validation
    hybrid           rag_path -> nl2sql_path                       -> compliance_validation
    high_risk_panel  rag_path -> nl2sql_path -> multi_agent_panel  -> compliance_validation

A request goes to a human only where the brief requires it: risk is High, the sources conflict,
confidence is under the threshold, the user asked for a human or legal review, or the answer could
not be certified (validation still failing after the permitted repairs, a budget stop, a rejected
release). When the evidence simply does not contain the answer, the reply is an honest
"I don't know" (``not_found`` -> src/nodes/no_answer.py) instead of an escalation.
"""

import logging

from configs.settings import settings
from src.auth.rbac import allowed_doc_types, allowed_tables
from src.core.budget import NODE_TOKEN_ESTIMATE, guard_from_state
from src.graph.path_policy import ESCALATE, NO_ACCESS, degrade_for_budget
from src.graph.state import AgentState
from src.schemas.enums import EvidencePath, ReviewDecision, RiskLevel, TerminalOutcome

logger = logging.getLogger(__name__)

# kept for backwards compatibility; the live values are settings.repair_headroom_seconds
# and repair_token_estimate(), because a repair is a whole reflection -> plan -> evidence
# -> validation pass and one second / 2000 tokens never covered it (the repair then got
# budget-stopped half way, after spending the time anyway)
REPAIR_HEADROOM_SECONDS = 1.0

REPAIR_HEADROOM_TOKENS = 2000

RETRIEVAL_NO_MATCH_CEILING = 0.05

RAG_PATHS = {"rag", "rag_path"}

SQL_PATHS = {"nl2sql", "nl2sql_path"}

# the reflection strategy for a records answer: keep the rows, rewrite the answer with the
# validator's defects (src/nodes/reflection.py, src/nodes/nl2sql_path.py)
SQL_RENARRATE = "sql_renarrate"

HYBRID_PATHS = {"hybrid", "hybrid_path"}

# the reflection strategy for a hybrid answer: keep the rows, search the policies wider and rewrite the
# answer with the validator's defects - neither evidence stage reads a re-plan (src/nodes/reflection.py)
HYBRID_REDRAFT = "hybrid_redraft"

# why the honest "I don't know" reply was given (state["not_found_reason"], src/nodes/no_answer.py)
NOT_FOUND_NO_MATCH = "no_match"
NOT_FOUND_NOT_IN_EXTRACTS = "not_in_extracts"
NOT_FOUND_NO_RECORDS = "no_records"
NOT_FOUND_NO_ACCESS = "no_access"
NOT_FOUND_NO_EVIDENCE = "no_evidence"

POLICY_NOT_FOUND = {NOT_FOUND_NO_MATCH, NOT_FOUND_NOT_IN_EXTRACTS}

RECORDS_NOT_FOUND = {NOT_FOUND_NO_RECORDS, NOT_FOUND_NO_EVIDENCE}

RAG_STAGE = "rag_path"

SQL_STAGE = "nl2sql_path"

PANEL_STAGE = "multi_agent_panel"

VALIDATION = "compliance_validation"

# the order the evidence stages always run in
STAGE_ORDER = (RAG_STAGE, SQL_STAGE, PANEL_STAGE)

# node names and retired routes that may still sit in an older checkpoint or a stub; the v4
# workflow has no agentic route, so an agentic decision runs the stages hybrid runs
ROUTE_ALIASES = {
    "rag_path": EvidencePath.RAG.value,
    "nl2sql_path": EvidencePath.NL2SQL.value,
    "hybrid_path": EvidencePath.HYBRID.value,
    EvidencePath.AGENTIC.value: EvidencePath.HYBRID.value,
    "agentic_rag": EvidencePath.HYBRID.value,
    "multi_agent_panel": EvidencePath.HIGH_RISK_PANEL.value,
}

RESUMABLE_DECISIONS = {ReviewDecision.ACCEPT.value, ReviewDecision.EDIT.value}


def current_evidence_path(state: AgentState) -> str:
    return state.get("evidence_path") or state.get("routed_path") or ""


def current_route(state: AgentState) -> str:
    """The evidence route the planner settled on, after clamps, capability gates and budget."""
    risk = state.get("risk")

    if risk and risk.final_level is RiskLevel.HIGH:
        return EvidencePath.HIGH_RISK_PANEL.value

    for candidate in (state.get("routed_path"), state.get("evidence_path")):
        if candidate:
            return ROUTE_ALIASES.get(candidate, candidate)

    return _plan_route(state)[1].path


def _plan_route(state: AgentState):
    """(the plan's route, its budget decision) - used only when the planner recorded no route.

    The router and every stage read this same decision, so an older checkpoint that is degraded from
    hybrid to rag runs rag's stages, not hybrid's. The deadline only gets closer, so a later read
    never undoes a degrade an earlier one made.
    """
    plan = state.get("plan")
    routed = ROUTE_ALIASES.get(plan.path.value, plan.path.value) if plan else EvidencePath.RAG.value
    guard = guard_from_state(state)
    intent = state["intent"].intent if state.get("intent") else None

    return routed, degrade_for_budget(
        path=EvidencePath(routed),
        intent=intent,
        seconds_remaining=guard.seconds_remaining,
        agentic_min_seconds=settings.agentic_min_seconds,
        hybrid_min_seconds=settings.hybrid_min_seconds,
    )


def evidence_stages(state: AgentState, route: str | None = None) -> list[str]:
    """The evidence stages a route needs, in the order they run."""
    route = ROUTE_ALIASES.get(route, route) if route else current_route(state)
    scopes = state.get("access_scopes") or []

    if route == EvidencePath.RAG.value:
        return [RAG_STAGE]

    if route == EvidencePath.NL2SQL.value:
        return [SQL_STAGE]

    if route == EvidencePath.HYBRID.value:
        return [RAG_STAGE, SQL_STAGE]

    if route == EvidencePath.HIGH_RISK_PANEL.value:
        # the panel reviews whatever this role may read: a source outside the grant is skipped,
        # not gathered and then hidden
        stages = [RAG_STAGE] if allowed_doc_types(scopes) else []

        if allowed_tables(scopes):
            stages.append(SQL_STAGE)

        return [*stages, PANEL_STAGE]

    return []


def next_stage(state: AgentState, after: str) -> str:
    stages = evidence_stages(state)

    if after in stages:
        index = stages.index(after)

        if index + 1 < len(stages):
            return stages[index + 1]

    return VALIDATION


def repair_attempts_allowed(state: AgentState) -> int:
    """How many reflection passes this request may take.

    A RAG repair searches the same documents again and an NL2SQL repair rewrites an answer
    over the same rows, so a second repair repeats the first one; a hybrid repair can switch
    source, so it keeps MAX_REFLECTION_RETRIES.
    """
    if current_evidence_path(state) in RAG_PATHS:
        return max(0, min(settings.rag_repair_attempts, settings.max_reflection_retries))

    if current_evidence_path(state) in SQL_PATHS:
        # the rows do not change between passes, so one rewrite of the answer is the only repair
        # that can change the verdict
        return max(0, min(settings.sql_repair_attempts, settings.max_reflection_retries))

    return settings.max_reflection_retries


def repair_token_estimate(state: AgentState) -> int:
    stages = evidence_stages(state) or [RAG_STAGE]

    return sum(
        NODE_TOKEN_ESTIMATE.get(name, 500) for name in ("reflection", "planner", *stages, VALIDATION)
    )


def best_retrieval_score(state: AgentState) -> float | None:
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        return None

    leading = chunks[:5]
    scored = [chunk.rerank_score for chunk in leading if chunk.rerank_score is not None]

    if scored:
        return max(scored)

    dense = [chunk.dense_score for chunk in leading if chunk.dense_score]

    return max(dense) if dense else None


def retrieval_found_nothing(state: AgentState) -> bool:
    """True when the documents this role can search do not answer the question at all."""
    best = best_retrieval_score(state)

    return best is not None and best <= RETRIEVAL_NO_MATCH_CEILING and state.get("sql_evidence") is None


def human_requested(state: AgentState) -> bool:
    from src.core.handoff import requested_human

    return requested_human(state.get("raw_query") or "")


def _is_high_risk(state: AgentState) -> bool:
    risk = state.get("risk")

    return bool(risk and risk.final_level is RiskLevel.HIGH)


def _stopped(state: AgentState) -> bool:
    return bool(state.get("escalation_reason")) or bool(state.get("budget_stops"))


def _edge(source: str, target: str, state: AgentState, reason: str = "", level: int = logging.DEBUG) -> str:
    logger.log(
        level,
        "edge %s -> %s request_id=%s%s",
        source,
        target,
        state.get("request_id"),
        f" reason={reason}" if reason else "",
    )

    return target


def after_guardrail(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.REFUSED.value:
        return _edge("input_guardrail", "refuse", state, state.get("refusal_reason") or "", logging.INFO)

    if _stopped(state):
        return _edge("input_guardrail", "escalate", state, _stop_reason(state), logging.INFO)

    return _edge("input_guardrail", "continue", state)


def after_intent(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.REFUSED.value:
        return _edge(
            "intent_classification", "refuse", state, state.get("refusal_reason") or "", logging.INFO
        )

    if _stopped(state):
        return _edge("intent_classification", "escalate", state, _stop_reason(state), logging.INFO)

    return _edge("intent_classification", "continue", state)


def after_entity_resolution(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.CLARIFICATION_REQUIRED.value:
        return _edge("entity_resolution", "clarify", state, "ambiguous or unknown entity", logging.INFO)

    if _stopped(state):
        return _edge("entity_resolution", "escalate", state, _stop_reason(state), logging.INFO)

    return _edge("entity_resolution", "continue", state)


def after_risk(state: AgentState) -> str:
    if _stopped(state):
        return _edge("risk_assessment", "escalate", state, _stop_reason(state), logging.INFO)

    return _edge("risk_assessment", "continue", state)


def route_evidence_path(state: AgentState) -> str:
    """The evidence route: rag, nl2sql, hybrid or high_risk_panel - or escalate / not_found."""
    if _stopped(state):
        return _edge("planner", "escalate", state, _stop_reason(state), logging.INFO)

    if _is_high_risk(state):
        return _edge("planner", EvidencePath.HIGH_RISK_PANEL.value, state, "risk is High", logging.INFO)

    routed = state.get("routed_path")

    if routed == ESCALATE:
        return _edge(
            "planner", "escalate", state, state.get("escalation_reason") or "path policy", logging.INFO
        )

    if routed == NO_ACCESS:
        reason = (state.get("path_decision") or {}).get("reason") or "the role cannot read the source"

        if human_requested(state):
            return _edge("planner", "escalate", state, f"{reason}; the user asked for a human", logging.INFO)

        return _edge("planner", "not_found", state, reason, logging.INFO)

    if routed is not None:
        # the planner already applied the clamps, the capability gates and the budget degrade and
        # recorded the result; every stage reads that same value (current_route). Deciding again here
        # on a later clock skipped rag_path on a hybrid question whenever the deadline crossed
        # HYBRID_MIN_SECONDS between the two reads.
        return _edge("planner", ROUTE_ALIASES.get(routed, routed), state, "route recorded by the planner", logging.INFO)

    # no recorded decision (an older checkpoint): route from the plan, degraded for the budget - the
    # same decision current_route() gives every stage
    routed, decision = _plan_route(state)

    if decision.clamped:
        logger.warning(
            "evidence path DEGRADED %s -> %s request_id=%s seconds_remaining=%.1f reason=%s",
            routed,
            decision.path,
            state.get("request_id"),
            guard_from_state(state).seconds_remaining,
            decision.reason,
        )

    return _edge("planner", decision.path, state, "" if decision.clamped else decision.reason, logging.INFO)


def after_planner(state: AgentState) -> str:
    """The first evidence stage the route needs (or escalate / not_found)."""
    route = route_evidence_path(state)

    if route in ("escalate", "not_found"):
        return route

    stages = evidence_stages(state, route) or [RAG_STAGE]

    logger.info(
        "evidence stages route=%s order=%s request_id=%s",
        route,
        " -> ".join([*stages, VALIDATION]),
        state.get("request_id"),
    )

    return stages[0]


def answer_not_in_documents(state: AgentState) -> bool:
    """True when the honest answer to a policy question is "I don't know" rather than a guess."""
    if current_route(state) != EvidencePath.RAG.value or _is_high_risk(state):
        return False

    if human_requested(state):
        return False

    if state.get("not_found_reason") in POLICY_NOT_FOUND:
        return True

    return retrieval_found_nothing(state)


def after_rag_path(state: AgentState) -> str:
    if _stopped(state):
        return _edge(RAG_STAGE, "escalate", state, _stop_reason(state), logging.INFO)

    if answer_not_in_documents(state):
        if state.get("not_found_reason") == NOT_FOUND_NOT_IN_EXTRACTS:
            reason = "the answer writer found nothing in the retrieved extracts that answers this"
        else:
            reason = (
                f"best retrieved extract scored {best_retrieval_score(state)}: "
                "nothing in the searchable documents answers this"
            )

        return _edge(RAG_STAGE, "not_found", state, reason, logging.INFO)

    return _edge(RAG_STAGE, next_stage(state, RAG_STAGE), state, f"route {current_route(state)}", logging.INFO)


def after_nl2sql_path(state: AgentState) -> str:
    if _stopped(state):
        return _edge(SQL_STAGE, "escalate", state, _stop_reason(state), logging.INFO)

    no_records = state.get("not_found_reason") in RECORDS_NOT_FOUND

    if current_route(state) in (EvidencePath.NL2SQL.value, EvidencePath.HYBRID.value) and no_records:
        failure = state.get("sql_failure") or "no rows"

        if human_requested(state):
            return _edge(
                SQL_STAGE,
                "escalate",
                state,
                f"the records path produced no evidence ({failure}) and the user asked for a human",
                logging.INFO,
            )

        return _edge(SQL_STAGE, "not_found", state, f"the records path produced no evidence: {failure}", logging.INFO)

    return _edge(SQL_STAGE, next_stage(state, SQL_STAGE), state, f"route {current_route(state)}", logging.INFO)


def after_panel(state: AgentState) -> str:
    if _stopped(state):
        return _edge(PANEL_STAGE, "escalate", state, _stop_reason(state), logging.INFO)

    return _edge(PANEL_STAGE, VALIDATION, state)


def after_validation(state: AgentState) -> str:
    if state.get("budget_stops"):
        return _edge(VALIDATION, "escalate", state, "budget stop", logging.INFO)

    validation = state.get("validation")

    if validation is None:
        return _edge(VALIDATION, "score", state, "no validation report")

    if validation.passed:
        return _edge(VALIDATION, "score", state)

    if validation.conflict_detected:
        # conflicting clauses (or a clause against a record) are the brief's own escalation trigger,
        # and the conflict check reads the retrieved extracts, not the draft - a repair cannot clear it
        return _edge(VALIDATION, "escalate", state, "validation detected a conflict between the sources", logging.INFO)

    if _is_high_risk(state):
        # a High-risk answer always goes to a human, and re-running the panel two more
        # times before that took 160-200s in the log; the reviewer gets the defects instead
        return _edge(
            VALIDATION,
            "escalate",
            state,
            "validation failed on a High-risk answer; the defects go to the human reviewer instead of a re-plan",
            logging.INFO,
        )

    if retrieval_found_nothing(state):
        if human_requested(state):
            return _edge(
                VALIDATION,
                "escalate",
                state,
                f"validation failed, the best retrieved extract scored {best_retrieval_score(state)} "
                "and the user asked for a human",
                logging.INFO,
            )

        # a re-plan cannot find evidence the searchable documents do not contain, and a human
        # cannot certify an answer nobody can source: the reply is an honest "I don't know"
        return _edge(
            VALIDATION,
            "not_found",
            state,
            f"validation failed and the best retrieved extract scored {best_retrieval_score(state)}; "
            "nothing in the searchable documents answers this",
            logging.INFO,
        )

    attempts = state.get("reflection_count", 0)
    allowed = repair_attempts_allowed(state)

    if attempts >= allowed:
        return _edge(
            VALIDATION,
            "escalate",
            state,
            f"validation failed and {attempts}/{allowed} repair attempts used",
            logging.INFO,
        )

    guard = guard_from_state(state)

    if guard.seconds_remaining < settings.repair_headroom_seconds:
        return _edge(
            VALIDATION,
            "escalate",
            state,
            f"validation failed with {guard.seconds_remaining:.2f}s left, under the "
            f"{settings.repair_headroom_seconds:.0f}s a repair needs",
            logging.INFO,
        )

    needed_tokens = repair_token_estimate(state)

    if guard.tokens_remaining < needed_tokens:
        return _edge(
            VALIDATION,
            "escalate",
            state,
            f"validation failed with {guard.tokens_remaining} tokens left, under the {needed_tokens} a repair needs",
            logging.INFO,
        )

    return _edge(
        VALIDATION,
        "reflect",
        state,
        f"validation failed, repair attempt {attempts + 1}/{allowed}",
        logging.INFO,
    )


def after_confidence(state: AgentState) -> str:
    confidence = state.get("confidence")
    validation = state.get("validation")
    panel = state.get("panel_verdict")

    if _stopped(state):
        return _edge("confidence_scoring", "escalate", state, _stop_reason(state), logging.INFO)

    if _is_high_risk(state):
        return _edge(
            "confidence_scoring", "escalate", state, "risk is High: human review is mandatory", logging.INFO
        )

    if validation and validation.conflict_detected:
        return _edge(
            "confidence_scoring", "escalate", state, "validation detected a source conflict", logging.INFO
        )

    if panel and panel.unresolved_conflict:
        return _edge("confidence_scoring", "escalate", state, "panel conflict unresolved", logging.INFO)

    if human_requested(state):
        return _edge(
            "confidence_scoring",
            "escalate",
            state,
            "the user explicitly asked for a human or legal review",
            logging.INFO,
        )

    if confidence is None or confidence.final_score < settings.confidence_threshold:
        score = confidence.final_score if confidence else None

        return _edge(
            "confidence_scoring",
            "escalate",
            state,
            f"confidence {score} below threshold {settings.confidence_threshold}",
            logging.INFO,
        )

    return _edge("confidence_scoring", "respond", state, f"confidence {confidence.final_score}", logging.INFO)


def after_output_guardrail(state: AgentState) -> str:
    if state.get("terminal_outcome") != TerminalOutcome.ESCALATED.value:
        return _edge("output_guardrail", "done", state, state.get("terminal_outcome") or "")

    if state.get("reviewer_decision"):
        return _edge("output_guardrail", "done", state, "already reviewed", logging.INFO)

    return _edge("output_guardrail", "escalate", state, state.get("escalation_reason") or "", logging.INFO)


def after_escalation(state: AgentState) -> str:
    if state.get("reviewer_decision") in RESUMABLE_DECISIONS:
        return _edge(
            "escalation_manager", "review", state, f"reviewer {state.get('reviewer_decision')}", logging.INFO
        )

    return _edge("escalation_manager", "wait", state, "awaiting human review", logging.INFO)


def _stop_reason(state: AgentState) -> str:
    if state.get("escalation_reason"):
        return str(state["escalation_reason"])

    stops = state.get("budget_stops") or []

    if stops:
        last = stops[-1]

        return f"budget stop at {last.get('node')}: {last.get('reason')}"

    return "stopped"
