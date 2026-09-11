import logging

from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from configs.settings import settings
from src.auth.rbac import allowed_doc_types, allowed_tables
from src.core.budget import guard_from_state, widen_for_path
from src.graph.path_policy import (
    NO_ACCESS,
    RECORDS,
    PathDecision,
    allowed_path_names,
    degrade_for_budget,
    resolve_path,
    rule_for,
)
from src.graph.routing import NOT_FOUND_NO_ACCESS
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import PLANNER
from src.schemas.enums import EntityStatus, EvidencePath, RiskLevel
from src.schemas.models import EvidencePlan, PlanStep
from src.sqlpath.selector import match_simple_template
from src.sqlpath.templates import templates_visible_to

logger = logging.getLogger(__name__)

SOURCE_FOR_EVIDENCE = {"policy": "policy_kb", "records": "compliance_db", "both": "both"}


def _fallback_plan(state: AgentState, path: EvidencePath) -> EvidencePlan:
    rule = rule_for(state["intent"].intent if state.get("intent") else None)

    return EvidencePlan(
        path=path,
        steps=[
            PlanStep(
                order=1,
                source=SOURCE_FOR_EVIDENCE[rule.evidence],
                objective="gather the evidence the question turns on",
                must_prove="the answer is supported by a retrieved clause or a returned row",
            )
        ],
        required_claims=[],
        revision=state.get("plan").revision + 1 if state.get("plan") else 0,
    )


SOURCE_FOR_PATH = {
    EvidencePath.RAG.value: "policy_kb",
    EvidencePath.NL2SQL.value: "compliance_db",
    EvidencePath.HYBRID.value: "both",
    EvidencePath.AGENTIC.value: "both",
    EvidencePath.HIGH_RISK_PANEL.value: "both",
}


def reachable_paths(intent, risk_level: RiskLevel | None, scopes: list[str]) -> set[str]:
    """Every path the router could end on for this intent, risk level and role.

    When this is a single path, asking the planner LLM which path to take cannot
    change anything: whatever it proposes is clamped back to the same place. On a
    slow Azure deployment that call alone was 6-11 seconds.
    """
    rule = rule_for(intent)
    outcomes = set()

    for proposed in rule.allowed:
        decision = resolve_path(
            intent=intent,
            risk_level=risk_level,
            proposed=proposed,
            has_tables=bool(allowed_tables(scopes)),
            has_documents=bool(allowed_doc_types(scopes)),
            agentic_enabled=settings.enable_agentic_path,
        )
        outcomes.add(decision.path)

    return outcomes


def _deterministic_plan(state: AgentState, intent, only_path: str) -> EvidencePlan:
    rule = rule_for(intent)

    # the plan proposes the intent's default path, so resolve_path applies exactly the
    # clamps, capability gates and partial-evidence flag it applied to an LLM plan;
    # the step source follows the one path that is actually reachable
    return EvidencePlan(
        path=rule.default,
        steps=[
            PlanStep(
                order=1,
                source=SOURCE_FOR_PATH.get(only_path, SOURCE_FOR_EVIDENCE[rule.evidence]),
                objective=f"gather the evidence for: {state['standalone_query'][:160]}",
                must_prove="every claim in the answer is backed by a retrieved clause or a returned row",
            )
        ],
        required_claims=[],
        revision=0,
    )


def vetted_record_match(state: AgentState, intent, reachable: set[str]) -> str | None:
    """The vetted template that answers a plain records question outright, or None.

    "how many vendors are approved and their names" is a VENDOR_STATUS question, which admits
    nl2sql and hybrid, so the planner model was asked - 17.5 s on the slow deployment - and chose
    nl2sql, as it must when a single vetted query answers the question and no policy word is in it.
    """
    if EvidencePath.NL2SQL.value not in reachable or rule_for(intent).evidence != RECORDS:
        return None

    scopes = state.get("access_scopes", [])
    named_vendor = any(
        entity.entity_type == "vendor" and entity.resolved_id is not None
        for entity in state.get("resolved_entities", [])
    )

    matched = match_simple_template(
        state.get("standalone_query") or "",
        templates_visible_to(allowed_tables(scopes)),
        named_vendor=named_vendor,
    )

    return matched[0].template_id if matched else None


def _record_plan(template_id: str) -> EvidencePlan:
    return EvidencePlan(
        path=EvidencePath.NL2SQL,
        steps=[
            PlanStep(
                order=1,
                source="compliance_db",
                objective=f"read the records with the vetted query {template_id}",
                must_prove="every count and name in the answer comes from a returned row",
            )
        ],
        required_claims=[],
        revision=0,
    )


def _unresolved_surfaces(state: AgentState) -> list[str]:
    return [
        entity.surface_form
        for entity in state.get("resolved_entities", [])
        if entity.status is not EntityStatus.EXACT
    ]


def _draft_plan(state: AgentState, intent, forced_path: EvidencePath) -> EvidencePlan:
    scopes = state.get("access_scopes", [])
    rule = rule_for(intent)

    prompt = render_prompt("PLANNER", PLANNER,
        risk_level=state["risk"].final_level.value,
        allowed_docs=", ".join(allowed_doc_types(scopes)) or "none",
        allowed_tables=", ".join(sorted(allowed_tables(scopes))) or "none",
        unresolved=", ".join(_unresolved_surfaces(state)) or "none",
        allowed_paths=", ".join(allowed_path_names(intent)),
        default_path=rule.default.value,
        path_rationale=rule.rationale,
        revision_context=state.get("replan_directive", ""),
        query=state["standalone_query"],
    )

    model = model_for("planner").with_structured_output(EvidencePlan)

    try:
        return model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
            config=runnable_config(state, "planner"),
        )
    except Exception:
        logger.warning(
            "planner LLM draft failed, using deterministic fallback plan request_id=%s",
            state.get("request_id"),
            exc_info=True,
        )

        return _fallback_plan(state, forced_path)


def _within_budget(state: AgentState, widened: dict, intent, decision: PathDecision) -> tuple[str, str, bool]:
    """The route after budget degradation, decided once here so every evidence stage reads the same one."""
    guard = guard_from_state({**state, **widened})

    budget = degrade_for_budget(
        path=EvidencePath(decision.path),
        intent=intent,
        seconds_remaining=guard.seconds_remaining,
        agentic_min_seconds=settings.agentic_min_seconds,
        hybrid_min_seconds=settings.hybrid_min_seconds,
    )

    if not budget.clamped:
        return decision.path, decision.reason, decision.partial_evidence

    logger.warning(
        "evidence path DEGRADED %s -> %s request_id=%s seconds_remaining=%.1f reason=%s",
        decision.path,
        budget.path,
        state.get("request_id"),
        guard.seconds_remaining,
        budget.reason,
    )

    return budget.path, f"{decision.reason}; {budget.reason}", True


@traced_node("planner")
def planner_node(state: AgentState) -> dict:
    risk = state["risk"]
    intent = state["intent"].intent if state.get("intent") else None
    previous = state.get("plan")
    replanned = bool(state.get("plan_from_reflection"))

    scopes = state.get("access_scopes", [])
    planner_mode = "llm"

    if replanned and previous is not None:
        plan = previous
        tokens = 0
        planner_mode = "reflection"
    else:
        reachable = reachable_paths(intent, risk.final_level, scopes)

        if settings.skip_planner_for_single_path and len(reachable) == 1:
            only_path = next(iter(reachable))
            plan = _deterministic_plan(state, intent, only_path)
            tokens = 0
            planner_mode = "deterministic"

            logger.info(
                "planner LLM skipped request_id=%s intent=%s risk=%s reason=only one path is possible (%s)",
                state.get("request_id"),
                intent.value if intent else "unknown",
                risk.final_level.value,
                only_path,
            )
        elif settings.skip_planner_for_single_path and (template_id := vetted_record_match(state, intent, reachable)):
            plan = _record_plan(template_id)
            tokens = 0
            planner_mode = "vetted_query"

            logger.info(
                "planner LLM skipped request_id=%s intent=%s risk=%s reason=the vetted query %s answers "
                "this records question directly",
                state.get("request_id"),
                intent.value if intent else "unknown",
                risk.final_level.value,
                template_id,
            )
        else:
            plan = _draft_plan(state, intent, rule_for(intent).default)
            tokens = 800

    decision = resolve_path(
        intent=intent,
        risk_level=risk.final_level,
        proposed=plan.path,
        has_tables=bool(allowed_tables(scopes)),
        has_documents=bool(allowed_doc_types(scopes)),
        agentic_enabled=settings.enable_agentic_path,
    )

    logger.info(
        "route intent=%s proposed=%s chosen=%s clamped=%s replanned=%s reason=%s request_id=%s",
        intent.value if intent else "unknown",
        plan.path.value,
        decision.path,
        decision.clamped,
        replanned,
        decision.reason,
        state.get("request_id"),
    )

    if decision.path == NO_ACCESS:
        # the only source that answers this is outside the role's grant; a reviewer could not hand
        # this role the data either, so the reply says so plainly instead of queueing a review
        logger.info(
            "route NO_ACCESS request_id=%s reason=%s -> honest reply, not an escalation",
            state.get("request_id"),
            decision.reason,
        )

        return {
            "plan": plan,
            "routed_path": NO_ACCESS,
            "path_decision": {
                "path": NO_ACCESS,
                "reason": decision.reason,
                "clamped": True,
                "planner": planner_mode,
            },
            "not_found_reason": NOT_FOUND_NO_ACCESS,
            "plan_from_reflection": False,
            "tokens_spent": tokens,
        }

    widened = widen_for_path(state, decision.path)
    route, reason, partial = _within_budget(state, widened, intent, decision)

    plan.path = EvidencePath(route)
    plan.revision = previous.revision + 1 if previous and not replanned else plan.revision

    result = {
        "plan": plan,
        "routed_path": route,
        "path_decision": {
            "path": route,
            "reason": reason,
            "clamped": decision.clamped or route != decision.path,
            "partial_evidence": partial,
            "replanned": replanned,
            "planner": planner_mode,
        },
        "plan_from_reflection": False,
        "degraded": partial or state.get("degraded", False),
        "tokens_spent": tokens,
    }
    result.update(widened)

    return result
