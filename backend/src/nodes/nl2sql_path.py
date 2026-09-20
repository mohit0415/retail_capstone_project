"""Evidence stage 2 of every route that reads records: one vetted (or guarded generated) query.

It runs after rag_path, so on the hybrid route the clauses are already in the state and this stage
writes the reconciled answer; on the high-risk route it only gathers the rows for the panel; on the
nl2sql route it narrates the rows, or answers "I don't know" when the records produce no evidence.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for_long_output
from configs.settings import settings
from src.auth.rbac import allowed_risk_categories, allowed_tables
from src.core.budget import guard_from_state
from src.graph.routing import (
    HYBRID_REDRAFT,
    NOT_FOUND_NO_EVIDENCE,
    NOT_FOUND_NO_RECORDS,
    SQL_RENARRATE,
    current_route,
    human_requested,
    retrieval_found_nothing,
)
from src.graph.state import AgentState
from src.nodes.hybrid_path import draft_hybrid_answer
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import SQL_NARRATION
from src.schemas.enums import EvidencePath
from src.schemas.models import DraftAnswer, SqlEvidence
from src.sqlpath.disclosure import rows_for_prompt, with_disclosures
from src.sqlpath.executor import SqlPolicyError, run_vetted_sql, sanity_check

logger = logging.getLogger(__name__)

# the narration model sees every row it reports (up to this cap) with the row count in front;
# it used to see 25 rows of a 31-row result and could not name the other six
NARRATION_MAX_ROWS = 60

# narrating a wide result ("list all 60 records with 8 columns") is a completion of thousands
# of tokens; the default per-call timeout aborted every attempt mid-stream. One attempt runs
# with the remaining route budget (up to this ceiling), keeping this much back for validation.
NARRATION_TIMEOUT_CEILING_SECONDS = 75.0
NARRATION_TIMEOUT_FLOOR_SECONDS = 15.0
NARRATION_VALIDATION_RESERVE_SECONDS = 15.0

# A hybrid or high-risk question is often two questions in one: "how often must a Low risk vendor be
# monitored under the vendor policy, and which of our vendors are Low risk?". The records query gets
# only the records half. Given the whole question, the template selector answered "the question asks
# about policy requirements, which are not stored in the database", no vetted query was used, and the
# generated fallback returned 21 rows on one pass and 28 on the next.
_PART_BREAK = re.compile(
    r"\s*[,;]\s*(?:and|but|also|plus)\s+|\s+and\s+(?=(?:which|what|how many|list|show|who|are|is|do|does)\b)|\?\s+",
    re.I,
)

_RECORDS_CUE = re.compile(r"\b(?:which|list|show|how many|count|our|we hold|do we have|are there|currently)\b", re.I)

_POLICY_CUE = re.compile(
    r"\b(?:polic(?:y|ies)|must|shall|required?|requires|how often|how long|what does|according to|under the)\b", re.I
)


def records_part(question: str) -> str:
    """The half of a policy-and-records question that the records query answers (else the whole question)."""
    parts = [part.strip(" ,;.?") for part in _PART_BREAK.split(question or "")]
    parts = [part for part in parts if part]

    if len(parts) < 2:
        return question

    records = [part for part in parts if _RECORDS_CUE.search(part) and not _POLICY_CUE.search(part)]

    if not records:
        return question

    return " and ".join(records) + "?"


def _records_state(state: AgentState) -> AgentState:
    """The state the records query runs on: the same, asking only the records half of the question."""
    question = state["standalone_query"]
    part = records_part(question)

    if part == question:
        return state

    logger.info("records stage asks only the records half request_id=%s question=%r", state.get("request_id"), part)

    return {**state, "standalone_query": part}


def _entity_hints(state: AgentState) -> str:
    hints = []

    for entity in state.get("resolved_entities", []):
        if entity.resolved_id is not None:
            hints.append(f"{entity.entity_type} \"{entity.surface_form}\" is id {entity.resolved_id}")
        elif entity.canonical_name:
            hints.append(f"{entity.entity_type} \"{entity.surface_form}\" is \"{entity.canonical_name}\"")

    if not hints:
        return ""

    return "\n- ".join(["", *hints]).strip()


def _presets(state: AgentState) -> dict:
    for entity in state.get("resolved_entities", []):
        if entity.entity_type == "vendor" and entity.resolved_id is not None:
            return {"vendor_id": entity.resolved_id}

    return {}


def run_sql_evidence(state: AgentState):
    scopes = state.get("access_scopes", [])
    tables = allowed_tables(scopes)

    if not tables:
        logger.info("sql evidence skipped: role %s has no table in scope", state.get("role"))

        return None, "this role has no access to the compliance database"

    try:
        evidence = run_vetted_sql(
            question=state["standalone_query"],
            allowed_tables=tables,
            departments=state.get("departments", []),
            as_of=settings.as_of_date,
            risk_categories=allowed_risk_categories(scopes),
            presets=_presets(state),
            entity_hints=_entity_hints(state),
            config=runnable_config(state, "sql_template_selector"),
        )
    except SqlPolicyError as exc:
        logger.warning("vetted sql refused: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.error("vetted sql failed: %s", exc)
        return None, "the database probe could not be completed"

    return evidence, ""


def is_sql_repair_pass(state: AgentState) -> bool:
    """A repair of a records answer the validator rejected: the rows are kept, the answer is rewritten.

    Running the query again returned the same rows, the selector and the narration came back from
    the LLM cache, and the validator rejected the same draft twice more (72 s, then escalated).
    """
    return (
        state.get("repair_strategy") == SQL_RENARRATE
        and bool(state.get("reflection_count"))
        and bool(state.get("replan_directive"))
        and state.get("sql_evidence") is not None
    )


def narrate_sql(state: AgentState, evidence: SqlEvidence, repair_directive: str = "") -> DraftAnswer:
    caveats = sanity_check(evidence)

    prompt = render_prompt("SQL_NARRATION", SQL_NARRATION,
        query=state["standalone_query"],
        template_id=evidence.template_id,
        as_of=evidence.as_of,
        row_count=evidence.row_count,
        rows=rows_for_prompt(evidence, NARRATION_MAX_ROWS),
        statement=evidence.statement,
        caveats="\n".join(f"- {item}" for item in caveats) or "- none",
    )

    messages = [SystemMessage(content=prompt)]

    if repair_directive:
        messages.append(SystemMessage(content=repair_directive))

    messages.append(HumanMessage(content=state["standalone_query"]))

    remaining = guard_from_state(state).seconds_remaining
    timeout = min(
        NARRATION_TIMEOUT_CEILING_SECONDS,
        max(NARRATION_TIMEOUT_FLOOR_SECONDS, remaining - NARRATION_VALIDATION_RESERVE_SECONDS),
    )
    model = model_for_long_output("nl2sql_intent", timeout).with_structured_output(DraftAnswer)

    draft: DraftAnswer = model.invoke(messages, config=runnable_config(state, "sql_narration"))

    answer, added = with_disclosures(evidence, draft.answer)

    if added:
        logger.info(
            "sql narration left out %d caveat(s), appended them to the answer request_id=%s caveats=%s",
            len(added),
            state.get("request_id"),
            added,
        )
        draft = draft.model_copy(update={"answer": answer})

    return draft


def _log_evidence(state: AgentState, evidence: SqlEvidence, route: str, repair: bool = False) -> None:
    logger.info(
        "sql evidence request_id=%s route=%s template=%s selection=%s generated=%s rows=%d truncated=%s "
        "scope_filtered=%d scope=%r repair=%s",
        state.get("request_id"),
        route,
        evidence.template_id,
        evidence.selection or "unknown",
        evidence.generated,
        evidence.row_count,
        evidence.truncated,
        evidence.rows_filtered_by_scope,
        evidence.scope_note,
        repair,
    )


def _no_policy_evidence(chunks) -> bool:
    return not chunks or retrieval_found_nothing({"retrieved_chunks": chunks, "sql_evidence": None})


def _records_for_panel(state: AgentState) -> dict:
    """High-risk route: the rows are gathered for the panel, which writes the answer."""
    evidence, failure = run_sql_evidence(_records_state(state))
    route = EvidencePath.HIGH_RISK_PANEL.value

    if evidence is not None:
        _log_evidence(state, evidence, route)
    else:
        logger.warning(
            "records stage produced no evidence for the panel request_id=%s failure=%s - the data "
            "verifier will say the data is silent",
            state.get("request_id"),
            failure,
        )

    return {"evidence_path": route, "sql_evidence": evidence, "sql_failure": failure, "tokens_spent": 600}


def _reconcile_with_policy(state: AgentState) -> dict:
    """Hybrid route: rag_path already gathered the clauses; the records are read and both reconciled."""
    route = EvidencePath.HYBRID.value
    chunks = list(state.get("retrieved_chunks") or [])
    repair = (
        state.get("repair_strategy") == HYBRID_REDRAFT
        and bool(state.get("reflection_count"))
        and state.get("sql_evidence") is not None
    )

    if repair:
        # the rows do not change between passes; the answer is rewritten over the same rows
        evidence, failure = state["sql_evidence"], ""
    else:
        evidence, failure = run_sql_evidence(_records_state(state))

    if evidence is not None:
        _log_evidence(state, evidence, route, repair)
    elif _no_policy_evidence(chunks) and not human_requested(state):
        # neither source has anything: a reconciled answer could only be a guess about both
        logger.info(
            "hybrid route found no matching clause and no record request_id=%s failure=%s -> honest 'I don't know'",
            state.get("request_id"),
            failure,
        )

        return {
            "evidence_path": route,
            "sql_evidence": None,
            "sql_failure": failure,
            "not_found_reason": NOT_FOUND_NO_EVIDENCE,
            "tokens_spent": 600,
        }
    else:
        logger.warning(
            "records stage produced no evidence on the hybrid route request_id=%s failure=%s - the answer "
            "is written from the %d clause(s) and says no record was available",
            state.get("request_id"),
            failure,
            len(chunks),
        )

    draft, decision, _caveats = draft_hybrid_answer(state, chunks, evidence, failure, NARRATION_MAX_ROWS)

    return {
        "evidence_path": route,
        "sql_evidence": evidence,
        "sql_failure": failure,
        "draft": draft,
        "model_routing": [decision.as_dict()],
        "tokens_spent": 4000,
    }


def _fallback_narration(evidence: SqlEvidence) -> DraftAnswer:
    """A deterministic reading of the rows for when the narration model does not answer.

    The rows are already fetched and scope-filtered; failing the whole request over a
    narration timeout would discard real evidence. This states the count and the leading
    rows verbatim - no interpretation, so nothing can be hallucinated - and the request
    is released as degraded instead of raising a 500.
    """
    shown = evidence.rows[:5]
    listed = "; ".join(", ".join(f"{key}={value}" for key, value in row.items()) for row in shown)

    answer = (
        f"As of {evidence.as_of}, the compliance records return {evidence.row_count} row(s) "
        "for this question."
    )

    if listed:
        answer += f" Leading rows: {listed}."

    if evidence.row_count > len(shown):
        # no arithmetic here: "55 further rows" is a figure the deterministic sanity check
        # cannot find in the rows, and it sent the first fallback round a repair loop
        answer += " The remaining rows are not listed here; the row count above is the full total."

    answer, _ = with_disclosures(evidence, answer)

    return DraftAnswer(
        answer=answer,
        uncertainty_note=(
            "the narration model did not respond in time; this is a direct, unnarrated "
            "reading of the query result"
        ),
    )


def _records_answer(state: AgentState) -> dict:
    """nl2sql route: the rows are the answer."""
    repair = is_sql_repair_pass(state)

    if repair:
        evidence, failure = state["sql_evidence"], ""

        logger.info(
            "nl2sql repair pass: rows kept, answer rewritten with the validator's defects "
            "request_id=%s template=%s rows=%d",
            state.get("request_id"),
            evidence.template_id,
            evidence.row_count,
        )
    else:
        evidence, failure = run_sql_evidence(state)

    if evidence is None:
        # nothing to report is an honest "I don't know" (src/nodes/no_answer.py), not a review:
        # a reviewer could only confirm that the records say nothing about it
        logger.warning(
            "nl2sql path produced no evidence request_id=%s failure=%s", state.get("request_id"), failure
        )

        return {
            "evidence_path": EvidencePath.NL2SQL.value,
            "sql_failure": failure,
            "not_found_reason": NOT_FOUND_NO_RECORDS,
            "draft": DraftAnswer(
                answer=f"This question could not be answered from the compliance records: {failure}.",
                uncertainty_note="no database evidence was produced",
                answer_found=False,
            ),
            "tokens_spent": 900,
        }

    _log_evidence(state, evidence, EvidencePath.NL2SQL.value, repair)

    degraded = False

    try:
        draft = narrate_sql(state, evidence, repair_directive=state.get("replan_directive", "") if repair else "")
    except Exception:
        # an Azure timeout here used to raise out of the graph and turn the whole /ask
        # into a 500, with the rows already fetched; the deterministic reading releases
        # the evidence instead
        logger.warning(
            "sql narration failed request_id=%s template=%s rows=%d - releasing the deterministic "
            "row reading as a degraded answer",
            state.get("request_id"),
            evidence.template_id,
            evidence.row_count,
            exc_info=True,
        )
        draft = _fallback_narration(evidence)
        degraded = True

    return {
        "evidence_path": EvidencePath.NL2SQL.value,
        "sql_evidence": evidence,
        "sql_failure": "",
        "draft": draft,
        "degraded": degraded or state.get("degraded", False),
        "tokens_spent": 1500,
    }


@traced_node("nl2sql_path")
def nl2sql_path_node(state: AgentState) -> dict:
    route = current_route(state)

    if route == EvidencePath.HIGH_RISK_PANEL.value:
        return _records_for_panel(state)

    if route == EvidencePath.HYBRID.value:
        return _reconcile_with_policy(state)

    return _records_answer(state)
