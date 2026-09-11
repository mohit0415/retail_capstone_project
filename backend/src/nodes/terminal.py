import logging

from src.core.audit import audit_from_state
from src.graph.state import AgentState
from src.guardrails.output_guard import run_output_guardrail
from src.observability.tracing import traced_node
from src.retrieval.citations import citable_citation_texts
from src.schemas.enums import TerminalOutcome

logger = logging.getLogger(__name__)


@traced_node("output_guardrail")
def output_guardrail_node(state: AgentState) -> dict:
    draft = state.get("draft")

    if draft is None:
        logger.warning("output guardrail: no draft to release request_id=%s", state.get("request_id"))

        result = {
            "terminal_outcome": TerminalOutcome.ESCALATED.value,
            "escalation_reason": "there was no drafted answer to release",
        }

        audit_from_state({**state, **result}, node="output_guardrail", event="rejected")

        return result

    chunks = state.get("retrieved_chunks") or []

    # a clause cited from inside an extract (§4.1 in the text of a §4 extract) was retrieved too
    allowed = citable_citation_texts(chunks)

    # an answer written only from database rows has no clause to cite - it names the query and the
    # as-of date - and demanding one rejected every records answer ("no clause citation"). The
    # "I don't know" reply cites nothing either. A reviewer who accepts or edits an answer takes
    # responsibility for it, so a plain-text edit is released; the PII scrub still runs and a
    # citation of a clause that was never retrieved is still refused.
    record_only = state.get("sql_evidence") is not None and not chunks
    reviewed = state.get("reviewer_decision") in ("accept", "edit")
    require_citations = not record_only and not reviewed and not state.get("answer_not_found", False)

    outcome = run_output_guardrail(draft.answer, allowed, require_citations=require_citations)

    if outcome.enforcement_failed:
        logger.warning(
            "output guardrail REJECTED request_id=%s reason=%s pii_removed=%s citations=%d after_review=%s",
            state.get("request_id"),
            outcome.failure_reason,
            outcome.pii_removed or "none",
            len(outcome.citations_found),
            bool(state.get("reviewer_decision")),
        )

        result = {
            "terminal_outcome": TerminalOutcome.ESCALATED.value,
            "escalation_reason": f"output guardrail rejected the answer: {outcome.failure_reason}",
        }

        audit_from_state(
            {**state, **result},
            node="output_guardrail",
            event="rejected",
            detail={
                "failure_reason": outcome.failure_reason,
                "pii_removed": outcome.pii_removed,
                "after_human_review": bool(state.get("reviewer_decision")),
            },
        )

        return result

    cleaned = draft.model_copy(update={"answer": outcome.answer, "cited_clauses": outcome.citations_found})

    logger.info(
        "output guardrail RELEASED request_id=%s path=%s citation_coverage=%.2f citations=%d pii_removed=%s "
        "after_review=%s answer_chars=%d",
        state.get("request_id"),
        state.get("evidence_path"),
        outcome.citation_coverage,
        len(outcome.citations_found),
        outcome.pii_removed or "none",
        bool(state.get("reviewer_decision")),
        len(outcome.answer),
    )

    result = {
        "draft": cleaned,
        "terminal_outcome": TerminalOutcome.ANSWERED.value,
        "escalation_reason": None,
    }

    audit_from_state(
        {**state, **result},
        node="output_guardrail",
        event="answered",
        detail={
            "citation_coverage": outcome.citation_coverage,
            "pii_removed": outcome.pii_removed,
            "evidence_path": state.get("evidence_path"),
            "after_human_review": bool(state.get("reviewer_decision")),
        },
    )

    return result


@traced_node("safe_refusal")
def refusal_node(state: AgentState) -> dict:
    logger.info("request REFUSED request_id=%s reason=%s", state.get("request_id"), state.get("refusal_reason"))

    result = {"terminal_outcome": TerminalOutcome.REFUSED.value}

    audit_from_state(
        {**state, **result},
        node="safe_refusal",
        event="refused",
        detail={"reason": state.get("refusal_reason")},
    )

    return result


@traced_node("clarification")
def clarification_node(state: AgentState) -> dict:
    logger.info(
        "request needs CLARIFICATION request_id=%s question=%r",
        state.get("request_id"),
        state.get("clarification_question"),
    )

    result = {"terminal_outcome": TerminalOutcome.CLARIFICATION_REQUIRED.value}

    audit_from_state(
        {**state, **result},
        node="clarification",
        event="clarification_requested",
        detail={"question": state.get("clarification_question")},
    )

    return result
