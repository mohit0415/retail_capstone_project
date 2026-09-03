from src.core.audit import audit_from_state
from src.graph.state import AgentState
from src.guardrails.output_guard import run_output_guardrail
from src.observability.tracing import traced_node
from src.schemas.enums import TerminalOutcome


@traced_node("output_guardrail")
def output_guardrail_node(state: AgentState) -> dict:
    draft = state["draft"]

    allowed = [f"{chunk.document_title} §{chunk.clause_number}" for chunk in state.get("retrieved_chunks", [])]

    outcome = run_output_guardrail(draft.answer, allowed)

    if outcome.enforcement_failed:
        result = {
            "terminal_outcome": TerminalOutcome.ESCALATED.value,
            "escalation_reason": f"output guardrail rejected the answer: {outcome.failure_reason}",
        }

        audit_from_state({**state, **result}, node="output_guardrail", event="rejected")

        return result

    cleaned = draft.model_copy(update={"answer": outcome.answer, "cited_clauses": outcome.citations_found})

    result = {
        "draft": cleaned,
        "terminal_outcome": TerminalOutcome.ANSWERED.value,
    }

    audit_from_state(
        {**state, **result},
        node="output_guardrail",
        event="answered",
        detail={
            "citation_coverage": outcome.citation_coverage,
            "pii_removed": outcome.pii_removed,
            "evidence_path": state.get("evidence_path"),
        },
    )

    return result


@traced_node("safe_refusal")
def refusal_node(state: AgentState) -> dict:
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
    result = {"terminal_outcome": TerminalOutcome.CLARIFICATION_REQUIRED.value}

    audit_from_state(
        {**state, **result},
        node="clarification",
        event="clarification_requested",
        detail={"question": state.get("clarification_question")},
    )

    return result
