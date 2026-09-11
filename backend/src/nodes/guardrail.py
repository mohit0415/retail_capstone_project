import logging

from src.graph.state import AgentState
from src.guardrails.input_guard import run_input_guardrail
from src.observability.tracing import traced_node
from src.schemas.enums import TerminalOutcome

logger = logging.getLogger(__name__)


@traced_node("input_guardrail")
def input_guardrail_node(state: AgentState) -> dict:
    outcome = run_input_guardrail(state["raw_query"])

    if outcome.blocked:
        logger.warning(
            "input guardrail BLOCKED request_id=%s user=%s role=%s reason=%s pii_entities=%s",
            state.get("request_id"),
            state.get("user_id"),
            state.get("role"),
            outcome.refusal_reason,
            outcome.pii_entities,
        )

        return {
            "sanitised_query": outcome.sanitised_query,
            "terminal_outcome": TerminalOutcome.REFUSED.value,
            "refusal_reason": outcome.refusal_reason,
        }

    logger.info(
        "input guardrail passed request_id=%s pii_redacted=%s risk_floor=%s risk_keywords=%s query_chars=%d",
        state.get("request_id"),
        outcome.pii_entities or "none",
        outcome.risk_floor,
        outcome.risk_keywords or "none",
        len(outcome.sanitised_query),
    )

    return {"sanitised_query": outcome.sanitised_query}
