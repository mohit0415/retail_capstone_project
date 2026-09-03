from src.graph.state import AgentState
from src.guardrails.input_guard import run_input_guardrail
from src.observability.tracing import traced_node
from src.schemas.enums import TerminalOutcome


@traced_node("input_guardrail")
def input_guardrail_node(state: AgentState) -> dict:
    outcome = run_input_guardrail(state["raw_query"])

    if outcome.blocked:
        return {
            "sanitised_query": outcome.sanitised_query,
            "terminal_outcome": TerminalOutcome.REFUSED.value,
            "refusal_reason": outcome.refusal_reason,
        }

    return {"sanitised_query": outcome.sanitised_query}
