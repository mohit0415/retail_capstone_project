from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import INTENT_CLASSIFICATION
from src.schemas.enums import Intent, TerminalOutcome
from src.schemas.models import IntentResult


@traced_node("intent_classification")
def intent_classification_node(state: AgentState) -> dict:
    query = state["standalone_query"]

    prompt = INTENT_CLASSIFICATION.format(query=query)
    model = model_for("intent_classification").with_structured_output(IntentResult)

    result: IntentResult = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=query)],
        config=runnable_config(state, "intent_classification"),
    )

    if result.intent is Intent.OUT_OF_SCOPE:
        return {
            "intent": result,
            "terminal_outcome": TerminalOutcome.REFUSED.value,
            "refusal_reason": "the question is not a retail policy or compliance matter",
            "tokens_spent": 500,
        }

    return {"intent": result, "tokens_spent": 500}
