from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import QUERY_REWRITE
from src.schemas.models import RewriteResult

RECENT_TURN_WINDOW = 5


def _format_turns(history: list[dict]) -> str:
    window = history[-RECENT_TURN_WINDOW:]

    if not window:
        return "(no earlier turns)"

    return "\n".join(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in window)


@traced_node("query_rewrite")
def query_rewrite_node(state: AgentState) -> dict:
    history = state.get("conversation_history", [])
    query = state["sanitised_query"]

    if not history:
        return {"standalone_query": query}

    prompt = QUERY_REWRITE.format(
        thread_summary=state.get("thread_summary") or "(none)",
        recent_turns=_format_turns(history),
        current_turn=query,
    )

    model = model_for("query_rewrite").with_structured_output(RewriteResult)

    result: RewriteResult = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=query)],
        config=runnable_config(state, "query_rewrite"),
    )

    rewritten = result.standalone_query.strip() or query

    return {
        "standalone_query": rewritten,
        "tokens_spent": 600,
    }
