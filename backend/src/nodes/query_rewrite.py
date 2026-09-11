import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from configs.llms import model_for
from configs.settings import settings
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import QUERY_REWRITE
from src.schemas.models import RewriteResult

logger = logging.getLogger(__name__)


# words that only make sense with the earlier turns in view ("does that apply...", "the same for...")
FOLLOW_UP_MARKERS = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|same|such|above|previous|previously|earlier|"
    r"mentioned|former|latter|also|too|else|again|instead|then|there|one|ones|he|she|him|her)\b",
    re.I,
)

FOLLOW_UP_OPENERS = re.compile(r"^\s*(and|but|so|or|what about|how about|what if|ok|okay|also)\b", re.I)

MIN_STANDALONE_WORDS = 7


def reads_as_standalone(query: str) -> bool:
    """True when a follow-up already names everything it asks about.

    The rewrite call exists to resolve pronouns and ellipsis from earlier turns;
    for a complete question it returns the text unchanged, and on the slow Azure
    deployment that no-op call took 7-19 seconds. Anything that looks like it
    leans on context still goes to the model.
    """
    words = re.findall(r"[A-Za-z0-9']+", query or "")

    if len(words) < MIN_STANDALONE_WORDS:
        return False

    return not (FOLLOW_UP_OPENERS.search(query) or FOLLOW_UP_MARKERS.search(query))


def _format_turns(history: list[dict]) -> str:
    window = history[-settings.conversation_rewrite_turns :]

    if not window:
        return "(no earlier turns)"

    return "\n".join(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in window)


@traced_node("query_rewrite")
def query_rewrite_node(state: AgentState) -> dict:
    history = state.get("conversation_history", [])
    query = state["sanitised_query"]

    if not history:
        logger.debug("query rewrite skipped: no conversation history request_id=%s", state.get("request_id"))

        return {"standalone_query": query, "rewrite_mode": "first_turn"}

    if settings.skip_rewrite_for_standalone and reads_as_standalone(query):
        logger.info(
            "query rewrite skipped: the follow-up already reads as a full question request_id=%s history_turns=%d",
            state.get("request_id"),
            len(history),
        )

        return {"standalone_query": query, "rewrite_mode": "standalone"}

    prompt = render_prompt(
        "QUERY_REWRITE",
        QUERY_REWRITE,
        thread_summary=state.get("thread_summary") or "(none)",
        recent_turns=_format_turns(history),
        current_turn=query,
    )

    chain = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), ("human", "{query}")]
    ) | model_for("query_rewrite").with_structured_output(RewriteResult)

    result: RewriteResult = chain.invoke(
        {"system_prompt": prompt, "query": query},
        config=runnable_config(state, "query_rewrite"),
    )

    rewritten = result.standalone_query.strip() or query

    if rewritten != query:
        logger.info(
            "query rewritten request_id=%s history_turns=%d summary=%s follow_up=%s carried=%s rewritten=%r",
            state.get("request_id"),
            len(history),
            bool(state.get("thread_summary")),
            result.is_follow_up,
            result.carried_entities,
            _preview(rewritten),
        )
    else:
        logger.info(
            "query unchanged by rewrite request_id=%s history_turns=%d follow_up=%s",
            state.get("request_id"),
            len(history),
            result.is_follow_up,
        )

    return {
        "standalone_query": rewritten,
        "rewrite_mode": "rewritten" if rewritten != query else "unchanged",
        "tokens_spent": 600,
    }


def _preview(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
