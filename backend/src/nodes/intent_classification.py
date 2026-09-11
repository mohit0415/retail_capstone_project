import logging
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate

from configs.llms import model_for
from configs.settings import settings
from src.graph.routing import RETRIEVAL_NO_MATCH_CEILING
from src.graph.state import AgentState
from src.nodes.rag_path import gather_policy_evidence
from src.observability.logging_config import with_request_context
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import INTENT_CLASSIFICATION
from src.schemas.enums import Intent, TerminalOutcome
from src.schemas.models import IntentResult

logger = logging.getLogger(__name__)

# The classifier said "not a policy question" but the policy corpus has a real match for it: a
# misspelt "what is the purpose of anti-brbrery?" was refused while the anti-bribery policy answers
# it. The rescue is judged on the cross-encoder's score of the policy extracts: twice the no-match
# ceiling for a question about any policy, just above it when the user picked the policy (the chips).
OUT_OF_SCOPE_RESCUE_SCORE = 0.10

# never rescued, whatever the extracts score: a creative-writing request, or an attempt to override
# the instructions, is off-topic even when it names a policy subject ("a haiku about bribery")
NOT_A_POLICY_QUESTION = re.compile(
    r"\b(?:write|compose|draft|generate|create|sing|rhyme)\b[^.?!]{0,40}\b(?:poem|haiku|limerick|song|story|joke"
    r"|rap|essay|tweet|lyrics)\b"
    r"|\b(?:joke|jokes|poem|haiku|limerick|riddle|lyrics)\b"
    r"|\bignore\b[^.?!]{0,40}\b(?:instructions|rules|prompt|guidelines)\b"
    r"|\b(?:system prompt|jailbreak|pretend (?:you are|to be)|act as|roleplay)\b",
    re.I,
)


def _rescue_policy_question(state: AgentState, result: IntentResult) -> tuple[IntentResult, str] | None:
    """A policy question the classifier refused, which the policy extracts clearly answer."""
    if NOT_A_POLICY_QUESTION.search(state.get("standalone_query") or ""):
        return None

    chosen = list(state.get("document_scope_request") or [])

    try:
        # scoped to the chosen policies when there are any - the same search, and the same retrieval
        # cache entry, that rag_path runs next
        chunks, _skipped = gather_policy_evidence(state)
    except Exception:
        logger.warning(
            "out-of-scope check against the policy corpus failed, refusing request_id=%s",
            state.get("request_id"),
            exc_info=True,
        )

        return None

    best = max((chunk.rerank_score for chunk in chunks[:5] if chunk.rerank_score is not None), default=None)

    if best is None:
        # no cross-encoder score to judge by (the reranker is off or was skipped): only the user's own
        # choice of policy, with something retrieved from it, is enough
        if not (chosen and chunks):
            return None

        why = f"the user narrowed the question to {chosen} and the search returned {len(chunks)} extract(s)"
    elif best > (RETRIEVAL_NO_MATCH_CEILING if chosen else OUT_OF_SCOPE_RESCUE_SCORE - 1e-9):
        why = f"the policy extracts match it (best extract {best:.2f}" + (f", narrowed to {chosen})" if chosen else ")")
    else:
        return None

    return (
        IntentResult(
            intent=Intent.POLICY_LOOKUP,
            entities=result.entities,
            document_scope=chosen,
            reasoning=f"{why}; classifier said: {result.reasoning}",
        ),
        why,
    )


def _classify_intent(state: AgentState, query: str) -> IntentResult:
    prompt = render_prompt("INTENT_CLASSIFICATION", INTENT_CLASSIFICATION, query=query)

    chain = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), ("human", "{query}")]
    ) | model_for("intent_classification").with_structured_output(IntentResult)

    return chain.invoke(
        {"system_prompt": prompt, "query": query},
        config=runnable_config(state, "intent_classification"),
    )


def _classify_risk(state: AgentState, query: str) -> dict | None:
    """The risk agent's LLM judgement, started next to the intent call.

    It reads only the query, so there is no reason to wait for intent and entity
    resolution first. The Risk Assessment Agent still fuses it with the lexical
    floor and the database probes in its own node; if this call fails the risk
    node simply asks again.
    """
    from src.nodes.risk_assessment import classify_risk_l2

    try:
        classification = classify_risk_l2(state, query)
    except Exception:
        logger.warning(
            "parallel risk classification failed, the risk node will retry request_id=%s",
            state.get("request_id"),
            exc_info=True,
        )

        return None

    return {
        "query": query,
        "level": classification.level.value,
        "scenario_id": classification.scenario_id or "",
        "rationale": classification.rationale or "",
    }


@traced_node("intent_classification")
def intent_classification_node(state: AgentState) -> dict:
    query = state["standalone_query"]
    risk_l2 = None

    if settings.parallel_intent_and_risk:
        with ThreadPoolExecutor(max_workers=2) as pool:
            intent_future = pool.submit(with_request_context(_classify_intent), state, query)
            risk_future = pool.submit(with_request_context(_classify_risk), state, query)

            result = intent_future.result()
            risk_l2 = risk_future.result()
    else:
        result = _classify_intent(state, query)

    if result.intent is Intent.OUT_OF_SCOPE:
        rescued = _rescue_policy_question(state, result)

        if rescued is None:
            logger.warning(
                "intent OUT_OF_SCOPE -> refusing request_id=%s reasoning=%r",
                state.get("request_id"),
                _preview(result.reasoning),
            )

            return {
                "intent": result,
                "risk_l2": risk_l2,
                "terminal_outcome": TerminalOutcome.REFUSED.value,
                "refusal_reason": "the question is not a retail policy or compliance matter",
                "tokens_spent": 500,
            }

        result, why = rescued

        logger.info(
            "intent OUT_OF_SCOPE overridden -> policy_lookup request_id=%s reason=%s",
            state.get("request_id"),
            why,
        )

    logger.info(
        "intent=%s entities=%s document_scope=%s risk_l2_ready=%s request_id=%s",
        result.intent.value,
        [f"{span.entity_type}:{span.text}" for span in result.entities] or "none",
        result.document_scope or "any",
        risk_l2 is not None,
        state.get("request_id"),
    )

    return {"intent": result, "risk_l2": risk_l2, "tokens_spent": 500}


def _preview(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
