"""Evidence stage 1 of every route that reads policy: retrieval over the policy corpus.

On the rag route this stage also writes the answer - cited on every sentence an extract supports,
or an honest "I don't know" when the extracts do not answer the question. On the hybrid and
high-risk routes it only gathers the clauses; the stage that has every source in front of it writes
the answer (nl2sql_path for hybrid, the multi-agent panel for High risk).
"""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import routed_model
from configs.settings import settings
from src.auth.rbac import allowed_doc_types
from src.core.budget import guard_from_state
from src.graph.routing import (
    NOT_FOUND_NO_MATCH,
    NOT_FOUND_NOT_IN_EXTRACTS,
    current_route,
    human_requested,
    retrieval_found_nothing,
)
from src.graph.state import AgentState
from src.guardrails.output_guard import CITATION_PATTERN
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import RAG_ANSWER
from src.retrieval.adapter import format_context, provenance_text
from src.retrieval.citations import (
    cite_uncited_sentences,
    cites_retrieved_clause,
    only_names_gaps,
    uncited_sentences,
    with_inline_citations,
)
from src.retrieval.hybrid_search import retrieve_policy_evidence
from src.retrieval.sibling_expansion import append_sibling_clauses
from src.schemas.enums import EvidencePath
from src.schemas.models import DraftAnswer, RetrievedChunk

logger = logging.getLogger(__name__)


def build_context(chunks: list[RetrievedChunk]) -> str:
    return format_context(chunks)


def effective_document_scope(state: AgentState) -> list[str] | None:
    grant = set(allowed_doc_types(state.get("access_scopes", [])))

    requested = [value for value in state.get("document_scope_request", []) if value in grant]

    intent = state.get("intent")
    inferred = [value for value in (intent.document_scope if intent else []) if value in grant]

    if requested and inferred:
        overlap = [value for value in requested if value in set(inferred)]

        return overlap or requested

    if requested:
        return requested

    return inferred or None


REPAIR_EXTRA_CANDIDATES = 4


def is_repair_pass(state: AgentState) -> bool:
    return bool(state.get("reflection_count")) and bool(state.get("replan_directive"))


def gather_policy_evidence(
    state: AgentState, widen: bool = False
) -> tuple[list[RetrievedChunk], list[str], list[RetrievedChunk]]:
    documents = allowed_doc_types(state.get("access_scopes", []))
    # a repair searches every document the role can read, with more candidates, because
    # repeating the first search returns exactly the extracts that produced the defects
    scope = None if widen else effective_document_scope(state)

    guard = guard_from_state(state)
    verdict = guard.check("reranker")

    if not verdict.allowed:
        logger.info(
            "reranker skipped under budget pressure request_id=%s reason=%s seconds_remaining=%.2f",
            state.get("request_id"),
            verdict.reason,
            guard.seconds_remaining,
        )

    started = time.perf_counter()

    chunks, _, skipped, raw_candidates = retrieve_policy_evidence(
        query=state["standalone_query"],
        allowed_doc_types=documents,
        doc_scope=scope,
        as_of=settings.as_of_date,
        top_k=settings.retrieval_top_k * 2 if widen else None,
        top_n=settings.rerank_top_n + REPAIR_EXTRA_CANDIDATES if widen else None,
        use_rerank=verdict.allowed,
    )

    best = (
        max((c.rerank_score if c.rerank_score is not None else c.dense_score) for c in chunks)
        if chunks
        else None
    )

    logger.info(
        "policy evidence request_id=%s chunks=%d scope=%s widened=%s allowed_docs=%d best_score=%s skipped=%s elapsed_ms=%.0f",
        state.get("request_id"),
        len(chunks),
        scope or "all",
        widen,
        len(documents),
        round(best, 4) if best is not None else None,
        skipped or "none",
        (time.perf_counter() - started) * 1000,
    )

    if not chunks:
        logger.warning(
            "policy evidence EMPTY request_id=%s scope=%s - the answer cannot be grounded in a clause",
            state.get("request_id"),
            scope or "all",
        )

    return chunks, skipped, raw_candidates


def _nothing_matched(state: AgentState, chunks: list[RetrievedChunk]) -> bool:
    """No extract came back, or the reranker says none is about the question (a no-match, not a weak match)."""
    if human_requested(state):
        return False

    return not chunks or retrieval_found_nothing({"retrieved_chunks": chunks, "sql_evidence": None})


def _writer_found_nothing(draft: DraftAnswer, chunks: list[RetrievedChunk]) -> bool:
    """The writer found nothing in the extracts that answers the question.

    A draft that says nothing, or only says what the extracts do not cover, is "I don't know"
    whatever cited_clauses holds (the writer lists the extracts it looked at). A draft that makes a
    claim and cites a retrieved clause is a (partial) answer whatever answer_found says, so it goes
    to validation.
    """
    answer = (draft.answer or "").strip()

    if not answer or only_names_gaps(answer):
        return True

    if cites_retrieved_clause(answer, draft.cited_clauses, chunks):
        return False

    return not getattr(draft, "answer_found", True)


def _cite_every_sentence(state: AgentState, draft: DraftAnswer, chunks: list[RetrievedChunk]) -> DraftAnswer:
    """Every substantive sentence of the answer carries the clause that supports it."""
    answer, attached = cite_uncited_sentences(draft.answer, draft.cited_clauses, chunks)

    if attached:
        logger.info(
            "rag draft left %d sentence(s) without a citation, placed the supporting clause on each "
            "request_id=%s citations=%s",
            len(attached),
            state.get("request_id"),
            attached,
        )

    added: list[str] = []

    if not CITATION_PATTERN.search(answer) and uncited_sentences(answer):
        # no claim could be matched to a clause: fall back to the clauses the writer itself said it
        # relied on, appended to the answer, rather than releasing claims with no citation at all
        # (never onto an answer that only names gaps - that one is "I don't know")
        answer, added = with_inline_citations(answer, draft.cited_clauses, chunks)

    if added:
        logger.info(
            "rag draft left out %d inline citation marker(s), appended them request_id=%s citations=%s",
            len(added),
            state.get("request_id"),
            added,
        )

    # section completeness: siblings of a cited clause (6.1 answered, 6.2 supplied)
    # are appended verbatim - the writer model leaves them out even when instructed
    answer, siblings = append_sibling_clauses(answer, [*draft.cited_clauses, *attached], chunks)

    if siblings:
        logger.info(
            "rag draft skipped %d sibling clause(s) of a cited section, appended them "
            "request_id=%s citations=%s",
            len(siblings),
            state.get("request_id"),
            siblings,
        )

    if answer == draft.answer:
        return draft

    cited = list(dict.fromkeys([*draft.cited_clauses, *attached, *siblings]))

    return draft.model_copy(update={"answer": answer, "cited_clauses": cited})


@traced_node("rag_path")
def rag_path_node(state: AgentState) -> dict:
    route = current_route(state)
    repairing = is_repair_pass(state)
    chunks, skipped, raw_candidates = gather_policy_evidence(state, widen=repairing)

    gathered = {
        "retrieved_chunks": chunks,
        # what fusion returned BEFORE the reranker cut, for RAGAS retrieval precision
        "retrieval_candidates": [provenance_text(c) for c in raw_candidates],
        "evidence_path": route,
        "degraded": bool(skipped) or state.get("degraded", False),
        "skipped_optional_nodes": skipped,
    }

    if route != EvidencePath.RAG.value:
        # stage 1 of a hybrid or high-risk route: the clauses are gathered here and the answer is
        # written by the stage that has every source in front of it, so no draft is spent here
        logger.info(
            "rag stage gathered %d extract(s) for route=%s request_id=%s; the answer is written by %s",
            len(chunks),
            route,
            state.get("request_id"),
            "the multi-agent panel" if route == EvidencePath.HIGH_RISK_PANEL.value else "nl2sql_path",
        )

        return {**gathered, "draft": None, "tokens_spent": 0}

    if _nothing_matched(state, chunks):
        # writing a draft from extracts that are not about the question can only produce a
        # guess; skip the model call and let the graph answer "I don't know" (no_answer node)
        logger.info(
            "rag draft skipped request_id=%s reason=no retrieved extract matches the question",
            state.get("request_id"),
        )

        return {**gathered, "draft": None, "not_found_reason": NOT_FOUND_NO_MATCH, "tokens_spent": 0}

    prompt = render_prompt(
        "RAG_ANSWER", RAG_ANSWER, query=state["standalone_query"], context=build_context(chunks)
    )

    model, decision = routed_model("rag_generate", state)

    messages = [SystemMessage(content=prompt)]

    if repairing:
        # the validator's defects go back to the writer; kept as its own message so a
        # Langfuse-managed RAG_ANSWER prompt still works unchanged
        messages.append(SystemMessage(content=state["replan_directive"]))

    messages.append(HumanMessage(content=state["standalone_query"]))

    draft: DraftAnswer = model.with_structured_output(DraftAnswer).invoke(
        messages,
        config=runnable_config(state, "rag_generate"),
    )

    if _writer_found_nothing(draft, chunks) and not human_requested(state):
        # the writer read the extracts and none of them answers the question: the honest reply is
        # "I don't know", released as such, not a guess sent through validation to a reviewer
        logger.info(
            "rag writer found no answer in %d extract(s) -> honest 'I don't know' request_id=%s",
            len(chunks),
            state.get("request_id"),
        )

        return {
            **gathered,
            "draft": None,
            "not_found_reason": NOT_FOUND_NOT_IN_EXTRACTS,
            "model_routing": [decision.as_dict()],
            "tokens_spent": 3000,
        }

    draft = _cite_every_sentence(state, draft, chunks)

    logger.info(
        "rag draft request_id=%s tier=%s repair=%s answer_chars=%d cited=%d uncertainty=%s",
        state.get("request_id"),
        decision.tier,
        repairing,
        len(draft.answer or ""),
        len(draft.cited_clauses),
        bool(draft.uncertainty_note),
    )

    return {
        **gathered,
        "draft": draft,
        "model_routing": [decision.as_dict()],
        "tokens_spent": 3000,
    }
