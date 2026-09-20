import logging
import time
from datetime import date

from llama_index.core.schema import QueryBundle

from configs.settings import settings
from src.cache.retrieval_cache import retrieval_cache, retrieval_key
from src.retrieval.adapter import to_retrieved_chunks
from src.retrieval.fusion import build_fusion_retriever
from src.retrieval.postprocessors import (
    CurrentVersionFilter,
    FlashRankRerank,
    KeepTopN,
    build_reranker,
)
from src.retrieval.sibling_expansion import expand_sibling_clauses
from src.schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)


def retrieve_policy_evidence(
    query: str,
    allowed_doc_types: list[str],
    doc_scope: list[str] | None = None,
    as_of: date | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    use_rerank: bool = True,
) -> tuple[list[RetrievedChunk], bool, list[str], list[RetrievedChunk]]:
    """Returns (kept chunks, fused?, skipped optimisations, raw candidates).

    The raw candidates are the current-version fusion results BEFORE the
    reranker cut - the set RAGAS retrieval precision is judged on, because
    precision measured after the cross-encoder has already filtered the list
    can only ever say the filter works, not that retrieval is precise.
    """
    if not allowed_doc_types:
        logger.info("retrieval skipped: no document type in scope")

        return [], False, [], []

    as_of = as_of or settings.as_of_date
    top_n = top_n or settings.rerank_top_n

    key = retrieval_key(query, allowed_doc_types, doc_scope, as_of, top_k, top_n, use_rerank)

    if settings.enable_retrieval_cache:
        cached = retrieval_cache.get(key)

        if cached is not None:
            return cached

    started = time.perf_counter()

    try:
        retriever, fused = build_fusion_retriever(allowed_doc_types, doc_scope, top_k)
    except Exception as exc:
        logger.error("retriever construction failed: %s", exc, exc_info=True)
        return [], False, [], []

    bundle = QueryBundle(query_str=query)

    try:
        nodes = retriever.retrieve(bundle)
    except Exception as exc:
        logger.error("retrieval failed: %s", exc, exc_info=True)
        return [], fused, [], []

    candidates = len(nodes)
    nodes = CurrentVersionFilter(as_of=str(as_of)).postprocess_nodes(nodes, query_bundle=bundle)
    current = len(nodes)
    raw_candidates = to_retrieved_chunks(nodes, fused)

    skipped: list[str] = []

    if use_rerank:
        postprocessor = build_reranker(top_n)
    else:
        postprocessor = KeepTopN(top_n=top_n)
        skipped.append("reranker")

    reranked = isinstance(postprocessor, FlashRankRerank)

    if use_rerank and not reranked:
        logger.info("no cross-encoder is available, so retrieval keeps the fusion order")

    rerank_started = time.perf_counter()
    nodes = postprocessor.postprocess_nodes(nodes, query_bundle=bundle)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000

    chunks = to_retrieved_chunks(nodes, fused, reranked)

    if settings.enable_sibling_expansion:
        chunks = expand_sibling_clauses(chunks, allowed_doc_types)

    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "retrieval candidates=%d current=%d kept=%d fused=%s reranked=%s scope=%s rerank_ms=%.0f total_ms=%.0f",
        candidates,
        current,
        len(chunks),
        fused,
        reranked,
        doc_scope or "all",
        rerank_ms,
        elapsed_ms,
    )

    result = (chunks, fused, skipped, raw_candidates)

    if settings.enable_retrieval_cache:
        retrieval_cache.put(key, result, elapsed_ms)

    return result


def hybrid_retrieve(
    query: str,
    allowed_doc_types: list[str],
    doc_scope: list[str] | None = None,
    as_of: date | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    chunks, _, _, _ = retrieve_policy_evidence(
        query=query,
        allowed_doc_types=allowed_doc_types,
        doc_scope=doc_scope,
        as_of=as_of,
        top_k=top_k,
    )

    return chunks


def widen_retrieval(
    query: str,
    allowed_doc_types: list[str],
    as_of: date | None = None,
) -> list[RetrievedChunk]:
    chunks, _, _, _ = retrieve_policy_evidence(
        query=query,
        allowed_doc_types=allowed_doc_types,
        doc_scope=None,
        as_of=as_of,
        top_k=settings.retrieval_top_k * 2,
        top_n=settings.rerank_top_n * 2,
    )

    return chunks


def reciprocal_rank_fusion(dense_hits, lexical_hits, k: int | None = None):
    k = k or settings.rrf_k
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_hits, start=1):
        merged[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, chunk in enumerate(lexical_hits, start=1):
        if chunk.chunk_id in merged:
            merged[chunk.chunk_id].lexical_score = chunk.lexical_score
        else:
            merged[chunk.chunk_id] = chunk

        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)

    for chunk_id, score in scores.items():
        merged[chunk_id].fused_score = round(score, 6)

    return sorted(merged.values(), key=lambda chunk: chunk.fused_score, reverse=True)
