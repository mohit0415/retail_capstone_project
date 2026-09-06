import logging
from datetime import date

from llama_index.core.schema import QueryBundle

from configs.settings import settings
from src.retrieval.adapter import to_retrieved_chunks
from src.retrieval.fusion import build_fusion_retriever
from src.retrieval.postprocessors import (
    CurrentVersionFilter,
    FlashRankRerank,
    KeepTopN,
    build_reranker,
)
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
) -> tuple[list[RetrievedChunk], bool, list[str]]:
    if not allowed_doc_types:
        return [], False, []

    as_of = as_of or settings.as_of_date
    top_n = top_n or settings.rerank_top_n

    try:
        retriever, fused = build_fusion_retriever(allowed_doc_types, doc_scope, top_k)
    except Exception as exc:
        logger.error("retriever construction failed: %s", exc)
        return [], False, []

    bundle = QueryBundle(query_str=query)

    try:
        nodes = retriever.retrieve(bundle)
    except Exception as exc:
        logger.error("retrieval failed: %s", exc)
        return [], fused, []

    nodes = CurrentVersionFilter(as_of=str(as_of)).postprocess_nodes(nodes, query_bundle=bundle)

    skipped: list[str] = []

    if use_rerank:
        postprocessor = build_reranker(top_n)
    else:
        postprocessor = KeepTopN(top_n=top_n)
        skipped.append("reranker")

    reranked = isinstance(postprocessor, FlashRankRerank)

    if use_rerank and not reranked:
        logger.info("no cross-encoder is available, so retrieval keeps the fusion order")

    nodes = postprocessor.postprocess_nodes(nodes, query_bundle=bundle)

    return to_retrieved_chunks(nodes, fused, reranked), fused, skipped


def hybrid_retrieve(
    query: str,
    allowed_doc_types: list[str],
    doc_scope: list[str] | None = None,
    as_of: date | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    chunks, _, _ = retrieve_policy_evidence(
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
    chunks, _, _ = retrieve_policy_evidence(
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
