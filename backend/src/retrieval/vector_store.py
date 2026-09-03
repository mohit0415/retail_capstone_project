import logging
from typing import List, Optional

from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from configs.settings import settings
from src.index.vector_index import load_or_create_index
from src.retrieval.adapter import to_retrieved_chunks
from src.retrieval.fusion import build_scope_filters
from src.schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)


def dense_search(
    query: str,
    allowed_doc_types: List[str],
    doc_scope: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    index = load_or_create_index()

    retriever = index.as_retriever(
        similarity_top_k=top_k or settings.retrieval_top_k,
        filters=build_scope_filters(allowed_doc_types, doc_scope),
    )

    nodes = retriever.retrieve(QueryBundle(query_str=query))

    return to_retrieved_chunks(nodes, fused=False)


def lexical_search(
    query: str,
    allowed_doc_types: List[str],
    doc_scope: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    from llama_index.retrievers.bm25 import BM25Retriever

    index = load_or_create_index()

    candidates = [
        node
        for node in index.docstore.docs.values()
        if (node.metadata or {}).get("doc_type") in allowed_doc_types
    ]

    if not candidates:
        return []

    retriever = BM25Retriever.from_defaults(
        nodes=candidates,
        similarity_top_k=min(top_k or settings.retrieval_top_k, len(candidates)),
    )

    nodes = retriever.retrieve(QueryBundle(query_str=query))
    chunks = to_retrieved_chunks(nodes, fused=False)

    for chunk in chunks:
        chunk.lexical_score = chunk.dense_score
        chunk.dense_score = 0.0

    return chunks


def fetch_clause(clause_number: str, allowed_doc_types: List[str]) -> RetrievedChunk | None:
    index = load_or_create_index()

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="clause_number", value=clause_number, operator=FilterOperator.EQ),
            MetadataFilter(key="doc_type", value=allowed_doc_types, operator=FilterOperator.IN),
        ]
    )

    try:
        retriever = index.as_retriever(similarity_top_k=1, filters=filters)
        nodes = retriever.retrieve(QueryBundle(query_str=f"clause {clause_number}"))
    except Exception as exc:
        logger.warning("clause lookup failed for %s: %s", clause_number, exc)
        return None

    chunks = to_retrieved_chunks(nodes, fused=False)

    return chunks[0] if chunks else None
