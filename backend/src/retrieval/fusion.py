import logging
from typing import List, Optional

from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever, VectorIndexAutoRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

from configs.settings import settings
from src.index.models import get_llm
from src.index.vector_index import load_or_create_index

logger = logging.getLogger(__name__)

VECTOR_STORE_INFO = VectorStoreInfo(
    content_info=(
        "Clause-level extracts from five internal retail policies (privacy, retention, vendor, "
        "anti-bribery, information security), GDPR articles and ISO 27001 Annex A controls, plus "
        "summaries of tables and captions of diagrams taken from those documents."
    ),
    metadata_info=[
        MetadataInfo(
            name="doc_type",
            type="str",
            description=(
                "Which document the clause belongs to. One of: privacy_policy, retention_policy, "
                "vendor_policy, anti_bribery_policy, infosec_policy, gdpr, iso_27001"
            ),
        ),
        MetadataInfo(name="document_title", type="str", description="Printed title of the document"),
        MetadataInfo(
            name="clause_number",
            type="str",
            description="Clause identifier as printed, for example 4.1, A.8.15 or 17",
        ),
        MetadataInfo(name="section", type="str", description="Heading of the clause"),
        MetadataInfo(name="version", type="str", description="Published version of the document"),
        MetadataInfo(
            name="effective_date",
            type="str",
            description="ISO date the version came into force",
        ),
        MetadataInfo(
            name="owning_department",
            type="str",
            description="Department that owns the clause: legal, finance, it, hr, marketing, logistics, facilities, all",
        ),
        MetadataInfo(
            name="content_type",
            type="str",
            description="Node modality: text, table_summary or image_caption",
        ),
        MetadataInfo(
            name="modality",
            type="str",
            description="Source modality: text, table or diagram",
        ),
        MetadataInfo(
            name="content_domain",
            type="str",
            description="obligation, definition, procedure, reference or mixed",
        ),
        MetadataInfo(
            name="intended_route",
            type="str",
            description="Route the clause is best answered by: rag, sql or hybrid",
        ),
    ],
)


def build_scope_filters(allowed_doc_types: List[str], doc_scope: Optional[List[str]]) -> MetadataFilters:
    effective = [value for value in (doc_scope or allowed_doc_types) if value in allowed_doc_types]

    if not effective:
        effective = allowed_doc_types

    return MetadataFilters(
        filters=[
            MetadataFilter(key="doc_type", value=effective, operator=FilterOperator.IN),
        ]
    )


class AutoWithFallbackRetriever(BaseRetriever):
    def __init__(self, auto_retriever: BaseRetriever, plain_retriever: BaseRetriever):
        self._auto = auto_retriever
        self._plain = plain_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        try:
            nodes = self._auto.retrieve(query_bundle)
        except Exception as exc:
            logger.warning("auto-retriever failed (%s); using plain vector retrieval", exc)
            nodes = []

        if not nodes:
            nodes = self._plain.retrieve(query_bundle)

        return nodes


def _build_vector_retriever(index, filters: MetadataFilters, top_k: int) -> BaseRetriever:
    plain = index.as_retriever(similarity_top_k=top_k, filters=filters)

    if not settings.use_auto_retriever:
        return plain

    auto = VectorIndexAutoRetriever(
        index=index,
        vector_store_info=VECTOR_STORE_INFO,
        similarity_top_k=top_k,
        llm=get_llm("planner"),
    )

    return AutoWithFallbackRetriever(auto, plain)


def _build_bm25_retriever(index, allowed_doc_types: List[str], top_k: int):
    try:
        from llama_index.retrievers.bm25 import BM25Retriever

        candidates = [
            node
            for node in index.docstore.docs.values()
            if (node.metadata or {}).get("doc_type") in allowed_doc_types
            and (node.metadata or {}).get("is_current") is not False
        ]
    except Exception as exc:
        logger.warning("could not read the docstore for BM25: %s", exc)
        return None

    if not candidates:
        logger.warning("docstore holds no nodes in scope; BM25 leg skipped, retrieval is vector only")
        return None

    return BM25Retriever.from_defaults(
        nodes=candidates,
        similarity_top_k=min(top_k, len(candidates)),
    )


def build_fusion_retriever(
    allowed_doc_types: List[str],
    doc_scope: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> tuple[BaseRetriever, bool]:
    index = load_or_create_index()
    top_k = top_k or settings.retrieval_top_k

    filters = build_scope_filters(allowed_doc_types, doc_scope)
    vector_retriever = _build_vector_retriever(index, filters, top_k)

    bm25_retriever = _build_bm25_retriever(index, allowed_doc_types, top_k)

    if bm25_retriever is None:
        return vector_retriever, False

    fusion = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=settings.fusion_num_queries,
        mode="reciprocal_rerank",
        use_async=False,
        llm=get_llm("query_rewrite") if settings.fusion_num_queries > 1 else None,
        verbose=False,
    )

    return fusion, True
