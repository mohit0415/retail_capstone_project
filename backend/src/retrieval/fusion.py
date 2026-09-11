import logging
import threading

from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever, VectorIndexAutoRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

from configs.settings import settings
from src.index.models import get_llm
from src.index.vector_index import load_lexical_corpus, load_or_create_index

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


def build_scope_filters(allowed_doc_types: list[str], doc_scope: list[str] | None) -> MetadataFilters:
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

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
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


def _in_scope(node, allowed_doc_types: list[str]) -> bool:
    metadata = node.metadata or {}

    return metadata.get("doc_type") in allowed_doc_types and metadata.get("is_current") is not False


# The BM25 index used to be rebuilt (tokenise + stem every node) on every single
# retrieval. It only changes when the corpus changes, so it is cached per role scope
# and per corpus snapshot; an ingest or rebuild produces a new corpus list and with
# it a new cache key.
_BM25_CACHE: dict[tuple, object] = {}
_BM25_CACHE_LIMIT = 16
_bm25_lock = threading.Lock()


def clear_bm25_cache() -> None:
    with _bm25_lock:
        _BM25_CACHE.clear()


def _build_bm25_retriever(index, allowed_doc_types: list[str], top_k: int):
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except Exception as exc:
        logger.warning("BM25 retriever unavailable (%s); retrieval is vector only", exc)
        return None

    source = "docstore"

    try:
        docstore_nodes = list(index.docstore.docs.values())
    except Exception as exc:
        logger.warning("could not read the docstore for BM25: %s", exc)
        docstore_nodes = []

    candidates = [node for node in docstore_nodes if _in_scope(node, allowed_doc_types)]
    corpus = getattr(index, "docstore", None)

    if not candidates:
        source = "vector_table"
        corpus = load_lexical_corpus()
        candidates = [node for node in corpus if _in_scope(node, allowed_doc_types)]

    # the cache entry holds on to the corpus list it was built from, so id(corpus) cannot be
    # reused by a newer corpus while the entry is alive
    size = len(docstore_nodes) if source == "docstore" else len(corpus)
    key = (tuple(sorted(allowed_doc_types)), min(top_k, len(candidates)), source, id(corpus), size)

    with _bm25_lock:
        cached = _BM25_CACHE.get(key)

    if cached is not None and candidates:
        return cached[0]

    if not candidates:
        logger.warning(
            "no nodes in scope for BM25 (docs=%s); leg skipped, retrieval is vector only", allowed_doc_types
        )
        return None

    logger.debug(
        "bm25 leg built from %s nodes=%d top_k=%d", source, len(candidates), min(top_k, len(candidates))
    )

    retriever = BM25Retriever.from_defaults(
        nodes=candidates,
        similarity_top_k=min(top_k, len(candidates)),
    )

    with _bm25_lock:
        if len(_BM25_CACHE) >= _BM25_CACHE_LIMIT:
            _BM25_CACHE.clear()

        _BM25_CACHE[key] = (retriever, corpus)

    return retriever


def build_fusion_retriever(
    allowed_doc_types: list[str],
    doc_scope: list[str] | None = None,
    top_k: int | None = None,
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
