import logging
from typing import List

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import FunctionTool, QueryEngineTool

from configs.settings import settings
from src.index.models import get_llm
from src.retrieval.adapter import to_retrieved_chunks
from src.retrieval.fusion import build_fusion_retriever
from src.retrieval.postprocessors import CurrentVersionFilter, build_reranker

logger = logging.getLogger(__name__)

POLICY_TOOL_DESCRIPTION = (
    "Semantic and keyword search over the retail policy corpus: the privacy, retention, vendor, "
    "anti-bribery and information security policies, the GDPR articles and the ISO 27001 Annex A "
    "controls the company follows. It also covers summaries of tables and captions of diagrams "
    "taken from those documents. "
    "Use this tool for any question about what a policy, clause, article or control requires, "
    "permits, forbids or defines, and for any retention period, approval rule, notification "
    "deadline or process step. Every result carries its document title and clause number, which "
    "is what the final answer must cite. "
    "Call this tool before answering any question about a rule, even when the question sounds "
    "informal. Never answer a policy question from your own knowledge."
)


def build_policy_tool(allowed_doc_types: List[str], doc_scope: List[str] | None = None):
    retriever, fused = build_fusion_retriever(allowed_doc_types, doc_scope)

    version_filter = CurrentVersionFilter(as_of=str(settings.as_of_date))
    reranker = build_reranker(settings.rerank_top_n)

    if settings.enable_citation_synthesis:
        engine = CitationQueryEngine.from_args(
            index=None,
            retriever=retriever,
            llm=get_llm("rag_generate"),
            citation_chunk_size=512,
            node_postprocessors=[version_filter, reranker],
        )

        return QueryEngineTool.from_defaults(
            query_engine=engine,
            name="policy_documents",
            description=POLICY_TOOL_DESCRIPTION,
        )

    def search_policy_documents(input: str) -> Response:
        bundle = QueryBundle(query_str=input)

        nodes = retriever.retrieve(bundle)
        nodes = version_filter.postprocess_nodes(nodes, query_bundle=bundle)
        nodes = reranker.postprocess_nodes(nodes, query_bundle=bundle)

        chunks = to_retrieved_chunks(nodes, fused)

        if not chunks:
            return Response(
                response="No clause in the documents available to this role covers that question.",
                source_nodes=nodes,
            )

        blocks = []

        for position, chunk in enumerate(chunks, start=1):
            blocks.append(
                f"[{position}] [{chunk.document_title} §{chunk.clause_number}] "
                f"(section: {chunk.section}, version: {chunk.version}, modality: {chunk.modality})\n"
                f"{chunk.content}"
            )

        return Response(response="\n\n".join(blocks), source_nodes=nodes)

    return FunctionTool.from_defaults(
        fn=search_policy_documents,
        name="policy_documents",
        description=POLICY_TOOL_DESCRIPTION,
    )
