from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from configs.settings import settings
from src.auth.rbac import allowed_doc_types
from src.core.budget import guard_from_state
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import RAG_ANSWER
from src.retrieval.adapter import format_context
from src.retrieval.hybrid_search import retrieve_policy_evidence
from src.schemas.models import DraftAnswer, RetrievedChunk


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


def gather_policy_evidence(state: AgentState) -> tuple[list[RetrievedChunk], list[str]]:
    documents = allowed_doc_types(state.get("access_scopes", []))
    scope = effective_document_scope(state)

    guard = guard_from_state(state)
    verdict = guard.check("reranker")

    chunks, _, skipped = retrieve_policy_evidence(
        query=state["standalone_query"],
        allowed_doc_types=documents,
        doc_scope=scope,
        as_of=settings.as_of_date,
        use_rerank=verdict.allowed,
    )

    return chunks, skipped


@traced_node("rag_path")
def rag_path_node(state: AgentState) -> dict:
    chunks, skipped = gather_policy_evidence(state)

    prompt = RAG_ANSWER.format(query=state["standalone_query"], context=build_context(chunks))

    model = model_for("rag_generate").with_structured_output(DraftAnswer)

    draft: DraftAnswer = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "rag_generate"),
    )

    return {
        "retrieved_chunks": chunks,
        "draft": draft,
        "evidence_path": "rag",
        "degraded": bool(skipped) or state.get("degraded", False),
        "skipped_optional_nodes": skipped,
        "tokens_spent": 3000,
    }
