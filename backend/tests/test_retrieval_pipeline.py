
from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval.adapter import format_context, to_retrieved_chunks
from src.retrieval.fusion import build_scope_filters
from src.retrieval.postprocessors import CurrentVersionFilter, KeepTopN


def make_node(
    node_id: str,
    doc_type: str = "retention_policy",
    clause: str = "4.1",
    content_type: str = "text",
    is_current: bool = True,
    effective_date: str = "2025-04-01",
    score: float = 0.5,
) -> NodeWithScore:
    node = TextNode(
        id_=node_id,
        text=f"Content of clause {clause}",
        metadata={
            "doc_type": doc_type,
            "document_title": "Records Retention Policy",
            "section": "Transaction records",
            "clause_number": clause,
            "version": "3.2",
            "effective_date": effective_date,
            "is_current": is_current,
            "content_type": content_type,
            "modality": "table" if content_type == "table_summary" else "text",
        },
    )

    return NodeWithScore(node=node, score=score)


def test_scope_filter_never_widens_beyond_the_role():
    filters = build_scope_filters(["privacy_policy", "retention_policy"], ["gdpr", "retention_policy"])

    values = filters.filters[0].value

    assert "gdpr" not in values
    assert "retention_policy" in values


def test_empty_scope_falls_back_to_the_full_role_grant():
    filters = build_scope_filters(["privacy_policy"], ["gdpr"])

    assert filters.filters[0].value == ["privacy_policy"]


def test_superseded_versions_are_dropped():
    nodes = [make_node("a", is_current=True), make_node("b", is_current=False)]

    kept = CurrentVersionFilter(as_of="2025-12-31").postprocess_nodes(nodes)

    assert [item.node.node_id for item in kept] == ["a"]


def test_documents_not_yet_in_force_are_dropped():
    nodes = [make_node("a", effective_date="2025-01-01"), make_node("b", effective_date="2026-06-01")]

    kept = CurrentVersionFilter(as_of="2025-12-31").postprocess_nodes(nodes)

    assert [item.node.node_id for item in kept] == ["a"]


def test_keep_top_n_carries_media_past_the_cut():
    nodes = [make_node(str(index)) for index in range(6)]
    nodes.append(make_node("table", content_type="table_summary"))

    kept = KeepTopN(top_n=3, max_media=2).postprocess_nodes(nodes)

    ids = [item.node.node_id for item in kept]

    assert ids[:3] == ["0", "1", "2"]
    assert "table" in ids


def test_adapter_carries_clause_identity_and_modality():
    chunks = to_retrieved_chunks([make_node("a", content_type="table_summary")], fused=True)

    assert len(chunks) == 1
    assert chunks[0].clause_number == "4.1"
    assert chunks[0].modality == "table"
    assert chunks[0].citation == "Records Retention Policy §4.1"


def test_adapter_deduplicates_by_node_id():
    chunks = to_retrieved_chunks([make_node("a"), make_node("a")], fused=True)

    assert len(chunks) == 1


def test_fused_scores_land_in_the_fusion_field():
    chunks = to_retrieved_chunks([make_node("a", score=0.031)], fused=True)

    assert chunks[0].fused_score == 0.031
    assert chunks[0].dense_score == 0.0


def test_context_wraps_every_chunk_as_data():
    context = format_context(to_retrieved_chunks([make_node("a")], fused=True))

    assert "<retrieved_document" in context
    assert "Records Retention Policy §4.1" in context


def test_empty_retrieval_says_so_rather_than_returning_blank():
    assert "no policy extract" in format_context([])
