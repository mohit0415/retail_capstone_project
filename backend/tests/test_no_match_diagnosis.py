import pytest
from fastapi import HTTPException

from src.api.routes import _validated_document_scope
from src.auth.rbac import access_scopes_for
from src.auth.security import Principal
from src.nodes.escalation import _derive_reason
from src.nodes.rag_path import effective_document_scope
from src.schemas.enums import Role
from src.schemas.models import ConfidenceBreakdown, IntentResult, RetrievedChunk


def _principal(role: Role) -> Principal:
    return Principal(
        user_id="mohit",
        role=role,
        access_scopes=access_scopes_for(role),
        departments=[],
    )


def _chunk(title: str, doc_type: str, rerank: float | None = None, dense: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_type}-1",
        doc_type=doc_type,
        document_title=title,
        section="Access",
        clause_number="2.1",
        version="1.0",
        content="Privileged accounts are reviewed quarterly.",
        dense_score=dense,
        rerank_score=rerank,
    )


def _scored_state(chunks: list[RetrievedChunk], score: float) -> dict:
    return {
        "raw_query": "How long do we keep point of sale transaction records?",
        "retrieved_chunks": chunks,
        "sql_evidence": None,
        "confidence": ConfidenceBreakdown(
            retrieval_score=score,
            validation_score=0.95,
            source_agreement=1.0,
            coverage=0.0,
            final_score=0.485,
        ),
    }


def test_a_no_match_says_so_instead_of_quoting_a_score():
    chunks = [_chunk("Iso 27001 Access Control Summary", "iso_27001", rerank=0.0001)]

    reason = _derive_reason(_scored_state(chunks, 0.0001))

    assert "nothing in it answers this question" in reason
    assert "no-match rather than a weak match" in reason
    assert "Iso 27001 Access Control Summary" in reason
    assert "ingest/status" in reason


def test_a_genuinely_weak_match_still_reports_the_confidence_breakdown():
    chunks = [_chunk("Data Retention And Archival Policy", "retention_policy", rerank=0.42)]

    reason = _derive_reason(_scored_state(chunks, 0.42))

    assert "below the" in reason
    assert "release threshold" in reason
    assert "no-match" not in reason


def test_sql_evidence_keeps_the_confidence_reason_even_with_poor_retrieval():
    state = _scored_state([_chunk("Vendor Policy", "vendor_policy", rerank=0.001)], 0.001)
    state["sql_evidence"] = object()

    assert "no-match" not in _derive_reason(state)


def test_an_empty_retrieval_keeps_its_own_reason():
    state = _scored_state([], 0.0)

    assert "no grounding evidence was retrieved" in _derive_reason(state)


def test_an_unknown_document_scope_is_rejected_at_the_boundary():
    with pytest.raises(HTTPException) as caught:
        _validated_document_scope(["string"], _principal(Role.STORE_MANAGER))

    assert caught.value.status_code == 422
    assert "string" in caught.value.detail
    assert "retention_policy" in caught.value.detail


def test_a_scope_the_role_cannot_read_is_rejected_too():
    with pytest.raises(HTTPException) as caught:
        _validated_document_scope(["gdpr"], _principal(Role.STORE_MANAGER))

    assert caught.value.status_code == 422


def test_a_valid_scope_passes_through():
    assert _validated_document_scope(["retention_policy"], _principal(Role.STORE_MANAGER)) == [
        "retention_policy"
    ]


def test_an_absent_scope_stays_absent():
    assert _validated_document_scope([], _principal(Role.STORE_MANAGER)) == []


def test_the_retrieval_scope_can_never_leave_the_role_grant():
    state = {
        "access_scopes": access_scopes_for(Role.STORE_MANAGER),
        "document_scope_request": ["gdpr", "retention_policy"],
        "intent": None,
    }

    assert effective_document_scope(state) == ["retention_policy"]


def test_an_entirely_invalid_scope_falls_back_to_the_whole_grant():
    state = {
        "access_scopes": access_scopes_for(Role.STORE_MANAGER),
        "document_scope_request": ["string"],
        "intent": None,
    }

    assert effective_document_scope(state) is None


def test_an_inferred_scope_outside_the_grant_is_dropped():
    state = {
        "access_scopes": access_scopes_for(Role.STORE_ASSOCIATE),
        "document_scope_request": [],
        "intent": IntentResult(intent="policy_lookup", document_scope=["gdpr", "privacy_policy"]),
    }

    assert effective_document_scope(state) == ["privacy_policy"]
