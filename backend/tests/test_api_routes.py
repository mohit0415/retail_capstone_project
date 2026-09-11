"""HTTP-level tests for every route in ``src/api/routes.py``.

The graph, the database and the ingestion pipeline are replaced at the module
boundary, so what is under test is the HTTP contract: auth, RBAC, request
validation, status codes, response shapes, idempotency and the review flow.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from configs.settings import settings
from src.api import escalation_service, routes
from src.auth import azure_credentials
from src.cache import response_cache as response_cache_module
from src.cache.response_cache import ResponseCache
from src.core import conversation
from src.core.idempotency import IdempotencyStore
from src.ingestion.pipeline import IngestionReport, IngestionResult
from src.schemas.api import QueueItem, ReviewPackage
from src.schemas.enums import EvidencePath, ReviewDecision, RiskLevel, Role, TerminalOutcome
from src.schemas.models import (
    ConfidenceBreakdown,
    DraftAnswer,
    ResolvedEntity,
    RetrievedChunk,
    RiskAssessment,
    SqlEvidence,
)


@pytest.fixture
def client(no_db, no_tracing, auth0, monkeypatch):
    from main import app

    monkeypatch.setattr(routes, "idempotency_store", IdempotencyStore())
    monkeypatch.setattr(response_cache_module, "response_cache", ResponseCache(embedder=None, semantic_threshold=1.0))
    monkeypatch.setattr(conversation, "record_exchange", lambda *args, **kwargs: None)
    monkeypatch.setattr(conversation, "record_user_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(conversation, "record_assistant_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(conversation, "load_thread", lambda thread_id: ([], ""))
    monkeypatch.setattr(routes.slo, "record_latency", lambda *args, **kwargs: None)

    return TestClient(app)


@pytest.fixture
def token(client, auth0):
    """Mint an Auth0-style RS256 token (signed with the test key planted in the JWKS cache)."""

    def _mint(role: Role = Role.COMPLIANCE_OFFICER, user_id: str = "mohit", departments=None, **kwargs) -> str:
        return auth0(role, user_id=user_id, departments=departments, **kwargs)

    return _mint


@pytest.fixture
def auth(token):
    def _headers(role: Role = Role.COMPLIANCE_OFFICER, **kwargs) -> dict:
        return {"Authorization": f"Bearer {token(role, **kwargs)}"}

    return _headers


class FakeGraph:
    """Stands in for the compiled LangGraph: returns a canned final state."""

    def __init__(self, final_state: dict):
        self.final_state = final_state
        self.calls: list[dict] = []

    def invoke(self, state, config=None):
        self.calls.append({"state": state, "config": config})

        return {**state, **self.final_state}


@pytest.fixture
def graph_returns(monkeypatch):
    def _install(final_state: dict) -> FakeGraph:
        fake = FakeGraph(final_state)
        monkeypatch.setattr(routes, "get_compiled_graph", lambda: fake)

        return fake

    return _install


def _chunk(title: str, clause: str, content: str = "Records are retained for seven years.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{title}-{clause}",
        doc_type="retention_policy",
        document_title=title,
        section="Retention",
        clause_number=clause,
        version="2.0",
        content=content,
    )


def _answered_state(**overrides) -> dict:
    state = {
        "terminal_outcome": TerminalOutcome.ANSWERED.value,
        "draft": DraftAnswer(
            answer="Invoices are kept for seven years [Retention Policy §4.2].",
            cited_clauses=["Retention Policy §4.2"],
            uncertainty_note="",
        ),
        "retrieved_chunks": [
            _chunk("Retention Policy", "4.2"),
            _chunk("Retention Policy", "4.2", content="duplicate of the same clause"),
            _chunk("Privacy Policy", "1.1", content="unrelated clause that was retrieved but not cited"),
        ],
        "confidence": ConfidenceBreakdown(final_score=0.91),
        "risk": RiskAssessment(final_level=RiskLevel.LOW),
        "evidence_path": EvidencePath.RAG.value,
        "marks": [
            {"node": "input_guardrail", "stage": "t1", "elapsed_ms": 10.0},
            {"node": "planner", "stage": "t2", "elapsed_ms": 20.0},
        ],
    }
    state.update(overrides)

    return state


def _ask(client, headers, query="What is the retention period for customer invoices?", **extra):
    return client.post("/ask", json={"query": query, **extra}, headers=headers)


def test_auth_me_returns_the_role_its_scopes_and_its_screens(client, auth):
    response = client.get("/auth/me", headers=auth(Role.STORE_MANAGER, user_id="auth0|u1", departments=["Finance"]))

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "auth0|u1"
    assert body["role"] == "store_manager"
    assert body["departments"] == ["Finance"]
    assert body["auth0_roles"] == ["store_manager"]
    assert "doc:vendor_policy" in body["access_scopes"]
    assert "doc:gdpr" not in body["access_scopes"]
    assert body["screens"] == ["chat", "dashboard"]
    assert body["permissions"]["can_review"] is False
    assert body["permissions"]["can_view_dashboard"] is True
    assert body["azure_configured"] is False


def test_auth_me_for_admin_opens_every_screen(client, auth):
    body = client.get("/auth/me", headers=auth(Role.ADMIN)).json()

    assert body["screens"] == ["chat", "dashboard", "review", "slo", "corpus"]
    assert body["permissions"] == {
        "can_chat": True,
        "can_view_dashboard": True,
        "can_review": True,
        "can_view_slo": True,
        "can_ingest": True,
    }


def test_the_most_privileged_auth0_role_wins_when_a_user_has_several(client, auth):
    body = client.get("/auth/me", headers=auth([Role.STORE_ASSOCIATE, "legal_reviewer", "something-else"])).json()

    assert body["role"] == "legal_reviewer"
    assert body["auth0_roles"] == ["store_associate", "legal_reviewer", "something-else"]


def test_the_plain_user_role_from_practice_1_maps_to_store_associate(client, auth):
    body = client.get("/auth/me", headers=auth("user")).json()

    assert body["role"] == "store_associate"
    assert body["screens"] == ["chat"]


def test_a_token_without_any_known_role_is_403_and_names_the_claim(client, auth):
    response = client.get("/auth/me", headers=auth("intern"))

    assert response.status_code == 403
    assert "no recognised role" in response.json()["detail"]
    assert "https://stateful-agent.com/roles" in response.json()["detail"]

    assert client.get("/auth/me", headers=auth(None)).status_code == 403


def test_an_empty_roles_claim_says_the_user_has_no_role_assigned(client, auth):
    response = client.get("/auth/me", headers=auth(None))

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "no role assigned" in detail
    assert "Assign Roles" in detail


def test_a_missing_roles_claim_points_at_the_auth0_action(client, auth0):
    token = auth0(Role.ADMIN, omit_roles_claim=True)

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "no 'https://stateful-agent.com/roles' claim at all" in detail
    assert "api.accessToken.setCustomClaim" in detail


def test_roles_under_another_namespace_are_named_in_the_403(client, auth0):
    token = auth0(
        Role.ADMIN,
        omit_roles_claim=True,
        extra_claims={"https://other-namespace.com/roles": ["admin"]},
    )

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "https://other-namespace.com/roles" in detail
    assert "AUTH0_ROLE_NAMESPACE" in detail


def test_ask_without_a_token_is_401_with_the_auth0_hint(client):
    response = client.post("/ask", json={"query": "what is the retention period"})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert "Auth0" in response.json()["detail"]


def test_a_garbled_token_is_401(client):
    response = _ask(client, {"Authorization": "Bearer not-a-jwt"})

    assert response.status_code == 401
    assert "Invalid token header" in response.json()["detail"]


def test_a_token_signed_with_another_key_is_401(client, auth0):
    forged = auth0(Role.ADMIN, wrong_key=True)

    response = _ask(client, {"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401
    assert "Unable to parse authentication token" in response.json()["detail"]


def test_a_token_from_an_unknown_signing_key_is_401(client, auth0):
    stranger = auth0(Role.ADMIN, kid="rotated-away")

    response = _ask(client, {"Authorization": f"Bearer {stranger}"})

    assert response.status_code == 401
    assert "Unable to find appropriate key" in response.json()["detail"]


def test_an_expired_token_is_401_and_says_so(client, auth0):
    expired = auth0(Role.ADMIN, expires_in=timedelta(hours=-1))

    response = _ask(client, {"Authorization": f"Bearer {expired}"})

    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_a_token_for_another_audience_or_issuer_is_401(client, auth0):
    other_api = auth0(Role.ADMIN, audience="https://some-other-api")
    other_tenant = auth0(Role.ADMIN, issuer="https://evil.example.com/")

    for token in (other_api, other_tenant):
        response = _ask(client, {"Authorization": f"Bearer {token}"})

        assert response.status_code == 401
        assert "audience and issuer" in response.json()["detail"]


def test_when_auth0_is_not_configured_every_protected_call_is_503(client, auth, monkeypatch):
    headers = auth()
    monkeypatch.setattr(settings, "auth0_domain", "")

    response = client.get("/auth/me", headers=headers)

    assert response.status_code == 503
    assert "AUTH0_DOMAIN" in response.json()["detail"]


def test_azure_credentials_from_the_login_page_are_applied_after_verification(client, auth, monkeypatch):
    probes = []

    def _verify(endpoint, api_key, api_version, small, embedding):
        probes.append((endpoint, api_key, api_version, small, embedding))

        return True, "ok"

    applied = {}

    def _apply(endpoint, api_key, api_version, small, strong, embedding):
        applied.update(endpoint=endpoint, api_key=api_key, small=small, strong=strong, embedding=embedding)
        monkeypatch.setattr(settings, "azure_openai_endpoint", endpoint)
        monkeypatch.setattr(settings, "azure_openai_api_key", api_key)

    monkeypatch.setattr(azure_credentials, "verify_azure_credentials", _verify)
    monkeypatch.setattr(azure_credentials, "apply_azure_credentials", _apply)

    response = client.post(
        "/auth/azure",
        json={"endpoint": "https://demo.openai.azure.com/", "api_key": "sk-test-1234", "strong_deployment": "gpt-4o"},
        headers=auth(Role.STORE_ASSOCIATE),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["azure_configured"] is True
    assert body["verified"] is True
    assert body["endpoint"] == "demo.openai.azure.com"
    assert probes == [("https://demo.openai.azure.com/", "sk-test-1234", "2024-10-21", "gpt-4o-mini", "text-embedding-3-small")]
    assert applied["strong"] == "gpt-4o"

    assert client.get("/auth/me", headers=auth()).json()["azure_configured"] is True
    assert client.get("/health").json()["azure_configured"] is True


def test_rejected_azure_credentials_are_a_400_and_nothing_is_applied(client, auth, monkeypatch):
    monkeypatch.setattr(azure_credentials, "verify_azure_credentials", lambda *a: (False, "chat deployment 'nope' was rejected: 404"))
    monkeypatch.setattr(azure_credentials, "apply_azure_credentials", lambda *a: pytest.fail("must not apply"))

    response = client.post(
        "/auth/azure",
        json={"endpoint": "https://demo.openai.azure.com/", "api_key": "sk-test-1234", "small_deployment": "nope"},
        headers=auth(),
    )

    assert response.status_code == 400
    assert "rejected" in response.json()["detail"]


def test_azure_credentials_need_an_https_endpoint_and_a_token(client, auth):
    body = {"endpoint": "http://demo.openai.azure.com/", "api_key": "sk-test-1234"}

    assert client.post("/auth/azure", json=body, headers=auth()).status_code == 422
    assert client.post("/auth/azure", json=body).status_code == 401


def test_azure_credentials_can_be_applied_without_verification(client, auth, monkeypatch):
    monkeypatch.setattr(azure_credentials, "verify_azure_credentials", lambda *a: pytest.fail("verify=false must skip the probe"))
    monkeypatch.setattr(azure_credentials, "apply_azure_credentials", lambda *a: None)

    response = client.post(
        "/auth/azure",
        json={"endpoint": "https://demo.openai.azure.com/", "api_key": "sk-test-1234", "verify": False},
        headers=auth(),
    )

    assert response.status_code == 200
    assert response.json()["verified"] is False


@pytest.mark.parametrize("query", ["", "hi", "x" * 2001])
def test_ask_rejects_queries_outside_the_length_bounds(client, auth, query):
    response = _ask(client, auth(), query=query)

    assert response.status_code == 422


def test_ask_rejects_a_document_scope_the_role_cannot_read(client, auth, graph_returns):
    fake = graph_returns(_answered_state())

    response = _ask(client, auth(Role.STORE_MANAGER), document_scope=["gdpr"])

    assert response.status_code == 422
    assert "gdpr" in response.json()["detail"]
    assert fake.calls == [], "the graph must not run for a rejected scope"


def test_ask_passes_a_valid_document_scope_into_the_graph_state(client, auth, graph_returns):
    fake = graph_returns(_answered_state())

    response = _ask(client, auth(Role.STORE_MANAGER), document_scope=["vendor_policy"])

    assert response.status_code == 200
    assert fake.calls[0]["state"]["document_scope_request"] == ["vendor_policy"]


def test_an_answered_run_returns_200_with_only_the_cited_clauses(client, auth, graph_returns):
    fake = graph_returns(_answered_state())

    response = _ask(client, auth())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"].startswith("Invoices are kept")
    assert body["confidence"] == 0.91
    assert body["risk_level"] == "Low"
    assert body["evidence_path"] == "rag"
    assert body["sql_evidence"] is None
    assert body["degraded"] is False

    assert [c["clause_number"] for c in body["citations"]] == ["4.2"]
    assert body["citations"][0]["document_title"] == "Retention Policy"

    assert body["timings"]["t1_ms"] == 10.0
    assert body["timings"]["t2_ms"] == 20.0
    assert body["timings"]["t3_ms"] == body["timings"]["t4_ms"] == 20.0
    assert body["timings"]["total_ms"] >= 0

    state = fake.calls[0]["state"]
    assert state["user_id"] == "mohit"
    assert state["role"] == "compliance_officer"
    assert "doc:gdpr" in state["access_scopes"]
    # the checkpoint is per request (thread:request) so follow-ups never inherit reducer state
    assert fake.calls[0]["config"]["configurable"]["thread_id"] == f"{body['thread_id']}:{body['request_id']}"
    assert fake.calls[0]["config"]["metadata"]["thread_id"] == body["thread_id"]


def test_when_the_draft_cites_nothing_every_retrieved_clause_is_returned_once(client, auth, graph_returns):
    graph_returns(_answered_state(draft=DraftAnswer(answer="Seven years.", cited_clauses=[])))

    body = _ask(client, auth()).json()

    assert sorted(c["clause_number"] for c in body["citations"]) == ["1.1", "4.2"]


def test_sql_evidence_is_surfaced_as_proof(client, auth, graph_returns):
    evidence = SqlEvidence(
        template_id="vendor_status_by_name",
        statement="SELECT ... WHERE name = %(name)s",
        parameters={"name": "Acme"},
        row_count=1,
        rows=[{"name": "Acme", "status": "approved"}],
        as_of=settings.as_of_date,
    )
    graph_returns(_answered_state(sql_evidence=evidence, evidence_path="nl2sql"))

    body = _ask(client, auth()).json()

    assert body["sql_evidence"] == {
        "template_id": "vendor_status_by_name",
        "statement": "SELECT ... WHERE name = %(name)s",
        "parameters": {"name": "Acme"},
        "row_count": 1,
        "as_of": str(settings.as_of_date),
    }


def test_a_supplied_thread_id_is_reused_and_its_history_loaded(client, auth, graph_returns, monkeypatch):
    fake = graph_returns(_answered_state())
    loaded = []

    def _load(thread_id):
        loaded.append(thread_id)

        return [{"role": "user", "content": "earlier question"}], "summary so far"

    monkeypatch.setattr(conversation, "load_thread", _load)

    body = _ask(client, auth(), thread_id="thread-77").json()

    assert body["thread_id"] == "thread-77"
    assert loaded == ["thread-77"]
    assert fake.calls[0]["state"]["conversation_history"] == [{"role": "user", "content": "earlier question"}]
    assert fake.calls[0]["state"]["thread_summary"] == "summary so far"


def test_a_refused_run_returns_200_with_the_refusal_reason(client, auth, graph_returns):
    graph_returns(
        {
            "terminal_outcome": TerminalOutcome.REFUSED.value,
            "refusal_reason": "request rejected by the prompt-injection filter",
        }
    )

    response = _ask(client, auth())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert "prompt-injection" in body["reason"]
    assert "answer" not in body


def test_a_clarification_returns_the_question_and_the_candidates(client, auth, graph_returns):
    graph_returns(
        {
            "terminal_outcome": TerminalOutcome.CLARIFICATION_REQUIRED.value,
            "clarification_question": "Which Acme do you mean?",
            "resolved_entities": [
                ResolvedEntity(
                    surface_form="Acme",
                    entity_type="vendor",
                    status="ambiguous",
                    candidates=[{"id": 1, "name": "Acme Ltd"}, {"id": 2, "name": "Acme Corp"}],
                )
            ],
        }
    )

    body = _ask(client, auth()).json()

    assert body["status"] == "clarification_required"
    assert body["question"] == "Which Acme do you mean?"
    assert [c["name"] for c in body["candidates"]] == ["Acme Ltd", "Acme Corp"]


def test_an_escalated_run_is_202_with_a_poll_url_and_no_answer(client, auth, graph_returns):
    graph_returns(
        {
            "terminal_outcome": TerminalOutcome.ESCALATED.value,
            "escalation_reason": "risk level is High",
            "risk": RiskAssessment(final_level=RiskLevel.HIGH),
            "draft": DraftAnswer(answer="this must never leak"),
        }
    )

    response = _ask(client, auth())

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending_review"
    assert body["risk_level"] == "High"
    assert body["reason"] == "risk level is High"
    assert body["poll_url"].endswith(f"/requests/{body['request_id']}")
    assert "answer" not in body
    assert "this must never leak" not in response.text


def test_a_run_that_ends_without_an_outcome_is_treated_as_escalated(client, auth, graph_returns):
    graph_returns({"terminal_outcome": None})

    response = _ask(client, auth())

    assert response.status_code == 202
    assert response.json()["risk_level"] == "High"


def test_an_answered_run_without_a_draft_is_a_500(client, auth, graph_returns):
    graph_returns({"terminal_outcome": TerminalOutcome.ANSWERED.value, "draft": None})

    response = _ask(client, auth())

    assert response.status_code == 500
    assert "without an answer" in response.json()["detail"]


def test_the_same_idempotency_key_replays_the_response_without_rerunning_the_graph(
    client, auth, graph_returns
):
    fake = graph_returns(_answered_state())
    headers = {**auth(), "Idempotency-Key": "abc-123"}

    first = _ask(client, headers)
    second = _ask(client, headers, query="a completely different question about retention")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert len(fake.calls) == 1


def test_without_a_key_an_escalated_question_is_run_again(client, auth, graph_returns):
    fake = graph_returns({"terminal_outcome": TerminalOutcome.ESCALATED.value})
    headers = auth()

    assert _ask(client, headers).status_code == 202
    assert _ask(client, headers).status_code == 202
    assert len(fake.calls) == 2


def test_an_escalated_replay_keeps_its_202(client, auth, graph_returns):
    graph_returns({"terminal_outcome": TerminalOutcome.ESCALATED.value})
    headers = {**auth(), "Idempotency-Key": "esc-1"}

    assert _ask(client, headers).status_code == 202
    assert _ask(client, headers).status_code == 202


def test_idempotency_keys_are_scoped_to_the_user(client, auth, graph_returns):
    fake = graph_returns(_answered_state())

    _ask(client, {**auth(user_id="alice"), "Idempotency-Key": "shared"}, use_cache=False)
    _ask(client, {**auth(user_id="bob"), "Idempotency-Key": "shared"}, use_cache=False)

    assert len(fake.calls) == 2


def test_without_a_key_an_identical_body_is_still_deduplicated(client, auth, graph_returns):
    fake = graph_returns(_answered_state())
    headers = auth()

    _ask(client, headers)
    _ask(client, headers)
    _ask(client, headers, query="a different question about vendor approvals")

    assert len(fake.calls) == 2


def _package(**overrides) -> ReviewPackage:
    fields = {
        "request_id": "req-1",
        "thread_id": "thread-1",
        "original_query": "is vendor Acme approved",
        "standalone_query": "is vendor Acme approved",
        "conversation_history": [],
        "retrieved_documents": [
            {
                "chunk_id": "vendor-2.1",
                "citation": "Vendor Policy §2.1",
                "section": "Approval",
                "version": "1.0",
                "content": "A vendor must be approved before it is onboarded.",
            }
        ],
        "sql_evidence": None,
        "validation_output": {},
        "reasoning_trace": [],
        "draft_answer": "Acme is approved [Vendor Policy §2.1].",
        "risk_level": "High",
        "confidence": 0.6,
        "evidence_path": "high_risk_panel",
    }
    fields.update(overrides)

    return ReviewPackage(**fields)


def _row(status: str = "pending", **overrides) -> dict:
    row = {
        "request_id": "req-1",
        "thread_id": "thread-1",
        "user_id": "asker",
        "risk_level": "High",
        "reason": "risk level is High",
        "status": status,
        "reviewer_decision": None,
        "reviewed_answer": None,
        "queued_at": "2026-09-01T09:00:00+00:00",
        "reviewed_at": None,
    }
    row.update(overrides)

    return row


def test_an_unknown_request_id_is_404(client, auth, monkeypatch):
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (None, None))

    assert client.get("/requests/nope", headers=auth(Role.STORE_ASSOCIATE)).status_code == 404


def test_a_pending_request_reports_pending_review(client, auth, monkeypatch):
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (_package(), _row()))

    body = client.get("/requests/req-1", headers=auth(Role.STORE_ASSOCIATE)).json()

    assert body["status"] == "pending_review"
    assert body["request_id"] == "req-1"
    assert body["risk_level"] == "High"
    assert body["thread_id"] == "thread-1"
    assert body["reason"] == "risk level is High"
    assert body["queued_at"] == "2026-09-01T09:00:00+00:00"


def test_a_reviewed_request_returns_the_released_answer(client, auth, monkeypatch):
    row = _row(
        status="reviewed",
        reviewer_decision="edit",
        reviewed_answer="Acme is approved until March [Vendor Policy §2.1].",
        reviewed_at="2026-09-01T10:00:00+00:00",
    )
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (_package(), row))

    body = client.get("/requests/req-1", headers=auth(Role.STORE_ASSOCIATE)).json()

    assert body["status"] == "answered"
    assert body["answer"].startswith("Acme is approved until March")
    assert body["reviewer_decision"] == "edit"
    assert body["thread_id"] == "thread-1"


def test_a_reviewed_request_falls_back_to_the_draft_when_nothing_was_edited(client, auth, monkeypatch):
    row = _row(status="reviewed", reviewer_decision="accept", reviewed_answer=None)
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (_package(), row))

    body = client.get("/requests/req-1", headers=auth()).json()

    assert body["answer"] == "Acme is approved [Vendor Policy §2.1]."


@pytest.mark.parametrize("role", [Role.STORE_ASSOCIATE, Role.STORE_MANAGER])
def test_non_reviewer_roles_cannot_see_the_queue(client, auth, role):
    response = client.get("/review/queue", headers=auth(role))

    assert response.status_code == 403
    assert role.value in response.json()["detail"]


@pytest.mark.parametrize("role", [Role.COMPLIANCE_OFFICER, Role.LEGAL_REVIEWER, Role.ADMIN])
def test_reviewer_roles_see_the_pending_queue(client, auth, monkeypatch, role):
    item = QueueItem(
        request_id="req-1",
        thread_id="thread-1",
        user_id="asker",
        risk_level="High",
        reason="risk level is High",
        queued_at=datetime.now(UTC),
    )
    monkeypatch.setattr(escalation_service, "list_pending", lambda limit=50: [item])

    response = client.get("/review/queue", headers=auth(role))

    assert response.status_code == 200
    assert response.json()[0]["request_id"] == "req-1"


def test_the_review_package_is_returned_to_a_reviewer(client, auth, monkeypatch):
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (_package(), _row()))

    body = client.get("/review/req-1", headers=auth()).json()

    assert body["draft_answer"] == "Acme is approved [Vendor Policy §2.1]."
    assert body["evidence_path"] == "high_risk_panel"


def test_the_review_package_404s_for_an_unknown_id(client, auth, monkeypatch):
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (None, None))

    assert client.get("/review/nope", headers=auth()).status_code == 404


@pytest.fixture
def review_setup(monkeypatch):
    """Wire the review endpoint to canned data and record what it does."""
    calls = {"record": [], "resume": [], "stored": [], "reopened": []}

    def _install(package=None, row=None, record_ok=True, resumed=True, final_state="answered"):
        package = package or _package()
        row = row or _row()

        monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (package, row))
        monkeypatch.setattr(
            escalation_service,
            "store_released_answer",
            lambda request_id, answer: calls["stored"].append((request_id, answer)),
        )
        monkeypatch.setattr(escalation_service, "reopen", lambda request_id: calls["reopened"].append(request_id) or True)

        def _record(**kwargs):
            calls["record"].append(kwargs)

            return record_ok

        monkeypatch.setattr(escalation_service, "record_decision", _record)

        def _resume(**kwargs):
            calls["resume"].append(kwargs)

            if not resumed:
                return False, None

            if final_state == "answered":
                return True, {
                    "terminal_outcome": TerminalOutcome.ANSWERED.value,
                    "draft": DraftAnswer(answer=kwargs["answer"] + " (guardrail-cleaned)"),
                }

            if final_state == "rejected_by_guardrail":
                return True, {
                    "terminal_outcome": TerminalOutcome.ESCALATED.value,
                    "escalation_reason": "output guardrail rejected the answer: cites unknown clauses",
                }

            return True, None

        monkeypatch.setattr(routes, "_apply_review_to_checkpoint", _resume)

        return calls

    return _install


def test_submitting_a_review_needs_a_reviewer_role(client, auth, review_setup):
    review_setup()

    response = client.post("/review/req-1", json={"decision": "accept"}, headers=auth(Role.STORE_MANAGER))

    assert response.status_code == 403


def test_reviewing_an_unknown_request_is_404(client, auth, monkeypatch):
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (None, None))

    response = client.post("/review/nope", json={"decision": "accept"}, headers=auth())

    assert response.status_code == 404


def test_reviewing_twice_is_409(client, auth, review_setup):
    review_setup(row=_row(status="reviewed"))

    response = client.post("/review/req-1", json={"decision": "accept"}, headers=auth())

    assert response.status_code == 409
    assert "already been reviewed" in response.json()["detail"]


def test_an_edit_without_text_is_422(client, auth, review_setup):
    calls = review_setup()

    response = client.post("/review/req-1", json={"decision": "edit", "edited_answer": "   "}, headers=auth())

    assert response.status_code == 422
    assert calls["record"] == []


def test_accepting_a_request_with_no_draft_is_422(client, auth, review_setup):
    review_setup(package=_package(draft_answer=""))

    response = client.post("/review/req-1", json={"decision": "accept"}, headers=auth())

    assert response.status_code == 422
    assert "no draft to accept" in response.json()["detail"]


def test_accept_releases_the_panel_consensus_and_ignores_a_stray_edit(client, auth, review_setup):
    calls = review_setup()

    response = client.post(
        "/review/req-1",
        json={"decision": "accept", "edited_answer": "string", "reviewer_notes": "looks right"},
        headers=auth(Role.LEGAL_REVIEWER),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "recorded"
    assert body["decision"] == "accept"
    assert body["resumed"] is True
    assert body["released"] is True
    assert body["outcome"] == "answered"
    assert body["answer_source"] == "panel_consensus"
    assert body["answer"].startswith("Acme is approved [Vendor Policy §2.1].")
    assert "string" not in body["answer"]

    recorded = calls["record"][0]
    assert recorded["decision"] is ReviewDecision.ACCEPT
    assert recorded["edited_answer"] == "Acme is approved [Vendor Policy §2.1]."
    assert recorded["reviewer_id"] == "mohit"
    assert recorded["notes"] == "looks right"


def test_accept_on_a_non_panel_path_is_labelled_system_draft(client, auth, review_setup):
    review_setup(package=_package(evidence_path="rag"))

    body = client.post("/review/req-1", json={"decision": "accept"}, headers=auth()).json()

    assert body["answer_source"] == "system_draft"


def test_edit_releases_the_reviewer_text(client, auth, review_setup):
    calls = review_setup()

    body = client.post(
        "/review/req-1",
        json={"decision": "edit", "edited_answer": "Acme is approved until March [Vendor Policy §2.1]."},
        headers=auth(),
    ).json()

    assert body["released"] is True
    assert body["answer_source"] == "reviewer_edit"
    assert body["answer"].startswith("Acme is approved until March")
    assert calls["resume"][0]["decision"] is ReviewDecision.EDIT
    assert calls["resume"][0]["thread_id"] == "thread-1"


def test_reject_releases_nothing(client, auth, review_setup):
    calls = review_setup()

    body = client.post("/review/req-1", json={"decision": "reject"}, headers=auth()).json()

    assert body["released"] is False
    assert body["outcome"] == "escalated"
    assert body["answer"] is None
    assert "rejected" in body["note"]
    assert calls["record"][0]["decision"] is ReviewDecision.REJECT


def test_a_lost_race_on_the_queue_row_is_409(client, auth, review_setup):
    review_setup(record_ok=False)

    response = client.post("/review/req-1", json={"decision": "accept"}, headers=auth())

    assert response.status_code == 409
    assert "someone else" in response.json()["detail"]


def test_a_reviewed_answer_is_released_even_when_the_checkpoint_cannot_resume(client, auth, review_setup):
    calls = review_setup(resumed=False)

    body = client.post("/review/req-1", json={"decision": "accept"}, headers=auth()).json()

    assert body["resumed"] is False
    assert body["released"] is True
    assert body["outcome"] == "answered"
    assert body["answer"] == "Acme is approved [Vendor Policy §2.1]."
    assert "could not be resumed" in body["note"]
    assert calls["stored"] == [("req-1", "Acme is approved [Vendor Policy §2.1].")]


def test_a_reviewed_answer_the_graph_guardrail_refuses_goes_back_to_the_queue(client, auth, review_setup):
    calls = review_setup(final_state="rejected_by_guardrail")

    body = client.post("/review/req-1", json={"decision": "accept"}, headers=auth()).json()

    assert body["resumed"] is True
    assert body["released"] is False
    assert body["outcome"] == "escalated"
    assert body["answer"] is None
    assert "output guardrail" in body["note"]
    assert "back in the review queue" in body["note"]
    assert calls["reopened"] == ["req-1"]
    assert calls["stored"] == []


def test_a_released_answer_is_stored_for_the_asker(client, auth, review_setup):
    calls = review_setup()

    client.post("/review/req-1", json={"decision": "accept"}, headers=auth())

    assert calls["stored"] == [("req-1", "Acme is approved [Vendor Policy §2.1]. (guardrail-cleaned)")]


def test_a_plain_text_edit_without_citations_is_released(client, auth, review_setup):
    review_setup(resumed=False)

    body = client.post(
        "/review/req-1",
        json={"decision": "edit", "edited_answer": "Yes. Acme may be onboarded once procurement signs the approval."},
        headers=auth(),
    ).json()

    assert body["released"] is True
    assert body["answer_source"] == "reviewer_edit"


def test_an_edit_citing_a_clause_that_was_never_retrieved_is_422_and_nothing_is_recorded(client, auth, review_setup):
    calls = review_setup()

    response = client.post(
        "/review/req-1",
        json={"decision": "edit", "edited_answer": "Acme must be deleted [GDPR §17]."},
        headers=auth(),
    )

    assert response.status_code == 422
    assert "never retrieved" in response.json()["detail"]
    assert "still in the queue" in response.json()["detail"]
    assert calls["record"] == []
    assert calls["resume"] == []


def test_a_rejected_request_releases_nothing_to_the_asker(client, auth, monkeypatch):
    row = _row(status="reviewed", reviewer_decision="reject", reviewed_answer=None)
    monkeypatch.setattr(escalation_service, "fetch_package", lambda request_id: (_package(), row))

    body = client.get("/requests/req-1", headers=auth(Role.STORE_MANAGER)).json()

    assert body["status"] == "rejected"
    assert "answer" not in body
    assert "rejected" in body["note"]


def test_a_rejection_stores_no_answer(client, auth, review_setup):
    calls = review_setup()

    client.post("/review/req-1", json={"decision": "reject", "reviewer_notes": "insufficient evidence"}, headers=auth())

    assert calls["record"][0]["edited_answer"] is None
    assert calls["stored"] == []


def _upload(
    client, headers, name="policy.md", payload=b"# Retention\n\n4.2 Keep invoices seven years.", **params
):
    return client.post(
        "/ingest",
        files={"file": (name, payload, "text/markdown")},
        headers=headers,
        params=params,
    )


@pytest.fixture
def ingest_ok(monkeypatch):
    monkeypatch.setattr(routes, "check_embed_model_compatibility", lambda: (True, "ok"))

    def _install(result: IngestionResult | Exception):
        calls = []

        def _ingest(path, original_name, force):
            calls.append({"path": path, "original_name": original_name, "force": force})

            if isinstance(result, Exception):
                raise result

            return result

        monkeypatch.setattr(routes, "ingest_file", _ingest)

        return calls

    return _install


@pytest.mark.parametrize("role", [Role.STORE_ASSOCIATE, Role.COMPLIANCE_OFFICER, Role.LEGAL_REVIEWER])
def test_only_admin_can_ingest(client, auth, role):
    response = _upload(client, auth(role))

    assert response.status_code == 403
    assert "'admin'" in response.json()["detail"]


def test_an_unsupported_file_type_is_415(client, auth):
    response = _upload(client, auth(Role.ADMIN), name="policy.xlsx")

    assert response.status_code == 415
    assert ".xlsx" in response.json()["detail"]


def test_an_empty_upload_is_400(client, auth):
    response = _upload(client, auth(Role.ADMIN), payload=b"")

    assert response.status_code == 400


def test_an_oversized_upload_is_413(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "upload_max_bytes", 10)

    response = _upload(client, auth(Role.ADMIN), payload=b"x" * 11)

    assert response.status_code == 413


def test_an_incompatible_embedding_model_blocks_ingestion_with_409(client, auth, monkeypatch):
    monkeypatch.setattr(
        routes, "check_embed_model_compatibility", lambda: (False, "table was built with another model")
    )

    response = _upload(client, auth(Role.ADMIN))

    assert response.status_code == 409
    assert "another model" in response.json()["detail"]


def test_a_successful_ingest_reports_the_node_counts(client, auth, ingest_ok):
    calls = ingest_ok(
        IngestionResult(
            file_name="policy.md",
            doc_type="retention_policy",
            version="2.0",
            parsed_with="markdown",
            text_nodes=4,
            table_nodes=1,
            image_nodes=0,
            superseded=2,
        )
    )

    response = _upload(client, auth(Role.ADMIN), force="true")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "indexed"
    assert body["doc_type"] == "retention_policy"
    assert body["total_nodes"] == 5
    assert body["superseded_nodes"] == 2

    assert calls[0]["original_name"] == "policy.md"
    assert calls[0]["force"] is True
    assert calls[0]["path"].endswith(".md")


def test_a_skipped_ingest_reports_why(client, auth, ingest_ok):
    ingest_ok(IngestionResult(file_name="policy.md", skipped=True, reason="already indexed (same hash)"))

    body = _upload(client, auth(Role.ADMIN)).json()

    assert body == {**body, "status": "skipped", "reason": "already indexed (same hash)"}


def test_a_document_the_parser_rejects_is_422(client, auth, ingest_ok):
    ingest_ok(ValueError("no version marker found in the document"))

    response = _upload(client, auth(Role.ADMIN))

    assert response.status_code == 422
    assert "version marker" in response.json()["detail"]


def test_an_unexpected_ingest_failure_is_500_and_names_the_file(client, auth, ingest_ok):
    ingest_ok(RuntimeError("embedding endpoint down"))

    response = _upload(client, auth(Role.ADMIN))

    assert response.status_code == 500
    assert "policy.md" in response.json()["detail"]


def test_corpus_status_is_admin_only_and_reports_the_index(client, auth, monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "check_embed_model_compatibility", lambda: (True, ""))
    monkeypatch.setattr(routes, "corpus_is_indexed", lambda: True)
    monkeypatch.setattr(routes, "corpus_dir", lambda: tmp_path)
    monkeypatch.setattr(routes, "active_embed_model_name", lambda: "text-embedding-3-small")
    monkeypatch.setattr(routes, "indexed_document_summary", lambda: [{"doc_type": "gdpr", "chunks": 12}])
    monkeypatch.setattr(routes, "unindexed_files", lambda: ["new_policy.pdf"])

    assert client.get("/ingest/status", headers=auth(Role.COMPLIANCE_OFFICER)).status_code == 403

    body = client.get("/ingest/status", headers=auth(Role.ADMIN)).json()

    assert body["indexed"] is True
    assert body["embed_model"] == "text-embedding-3-small"
    assert body["documents"] == [{"doc_type": "gdpr", "chunks": 12}]
    assert body["unindexed_files"] == ["new_policy.pdf"]


def test_rebuild_refuses_to_run_without_confirmation(client, auth, monkeypatch):
    cleared = []
    monkeypatch.setattr(routes, "clear_vector_table", lambda: cleared.append(1) or 0)

    response = client.post("/ingest/rebuild", headers=auth(Role.ADMIN))

    assert response.status_code == 400
    assert "confirm=true" in response.json()["detail"]
    assert cleared == []


def test_rebuild_409s_when_the_corpus_directory_is_missing(client, auth, monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "corpus_dir", lambda: tmp_path / "missing")

    response = client.post("/ingest/rebuild", params={"confirm": "true"}, headers=auth(Role.ADMIN))

    assert response.status_code == 409


def test_rebuild_clears_then_reingests_and_reports_both(client, auth, monkeypatch, tmp_path):
    order = []

    monkeypatch.setattr(routes, "corpus_dir", lambda: tmp_path)
    monkeypatch.setattr(routes, "clear_vector_table", lambda: order.append("clear") or 42)
    monkeypatch.setattr(routes, "active_embed_model_name", lambda: "text-embedding-3-small")
    monkeypatch.setattr(routes, "indexed_document_summary", lambda: [])
    monkeypatch.setattr(routes, "unindexed_files", lambda: [])

    def _ingest(directory):
        order.append("ingest")

        return IngestionReport(
            results=[
                IngestionResult(file_name="a.md", text_nodes=3),
                IngestionResult(file_name="b.md", skipped=True, reason="unsupported"),
            ]
        )

    monkeypatch.setattr(routes, "ingest_directory", _ingest)

    body = client.post("/ingest/rebuild", params={"confirm": "true"}, headers=auth(Role.ADMIN)).json()

    assert order == ["clear", "ingest"]
    assert body["cleared_chunks"] == 42
    assert body["indexed_files"] == ["a.md"]
    assert body["skipped_files"] == [{"file_name": "b.md", "reason": "unsupported"}]
    assert body["total_nodes"] == 3


def test_rebuild_reports_a_failure_after_the_table_was_cleared(client, auth, monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "corpus_dir", lambda: tmp_path)
    monkeypatch.setattr(routes, "clear_vector_table", lambda: 7)

    def _boom(directory):
        raise RuntimeError("embedding endpoint down")

    monkeypatch.setattr(routes, "ingest_directory", _boom)

    response = client.post("/ingest/rebuild", params={"confirm": "true"}, headers=auth(Role.ADMIN))

    assert response.status_code == 500
    assert "was cleared but re-ingestion failed" in response.json()["detail"]


def _slo_report(hours: int) -> dict:
    return {
        "window_hours": hours,
        "sample_size": 3,
        "measured_from": "request_latency",
        "total_target_basis": "SLO_TOTAL_P95_MS",
        "stages": [
            {
                "stage": "t1",
                "meaning": "guardrail+rewrite",
                "p95_ms": 100.0,
                "target_p95_ms": 6000.0,
                "meets_slo": True,
            }
        ],
        "total": {"p95_ms": 900.0, "target_p95_ms": 28000.0, "meets_slo": True},
        "breached_stages": [],
        "breached_paths": [],
        "outcomes": [{"outcome": "answered", "count": 3}],
        "evidence_paths": [],
    }


def test_slo_metrics_default_to_the_configured_window(client, auth, monkeypatch):
    monkeypatch.setattr(routes.slo, "latency_report", _slo_report)

    body = client.get("/metrics/slo", headers=auth()).json()

    assert body["window_hours"] == settings.slo_window_hours
    assert body["total"]["meets_slo"] is True


def test_slo_metrics_accept_a_window_and_reject_a_silly_one(client, auth, monkeypatch):
    monkeypatch.setattr(routes.slo, "latency_report", _slo_report)

    assert client.get("/metrics/slo", params={"hours": 48}, headers=auth()).json()["window_hours"] == 48
    assert client.get("/metrics/slo", params={"hours": 0}, headers=auth()).status_code == 422
    assert client.get("/metrics/slo", params={"hours": 721}, headers=auth()).status_code == 422


def test_slo_metrics_need_a_reviewer_role(client, auth):
    assert client.get("/metrics/slo", headers=auth(Role.STORE_MANAGER)).status_code == 403


def test_slo_metrics_are_503_when_history_is_unavailable(client, auth, monkeypatch):
    def _boom(hours):
        raise RuntimeError("request_latency table missing")

    monkeypatch.setattr(routes.slo, "latency_report", _boom)

    response = client.get("/metrics/slo", headers=auth())

    assert response.status_code == 503
    assert "request_latency" in response.json()["detail"]


def test_health_is_degraded_when_the_database_probe_fails(client):
    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["database"] == "down"
    assert body["escalation_queue_depth"] is None
    assert body["audit_log"]["writing"] in (True, False)
    assert body["deadlines_seconds"]["standard"] == settings.deadline_seconds_standard


def test_health_is_ok_when_the_database_answers(client, monkeypatch):
    monkeypatch.setattr(escalation_service, "queue_depth", lambda: 4)

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "up"
    assert body["escalation_queue_depth"] == 4
    assert body["as_of_date"] == str(settings.as_of_date)


def test_health_needs_no_token(client):
    assert client.get("/health").status_code == 200


def test_a_repeated_question_is_served_from_the_response_cache(client, auth, graph_returns):
    fake = graph_returns(_answered_state())
    headers = auth(user_id="alice")

    first = _ask(client, headers).json()
    second = _ask(client, auth(user_id="bob")).json()

    assert len(fake.calls) == 1
    assert first.get("cache") is None
    assert second["status"] == "answered"
    assert second["answer"] == first["answer"]
    assert second["request_id"] != first["request_id"]
    assert second["cache"]["hit"] is True
    assert second["cache"]["kind"] == "exact"
    assert second["cache"]["source_request_id"] == first["request_id"]
    assert second["cache"]["saved_tokens_estimate"] >= 0


def test_the_cache_never_crosses_roles(client, auth, graph_returns):
    fake = graph_returns(_answered_state())

    _ask(client, auth(Role.COMPLIANCE_OFFICER, user_id="officer"))
    response = _ask(client, auth(Role.STORE_ASSOCIATE, user_id="associate"))

    assert len(fake.calls) == 2
    assert response.json().get("cache") is None


def test_use_cache_false_bypasses_the_cache(client, auth, graph_returns):
    fake = graph_returns(_answered_state())

    _ask(client, auth())
    response = _ask(client, auth(), use_cache=False)

    assert len(fake.calls) == 2
    assert response.json().get("cache") is None


def test_a_follow_up_with_history_is_not_looked_up_in_the_cache(client, auth, graph_returns, monkeypatch):
    fake = graph_returns(_answered_state())
    _ask(client, auth())

    monkeypatch.setattr(conversation, "load_thread", lambda thread_id: ([{"role": "user", "content": "earlier"}], ""))
    response = _ask(client, auth(), thread_id="thread-with-history")

    assert len(fake.calls) == 2
    assert response.json().get("cache") is None


def test_escalations_and_degraded_answers_are_not_cached(client, auth, graph_returns):
    fake = graph_returns(_answered_state(degraded=True))

    _ask(client, auth())
    _ask(client, auth(user_id="someone-else"))

    assert len(fake.calls) == 2


def test_a_cache_hit_is_audited_and_recorded_in_the_slo_history(client, auth, graph_returns, monkeypatch):
    graph_returns(_answered_state())
    audited: list[tuple[str, str]] = []
    recorded: list[str] = []

    monkeypatch.setattr(routes.audit, "audit_from_state", lambda state, node, event, detail=None: audited.append((node, event)))
    monkeypatch.setattr(routes.slo, "record_latency", lambda state, total_ms, outcome: recorded.append(state.get("evidence_path")))

    _ask(client, auth())
    _ask(client, auth(user_id="second"))

    assert ("response_cache", "cache_hit") in audited
    assert recorded[-1] == "cache"


def test_the_trace_carries_model_routing_and_llm_usage(client, auth, graph_returns):
    graph_returns(
        _answered_state(
            model_routing=[{"node": "rag_generate", "tier": "small", "strategy": "heuristic", "reason": "simple"}],
        )
    )

    body = _ask(client, auth()).json()

    assert body["trace"]["model_routing"][0]["tier"] == "small"
    assert body["trace"]["llm_usage"]["calls"] == 0
    assert "usd" in body["trace"]["llm_usage"]


def test_optimization_metrics_need_a_reviewer(client, auth):
    assert client.get("/metrics/optimization", headers=auth(Role.STORE_ASSOCIATE)).status_code == 403


def test_optimization_metrics_report_caches_routing_and_cost(client, auth, graph_returns):
    graph_returns(_answered_state())
    _ask(client, auth())
    _ask(client, auth(user_id="second"))

    body = client.get("/metrics/optimization", headers=auth()).json()

    assert set(body) == {"generated_at", "caches", "routing", "cost"}
    assert body["caches"]["enabled"]["response"] is True
    assert body["routing"]["strategy"] == settings.model_routing_strategy
    assert "small_tier_share" in body["cost"]
    assert body["cost"]["budget_usd_per_request"] == settings.cost_budget_usd_per_request


def test_cache_invalidation_is_admin_only_and_empties_the_caches(client, auth, graph_returns):
    fake = graph_returns(_answered_state())
    _ask(client, auth(user_id="first"))

    assert client.post("/cache/invalidate", headers=auth(Role.COMPLIANCE_OFFICER)).status_code == 403

    response = client.post("/cache/invalidate?reason=test", headers=auth(Role.ADMIN))

    assert response.status_code == 200
    assert response.json()["reason"] == "test"
    assert "response" in response.json()["cleared"]

    _ask(client, auth(user_id="second"))

    assert len(fake.calls) == 2


def test_ingesting_a_document_invalidates_the_caches(client, auth, graph_returns, monkeypatch, tmp_path):
    fake = graph_returns(_answered_state())
    _ask(client, auth(user_id="first"))

    monkeypatch.setattr(routes, "check_embed_model_compatibility", lambda: (True, "ok"))
    monkeypatch.setattr(
        routes,
        "ingest_file",
        lambda path, name, force: IngestionResult(
            file_name=name, doc_type="gdpr", version="1.0", parsed_with="pymupdf",
            text_nodes=3, table_nodes=0, image_nodes=0, superseded=0, skipped=False,
        ),
    )

    response = client.post(
        "/ingest",
        files={"file": ("gdpr.pdf", b"%PDF-1.4 fake", "application/pdf")},
        headers=auth(Role.ADMIN),
    )

    assert response.status_code == 200, response.text

    _ask(client, auth(user_id="second"))

    assert len(fake.calls) == 2


def test_health_reports_the_optimisation_summary(client):
    body = client.get("/health").json()

    assert body["optimization"]["routing_strategy"] == settings.model_routing_strategy
    assert set(body["optimization"]["caches_enabled"]) == {"response", "semantic", "retrieval", "llm"}
    assert "usd_spent" in body["optimization"]


class _Snapshot:
    def __init__(self, values):
        self.values = values


class _CheckpointGraph:
    def __init__(self, states: dict):
        self.states = states

    def get_state(self, config):
        return _Snapshot(self.states.get(config["configurable"]["thread_id"], {}))


def test_the_review_resume_never_picks_up_another_requests_checkpoint():
    graph = _CheckpointGraph({"thread-1": {"request_id": "a-later-request", "draft": None}})

    _config, snapshot = routes._review_snapshot(graph, {"configurable": {}}, "thread-1", "req-1")

    assert snapshot is None


def test_the_review_resume_prefers_the_per_request_checkpoint():
    graph = _CheckpointGraph(
        {
            "thread-1:req-1": {"request_id": "req-1"},
            "thread-1": {"request_id": "a-later-request"},
        }
    )

    config, snapshot = routes._review_snapshot(graph, {"configurable": {}}, "thread-1", "req-1")

    assert snapshot.values["request_id"] == "req-1"
    assert config["configurable"]["thread_id"] == "thread-1:req-1"


def test_a_reviewed_policy_answer_needs_no_clause_citation_but_a_fabricated_one_is_refused(no_db):
    from src.nodes.terminal import output_guardrail_node
    from tests.conftest import base_state

    chunk = RetrievedChunk(
        chunk_id="vendor-2.1",
        doc_type="vendor_policy",
        document_title="Vendor Policy",
        section="Approval",
        clause_number="2.1",
        version="1.0",
        content="A vendor must be approved before it is onboarded.",
    )

    plain = base_state(
        "can we onboard Acme",
        draft=DraftAnswer(answer="Yes, once procurement signs the approval Acme can be onboarded."),
        retrieved_chunks=[chunk],
        reviewer_decision="edit",
    )
    fabricated = {**plain, "draft": DraftAnswer(answer="Yes, onboarding is allowed [GDPR §17].")}

    assert output_guardrail_node(plain)["terminal_outcome"] == TerminalOutcome.ANSWERED.value
    assert output_guardrail_node(fabricated)["terminal_outcome"] == TerminalOutcome.ESCALATED.value
