"""Logging: request correlation, the access log, and the node-level log lines."""

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from src.observability.logging_config import (
    RequestContextFilter,
    bind_request_context,
    current_request_context,
    current_request_id,
    request_context,
    reset_request_context,
    run_with_context,
    with_request_context,
)


def _record(message: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, message, None, None)


def test_the_filter_stamps_a_dash_outside_any_request():
    record = _record()

    assert RequestContextFilter().filter(record) is True
    assert record.request_id == "-"
    assert record.thread_id == "-"
    assert record.user_id == "-"


def test_the_filter_stamps_the_bound_request_context():
    with request_context(request_id="req-1", thread_id="thr-1", user_id="mohit"):
        record = _record()
        RequestContextFilter().filter(record)

        assert (record.request_id, record.thread_id, record.user_id) == ("req-1", "thr-1", "mohit")
        assert current_request_id() == "req-1"
        assert current_request_context()["user_id"] == "mohit"

    assert current_request_id() is None


def test_the_filter_keeps_an_explicit_extra():
    with request_context(request_id="req-1"):
        record = _record()
        record.request_id = "explicit"
        RequestContextFilter().filter(record)

        assert record.request_id == "explicit"


def test_bind_and_reset_round_trip():
    tokens = bind_request_context(request_id="req-2")

    assert current_request_id() == "req-2"

    reset_request_context(tokens)

    assert current_request_id() is None


def test_worker_threads_inherit_the_context_only_through_with_request_context():
    with request_context(request_id="req-3"), ThreadPoolExecutor(max_workers=1) as pool:
        inherited = pool.submit(with_request_context(current_request_id)).result()
        bare = pool.submit(current_request_id).result()

    assert inherited == "req-3"
    assert bare is None


def test_a_wrapper_can_be_called_more_than_once_and_concurrently():
    with request_context(request_id="req-4"):
        wrapped = with_request_context(current_request_id)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [future.result() for future in [pool.submit(wrapped) for _ in range(6)]]

    assert results == ["req-4"] * 6


def test_run_with_context_runs_in_a_copy_on_the_same_thread():
    with request_context(request_id="req-5"):
        assert run_with_context(current_request_id) == "req-5"


@pytest.fixture
def client(no_db, no_tracing):
    from main import app

    return TestClient(app)


def test_every_response_carries_a_request_id_header(client):
    response = client.get("/health")

    assert response.status_code in (200, 503)
    assert len(response.headers["X-Request-ID"]) >= 8


def test_an_incoming_request_id_is_propagated_and_logged(client, auth0, caplog):
    with caplog.at_level("INFO", logger="rpids.access"):
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {auth0('admin')}", "X-Request-ID": "trace-abc"},
        )

    assert response.headers["X-Request-ID"] == "trace-abc"

    access = [r for r in caplog.records if r.name == "rpids.access" and "http_request " in r.getMessage()]

    assert access, "no access log line was written"
    assert access[-1].request_id == "trace-abc"
    assert "method=GET path=/auth/me status=200" in access[-1].getMessage()


def test_health_polls_are_demoted_to_debug(client, caplog):
    with caplog.at_level("DEBUG", logger="rpids.access"):
        client.get("/health")

    lines = [
        r
        for r in caplog.records
        if r.name == "rpids.access" and "path=/health" in r.getMessage() and "http_request " in r.getMessage()
    ]

    assert lines and lines[-1].levelno == logging.DEBUG


def test_client_errors_log_at_warning(client, caplog):
    with caplog.at_level("WARNING", logger="rpids.access"):
        client.post("/ask", json={"query": "what is the retention period"})

    lines = [r for r in caplog.records if r.name == "rpids.access" and "status=401" in r.getMessage()]

    assert lines and lines[-1].levelno == logging.WARNING


def test_the_graph_request_id_matches_the_http_request_id(client, auth0, monkeypatch, caplog):
    from src.api import routes
    from src.core import conversation
    from src.schemas.enums import TerminalOutcome

    monkeypatch.setattr(conversation, "load_thread", lambda thread_id: ([], ""))
    monkeypatch.setattr(conversation, "record_user_turn", lambda *a, **k: None)
    monkeypatch.setattr(routes.slo, "record_latency", lambda *a, **k: None)

    class _Graph:
        def invoke(self, state, config=None):
            return {**state, "terminal_outcome": TerminalOutcome.REFUSED.value, "refusal_reason": "test"}

    monkeypatch.setattr(routes, "get_compiled_graph", lambda: _Graph())

    token = auth0("admin", user_id="u")

    with caplog.at_level("INFO"):
        response = client.post(
            "/ask",
            json={"query": "what is the retention period", "use_cache": False},
            headers={"Authorization": f"Bearer {token}", "X-Request-ID": "trace-xyz"},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "trace-xyz"

    node_lines = [r for r in caplog.records if r.name == "src.api.routes" and "ask END" in r.getMessage()]

    assert node_lines and node_lines[-1].request_id == "trace-xyz"


def test_confidence_node_logs_the_breakdown_against_the_threshold(caplog):
    from src.nodes.confidence import confidence_node
    from tests.conftest import base_state

    state = base_state()

    with caplog.at_level("INFO", logger="src.nodes.confidence"):
        confidence_node(state)

    line = next(r.getMessage() for r in caplog.records if "confidence " in r.getMessage())

    assert "BELOW_THRESHOLD" in line
    assert "threshold=" in line


def test_graph_edges_are_logged_with_their_reason(caplog):
    from src.graph.routing import after_confidence
    from src.schemas.enums import RiskLevel
    from src.schemas.models import RiskAssessment

    state = {"request_id": "req-edge", "risk": RiskAssessment(final_level=RiskLevel.HIGH)}

    with caplog.at_level("INFO", logger="src.graph.routing"):
        assert after_confidence(state) == "escalate"

    line = next(r.getMessage() for r in caplog.records if "edge confidence_scoring" in r.getMessage())

    assert "-> escalate" in line
    assert "risk is High" in line
    assert "req-edge" in line


def test_configs_logger_shim_delegates_to_the_central_configuration():
    from configs import logger as shim

    app_logger = shim.setup_logging("INFO")

    assert app_logger.name == "rpids"
    assert shim.logger.name == "rpids"
    assert all(
        isinstance(h.filters[0], RequestContextFilter) for h in logging.getLogger().handlers if h.filters
    )
