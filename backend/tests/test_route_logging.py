import time

import pytest

from src.nodes import planner as planner_module
from src.observability.tracing import _span
from src.schemas.enums import EvidencePath, Intent, RiskLevel
from src.schemas.models import EvidencePlan, IntentResult, PlanStep, RiskAssessment

SCOPES = ["doc:privacy_policy", "doc:vendor_policy", "table:vendors", "risk:High"]


def _plan(path: EvidencePath) -> EvidencePlan:
    return EvidencePlan(
        path=path,
        steps=[PlanStep(order=1, source="policy_kb", objective="o", must_prove="p")],
    )


def _state(intent: Intent) -> dict:
    return {
        "request_id": "req-42",
        # two status filters: no single vetted query answers it, so the planner model is asked
        "standalone_query": "which vendors are non compliant and still approved",
        "access_scopes": SCOPES,
        "risk": RiskAssessment(final_level=RiskLevel.LOW),
        "intent": IntentResult(intent=intent),
        "resolved_entities": [],
        "started_ts": time.monotonic(),
        "deadline_ts": time.monotonic() + 30,
        "token_budget": 48000,
        "tokens_spent": 0,
    }


@pytest.fixture
def planner_returns(monkeypatch):
    def _install(plan: EvidencePlan):
        class _Model:
            def invoke(self, _messages, config=None):
                return plan

        class _Factory:
            def with_structured_output(self, _schema):
                return _Model()

        monkeypatch.setattr(planner_module, "model_for", lambda _name: _Factory())

    return _install


def test_a_clamped_route_is_logged_with_both_paths_and_the_reason(planner_returns, caplog):
    planner_returns(_plan(EvidencePath.RAG))

    with caplog.at_level("INFO", logger="src.nodes.planner"):
        planner_module.planner_node(_state(Intent.VENDOR_STATUS))

    line = next(record.getMessage() for record in caplog.records if "route intent=" in record.getMessage())

    assert "intent=vendor_status" in line
    assert "proposed=rag" in line
    assert "chosen=nl2sql" in line
    assert "clamped=True" in line
    assert "req-42" in line


def test_an_unclamped_route_is_logged_too(planner_returns, caplog):
    planner_returns(_plan(EvidencePath.RAG))

    with caplog.at_level("INFO", logger="src.nodes.planner"):
        planner_module.planner_node(_state(Intent.POLICY_LOOKUP))

    line = next(record.getMessage() for record in caplog.records if "route intent=" in record.getMessage())

    assert "chosen=rag" in line
    assert "clamped=False" in line


def test_the_decision_reaches_the_trace_span(planner_returns):
    planner_returns(_plan(EvidencePath.AGENTIC))

    result = planner_module.planner_node(_state(Intent.VENDOR_STATUS))
    span = _span("planner", result, 12.0, 34.0, "completed")

    assert span["routed_path"] == EvidencePath.NL2SQL.value
    assert span["route_clamped"]
    assert "agentic" in span["route_reason"]


def test_a_node_without_a_routing_decision_leaves_the_span_clean():
    span = _span("rag_path", {"request_id": "req-42"}, 1.0, 2.0, "completed")

    assert "routed_path" not in span
