import time

import pytest

from src.nodes import planner as planner_module
from src.schemas.enums import EvidencePath, Intent, RiskLevel
from src.schemas.models import EvidencePlan, IntentResult, PlanStep, RiskAssessment, ValidationReport

COMPLIANCE_OFFICER_SCOPES = [
    "doc:privacy_policy",
    "doc:vendor_policy",
    "table:vendors",
    "table:audit_logs",
    "risk:High",
]

ASSOCIATE_SCOPES = ["doc:privacy_policy"]


def _plan(path: EvidencePath, source: str = "policy_kb", revision: int = 0) -> EvidencePlan:
    return EvidencePlan(
        path=path,
        steps=[PlanStep(order=1, source=source, objective="o", must_prove="p")],
        revision=revision,
    )


def _state(intent: Intent, scopes=None, **overrides) -> dict:
    state = {
        "standalone_query": "which vendors are non compliant",
        "access_scopes": scopes if scopes is not None else COMPLIANCE_OFFICER_SCOPES,
        "risk": RiskAssessment(final_level=RiskLevel.LOW),
        "intent": IntentResult(intent=intent),
        "resolved_entities": [],
        "started_ts": time.monotonic(),
        "deadline_ts": time.monotonic() + 30,
        "token_budget": 48000,
        "tokens_spent": 0,
    }
    state.update(overrides)

    return state


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


def test_a_records_question_is_clamped_even_when_the_model_asks_for_rag(planner_returns):
    planner_returns(_plan(EvidencePath.RAG))

    result = planner_module.planner_node(_state(Intent.RECORD_LOOKUP))

    assert result["routed_path"] == EvidencePath.NL2SQL.value
    assert result["path_decision"]["clamped"]
    assert result["plan"].path is EvidencePath.NL2SQL


def test_a_policy_question_is_clamped_even_when_the_model_asks_for_agentic(planner_returns):
    planner_returns(_plan(EvidencePath.AGENTIC))

    result = planner_module.planner_node(_state(Intent.POLICY_LOOKUP))

    assert result["routed_path"] == EvidencePath.RAG.value


def test_a_permitted_choice_is_left_alone(planner_returns):
    planner_returns(_plan(EvidencePath.AGENTIC, source="both"))

    result = planner_module.planner_node(_state(Intent.COMPLIANCE_CHECK))

    assert result["routed_path"] == EvidencePath.AGENTIC.value
    assert not result["path_decision"]["clamped"]


def test_high_risk_overrides_the_model_entirely(planner_returns):
    planner_returns(_plan(EvidencePath.RAG))

    state = _state(Intent.POLICY_LOOKUP, risk=RiskAssessment(final_level=RiskLevel.HIGH))
    result = planner_module.planner_node(state)

    assert result["routed_path"] == EvidencePath.HIGH_RISK_PANEL.value


def test_a_role_without_tables_cannot_be_routed_at_a_records_question(planner_returns):
    planner_returns(_plan(EvidencePath.NL2SQL))

    result = planner_module.planner_node(_state(Intent.RECORD_LOOKUP, scopes=ASSOCIATE_SCOPES))

    assert result["routed_path"] == "escalate"
    assert result["escalation_reason"]


def test_a_model_failure_falls_back_to_the_intent_default(monkeypatch):
    class _Model:
        def invoke(self, _messages, config=None):
            raise RuntimeError("azure is down")

    class _Factory:
        def with_structured_output(self, _schema):
            return _Model()

    monkeypatch.setattr(planner_module, "model_for", lambda _name: _Factory())

    result = planner_module.planner_node(_state(Intent.VENDOR_STATUS))

    assert result["routed_path"] == EvidencePath.NL2SQL.value


def test_a_reflection_plan_is_honoured_instead_of_being_re_drafted(monkeypatch):
    def _explode(_name):
        raise AssertionError("the planner must not call the model on a reflection re-plan")

    monkeypatch.setattr(planner_module, "model_for", _explode)

    state = _state(
        Intent.COMPLIANCE_CHECK,
        plan=_plan(EvidencePath.AGENTIC, source="both", revision=1),
        plan_from_reflection=True,
        validation=ValidationReport(passed=False),
        replan_directive="widen retrieval to the vendor policy",
    )

    result = planner_module.planner_node(state)

    assert result["routed_path"] == EvidencePath.AGENTIC.value
    assert result["path_decision"]["replanned"]
    assert result["plan"].revision == 1
    assert result["tokens_spent"] == 0
    assert result["plan_from_reflection"] is False


def test_a_reflection_plan_is_still_clamped_to_the_allowed_set(monkeypatch):
    monkeypatch.setattr(planner_module, "model_for", lambda _name: None)

    state = _state(
        Intent.RECORD_LOOKUP,
        plan=_plan(EvidencePath.RAG, revision=1),
        plan_from_reflection=True,
    )

    result = planner_module.planner_node(state)

    assert result["routed_path"] == EvidencePath.NL2SQL.value
