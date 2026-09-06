import time

from src.graph.routing import after_confidence, after_validation, route_evidence_path
from src.nodes.risk_assessment import fuse
from src.schemas.enums import DefectType, EvidencePath, Intent, RiskLevel
from src.schemas.models import (
    ConfidenceBreakdown,
    Defect,
    EvidencePlan,
    IntentResult,
    PlanStep,
    RiskAssessment,
    RiskSignal,
    ValidationReport,
)


def _plan(path: EvidencePath) -> EvidencePlan:
    return EvidencePlan(
        path=path,
        steps=[PlanStep(order=1, source="policy_kb", objective="o", must_prove="p")],
    )


def _base_state(**overrides) -> dict:
    state = {
        "deadline_ts": time.monotonic() + 10,
        "token_budget": 12000,
        "tokens_spent": 0,
        "reflection_count": 0,
    }
    state.update(overrides)

    return state


def test_probe_overrides_a_lower_classifier():
    signals = [
        RiskSignal(layer="L1", level=RiskLevel.LOW),
        RiskSignal(layer="L2", level=RiskLevel.LOW),
        RiskSignal(layer="L3", level=RiskLevel.HIGH, scenario_id="vendor_engagement"),
    ]

    assessment = fuse(signals, unresolved_entity=False)

    assert assessment.final_level is RiskLevel.HIGH
    assert assessment.disagreement


def test_classifier_may_raise_above_lexical_floor():
    signals = [
        RiskSignal(layer="L1", level=RiskLevel.MEDIUM),
        RiskSignal(layer="L2", level=RiskLevel.HIGH),
    ]

    assert fuse(signals, unresolved_entity=False).final_level is RiskLevel.HIGH


def test_unresolved_entity_floors_at_medium():
    signals = [RiskSignal(layer="L1", level=RiskLevel.LOW), RiskSignal(layer="L2", level=RiskLevel.LOW)]

    assert fuse(signals, unresolved_entity=True).final_level is RiskLevel.MEDIUM


def test_high_risk_always_routes_to_the_panel():
    state = _base_state(
        risk=RiskAssessment(final_level=RiskLevel.HIGH),
        plan=_plan(EvidencePath.RAG),
    )

    assert route_evidence_path(state) == EvidencePath.HIGH_RISK_PANEL.value


def test_hybrid_downgrades_to_policy_when_the_intent_turns_on_policy():
    state = _base_state(
        risk=RiskAssessment(final_level=RiskLevel.MEDIUM),
        intent=IntentResult(intent=Intent.INCIDENT_GUIDANCE),
        routed_path=EvidencePath.HYBRID.value,
        deadline_ts=time.monotonic() + 0.5,
    )

    assert route_evidence_path(state) == EvidencePath.RAG.value


def test_hybrid_downgrades_to_records_when_the_intent_turns_on_records():
    state = _base_state(
        risk=RiskAssessment(final_level=RiskLevel.MEDIUM),
        intent=IntentResult(intent=Intent.RETENTION_QUERY),
        routed_path=EvidencePath.HYBRID.value,
        deadline_ts=time.monotonic() + 0.5,
    )

    assert route_evidence_path(state) == EvidencePath.NL2SQL.value


def test_failed_validation_reflects_then_escalates():
    defects = [Defect(defect_type=DefectType.UNGROUNDED_CLAIM, description="d")]
    report = ValidationReport(passed=False, defects=defects)

    first = _base_state(validation=report, reflection_count=0)
    assert after_validation(first) == "reflect"

    exhausted = _base_state(validation=report, reflection_count=2)
    assert after_validation(exhausted) == "escalate"


def test_low_confidence_escalates():
    state = _base_state(
        risk=RiskAssessment(final_level=RiskLevel.LOW),
        confidence=ConfidenceBreakdown(final_score=0.4),
        validation=ValidationReport(passed=True),
    )

    assert after_confidence(state) == "escalate"


def test_conflict_escalates_even_with_high_confidence():
    state = _base_state(
        risk=RiskAssessment(final_level=RiskLevel.LOW),
        confidence=ConfidenceBreakdown(final_score=0.95),
        validation=ValidationReport(passed=True, conflict_detected=True),
    )

    assert after_confidence(state) == "escalate"


def test_clean_answer_responds():
    state = _base_state(
        risk=RiskAssessment(final_level=RiskLevel.LOW),
        confidence=ConfidenceBreakdown(final_score=0.9),
        validation=ValidationReport(passed=True),
    )

    assert after_confidence(state) == "respond"
