import time

import pytest

from src.graph.path_policy import (
    ESCALATE,
    NO_ACCESS,
    allowed_path_names,
    degrade_for_budget,
    resolve_path,
    rule_for,
)
from src.graph.routing import route_evidence_path
from src.schemas.enums import EvidencePath, Intent, RiskLevel
from src.schemas.models import EvidencePlan, PlanStep, RiskAssessment


def _resolve(intent, proposed, has_tables=True, has_documents=True, agentic_enabled=True):
    return resolve_path(
        intent=intent,
        risk_level=RiskLevel.LOW,
        proposed=proposed,
        has_tables=has_tables,
        has_documents=has_documents,
        agentic_enabled=agentic_enabled,
    )


def _state(**overrides) -> dict:
    state = {
        "deadline_ts": time.monotonic() + 30,
        "token_budget": 48000,
        "tokens_spent": 0,
        "risk": RiskAssessment(final_level=RiskLevel.LOW),
    }
    state.update(overrides)

    return state


DEFAULTS = [
    (Intent.POLICY_LOOKUP, EvidencePath.RAG),
    (Intent.RECORD_LOOKUP, EvidencePath.NL2SQL),
    (Intent.VENDOR_STATUS, EvidencePath.NL2SQL),
    (Intent.RETENTION_QUERY, EvidencePath.HYBRID),
    (Intent.COMPLIANCE_CHECK, EvidencePath.HYBRID),
    (Intent.INCIDENT_GUIDANCE, EvidencePath.HYBRID),
]


@pytest.mark.parametrize(("intent", "expected"), DEFAULTS)
def test_every_intent_has_one_default_path(intent, expected):
    assert rule_for(intent).default is expected


@pytest.mark.parametrize(("intent", "expected"), DEFAULTS)
def test_the_default_is_kept_when_the_planner_proposes_nothing(intent, expected):
    decision = _resolve(intent, proposed=None)

    assert decision.path == expected.value


@pytest.mark.parametrize(("intent", "expected"), DEFAULTS)
def test_a_path_outside_the_allowed_set_is_clamped_to_the_default(intent, expected):
    outsider = next(
        path
        for path in (EvidencePath.RAG, EvidencePath.NL2SQL, EvidencePath.AGENTIC)
        if path not in rule_for(intent).allowed
    )

    decision = _resolve(intent, proposed=outsider)

    assert decision.path == expected.value
    assert decision.clamped
    assert outsider.value in decision.reason


def test_a_record_question_can_never_be_answered_from_policy_text():
    decision = _resolve(Intent.RECORD_LOOKUP, proposed=EvidencePath.RAG)

    assert decision.path == EvidencePath.NL2SQL.value


def test_a_policy_question_can_never_be_answered_from_records():
    decision = _resolve(Intent.POLICY_LOOKUP, proposed=EvidencePath.NL2SQL)

    assert decision.path == EvidencePath.RAG.value


def test_a_choice_inside_the_allowed_set_survives():
    decision = _resolve(Intent.INCIDENT_GUIDANCE, proposed=EvidencePath.RAG)

    assert decision.path == EvidencePath.RAG.value
    assert not decision.clamped


@pytest.mark.parametrize("intent", [*Intent])
def test_no_intent_admits_the_agentic_route_the_v4_workflow_does_not_have(intent):
    decision = _resolve(intent, proposed=EvidencePath.AGENTIC)

    assert decision.path != EvidencePath.AGENTIC.value
    assert decision.clamped


def test_high_risk_beats_every_other_consideration():
    decision = resolve_path(
        intent=Intent.POLICY_LOOKUP,
        risk_level=RiskLevel.HIGH,
        proposed=EvidencePath.RAG,
        has_tables=False,
        has_documents=False,
    )

    assert decision.path == EvidencePath.HIGH_RISK_PANEL.value


def test_a_record_question_is_answered_honestly_when_the_role_reads_no_table():
    decision = _resolve(Intent.RECORD_LOOKUP, proposed=EvidencePath.NL2SQL, has_tables=False)

    assert decision.path == NO_ACCESS
    assert "no table" in decision.reason


def test_a_policy_question_is_answered_honestly_when_the_role_reads_no_document():
    decision = _resolve(Intent.POLICY_LOOKUP, proposed=EvidencePath.RAG, has_documents=False)

    assert decision.path == NO_ACCESS
    assert "no document" in decision.reason


def test_a_two_sided_question_falls_back_to_policy_and_flags_partial_evidence():
    decision = _resolve(Intent.COMPLIANCE_CHECK, proposed=EvidencePath.HYBRID, has_tables=False)

    assert decision.path == EvidencePath.RAG.value
    assert decision.partial_evidence


def test_a_two_sided_question_falls_back_to_records_when_no_document_is_granted():
    decision = _resolve(Intent.COMPLIANCE_CHECK, proposed=EvidencePath.HYBRID, has_documents=False)

    assert decision.path == EvidencePath.NL2SQL.value
    assert decision.partial_evidence


def test_disabling_the_agentic_path_falls_back_to_that_intents_fallback():
    decision = _resolve(Intent.INCIDENT_GUIDANCE, proposed=EvidencePath.AGENTIC, agentic_enabled=False)

    assert decision.path == EvidencePath.HYBRID.value


def test_agentic_degrades_to_hybrid_not_to_rag():
    decision = degrade_for_budget(
        path=EvidencePath.AGENTIC,
        intent=Intent.INCIDENT_GUIDANCE,
        seconds_remaining=1.0,
        agentic_min_seconds=3.0,
        hybrid_min_seconds=2.0,
    )

    assert decision.path == EvidencePath.HYBRID.value


def test_hybrid_degrades_to_the_source_the_intent_turns_on():
    records = degrade_for_budget(
        path=EvidencePath.HYBRID,
        intent=Intent.RETENTION_QUERY,
        seconds_remaining=0.5,
        agentic_min_seconds=3.0,
        hybrid_min_seconds=2.0,
    )

    policy = degrade_for_budget(
        path=EvidencePath.HYBRID,
        intent=Intent.INCIDENT_GUIDANCE,
        seconds_remaining=0.5,
        agentic_min_seconds=3.0,
        hybrid_min_seconds=2.0,
    )

    assert records.path == EvidencePath.NL2SQL.value
    assert policy.path == EvidencePath.RAG.value


def test_headroom_leaves_the_path_alone():
    decision = degrade_for_budget(
        path=EvidencePath.HYBRID,
        intent=Intent.COMPLIANCE_CHECK,
        seconds_remaining=20.0,
        agentic_min_seconds=3.0,
        hybrid_min_seconds=2.0,
    )

    assert decision.path == EvidencePath.HYBRID.value
    assert not decision.clamped


def test_the_router_reads_the_decision_the_planner_recorded():
    state = _state(routed_path=EvidencePath.NL2SQL.value, intent=None)

    assert route_evidence_path(state) == EvidencePath.NL2SQL.value


def test_the_router_escalates_when_the_planner_could_not_place_the_question():
    state = _state(routed_path=ESCALATE)

    assert route_evidence_path(state) == "escalate"


def test_the_router_answers_honestly_when_the_role_cannot_read_the_source():
    assert route_evidence_path(_state(routed_path=NO_ACCESS)) == "not_found"


def test_the_router_still_escalates_a_no_access_question_when_a_human_was_asked_for():
    state = _state(routed_path=NO_ACCESS, raw_query="which vendors are overdue? escalate this to legal review")

    assert route_evidence_path(state) == "escalate"


def test_the_router_falls_back_to_the_plan_when_no_decision_was_recorded():
    plan = EvidencePlan(
        path=EvidencePath.HYBRID,
        steps=[PlanStep(order=1, source="both", objective="o", must_prove="p")],
    )

    assert route_evidence_path(_state(plan=plan)) == EvidencePath.HYBRID.value


def test_high_risk_reaches_the_panel_through_the_router_too():
    state = _state(
        risk=RiskAssessment(final_level=RiskLevel.HIGH),
        routed_path=EvidencePath.RAG.value,
    )

    assert route_evidence_path(state) == EvidencePath.HIGH_RISK_PANEL.value


def test_the_allowed_set_is_reported_for_the_planner_prompt():
    assert allowed_path_names(Intent.COMPLIANCE_CHECK) == ["hybrid"]
    assert allowed_path_names(Intent.INCIDENT_GUIDANCE) == ["hybrid", "rag"]
    assert allowed_path_names(Intent.POLICY_LOOKUP) == ["rag"]


def test_an_unknown_intent_reads_policy_rather_than_guessing():
    assert _resolve(None, proposed=None).path == EvidencePath.RAG.value
