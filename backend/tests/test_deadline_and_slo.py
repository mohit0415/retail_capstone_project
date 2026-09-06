import time

import pytest

from configs.settings import MINIMUM_DEADLINES, MINIMUM_TOKEN_BUDGET, Settings, settings
from src.core.budget import (
    NODE_TOKEN_ESTIMATE,
    BudgetGuard,
    path_deadline_seconds,
    widen_for_path,
    widened_deadline,
)
from src.observability.slo import path_target_ms, stage_marks
from src.schemas.enums import EvidencePath


def _state(seconds: float) -> dict:
    started = time.monotonic()

    return {"started_ts": started, "deadline_ts": started + seconds}


def test_every_evidence_path_has_a_deadline_of_its_own():
    for path in EvidencePath:
        assert path.value in ("rag", "nl2sql") or path_deadline_seconds(
            path.value
        ) > settings.deadline_seconds_standard


def test_the_panel_is_given_the_high_risk_budget_not_the_standard_one():
    state = _state(settings.deadline_seconds_standard)

    widened = widen_for_path(state, EvidencePath.HIGH_RISK_PANEL.value)

    assert "deadline_ts" in widened
    assert widened["deadline_ts"] == pytest.approx(
        state["started_ts"] + settings.deadline_seconds_high_risk
    )


def test_the_deadline_is_measured_from_t0_not_from_the_moment_it_widens():
    state = _state(settings.deadline_seconds_standard)

    widened = widen_for_path(state, EvidencePath.AGENTIC.value)

    elapsed_budget = widened["deadline_ts"] - state["started_ts"]

    assert elapsed_budget == pytest.approx(settings.deadline_seconds_agentic)


def test_a_cheaper_path_can_never_shrink_a_deadline_already_widened():
    state = _state(settings.deadline_seconds_high_risk)

    assert widen_for_path(state, EvidencePath.RAG.value) == {}
    assert widen_for_path(state, EvidencePath.HYBRID.value) == {}


def test_widening_is_skipped_when_there_is_no_start_stamp():
    assert widened_deadline({}, 90.0) is None


def test_the_standard_budget_survives_a_full_pass_with_one_repair():
    guard = BudgetGuard(
        deadline_ts=time.monotonic() + settings.deadline_seconds_standard,
        token_budget=settings.default_token_budget,
        tokens_spent=0,
    )

    intake = ("query_rewrite", "intent_classification", "entity_resolution", "risk_assessment")
    spend = 0

    for node in (*intake, "planner", "rag_path", "compliance_validation", "reflection"):
        verdict = guard.check(node)

        assert verdict.allowed, f"{node} was stopped with {guard.tokens_remaining} tokens left"

        spend += NODE_TOKEN_ESTIMATE.get(node, 500)
        guard.tokens_spent = spend

    for node in ("planner", "rag_path", "compliance_validation"):
        assert guard.check(node).allowed, f"the repair pass could not afford {node}"

        spend += NODE_TOKEN_ESTIMATE.get(node, 500)
        guard.tokens_spent = spend


def test_the_release_tier_is_never_stopped_by_an_exhausted_budget():
    guard = BudgetGuard(deadline_ts=time.monotonic() - 5.0, token_budget=100, tokens_spent=100)

    for node in ("output_guardrail", "escalation_manager", "safe_refusal", "clarification"):
        assert guard.check(node).allowed


def test_the_configured_budget_clears_the_floor_that_raises_a_warning():
    for field, floor in MINIMUM_DEADLINES.items():
        assert getattr(settings, field) >= floor

    assert settings.default_token_budget >= MINIMUM_TOKEN_BUDGET


def test_an_unworkable_env_is_raised_to_the_floor_rather_than_trusted():
    raised = Settings(
        _env_file=None,
        deadline_seconds_standard=4.0,
        deadline_seconds_hybrid=6.0,
        deadline_seconds_agentic=10.0,
        deadline_seconds_high_risk=12.0,
        default_token_budget=12000,
    )

    assert raised.deadline_seconds_standard == MINIMUM_DEADLINES["deadline_seconds_standard"]
    assert raised.default_token_budget == MINIMUM_TOKEN_BUDGET


def test_each_path_is_scored_against_the_budget_that_path_was_given():
    panel = path_target_ms(EvidencePath.HIGH_RISK_PANEL.value)
    rag = path_target_ms(EvidencePath.RAG.value)

    assert panel > rag
    assert panel < settings.deadline_seconds_high_risk * 1000
    assert rag < settings.deadline_seconds_standard * 1000


def test_a_path_that_never_ran_falls_back_to_the_standard_objective():
    assert path_target_ms(None) == settings.slo_total_p95_ms
    assert path_target_ms("none") == settings.slo_total_p95_ms


def test_stage_marks_carry_forward_so_percentiles_share_one_population():
    marks = [
        {"stage": "t1", "elapsed_ms": 900.0},
        {"stage": "t2", "elapsed_ms": 1400.0},
    ]

    reached = stage_marks(marks)

    assert reached["t3"] == 1400.0
    assert reached["t4"] == 1400.0


def test_a_stage_reached_twice_keeps_the_later_mark():
    marks = [
        {"stage": "t3", "elapsed_ms": 5000.0},
        {"stage": "t3", "elapsed_ms": 11000.0},
    ]

    assert stage_marks(marks)["t3"] == 11000.0


def test_audit_writes_stop_blocking_the_deadline_once_the_database_is_down():
    from src.core.audit import AuditCircuit

    breaker = AuditCircuit(threshold=3, cooldown=60.0)

    for _ in range(3):
        assert breaker.should_attempt()
        breaker.record_failure()

    assert not breaker.should_attempt()
    assert breaker.status()["writing"] is False


def test_audit_writes_resume_after_the_cooldown():
    from src.core.audit import AuditCircuit

    breaker = AuditCircuit(threshold=2, cooldown=0.0)

    for _ in range(2):
        breaker.record_failure()

    assert breaker.should_attempt()


def test_one_success_re_arms_the_audit_circuit():
    from src.core.audit import AuditCircuit

    breaker = AuditCircuit(threshold=3, cooldown=60.0)

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()

    assert breaker.should_attempt()
    assert breaker.status()["consecutive_failures"] == 1
