import time

from src.core.budget import BudgetGuard
from src.nodes.confidence import confidence_node
from src.schemas.models import (
    ConfidenceBreakdown,
    DraftAnswer,
    RetrievedChunk,
    ValidationReport,
)


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_type="retention_policy",
        document_title="Records Retention Policy",
        section="Transaction records",
        clause_number="4.1",
        version="3.2",
        content="Transaction records are retained for seven financial years.",
        fused_score=score,
        rerank_score=score,
    )


def test_optional_node_is_dropped_instead_of_failing():
    guard = BudgetGuard(deadline_ts=time.monotonic() + 0.4, token_budget=12000, tokens_spent=0)
    verdict = guard.check("reranker")

    assert not verdict.allowed
    assert verdict.degrade


def test_required_node_is_not_marked_degradable():
    guard = BudgetGuard(deadline_ts=time.monotonic() - 1, token_budget=12000, tokens_spent=0)
    verdict = guard.check("compliance_validation")

    assert not verdict.allowed
    assert not verdict.degrade


def test_token_exhaustion_blocks_an_expensive_node():
    guard = BudgetGuard(deadline_ts=time.monotonic() + 30, token_budget=1000, tokens_spent=900)

    assert not guard.check("multi_agent_panel").allowed


def test_confidence_is_zero_when_validation_failed():
    state = {
        "retrieved_chunks": [_chunk(0.9)],
        "validation": ValidationReport(passed=False),
        "draft": DraftAnswer(answer="Records are kept for seven years [Records Retention Policy §4.1]."),
        "plan": None,
        "degraded": False,
    }

    result = confidence_node(state)
    breakdown: ConfidenceBreakdown = result["confidence"]

    assert breakdown.validation_score == 0.0
    assert breakdown.final_score < 0.75


def test_degradation_costs_confidence():
    base_state = {
        "retrieved_chunks": [_chunk(0.9)],
        "validation": ValidationReport(passed=True),
        "draft": DraftAnswer(answer="Records are kept for seven years [Records Retention Policy §4.1]."),
        "plan": None,
        "degraded": False,
    }

    clean = confidence_node(dict(base_state))["confidence"].final_score
    degraded = confidence_node({**base_state, "degraded": True, "skipped_optional_nodes": ["reranker"]})[
        "confidence"
    ].final_score

    assert degraded < clean


def test_no_evidence_means_no_coverage():
    state = {
        "retrieved_chunks": [],
        "sql_evidence": None,
        "validation": ValidationReport(passed=True),
        "draft": DraftAnswer(answer="Records are probably kept for a while."),
        "plan": None,
        "degraded": False,
    }

    breakdown = confidence_node(state)["confidence"]

    assert breakdown.coverage == 0.0
    assert breakdown.retrieval_score == 0.0


def _policy_plan():
    from src.schemas.enums import EvidencePath
    from src.schemas.models import EvidencePlan, PlanStep

    return EvidencePlan(
        path=EvidencePath.RAG,
        steps=[PlanStep(order=1, source="policy_kb", objective="find the clause", must_prove="period")],
        required_claims=["seven financial years"],
    )


def _records_plan():
    from src.schemas.enums import EvidencePath
    from src.schemas.models import EvidencePlan, PlanStep

    return EvidencePlan(
        path=EvidencePath.HYBRID,
        steps=[PlanStep(order=1, source="both", objective="check records", must_prove="due date")],
        required_claims=["seven financial years"],
    )


def _answer_state(chunk, plan):
    return {
        "retrieved_chunks": [chunk],
        "sql_evidence": None,
        "validation": ValidationReport(passed=True),
        "draft": DraftAnswer(answer="Records are kept for seven financial years [Records Retention Policy §4.1]."),
        "plan": plan,
        "degraded": False,
    }


def test_a_strong_cross_encoder_score_carries_the_answer_through():
    breakdown = confidence_node(_answer_state(_chunk(0.86), _policy_plan()))["confidence"]

    assert breakdown.retrieval_score == 1.0
    assert breakdown.final_score >= 0.75


def test_a_no_match_cross_encoder_score_escalates():
    # 0.02 is under the no-match ceiling: the documents do not answer this, however well cited
    breakdown = confidence_node(_answer_state(_chunk(0.02), _policy_plan()))["confidence"]

    assert breakdown.retrieval_score == 0.0
    assert breakdown.final_score < 0.75


def test_a_lukewarm_reranker_score_does_not_sink_a_cited_validated_answer():
    # the log: "tell me about vendor risk classification framework" passed validation on the
    # rewrite and then escalated on confidence 0.6998 because the reranker gave the best
    # extract 0.18 - the raw cross-encoder score was 30% of the release gate
    breakdown = confidence_node(_answer_state(_chunk(0.18), _policy_plan()))["confidence"]

    assert 0.6 < breakdown.retrieval_score < 1.0
    assert breakdown.final_score >= 0.75


def test_a_lukewarm_score_with_an_uncited_answer_still_escalates():
    state = _answer_state(_chunk(0.18), _policy_plan())
    state["draft"] = DraftAnswer(answer="Records are kept for seven financial years, according to the policy.")

    breakdown = confidence_node(state)["confidence"]

    assert breakdown.retrieval_score < 0.2
    assert breakdown.final_score < 0.75


def test_a_validator_coverage_gap_lowers_coverage():
    from src.schemas.enums import DefectType
    from src.schemas.models import Defect

    # a plan written without the planner model carries no claim list, so the validator's
    # coverage judgement is the coverage signal
    state = _answer_state(_chunk(0.86), None)
    state["validation"] = ValidationReport(
        passed=True,
        defects=[Defect(defect_type=DefectType.COVERAGE_GAP, description="the second half of the question is unanswered")],
    )

    breakdown = confidence_node(state)["confidence"]

    assert breakdown.coverage == 0.75
    assert confidence_node(_answer_state(_chunk(0.86), None))["confidence"].coverage == 1.0


def test_a_document_only_plan_is_not_penalised_for_having_no_records():
    breakdown = confidence_node(_answer_state(_chunk(0.86), _policy_plan()))["confidence"]

    assert breakdown.source_agreement == 1.0


def test_a_plan_that_wanted_records_is_penalised_when_none_arrive():
    breakdown = confidence_node(_answer_state(_chunk(0.86), _records_plan()))["confidence"]

    assert breakdown.source_agreement == 0.7


def test_an_empty_retrieval_scores_no_agreement_at_all():
    state = _answer_state(_chunk(0.86), _policy_plan())
    state["retrieved_chunks"] = []

    breakdown = confidence_node(state)["confidence"]

    assert breakdown.source_agreement == 0.0
    assert breakdown.final_score < 0.75
