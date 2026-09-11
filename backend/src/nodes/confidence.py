import logging
from math import exp

from configs.settings import settings
from src.graph.routing import RETRIEVAL_NO_MATCH_CEILING
from src.graph.state import AgentState
from src.guardrails.output_guard import extract_citations
from src.observability.tracing import traced_node
from src.retrieval.citations import citation_grounding
from src.schemas.enums import DefectType
from src.schemas.models import ConfidenceBreakdown

logger = logging.getLogger(__name__)

WEIGHTS = {
    "retrieval": 0.30,
    "validation": 0.30,
    "agreement": 0.20,
    "coverage": 0.20,
}

DEGRADED_PENALTY = 0.10

EXECUTED_TEMPLATE_EVIDENCE = 0.90

EMPTY_RESULT_EVIDENCE = 0.75

PARTIAL_RESULT_PENALTY = 0.10

GENERATED_QUERY_PENALTY = 0.15

MISSING_EXPECTED_RECORDS = 0.7

SIMILARITY_FLOOR = 0.20

SIMILARITY_CEILING = 0.65

# ---------------------------------------------------------------------------------------------
# The retrieval signal for a policy answer used to be the raw FlashRank cross-encoder score of the
# best extract. That score ranks passages; it is not a probability that the answer is right. In
# the log the same corpus gave 0.999 for one question and 0.11-0.18 for others whose answers the
# validator had just passed - and with retrieval weighted 0.30, a 0.18 caps the whole score at
# 0.72, under the 0.75 threshold, so every "tell me about ..." question escalated after passing
# validation. The signal is now evidence support: how much of the answer is cited to a retrieved
# extract (deterministic, and what validation actually checked) plus the reranker's relevance
# rescaled so that anything at or above RELEVANCE_CEILING counts as fully relevant. A best match
# at or below the no-match ceiling still scores zero - the extracts do not answer the question.
# ---------------------------------------------------------------------------------------------

RELEVANCE_FLOOR = RETRIEVAL_NO_MATCH_CEILING

RELEVANCE_CEILING = 0.50

EVIDENCE_RELEVANCE_WEIGHT = 0.40

EVIDENCE_GROUNDING_WEIGHT = 0.60

# coverage: a validator coverage_gap costs this much, down to the floor
COVERAGE_GAP_PENALTY = 0.25

COVERAGE_FLOOR = 0.25

RECORD_SOURCES = {"compliance_db", "both"}


def _rescale(value: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0

    return min(1.0, max(0.0, (value - floor) / (ceiling - floor)))


def _sql_evidence_score(evidence) -> float:
    if evidence is None:
        return 0.0

    score = EMPTY_RESULT_EVIDENCE if evidence.row_count == 0 else EXECUTED_TEMPLATE_EVIDENCE

    if evidence.truncated:
        score -= PARTIAL_RESULT_PENALTY

    if evidence.rows_filtered_by_scope:
        score -= PARTIAL_RESULT_PENALTY

    if evidence.generated:
        score -= GENERATED_QUERY_PENALTY

    return round(max(0.0, score), 4)


def _relevance(chunks: list) -> float | None:
    """How relevant the best extract looked to the retriever, rescaled to 0..1 (None: no score)."""
    leading = chunks[:5]

    reranked = [chunk.rerank_score for chunk in leading if chunk.rerank_score is not None]

    if reranked:
        top = max(reranked)

        if top > 1.0 or top < 0.0:
            top = 1 / (1 + exp(-top))

        return round(_rescale(top, RELEVANCE_FLOOR, RELEVANCE_CEILING), 4)

    dense = [chunk.dense_score for chunk in leading if chunk.dense_score]

    if dense:
        return round(_rescale(max(dense), SIMILARITY_FLOOR, SIMILARITY_CEILING), 4)

    fused = [chunk.fused_score for chunk in leading if chunk.fused_score]

    if not fused:
        return None

    return round(min(1.0, max(fused) * (settings.rrf_k + 1)), 4)


def _grounding(state: AgentState, chunks: list) -> float:
    """How much of the answer is cited to the extracts (the validator's grounded ratio if higher).

    On an answer written over records too, only the policy sentences are counted: the sentences that
    report rows are sourced by the as-of date. Counting them left a hybrid answer's score to the
    validator's grounded ratio, which moved between 0.0 and 0.86 on three runs of one question and
    escalated one of them (confidence 0.7186 against the 0.75 threshold).
    """
    draft = state.get("draft")
    rows_are_sourced = state.get("sql_evidence") is not None
    cited = citation_grounding(draft.answer, chunks, skip_records=rows_are_sourced) if draft is not None else 0.0

    validation = state.get("validation")
    judged = float(validation.grounded_claim_ratio or 0.0) if validation is not None and validation.passed else 0.0

    return round(max(cited, min(1.0, judged)), 4)


def _retrieval_score(state: AgentState) -> float:
    chunks = state.get("retrieved_chunks", [])
    evidence = state.get("sql_evidence")

    if not chunks:
        return _sql_evidence_score(evidence)

    relevance = _relevance(chunks)

    if relevance is not None and relevance <= 0.0 and evidence is None:
        # the best extract sits at or below the no-match ceiling: the documents do not answer this
        return 0.0

    grounding = _grounding(state, chunks)

    return round(EVIDENCE_RELEVANCE_WEIGHT * (relevance or 0.0) + EVIDENCE_GROUNDING_WEIGHT * grounding, 4)


def _validation_score(state: AgentState) -> float:
    validation = state.get("validation")

    if validation is None:
        return 0.0

    if not validation.passed:
        return 0.0

    penalty = 0.05 * len(validation.defects)

    return round(max(0.0, 1.0 - penalty), 4)


def _plan_expects_records(state: AgentState) -> bool:
    plan = state.get("plan")

    if plan is None:
        return False

    return any(step.source in RECORD_SOURCES for step in plan.steps)


def _agreement_score(state: AgentState) -> float:
    panel = state.get("panel_verdict")

    if panel is not None:
        if panel.unresolved_conflict:
            return 0.0

        return round(max(0.0, 1.0 - 0.2 * len(panel.dissent)), 4)

    has_policy = bool(state.get("retrieved_chunks"))
    has_records = state.get("sql_evidence") is not None

    if not has_policy and not has_records:
        return 0.0

    if has_policy and has_records:
        validation = state.get("validation")

        if validation and validation.conflict_detected:
            return 0.2

        return 1.0

    if _plan_expects_records(state) and not has_records:
        return MISSING_EXPECTED_RECORDS

    return 1.0


def _coverage_score(state: AgentState) -> float:
    draft = state.get("draft")
    plan = state.get("plan")

    if draft is None:
        return 0.0

    citations = extract_citations(draft.answer)
    has_evidence = bool(citations) or state.get("sql_evidence") is not None

    if not has_evidence:
        return 0.0

    if plan is None or not plan.required_claims:
        # no claim list to check against (the plan was written without a model call): the
        # validator's own coverage judgement stands - full marks unless it flagged a gap
        validation = state.get("validation")
        gaps = [d for d in (validation.defects if validation else []) if d.defect_type is DefectType.COVERAGE_GAP]

        return round(max(COVERAGE_FLOOR, 1.0 - COVERAGE_GAP_PENALTY * len(gaps)), 4)

    answer_lower = draft.answer.lower()
    covered = sum(1 for claim in plan.required_claims if any(
        token in answer_lower for token in claim.lower().split() if len(token) > 4
    ))

    return round(covered / len(plan.required_claims), 4)


@traced_node("confidence_scoring")
def confidence_node(state: AgentState) -> dict:
    retrieval = _retrieval_score(state)
    validation = _validation_score(state)
    agreement = _agreement_score(state)
    coverage = _coverage_score(state)

    final = (
        WEIGHTS["retrieval"] * retrieval
        + WEIGHTS["validation"] * validation
        + WEIGHTS["agreement"] * agreement
        + WEIGHTS["coverage"] * coverage
    )

    degraded = state.get("degraded", False)

    if degraded:
        final -= DEGRADED_PENALTY

    breakdown = ConfidenceBreakdown(
        retrieval_score=retrieval,
        validation_score=validation,
        source_agreement=agreement,
        coverage=coverage,
        final_score=round(max(0.0, min(1.0, final)), 4),
        degraded=degraded,
    )

    releasable = breakdown.final_score >= settings.confidence_threshold
    log = logger.info if releasable else logger.warning

    log(
        "confidence %s request_id=%s final=%.4f threshold=%.2f retrieval=%.3f validation=%.3f "
        "agreement=%.3f coverage=%.3f degraded=%s path=%s",
        "OK" if releasable else "BELOW_THRESHOLD",
        state.get("request_id"),
        breakdown.final_score,
        settings.confidence_threshold,
        retrieval,
        validation,
        agreement,
        coverage,
        degraded,
        state.get("evidence_path"),
    )

    return {"confidence": breakdown}
