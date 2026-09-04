from math import exp

from configs.settings import settings
from src.graph.state import AgentState
from src.guardrails.output_guard import extract_citations
from src.observability.tracing import traced_node
from src.schemas.models import ConfidenceBreakdown

WEIGHTS = {
    "retrieval": 0.30,
    "validation": 0.30,
    "agreement": 0.20,
    "coverage": 0.20,
}

DEGRADED_PENALTY = 0.10

SQL_ONLY_RETRIEVAL = 0.5

SIMILARITY_FLOOR = 0.20

SIMILARITY_CEILING = 0.65

RECORD_SOURCES = {"compliance_db", "both"}


def _rescale(value: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0

    return min(1.0, max(0.0, (value - floor) / (ceiling - floor)))


def _retrieval_score(state: AgentState) -> float:
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return SQL_ONLY_RETRIEVAL if state.get("sql_evidence") else 0.0

    leading = chunks[:5]

    reranked = [chunk.rerank_score for chunk in leading if chunk.rerank_score is not None]

    if reranked:
        top = max(reranked)

        if top > 1.0 or top < 0.0:
            return round(1 / (1 + exp(-top)), 4)

        return round(top, 4)

    dense = [chunk.dense_score for chunk in leading if chunk.dense_score]

    if dense:
        return round(_rescale(max(dense), SIMILARITY_FLOOR, SIMILARITY_CEILING), 4)

    fused = [chunk.fused_score for chunk in leading if chunk.fused_score]

    if not fused:
        return 0.0

    return round(min(1.0, max(fused) * (settings.rrf_k + 1)), 4)


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

    if _plan_expects_records(state):
        return 0.7

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
        return 0.8

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

    return {"confidence": breakdown}
