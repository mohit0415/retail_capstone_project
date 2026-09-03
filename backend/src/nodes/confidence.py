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


def _retrieval_score(state: AgentState) -> float:
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return 0.5 if state.get("sql_evidence") else 0.0

    scored = [c.rerank_score if c.rerank_score is not None else c.fused_score for c in chunks[:5]]

    if not scored:
        return 0.0

    top = max(scored)

    if state.get("skipped_optional_nodes"):
        normalised = min(1.0, top * 40) if top < 1 else min(1.0, top)
    else:
        normalised = 1 / (1 + pow(2.718281828, -top)) if top <= 10 else 1.0

    return round(min(1.0, max(0.0, normalised)), 4)


def _validation_score(state: AgentState) -> float:
    validation = state.get("validation")

    if validation is None:
        return 0.0

    if not validation.passed:
        return 0.0

    penalty = 0.05 * len(validation.defects)

    return round(max(0.0, 1.0 - penalty), 4)


def _agreement_score(state: AgentState) -> float:
    panel = state.get("panel_verdict")

    if panel is not None:
        if panel.unresolved_conflict:
            return 0.0

        return round(1.0 - 0.2 * len(panel.dissent), 4)

    has_policy = bool(state.get("retrieved_chunks"))
    has_records = state.get("sql_evidence") is not None

    if has_policy and has_records:
        validation = state.get("validation")

        if validation and validation.conflict_detected:
            return 0.2

        return 1.0

    return 0.7


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
        return 0.8 if has_evidence else 0.0

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
