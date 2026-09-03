import operator
from typing import Annotated, TypedDict

from src.schemas.models import (
    ConfidenceBreakdown,
    DraftAnswer,
    EvidencePlan,
    IntentResult,
    PanelVerdict,
    ResolvedEntity,
    RetrievedChunk,
    RiskAssessment,
    SqlEvidence,
    ValidationReport,
)


def keep_last(current, incoming):
    return incoming if incoming is not None else current


class AgentState(TypedDict, total=False):
    request_id: str
    thread_id: str
    user_id: str
    role: str
    access_scopes: list[str]

    raw_query: str
    sanitised_query: str
    standalone_query: str
    conversation_history: list[dict]
    thread_summary: str

    intent: IntentResult | None
    resolved_entities: list[ResolvedEntity]
    clarification_question: str | None

    risk: RiskAssessment | None
    plan: EvidencePlan | None
    evidence_path: str | None

    retrieved_chunks: Annotated[list[RetrievedChunk], operator.add]
    sql_evidence: SqlEvidence | None
    panel_verdict: PanelVerdict | None

    draft: DraftAnswer | None
    validation: ValidationReport | None
    reflection_count: int
    panel_repair_count: int

    confidence: ConfidenceBreakdown | None
    terminal_outcome: str | None
    refusal_reason: str | None
    escalation_reason: str | None

    deadline_ts: float
    token_budget: int
    tokens_spent: Annotated[int, operator.add]
    degraded: bool
    skipped_optional_nodes: Annotated[list[str], operator.add]

    trace: Annotated[list[dict], operator.add]


def initial_state(
    request_id: str,
    thread_id: str,
    user_id: str,
    role: str,
    access_scopes: list[str],
    raw_query: str,
    deadline_ts: float,
    token_budget: int,
) -> AgentState:
    return AgentState(
        request_id=request_id,
        thread_id=thread_id,
        user_id=user_id,
        role=role,
        access_scopes=access_scopes,
        raw_query=raw_query,
        sanitised_query=raw_query,
        standalone_query=raw_query,
        conversation_history=[],
        thread_summary="",
        intent=None,
        resolved_entities=[],
        clarification_question=None,
        risk=None,
        plan=None,
        evidence_path=None,
        retrieved_chunks=[],
        sql_evidence=None,
        panel_verdict=None,
        draft=None,
        validation=None,
        reflection_count=0,
        panel_repair_count=0,
        confidence=None,
        terminal_outcome=None,
        refusal_reason=None,
        escalation_reason=None,
        deadline_ts=deadline_ts,
        token_budget=token_budget,
        tokens_spent=0,
        degraded=False,
        skipped_optional_nodes=[],
        trace=[],
    )
