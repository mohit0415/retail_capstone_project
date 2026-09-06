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


def merge_chunks(current, incoming):
    if incoming is None:
        return []

    return list(current or []) + list(incoming)


class AgentState(TypedDict, total=False):
    request_id: str
    thread_id: str
    user_id: str
    role: str
    access_scopes: list[str]
    departments: list[str]

    raw_query: str
    sanitised_query: str
    standalone_query: str
    conversation_history: list[dict]
    thread_summary: str

    intent: IntentResult | None
    document_scope_request: list[str]
    resolved_entities: list[ResolvedEntity]
    clarification_question: str | None

    risk: RiskAssessment | None
    plan: EvidencePlan | None
    routed_path: str | None
    path_decision: dict | None
    plan_from_reflection: bool
    replan_directive: str
    evidence_path: str | None

    retrieved_chunks: Annotated[list[RetrievedChunk], merge_chunks]
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
    escalation_reference: str | None

    reviewer_decision: str | None
    reviewer_id: str | None
    reviewer_notes: str

    started_ts: float
    deadline_ts: float
    token_budget: int
    tokens_spent: Annotated[int, operator.add]
    degraded: bool
    skipped_optional_nodes: Annotated[list[str], operator.add]
    budget_stops: Annotated[list[dict], operator.add]

    marks: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]


def initial_state(
    request_id: str,
    thread_id: str,
    user_id: str,
    role: str,
    access_scopes: list[str],
    raw_query: str,
    started_ts: float,
    deadline_ts: float,
    token_budget: int,
    departments: list[str] | None = None,
    document_scope_request: list[str] | None = None,
    conversation_history: list[dict] | None = None,
    thread_summary: str = "",
) -> AgentState:
    return AgentState(
        request_id=request_id,
        thread_id=thread_id,
        user_id=user_id,
        role=role,
        access_scopes=access_scopes,
        departments=departments or [],
        raw_query=raw_query,
        sanitised_query=raw_query,
        standalone_query=raw_query,
        conversation_history=conversation_history or [],
        thread_summary=thread_summary,
        intent=None,
        document_scope_request=document_scope_request or [],
        resolved_entities=[],
        clarification_question=None,
        risk=None,
        plan=None,
        routed_path=None,
        path_decision=None,
        plan_from_reflection=False,
        replan_directive="",
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
        escalation_reference=None,
        reviewer_decision=None,
        reviewer_id=None,
        reviewer_notes="",
        started_ts=started_ts,
        deadline_ts=deadline_ts,
        token_budget=token_budget,
        tokens_spent=0,
        degraded=False,
        skipped_optional_nodes=[],
        budget_stops=[],
        marks=[],
        trace=[],
    )
