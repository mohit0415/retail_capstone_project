from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from src.schemas.enums import ReviewDecision, Role


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    thread_id: str | None = None
    document_scope: list[str] = Field(default_factory=list)
    use_cache: bool = True


class Citation(BaseModel):
    document_title: str
    clause_number: str
    section: str
    version: str
    excerpt: str


class SqlProof(BaseModel):
    template_id: str
    statement: str
    parameters: dict
    row_count: int
    as_of: str


class StageTimings(BaseModel):
    t1_ms: float | None = None
    t2_ms: float | None = None
    t3_ms: float | None = None
    t4_ms: float | None = None
    total_ms: float


class CacheInfo(BaseModel):
    """Set on an answer served from the response cache instead of a graph run."""

    hit: bool = True
    kind: Literal["exact", "semantic"]
    similarity: float = 1.0
    age_seconds: float = 0.0
    matched_query: str = ""
    source_request_id: str
    saved_ms_estimate: float = 0.0
    saved_tokens_estimate: int = 0


class DecisionTrace(BaseModel):
    """How the graph reached its decision, in the order the nodes ran.

    Attached to every /ask response so the caller can see the routing and the
    certification evidence without opening the reviewer console. The draft
    answer itself is *not* part of the trace: on a 202 it stays withheld
    because it could not be certified.
    """

    guardrail: str = "pass"
    standalone_query: str = ""
    intent: dict | None = None
    resolved_entities: list[dict] = Field(default_factory=list)
    risk: dict | None = None
    plan: dict | None = None
    routed_path: str | None = None
    path_decision: dict | None = None
    evidence_path: str | None = None
    retrieved_chunks: list[str] = Field(default_factory=list)
    sql_evidence: SqlProof | None = None
    panel_verdict: dict | None = None
    cited_clauses: list[str] = Field(default_factory=list)
    validation: dict | None = None
    confidence: dict | None = None
    tokens_spent: int = 0
    token_budget: int = 0
    deadline_seconds: float | None = None
    degraded: bool = False
    skipped_optional_nodes: list[str] = Field(default_factory=list)
    budget_stops: list[dict] = Field(default_factory=list)
    reflection_count: int = 0
    panel_repair_count: int = 0
    terminal_outcome: str | None = None
    escalation_reason: str | None = None
    escalation_reference: str | None = None
    model_routing: list[dict] = Field(default_factory=list)
    llm_usage: dict | None = None
    agent_steps: list[dict] = Field(
        default_factory=list,
        description="the multi-agent workflow in run order: agent, status, elapsed_ms, summary, model tier",
    )


class AnswerResponse(BaseModel):
    status: Literal["answered"] = "answered"
    request_id: str
    thread_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sql_evidence: SqlProof | None = None
    confidence: float
    risk_level: str
    uncertainty_note: str = ""
    evidence_path: str
    degraded: bool = False
    timings: StageTimings | None = None
    history_turns_used: int = 0
    thread_summary_used: bool = False
    standalone_query: str = ""
    not_found: bool = False
    # why the reply is "I don't know": no_match, not_in_extracts, no_records, no_evidence or no_access
    not_found_reason: str = ""
    trace: DecisionTrace | None = None
    cache: CacheInfo | None = None


class PendingReviewResponse(BaseModel):
    status: Literal["pending_review"] = "pending_review"
    request_id: str
    thread_id: str
    risk_level: str
    reason: str
    escalation_reference: str | None = None
    queued_at: datetime
    poll_url: str
    timings: StageTimings | None = None
    trace: DecisionTrace | None = None


class RefusalResponse(BaseModel):
    status: Literal["refused"] = "refused"
    request_id: str
    reason: str


class ClarificationResponse(BaseModel):
    status: Literal["clarification_required"] = "clarification_required"
    request_id: str
    thread_id: str
    question: str
    candidates: list[dict] = Field(default_factory=list)


AskReleasedResponse = Annotated[
    AnswerResponse | RefusalResponse | ClarificationResponse,
    Field(discriminator="status"),
]


class MeResponse(BaseModel):
    """What the Auth0 token resolved to on the backend - the frontend uses it for role-based screens."""

    user_id: str
    role: Role
    departments: list[str] = Field(default_factory=list)
    access_scopes: list[str] = Field(default_factory=list)
    auth0_roles: list[str] = Field(default_factory=list)
    email: str = ""
    name: str = ""
    permissions: dict[str, bool] = Field(default_factory=dict)
    screens: list[str] = Field(default_factory=list)
    azure_configured: bool = False


class AzureCredentialsRequest(BaseModel):
    """Azure OpenAI credentials typed on the login page (never stored in the frontend .env)."""

    endpoint: str = Field(min_length=12, description="https://<resource>.openai.azure.com/")
    api_key: str = Field(min_length=8)
    api_version: str = "2024-10-21"
    small_deployment: str = "gpt-4o-mini"
    strong_deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-small"
    verify: bool = True


class AzureCredentialsResponse(BaseModel):
    azure_configured: bool
    verified: bool = False
    endpoint: str = ""
    api_version: str = ""
    small_deployment: str = ""
    strong_deployment: str = ""
    embedding_deployment: str = ""
    message: str = ""


class QueueItem(BaseModel):
    request_id: str
    thread_id: str
    user_id: str
    risk_level: str
    reason: str
    queued_at: datetime


class ReviewPackage(BaseModel):
    request_id: str
    thread_id: str
    original_query: str
    standalone_query: str
    conversation_history: list[dict]
    retrieved_documents: list[dict]
    sql_evidence: dict | None
    validation_output: dict
    reasoning_trace: list[dict]
    draft_answer: str
    risk_level: str
    confidence: float
    evidence_path: str | None = None
    panel_verdict: dict | None = None
    plan: dict | None = None
    panel_repair_count: int = 0
    reflection_count: int = 0
    reference_id: str | None = None
    budget_stops: list[dict] = Field(default_factory=list)
    degraded: bool = False


class ReviewSubmission(BaseModel):
    """Reviewer decision.

    - accept: release ``draft_answer`` exactly as the system produced it
      (for High-risk requests that is the multi-agent panel consensus).
      ``edited_answer`` is ignored.
    - edit:   release ``edited_answer`` instead; it is required and must be non-empty.
    - reject: release nothing.
    """

    decision: ReviewDecision
    edited_answer: str | None = None
    reviewer_notes: str = ""


class ReviewOutcome(BaseModel):
    status: Literal["recorded"] = "recorded"
    request_id: str
    decision: str
    resumed: bool
    released: bool
    outcome: str | None = None
    answer: str | None = None
    answer_source: Literal["panel_consensus", "system_draft", "reviewer_edit"] | None = None
    evidence_path: str | None = None
    note: str = ""


class IngestResponse(BaseModel):
    status: Literal["indexed", "skipped"]
    file_name: str
    doc_type: str = ""
    version: str = ""
    parsed_with: str = ""
    total_nodes: int = 0
    text_nodes: int = 0
    table_nodes: int = 0
    image_nodes: int = 0
    superseded_nodes: int = 0
    reason: str = ""


class CorpusStatusResponse(BaseModel):
    indexed: bool
    corpus_dir: str
    embed_model: str
    embed_model_compatible: bool
    message: str = ""
    documents: list[dict] = Field(default_factory=list)
    unindexed_files: list[str] = Field(default_factory=list)


class RebuildResponse(BaseModel):
    cleared_chunks: int
    corpus_dir: str
    embed_model: str
    indexed_files: list[str] = Field(default_factory=list)
    skipped_files: list[dict] = Field(default_factory=list)
    total_nodes: int = 0
    documents: list[dict] = Field(default_factory=list)
    unindexed_files: list[str] = Field(default_factory=list)


class SloStage(BaseModel):
    stage: str
    meaning: str
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    target_p95_ms: float
    meets_slo: bool | None = None


class SloTotal(BaseModel):
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    target_p95_ms: float
    meets_slo: bool | None = None


class SloOutcomeCount(BaseModel):
    outcome: str
    count: int


class SloPathLatency(BaseModel):
    evidence_path: str
    count: int
    total_p95_ms: float | None = None
    target_p95_ms: float
    meets_slo: bool | None = None


class SloReport(BaseModel):
    window_hours: int
    sample_size: int
    measured_from: str
    total_target_basis: str
    stages: list[SloStage]
    total: SloTotal
    breached_stages: list[str] = Field(default_factory=list)
    breached_paths: list[str] = Field(default_factory=list)
    outcomes: list[SloOutcomeCount] = Field(default_factory=list)
    evidence_paths: list[SloPathLatency] = Field(default_factory=list)


class OptimizationReport(BaseModel):
    """Cost and latency optimisation evidence: cache effectiveness, routing mix, spend."""

    generated_at: datetime
    caches: dict
    routing: dict
    cost: dict


class CacheInvalidationResponse(BaseModel):
    reason: str
    cleared: dict[str, int]
