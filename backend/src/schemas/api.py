from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from src.schemas.enums import ReviewDecision, Role


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    thread_id: str | None = None
    document_scope: list[str] = Field(default_factory=list)


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


class PendingReviewResponse(BaseModel):
    status: Literal["pending_review"] = "pending_review"
    request_id: str
    thread_id: str
    risk_level: str
    reason: str
    queued_at: datetime
    poll_url: str
    timings: StageTimings | None = None


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


class TokenRequest(BaseModel):
    user_id: str
    role: Role
    departments: list[str] = Field(default_factory=list)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


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


class ReviewSubmission(BaseModel):
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
