from datetime import datetime
from typing import Literal

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


class PendingReviewResponse(BaseModel):
    status: Literal["pending_review"] = "pending_review"
    request_id: str
    thread_id: str
    risk_level: str
    reason: str
    queued_at: datetime
    poll_url: str


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
