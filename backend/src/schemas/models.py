from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.enums import (
    DefectType,
    EntityStatus,
    EvidencePath,
    Intent,
    RiskLevel,
)


class RewriteResult(BaseModel):
    standalone_query: str = Field(description="The follow-up rewritten as a query that stands on its own")
    carried_entities: list[str] = Field(default_factory=list)
    is_follow_up: bool = False


class EntitySpan(BaseModel):
    text: str
    entity_type: Literal["vendor", "department", "policy", "document", "date", "record_id"]


class IntentResult(BaseModel):
    intent: Intent
    entities: list[EntitySpan] = Field(default_factory=list)
    document_scope: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ResolvedEntity(BaseModel):
    surface_form: str
    entity_type: str
    resolved_id: int | None = None
    canonical_name: str | None = None
    status: EntityStatus
    candidates: list[dict] = Field(default_factory=list)


class RiskSignal(BaseModel):
    layer: Literal["L1", "L2", "L3"]
    level: RiskLevel
    scenario_id: str | None = None
    rationale: str = ""


class RiskAssessment(BaseModel):
    final_level: RiskLevel
    scenario_id: str | None = None
    signals: list[RiskSignal] = Field(default_factory=list)
    disagreement: bool = False


class PlanStep(BaseModel):
    order: int
    source: Literal["policy_kb", "compliance_db", "both"]
    objective: str
    must_prove: str


class EvidencePlan(BaseModel):
    path: EvidencePath
    steps: list[PlanStep]
    required_claims: list[str] = Field(default_factory=list)
    revision: int = 0


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_type: str
    document_title: str
    section: str
    clause_number: str
    version: str
    effective_date: date | None = None
    content: str
    content_type: Literal["text", "table_summary", "image_caption"] = "text"
    modality: Literal["text", "table", "diagram"] = "text"
    image_path: str | None = None
    original_table: str | None = None
    dense_score: float = 0.0
    lexical_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        if not self.clause_number:
            return self.document_title

        return f"{self.document_title} §{self.clause_number}"


class SqlEvidence(BaseModel):
    template_id: str
    statement: str
    parameters: dict
    row_count: int
    rows: list[dict]
    as_of: date
    truncated: bool = False
    rows_filtered_by_scope: int = 0
    generated: bool = False


class PanelOpinion(BaseModel):
    agent: Literal["policy_interpreter", "data_verifier", "challenger"]
    position: str
    supporting_citations: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)


class PanelVerdict(BaseModel):
    consensus: str
    dissent: list[str] = Field(default_factory=list)
    unresolved_conflict: bool = False
    opinions: list[PanelOpinion] = Field(default_factory=list)


class Defect(BaseModel):
    defect_type: DefectType
    description: str
    offending_claim: str | None = None
    suggested_repair: str = ""


class ValidationReport(BaseModel):
    passed: bool
    defects: list[Defect] = Field(default_factory=list)
    grounded_claim_ratio: float = 0.0
    conflict_detected: bool = False


class ConfidenceBreakdown(BaseModel):
    retrieval_score: float = 0.0
    validation_score: float = 0.0
    source_agreement: float = 0.0
    coverage: float = 0.0
    final_score: float = 0.0
    degraded: bool = False


class DraftAnswer(BaseModel):
    answer: str
    cited_clauses: list[str] = Field(default_factory=list)
    uncertainty_note: str = ""


class AuditRecord(BaseModel):
    request_id: str
    thread_id: str
    user_id: str
    role: str
    node: str
    event: str
    risk_level: str | None = None
    confidence: float | None = None
    outcome: str | None = None
    detail: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
