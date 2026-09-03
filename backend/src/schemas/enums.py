from enum import Enum


class Role(str, Enum):
    STORE_ASSOCIATE = "store_associate"
    STORE_MANAGER = "store_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    LEGAL_REVIEWER = "legal_reviewer"
    ADMIN = "admin"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def rank(self) -> int:
        return {"Low": 0, "Medium": 1, "High": 2}[self.value]


class Intent(str, Enum):
    POLICY_LOOKUP = "policy_lookup"
    RECORD_LOOKUP = "record_lookup"
    COMPLIANCE_CHECK = "compliance_check"
    VENDOR_STATUS = "vendor_status"
    RETENTION_QUERY = "retention_query"
    INCIDENT_GUIDANCE = "incident_guidance"
    OUT_OF_SCOPE = "out_of_scope"


class EvidencePath(str, Enum):
    RAG = "rag"
    NL2SQL = "nl2sql"
    HYBRID = "hybrid"
    AGENTIC = "agentic"
    HIGH_RISK_PANEL = "high_risk_panel"


class EntityStatus(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


class DefectType(str, Enum):
    UNGROUNDED_CLAIM = "ungrounded_claim"
    MISSING_CITATION = "missing_citation"
    POLICY_RULE_BREACH = "policy_rule_breach"
    CLAUSE_CONFLICT = "clause_conflict"
    POLICY_RECORD_CONFLICT = "policy_record_conflict"
    SQL_SANITY_FAILURE = "sql_sanity_failure"
    COVERAGE_GAP = "coverage_gap"


class TerminalOutcome(str, Enum):
    ANSWERED = "answered"
    ESCALATED = "escalated"
    REFUSED = "refused"
    CLARIFICATION_REQUIRED = "clarification_required"


class ReviewDecision(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"
