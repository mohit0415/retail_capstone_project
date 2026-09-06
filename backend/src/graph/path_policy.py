from dataclasses import dataclass

from src.schemas.enums import EvidencePath, Intent, RiskLevel

POLICY = "policy"
RECORDS = "records"
BOTH = "both"

ESCALATE = "escalate"


@dataclass(frozen=True)
class PathRule:
    default: EvidencePath
    allowed: frozenset[EvidencePath]
    fallback: EvidencePath
    evidence: str
    rationale: str


@dataclass(frozen=True)
class PathDecision:
    path: str
    reason: str
    clamped: bool = False
    partial_evidence: bool = False


POLICY_LOOKUP = PathRule(
    default=EvidencePath.RAG,
    allowed=frozenset({EvidencePath.RAG}),
    fallback=EvidencePath.RAG,
    evidence=POLICY,
    rationale="the answer is a clause; the database holds no policy text",
)

RECORD_LOOKUP = PathRule(
    default=EvidencePath.NL2SQL,
    allowed=frozenset({EvidencePath.NL2SQL}),
    fallback=EvidencePath.NL2SQL,
    evidence=RECORDS,
    rationale="the answer is a row; policy text cannot state what a record holds",
)

VENDOR_STATUS = PathRule(
    default=EvidencePath.NL2SQL,
    allowed=frozenset({EvidencePath.NL2SQL, EvidencePath.HYBRID}),
    fallback=EvidencePath.NL2SQL,
    evidence=RECORDS,
    rationale="status is a stored column; policy only enters when the question asks whether that status is allowed",
)

RETENTION_QUERY = PathRule(
    default=EvidencePath.HYBRID,
    allowed=frozenset({EvidencePath.RAG, EvidencePath.NL2SQL, EvidencePath.HYBRID}),
    fallback=EvidencePath.NL2SQL,
    evidence=BOTH,
    rationale="a retention period is written in policy and the review cycle is stored in a record",
)

COMPLIANCE_CHECK = PathRule(
    default=EvidencePath.HYBRID,
    allowed=frozenset({EvidencePath.HYBRID, EvidencePath.AGENTIC}),
    fallback=EvidencePath.HYBRID,
    evidence=BOTH,
    rationale="the question is exactly a rule checked against a record, so both sources are load bearing",
)

INCIDENT_GUIDANCE = PathRule(
    default=EvidencePath.AGENTIC,
    allowed=frozenset({EvidencePath.AGENTIC, EvidencePath.HYBRID}),
    fallback=EvidencePath.HYBRID,
    evidence=BOTH,
    rationale="what to look up next depends on what the first lookup returns, which is the only case agentic earns",
)

RULES: dict[Intent, PathRule] = {
    Intent.POLICY_LOOKUP: POLICY_LOOKUP,
    Intent.RECORD_LOOKUP: RECORD_LOOKUP,
    Intent.VENDOR_STATUS: VENDOR_STATUS,
    Intent.RETENTION_QUERY: RETENTION_QUERY,
    Intent.COMPLIANCE_CHECK: COMPLIANCE_CHECK,
    Intent.INCIDENT_GUIDANCE: INCIDENT_GUIDANCE,
}

UNKNOWN_INTENT = POLICY_LOOKUP

NEEDS_RECORDS = frozenset({EvidencePath.NL2SQL, EvidencePath.HYBRID, EvidencePath.AGENTIC})

NEEDS_DOCUMENTS = frozenset({EvidencePath.RAG, EvidencePath.HYBRID, EvidencePath.AGENTIC})

DEGRADE_LADDER = {
    EvidencePath.AGENTIC: EvidencePath.HYBRID,
    EvidencePath.HYBRID: None,
    EvidencePath.NL2SQL: None,
    EvidencePath.RAG: None,
}


def rule_for(intent: Intent | None) -> PathRule:
    if intent is None:
        return UNKNOWN_INTENT

    return RULES.get(intent, UNKNOWN_INTENT)


def allowed_path_names(intent: Intent | None) -> list[str]:
    return sorted(path.value for path in rule_for(intent).allowed)


def resolve_path(
    intent: Intent | None,
    risk_level: RiskLevel | None,
    proposed: EvidencePath | None,
    has_tables: bool,
    has_documents: bool,
    agentic_enabled: bool = True,
) -> PathDecision:
    if risk_level is RiskLevel.HIGH:
        return PathDecision(
            path=EvidencePath.HIGH_RISK_PANEL.value,
            reason="risk fused to High, so the multi-agent panel is mandatory",
        )

    rule = rule_for(intent)
    clamped = False

    if proposed is None or proposed not in rule.allowed:
        rejected = proposed.value if proposed else "none"
        path = rule.default
        clamped = proposed is not None
        reason = (
            f"intent {_intent_name(intent)} admits {allowed_path_names(intent)}; "
            f"the planner proposed {rejected}, so it was clamped to {path.value} because {rule.rationale}"
        )
    else:
        path = proposed
        reason = f"planner chose {path.value}, which intent {_intent_name(intent)} admits"

    if path is EvidencePath.AGENTIC and not agentic_enabled:
        path = rule.fallback
        reason = f"the agentic path is disabled, so this falls back to {path.value}"
        clamped = True

    return _apply_capability_gates(rule, path, reason, clamped, has_tables, has_documents)


def degrade_for_budget(
    path: EvidencePath,
    intent: Intent | None,
    seconds_remaining: float,
    agentic_min_seconds: float,
    hybrid_min_seconds: float,
) -> PathDecision:
    rule = rule_for(intent)

    if path is EvidencePath.AGENTIC and seconds_remaining < agentic_min_seconds:
        target = DEGRADE_LADDER[EvidencePath.AGENTIC]

        return PathDecision(
            path=target.value,
            reason=(
                f"{seconds_remaining:.1f}s left is under the {agentic_min_seconds:.1f}s an agentic run needs, "
                f"so it degrades to {target.value}, which still reads both sources"
            ),
            clamped=True,
            partial_evidence=True,
        )

    if path is EvidencePath.HYBRID and seconds_remaining < hybrid_min_seconds:
        target = rule.fallback if rule.fallback is not EvidencePath.HYBRID else EvidencePath.RAG

        return PathDecision(
            path=target.value,
            reason=(
                f"{seconds_remaining:.1f}s left is under the {hybrid_min_seconds:.1f}s hybrid needs, so it "
                f"degrades to {target.value}, the source this intent actually turns on"
            ),
            clamped=True,
            partial_evidence=True,
        )

    return PathDecision(path=path.value, reason="budget headroom is sufficient for the chosen path")


def _apply_capability_gates(
    rule: PathRule,
    path: EvidencePath,
    reason: str,
    clamped: bool,
    has_tables: bool,
    has_documents: bool,
) -> PathDecision:
    if path in NEEDS_RECORDS and not has_tables:
        if rule.evidence == RECORDS:
            return PathDecision(
                path=ESCALATE,
                reason="this question can only be answered from records and the role is granted no table",
                clamped=True,
            )

        return PathDecision(
            path=EvidencePath.RAG.value,
            reason=f"{reason}; the role reads no table, so only the policy half of the answer is available",
            clamped=True,
            partial_evidence=True,
        )

    if path in NEEDS_DOCUMENTS and not has_documents:
        if rule.evidence == POLICY:
            return PathDecision(
                path=ESCALATE,
                reason="this question can only be answered from policy text and the role is granted no document",
                clamped=True,
            )

        return PathDecision(
            path=EvidencePath.NL2SQL.value,
            reason=f"{reason}; the role reads no document, so only the record half of the answer is available",
            clamped=True,
            partial_evidence=True,
        )

    return PathDecision(path=path.value, reason=reason, clamped=clamped)


def _intent_name(intent: Intent | None) -> str:
    return intent.value if intent else "unknown"
