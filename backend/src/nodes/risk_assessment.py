import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.database import read_only_connection
from configs.llms import model_for
from configs.settings import settings
from src.core.budget import widen_for_path
from src.graph.state import AgentState
from src.guardrails.scope import lexical_risk_floor
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import RISK_CLASSIFIER
from src.schemas.enums import EntityStatus, EvidencePath, RiskLevel
from src.schemas.models import RiskAssessment, RiskSignal

logger = logging.getLogger(__name__)


class RiskClassification(BaseModel):
    level: RiskLevel
    scenario_id: str = Field(
        default="",
        description="one scenario id from the list in the prompt, or empty when none fits",
    )
    rationale: str = ""


PROBES = {
    "vendor_non_compliant": """
        SELECT COUNT(*) AS hits
        FROM vendors
        WHERE vendor_id = %(vendor_id)s
          AND compliance_status = 'Non-Compliant'
    """,
    "vendor_not_approved": """
        SELECT COUNT(*) AS hits
        FROM vendors
        WHERE vendor_id = %(vendor_id)s
          AND approval_status IN ('Pending', 'Rejected')
    """,
    "vendor_high_risk_band": """
        SELECT COUNT(*) AS hits
        FROM vendors
        WHERE vendor_id = %(vendor_id)s
          AND risk_category IN ('High', 'Critical')
    """,
    "vendor_review_overdue": """
        SELECT COUNT(*) AS hits
        FROM vendors
        WHERE vendor_id = %(vendor_id)s
          AND next_review_due < %(as_of)s
    """,
    "vendor_severe_open_finding": """
        SELECT COUNT(*) AS hits
        FROM audit_logs
        WHERE vendor_id = %(vendor_id)s
          AND issue_severity IN ('High', 'Critical')
          AND remediation_status <> 'Closed'
    """,
    "vendor_overdue_remediation": """
        SELECT COUNT(*) AS hits
        FROM audit_logs
        WHERE vendor_id = %(vendor_id)s
          AND remediation_status <> 'Closed'
          AND target_resolution_date < %(as_of)s
    """,
    "escalated_finding_present": """
        SELECT COUNT(*) AS hits
        FROM audit_logs
        WHERE vendor_id = %(vendor_id)s
          AND escalation_flag = TRUE
          AND remediation_status <> 'Closed'
    """,
    # the retention probes used to count every overdue row in the whole table, so ANY
    # retention question ("what is the retention period for X?") fused to High and ran
    # the multi-agent panel. They now only count rows for the vendor or department the
    # question actually names.
    "retention_review_overdue": """
        SELECT COUNT(*) AS hits
        FROM retention_records
        WHERE next_review_due < %(as_of)s
          AND legal_hold_flag = FALSE
          AND (%(vendor_id)s::int IS NULL OR vendor_id = %(vendor_id)s::int)
          AND (%(department)s::text IS NULL OR lower(department) = lower(%(department)s::text))
    """,
    "retention_pending_approval": """
        SELECT COUNT(*) AS hits
        FROM retention_records
        WHERE approval_status = 'Pending'
          AND next_review_due < %(as_of)s
          AND (%(vendor_id)s::int IS NULL OR vendor_id = %(vendor_id)s::int)
          AND (%(department)s::text IS NULL OR lower(department) = lower(%(department)s::text))
    """,
    "legal_hold_present": """
        SELECT COUNT(*) AS hits
        FROM retention_records
        WHERE legal_hold_flag = TRUE
          AND (%(vendor_id)s::int IS NULL OR vendor_id = %(vendor_id)s::int)
          AND (%(department)s::text IS NULL OR lower(department) = lower(%(department)s::text))
    """,
    "open_escalation_review": """
        SELECT COUNT(*) AS hits
        FROM compliance_reviews
        WHERE vendor_id = %(vendor_id)s
          AND review_type = 'Escalation Review'
          AND review_status <> 'Closed'
    """,
}

# probes that can be anchored by a department as well as by a vendor
DEPARTMENT_ANCHORED_PROBES = {"retention_review_overdue", "retention_pending_approval", "legal_hold_present"}

# an overdue review is a "Medium" situation in the classifier's own definitions
# (see RISK_CLASSIFIER); only the genuinely live exposures raise the level to High
PROBE_LEVEL = {
    "retention_review_overdue": RiskLevel.MEDIUM,
    "retention_pending_approval": RiskLevel.MEDIUM,
    "vendor_review_overdue": RiskLevel.MEDIUM,
}

SCENARIO_TO_PROBES = {
    "vendor_engagement": [
        "vendor_non_compliant",
        "vendor_not_approved",
        "vendor_severe_open_finding",
    ],
    "vendor_data_sharing": [
        "vendor_non_compliant",
        "vendor_high_risk_band",
        "escalated_finding_present",
    ],
    "vendor_termination": [
        "vendor_severe_open_finding",
        "open_escalation_review",
        "legal_hold_present",
    ],
    "audit_finding": [
        "vendor_overdue_remediation",
        "escalated_finding_present",
        "vendor_severe_open_finding",
    ],
    "data_retention": [
        "retention_review_overdue",
        "retention_pending_approval",
    ],
    "data_erasure": [
        "legal_hold_present",
        "retention_review_overdue",
    ],
    "vendor_review": [
        "vendor_review_overdue",
        "open_escalation_review",
    ],
}


def _probe_anchor(name: str, vendor_id: int | None, department: str | None) -> dict | None:
    """The parameters a probe runs with, or None when the question names nothing it can check."""
    if vendor_id is not None:
        return {"vendor_id": vendor_id, "department": None}

    if department and name in DEPARTMENT_ANCHORED_PROBES:
        return {"vendor_id": None, "department": department}

    return None


def run_probes(
    scenario_id: str, vendor_id: int | None, department: str | None = None
) -> tuple[list[RiskSignal], list[str], list[str]]:
    signals: list[RiskSignal] = []
    ran: list[str] = []
    skipped: list[str] = []

    for name in SCENARIO_TO_PROBES.get(scenario_id, []):
        anchor = _probe_anchor(name, vendor_id, department)

        if anchor is None:
            skipped.append(name)
            continue

        params = {"as_of": settings.as_of_date, **anchor}

        try:
            with read_only_connection() as conn:
                row = conn.execute(PROBES[name], params).fetchone()
        except Exception as exc:
            logger.warning("risk probe %s failed: %s", name, exc)
            continue

        ran.append(name)
        hits = int(row["hits"]) if row else 0
        logger.debug(
            "risk probe %s scenario=%s vendor_id=%s department=%s hits=%d",
            name,
            scenario_id,
            vendor_id,
            department,
            hits,
        )

        if hits > 0:
            anchor_text = f"vendor {vendor_id}" if vendor_id is not None else f"department {department}"
            signals.append(
                RiskSignal(
                    layer="L3",
                    level=PROBE_LEVEL.get(name, RiskLevel.HIGH),
                    scenario_id=scenario_id,
                    rationale=(
                        f"probe {name} confirmed {row['hits']} matching row(s) for {anchor_text} "
                        f"as of {settings.as_of_date}"
                    ),
                )
            )

    if skipped:
        logger.info(
            "risk probes skipped scenario=%s probes=%s reason=the question names no vendor or department to check",
            scenario_id,
            skipped,
        )

    return signals, ran, skipped


def _run_probes(scenario_id: str, vendor_id: int | None) -> list[RiskSignal]:
    """Kept for older callers: vendor-anchored probes only."""
    signals, _, _ = run_probes(scenario_id, vendor_id, None)

    return signals


def _first_vendor_id(state: AgentState) -> int | None:
    for entity in state.get("resolved_entities", []):
        if entity.entity_type == "vendor" and entity.resolved_id is not None:
            return entity.resolved_id

    return None


def _first_department(state: AgentState) -> str | None:
    for entity in state.get("resolved_entities", []):
        if entity.entity_type == "department" and entity.canonical_name:
            return entity.canonical_name

    return None


def fuse(signals: list[RiskSignal], unresolved_entity: bool) -> RiskAssessment:
    levels = [signal.level for signal in signals]

    if unresolved_entity:
        levels.append(RiskLevel.MEDIUM)

    final = max(levels, key=lambda level: level.rank) if levels else RiskLevel.LOW

    l2 = next((s for s in signals if s.layer == "L2"), None)
    l3 = next((s for s in signals if s.layer == "L3"), None)

    disagreement = bool(l2 and l3 and l2.level.rank < l3.level.rank)

    scenario = next((s.scenario_id for s in signals if s.scenario_id), None)

    return RiskAssessment(
        final_level=final,
        scenario_id=scenario,
        signals=signals,
        disagreement=disagreement,
    )


def classify_risk_l2(state: AgentState, query: str) -> RiskClassification:
    """The L2 (LLM) risk judgement. It only reads the query, so it can run next to the intent classifier."""
    floor_value, triggers = lexical_risk_floor(query)

    prompt = render_prompt(
        "RISK_CLASSIFIER",
        RISK_CLASSIFIER,
        lexical_floor=RiskLevel(floor_value).value,
        triggers=triggers or "none",
        query=query,
    )
    model = model_for("risk_l2_classifier").with_structured_output(RiskClassification)

    return model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=query)],
        config=runnable_config(state, "risk_l2_classifier"),
    )


def _precomputed_l2(state: AgentState, query: str) -> RiskClassification | None:
    cached = state.get("risk_l2")

    if not cached or cached.get("query") != query:
        return None

    try:
        return RiskClassification(
            level=RiskLevel(cached["level"]),
            scenario_id=cached.get("scenario_id") or "",
            rationale=cached.get("rationale") or "",
        )
    except Exception:
        logger.debug("precomputed risk classification unusable, classifying again", exc_info=True)

        return None


@traced_node("risk_assessment")
def risk_assessment_node(state: AgentState) -> dict:
    query = state["standalone_query"]

    floor_value, triggers = lexical_risk_floor(query)
    floor = RiskLevel(floor_value)

    signals = [
        RiskSignal(
            layer="L1",
            level=floor,
            rationale=f"lexical triggers matched: {triggers}" if triggers else "no lexical trigger matched",
        )
    ]

    classification = _precomputed_l2(state, query)
    precomputed = classification is not None

    if classification is None:
        classification = classify_risk_l2(state, query)

    signals.append(
        RiskSignal(
            layer="L2",
            level=classification.level,
            scenario_id=classification.scenario_id or None,
            rationale=classification.rationale,
        )
    )

    probes_run: list[str] = []
    probes_skipped: list[str] = []

    if classification.scenario_id:
        probe_signals, probes_run, probes_skipped = run_probes(
            classification.scenario_id, _first_vendor_id(state), _first_department(state)
        )
        signals.extend(probe_signals)

    unresolved = any(
        entity.status in (EntityStatus.UNRESOLVED, EntityStatus.AMBIGUOUS)
        for entity in state.get("resolved_entities", [])
    )

    assessment = fuse(signals, unresolved)
    assessment.probes_run = probes_run
    assessment.probes_skipped = probes_skipped

    logger.info(
        "risk fused=%s scenario=%s l1=%s l2=%s l3_hits=%d probes_run=%s probes_skipped=%s "
        "unresolved_entity=%s disagreement=%s l2_precomputed=%s request_id=%s",
        assessment.final_level.value,
        assessment.scenario_id or "none",
        floor.value,
        classification.level.value,
        sum(1 for s in signals if s.layer == "L3"),
        probes_run or "none",
        probes_skipped or "none",
        unresolved,
        assessment.disagreement,
        precomputed,
        state.get("request_id"),
    )

    result = {"risk": assessment, "tokens_spent": 700}

    if assessment.final_level is RiskLevel.HIGH:
        widened = widen_for_path(state, EvidencePath.HIGH_RISK_PANEL.value)
        result.update(widened)

        logger.warning(
            "risk HIGH -> mandatory multi-agent panel request_id=%s scenario=%s deadline_widened=%s",
            state.get("request_id"),
            assessment.scenario_id or "none",
            bool(widened),
        )

    if assessment.disagreement:
        logger.info(
            "risk disagreement request_id=%s l2=%s l3=%s -> fused %s, continuing to evidence",
            state.get("request_id"),
            next((s.level.value for s in signals if s.layer == "L2"), None),
            next((s.level.value for s in signals if s.layer == "L3"), None),
            assessment.final_level.value,
        )

    return result
