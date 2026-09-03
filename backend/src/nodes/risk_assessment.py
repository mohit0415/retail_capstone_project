import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.database import read_only_connection
from configs.llms import model_for
from configs.settings import settings
from src.graph.state import AgentState
from src.guardrails.scope import lexical_risk_floor
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import RISK_CLASSIFIER
from src.schemas.enums import EntityStatus, RiskLevel
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
    "retention_review_overdue": """
        SELECT COUNT(*) AS hits
        FROM retention_records
        WHERE next_review_due < %(as_of)s
          AND legal_hold_flag = FALSE
    """,
    "retention_pending_approval": """
        SELECT COUNT(*) AS hits
        FROM retention_records
        WHERE approval_status = 'Pending'
          AND next_review_due < %(as_of)s
    """,
    "legal_hold_present": """
        SELECT COUNT(*) AS hits
        FROM retention_records
        WHERE vendor_id = %(vendor_id)s
          AND legal_hold_flag = TRUE
    """,
    "open_escalation_review": """
        SELECT COUNT(*) AS hits
        FROM compliance_reviews
        WHERE vendor_id = %(vendor_id)s
          AND review_type = 'Escalation Review'
          AND review_status <> 'Closed'
    """,
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


def _run_probes(scenario_id: str, vendor_id: int | None) -> list[RiskSignal]:
    signals: list[RiskSignal] = []
    probe_names = SCENARIO_TO_PROBES.get(scenario_id, [])

    for name in probe_names:
        statement = PROBES[name]

        if "vendor_id" in statement and vendor_id is None:
            continue

        params = {"as_of": settings.as_of_date}

        if "vendor_id" in statement:
            params["vendor_id"] = vendor_id

        try:
            with read_only_connection() as conn:
                row = conn.execute(statement, params).fetchone()
        except Exception as exc:
            logger.warning("risk probe %s failed: %s", name, exc)
            continue

        if row and int(row["hits"]) > 0:
            signals.append(
                RiskSignal(
                    layer="L3",
                    level=RiskLevel.HIGH,
                    scenario_id=scenario_id,
                    rationale=f"probe {name} confirmed {row['hits']} matching row(s) as of {settings.as_of_date}",
                )
            )

    return signals


def _first_vendor_id(state: AgentState) -> int | None:
    for entity in state.get("resolved_entities", []):
        if entity.entity_type == "vendor" and entity.resolved_id is not None:
            return entity.resolved_id

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

    prompt = RISK_CLASSIFIER.format(lexical_floor=floor.value, triggers=triggers or "none", query=query)
    model = model_for("risk_l2_classifier").with_structured_output(RiskClassification)

    classification: RiskClassification = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=query)],
        config=runnable_config(state, "risk_l2_classifier"),
    )

    signals.append(
        RiskSignal(
            layer="L2",
            level=classification.level,
            scenario_id=classification.scenario_id or None,
            rationale=classification.rationale,
        )
    )

    if classification.scenario_id:
        signals.extend(_run_probes(classification.scenario_id, _first_vendor_id(state)))

    unresolved = any(
        entity.status in (EntityStatus.UNRESOLVED, EntityStatus.AMBIGUOUS)
        for entity in state.get("resolved_entities", [])
    )

    assessment = fuse(signals, unresolved)

    if assessment.disagreement:
        return {
            "risk": assessment,
            "escalation_reason": "risk classifier and database probe disagreed",
            "tokens_spent": 700,
        }

    return {"risk": assessment, "tokens_spent": 700}
