import json
import logging
from datetime import datetime, timezone

from configs.database import writable_connection
from configs.settings import settings
from src.core.audit import audit_from_state
from src.core.handoff import generate_reference_id, requested_human, send_escalation_email
from src.graph.state import AgentState
from src.observability.tracing import traced_node
from src.schemas.enums import RiskLevel, TerminalOutcome

logger = logging.getLogger(__name__)

ENQUEUE = """
INSERT INTO escalation_queue (
    request_id, thread_id, user_id, role, risk_level, reason,
    context_package, status, queued_at
)
VALUES (
    %(request_id)s, %(thread_id)s, %(user_id)s, %(role)s, %(risk_level)s, %(reason)s,
    %(context_package)s, 'pending', %(queued_at)s
)
ON CONFLICT (request_id) DO NOTHING
"""


def build_context_package(state: AgentState) -> dict:
    draft = state.get("draft")
    validation = state.get("validation")
    confidence = state.get("confidence")
    evidence = state.get("sql_evidence")
    panel = state.get("panel_verdict")

    return {
        "original_query": state.get("raw_query"),
        "standalone_query": state.get("standalone_query"),
        "conversation_history": state.get("conversation_history", []),
        "retrieved_documents": [
            {
                "chunk_id": chunk.chunk_id,
                "citation": f"{chunk.document_title} §{chunk.clause_number}",
                "section": chunk.section,
                "version": chunk.version,
                "content": chunk.content,
            }
            for chunk in state.get("retrieved_chunks", [])
        ],
        "sql_evidence": evidence.model_dump(mode="json") if evidence else None,
        "validation_output": validation.model_dump(mode="json") if validation else {},
        "panel_verdict": panel.model_dump(mode="json") if panel else None,
        "plan": state["plan"].model_dump(mode="json") if state.get("plan") else None,
        "reasoning_trace": state.get("trace", []),
        "draft_answer": draft.answer if draft else "",
        "risk_level": state["risk"].final_level.value if state.get("risk") else None,
        "confidence": confidence.final_score if confidence else None,
        "reflection_count": state.get("reflection_count", 0),
        "degraded": state.get("degraded", False),
    }


def _derive_reason(state: AgentState) -> str:
    if requested_human(state.get("raw_query", "")):
        return "the user explicitly asked for a human reviewer"

    if state.get("escalation_reason"):
        return state["escalation_reason"]

    if not state.get("retrieved_chunks") and state.get("sql_evidence") is None:
        return "no grounding evidence was retrieved, so no answer could be supported"

    risk = state.get("risk")

    if risk and risk.final_level is RiskLevel.HIGH:
        return "risk level is High, which always requires human review before release"

    validation = state.get("validation")

    if validation and not validation.passed:
        return f"validation failed after {state.get('reflection_count', 0)} repair attempt(s)"

    if validation and validation.conflict_detected:
        return "an unresolved conflict was detected between the cited sources"

    panel = state.get("panel_verdict")

    if panel and panel.unresolved_conflict:
        return "the high-risk panel could not reach consensus"

    confidence = state.get("confidence")

    if confidence:
        breakdown = (
            f"retrieval {confidence.retrieval_score}, "
            f"validation {confidence.validation_score}, "
            f"agreement {confidence.source_agreement}, "
            f"coverage {confidence.coverage}"
        )

        if confidence.degraded:
            breakdown += ", degraded"

        return (
            f"confidence {confidence.final_score} is below the "
            f"{settings.confidence_threshold} release threshold ({breakdown})"
        )

    return "the system could not certify this answer"


@traced_node("escalation_manager")
def escalation_node(state: AgentState) -> dict:
    reason = _derive_reason(state)
    reference_id = generate_reference_id()

    package = build_context_package(state)
    package["reference_id"] = reference_id

    payload = {
        "request_id": state["request_id"],
        "thread_id": state["thread_id"],
        "user_id": state["user_id"],
        "role": state["role"],
        "risk_level": state["risk"].final_level.value if state.get("risk") else "Unknown",
        "reason": reason,
        "context_package": json.dumps(package, default=str),
        "queued_at": datetime.now(timezone.utc),
    }

    try:
        with writable_connection() as conn:
            conn.execute(ENQUEUE, payload)
    except Exception as exc:
        logger.error("escalation enqueue failed request_id=%s error=%s", state["request_id"], exc)

    confidence = state.get("confidence")

    emailed = send_escalation_email(
        {
            "reference_id": reference_id,
            "request_id": state["request_id"],
            "thread_id": state["thread_id"],
            "user_id": state["user_id"],
            "role": state["role"],
            "risk_level": payload["risk_level"],
            "confidence": confidence.final_score if confidence else None,
            "evidence_path": state.get("evidence_path"),
            "reason": reason,
            "standalone_query": state.get("standalone_query"),
            "draft_answer": package.get("draft_answer"),
            "validation_output": package.get("validation_output"),
            "citations": [item["citation"] for item in package.get("retrieved_documents", [])],
            "sql_evidence": package.get("sql_evidence"),
            "reasoning_trace": package.get("reasoning_trace"),
        }
    )

    result = {
        "terminal_outcome": TerminalOutcome.ESCALATED.value,
        "escalation_reason": reason,
        "escalation_reference": reference_id,
    }

    audit_from_state(
        {**state, **result},
        node="escalation_manager",
        event="escalated",
        detail={"reason": reason, "reference_id": reference_id, "email_sent": emailed},
    )

    return result
