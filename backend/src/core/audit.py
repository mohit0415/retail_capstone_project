import json
import logging

from configs.database import writable_connection
from src.schemas.models import AuditRecord

logger = logging.getLogger(__name__)

INSERT_AUDIT = """
INSERT INTO system_audit_log (
    request_id, thread_id, user_id, role, node, event,
    risk_level, confidence, outcome, detail, created_at
)
VALUES (
    %(request_id)s, %(thread_id)s, %(user_id)s, %(role)s, %(node)s, %(event)s,
    %(risk_level)s, %(confidence)s, %(outcome)s, %(detail)s::jsonb, %(created_at)s
)
"""


def write_audit(record: AuditRecord) -> None:
    try:
        payload = record.model_dump()
        payload["detail"] = json.dumps(payload["detail"], default=str)

        with writable_connection() as conn:
            conn.execute(INSERT_AUDIT, payload)
    except Exception as exc:
        logger.error(
            "audit write failed request_id=%s node=%s event=%s error=%s",
            record.request_id,
            record.node,
            record.event,
            exc,
        )


def audit_from_state(state: dict, node: str, event: str, detail: dict | None = None) -> None:
    risk = state.get("risk")
    confidence = state.get("confidence")

    record = AuditRecord(
        request_id=state.get("request_id", "unknown"),
        thread_id=state.get("thread_id", "unknown"),
        user_id=state.get("user_id", "unknown"),
        role=state.get("role", "unknown"),
        node=node,
        event=event,
        risk_level=risk.final_level.value if risk else None,
        confidence=confidence.final_score if confidence else None,
        outcome=state.get("terminal_outcome"),
        detail=detail or {},
    )

    write_audit(record)
