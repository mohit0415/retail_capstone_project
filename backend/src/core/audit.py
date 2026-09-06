import json
import logging
import threading
import time

from configs.database import writable_connection
from src.schemas.models import AuditRecord

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = 3

COOLDOWN_SECONDS = 30.0

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


class AuditCircuit:
    def __init__(self, threshold: int = FAILURE_THRESHOLD, cooldown: float = COOLDOWN_SECONDS):
        self.threshold = threshold
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at = 0.0
        self._skipped = 0

    def should_attempt(self) -> bool:
        with self._lock:
            if self._failures < self.threshold:
                return True

            if time.monotonic() - self._opened_at >= self.cooldown:
                self._failures = 0

                return True

            self._skipped += 1

            return False

    def record_success(self) -> None:
        with self._lock:
            if self._failures:
                logger.info("audit writes recovered after %s skipped row(s)", self._skipped)

            self._failures = 0
            self._skipped = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1

            if self._failures == self.threshold:
                self._opened_at = time.monotonic()

                logger.error(
                    "audit writes suspended for %.0fs after %s consecutive failures; rows in this "
                    "window are lost and the append-only log will have a gap. The release path is "
                    "kept inside its deadline rather than blocked on the database",
                    self.cooldown,
                    self._failures,
                )

    def status(self) -> dict:
        with self._lock:
            open_now = self._failures >= self.threshold and (
                time.monotonic() - self._opened_at < self.cooldown
            )

            return {
                "writing": not open_now,
                "consecutive_failures": self._failures,
                "rows_skipped_in_window": self._skipped,
            }


circuit = AuditCircuit()


def write_audit(record: AuditRecord) -> None:
    if not circuit.should_attempt():
        return

    try:
        payload = record.model_dump()
        payload["detail"] = json.dumps(payload["detail"], default=str)

        with writable_connection() as conn:
            conn.execute(INSERT_AUDIT, payload)
    except Exception as exc:
        circuit.record_failure()

        logger.error(
            "audit write failed request_id=%s node=%s event=%s error=%s",
            record.request_id,
            record.node,
            record.event,
            exc,
        )

        return

    circuit.record_success()


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
