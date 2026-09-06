import json
from datetime import UTC, datetime

from configs.database import read_only_connection, writable_connection
from src.schemas.api import QueueItem, ReviewPackage
from src.schemas.enums import ReviewDecision

LIST_PENDING = """
SELECT request_id, thread_id, user_id, risk_level, reason, queued_at
FROM escalation_queue
WHERE status = 'pending'
ORDER BY
    CASE risk_level WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END,
    queued_at ASC
LIMIT %(limit)s
"""

FETCH_ONE = """
SELECT request_id, thread_id, user_id, risk_level, reason, status,
       context_package, reviewer_decision, reviewed_answer, queued_at, reviewed_at
FROM escalation_queue
WHERE request_id = %(request_id)s
"""

RECORD_DECISION = """
UPDATE escalation_queue
SET status = 'reviewed',
    reviewer_decision = %(decision)s,
    reviewed_answer = %(answer)s,
    reviewer_notes = %(notes)s,
    reviewer_id = %(reviewer_id)s,
    reviewed_at = %(reviewed_at)s
WHERE request_id = %(request_id)s
  AND status = 'pending'
"""

QUEUE_DEPTH = "SELECT COUNT(*) AS depth FROM escalation_queue WHERE status = 'pending'"


def list_pending(limit: int = 50) -> list[QueueItem]:
    with read_only_connection() as conn:
        rows = conn.execute(LIST_PENDING, {"limit": limit}).fetchall()

    return [QueueItem(**dict(row)) for row in rows]


def fetch_package(request_id: str) -> tuple[ReviewPackage | None, dict | None]:
    with read_only_connection() as conn:
        row = conn.execute(FETCH_ONE, {"request_id": request_id}).fetchone()

    if row is None:
        return None, None

    package = row["context_package"]

    if isinstance(package, str):
        package = json.loads(package)

    review = ReviewPackage(
        request_id=row["request_id"],
        thread_id=row["thread_id"],
        original_query=package.get("original_query", ""),
        standalone_query=package.get("standalone_query", ""),
        conversation_history=package.get("conversation_history", []),
        retrieved_documents=package.get("retrieved_documents", []),
        sql_evidence=package.get("sql_evidence"),
        validation_output=package.get("validation_output", {}),
        reasoning_trace=package.get("reasoning_trace", []),
        draft_answer=package.get("draft_answer", ""),
        risk_level=row["risk_level"],
        confidence=package.get("confidence") or 0.0,
    )

    return review, dict(row)


def record_decision(
    request_id: str,
    reviewer_id: str,
    decision: ReviewDecision,
    edited_answer: str | None,
    notes: str,
) -> bool:
    payload = {
        "request_id": request_id,
        "reviewer_id": reviewer_id,
        "decision": decision.value,
        "answer": edited_answer,
        "notes": notes,
        "reviewed_at": datetime.now(UTC),
    }

    with writable_connection() as conn:
        result = conn.execute(RECORD_DECISION, payload)

        return result.rowcount > 0


def queue_depth() -> int:
    with read_only_connection() as conn:
        row = conn.execute(QUEUE_DEPTH).fetchone()

    return int(row["depth"]) if row else 0
