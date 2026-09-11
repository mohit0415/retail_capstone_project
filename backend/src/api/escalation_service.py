import json
import logging
from datetime import UTC, datetime

from configs.database import read_only_connection, writable_connection
from src.schemas.api import QueueItem, ReviewPackage
from src.schemas.enums import ReviewDecision

logger = logging.getLogger(__name__)

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

STORE_RELEASED_ANSWER = """
UPDATE escalation_queue
SET reviewed_answer = %(answer)s
WHERE request_id = %(request_id)s
  AND status = 'reviewed'
"""

REOPEN = """
UPDATE escalation_queue
SET status = 'pending',
    reviewer_decision = NULL,
    reviewed_answer = NULL,
    reviewer_notes = NULL,
    reviewer_id = NULL,
    reviewed_at = NULL
WHERE request_id = %(request_id)s
  AND status = 'reviewed'
"""


def list_pending(limit: int = 50) -> list[QueueItem]:
    with read_only_connection() as conn:
        rows = conn.execute(LIST_PENDING, {"limit": limit}).fetchall()

    logger.info("review queue listed pending=%d limit=%d", len(rows), limit)

    return [QueueItem(**dict(row)) for row in rows]


def fetch_package(request_id: str) -> tuple[ReviewPackage | None, dict | None]:
    with read_only_connection() as conn:
        row = conn.execute(FETCH_ONE, {"request_id": request_id}).fetchone()

    if row is None:
        logger.info("review package not found request_id=%s", request_id)

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
        evidence_path=package.get("evidence_path"),
        panel_verdict=package.get("panel_verdict"),
        plan=package.get("plan"),
        panel_repair_count=package.get("panel_repair_count") or 0,
        reflection_count=package.get("reflection_count") or 0,
        reference_id=package.get("reference_id"),
        budget_stops=package.get("budget_stops") or [],
        degraded=bool(package.get("degraded", False)),
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
        updated = result.rowcount > 0

    if updated:
        logger.info(
            "review decision recorded request_id=%s reviewer=%s decision=%s edited=%s notes_chars=%d",
            request_id,
            reviewer_id,
            decision.value,
            bool(edited_answer),
            len(notes or ""),
        )
    else:
        logger.warning(
            "review decision NOT recorded request_id=%s reviewer=%s (already reviewed or unknown)",
            request_id,
            reviewer_id,
        )

    return updated


def queue_depth() -> int:
    with read_only_connection() as conn:
        row = conn.execute(QUEUE_DEPTH).fetchone()

    return int(row["depth"]) if row else 0


def store_released_answer(request_id: str, answer: str) -> None:
    """Keep the answer the output guardrail released (PII scrubbed), which is what the asker is shown."""
    with writable_connection() as conn:
        conn.execute(STORE_RELEASED_ANSWER, {"request_id": request_id, "answer": answer})

    logger.info("released answer stored request_id=%s answer_chars=%d", request_id, len(answer or ""))


def reopen(request_id: str) -> bool:
    """Put a reviewed request back in the queue when the reviewed answer could not be released.

    The decision is recorded before the graph resumes (that UPDATE is what stops two reviewers
    releasing the same request). Without this, an answer the output guardrail refused left the
    request marked reviewed with nothing released: the reviewer got 409 on a second try and the
    asker never received an answer.
    """
    with writable_connection() as conn:
        result = conn.execute(REOPEN, {"request_id": request_id})
        reopened = result.rowcount > 0

    logger.warning("review reopened request_id=%s reopened=%s", request_id, reopened)

    return reopened
