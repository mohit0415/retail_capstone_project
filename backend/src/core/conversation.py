import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from configs.database import read_only_connection, writable_connection
from configs.llms import model_for
from configs.settings import settings
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import THREAD_SUMMARY

logger = logging.getLogger(__name__)

APPEND_TURN = """
INSERT INTO conversation_turns (thread_id, user_id, request_id, speaker, content)
VALUES (%(thread_id)s, %(user_id)s, %(request_id)s, %(speaker)s, %(content)s)
"""

RECENT_TURNS = """
SELECT speaker, content, created_at
FROM conversation_turns
WHERE thread_id = %(thread_id)s
ORDER BY turn_id DESC
LIMIT %(limit)s
"""

TURN_COUNT = "SELECT COUNT(*) AS turns FROM conversation_turns WHERE thread_id = %(thread_id)s"

READ_SUMMARY = "SELECT summary FROM conversation_threads WHERE thread_id = %(thread_id)s"

UPSERT_SUMMARY = """
INSERT INTO conversation_threads (thread_id, summary, turn_count, updated_at)
VALUES (%(thread_id)s, %(summary)s, %(turn_count)s, now())
ON CONFLICT (thread_id) DO UPDATE
SET summary = EXCLUDED.summary,
    turn_count = EXCLUDED.turn_count,
    updated_at = now()
"""

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "sql" / "conversation_schema.sql"


def ensure_conversation_tables() -> bool:
    """Create the runtime tables if missing: conversation memory, audit log,
    escalation queue and SLO latency history (``sql/conversation_schema.sql``).

    Idempotent (IF NOT EXISTS). Called once at app startup so a database
    initialised from an older ``data/sql/schema.sql`` still has every table the
    request path writes to - without this, ``request_latency`` writes fail on
    every request and ``GET /metrics/slo`` answers 503.
    """
    try:
        ddl = SCHEMA_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("conversation schema file missing at %s: %s", SCHEMA_FILE, exc)

        return False

    try:
        with writable_connection() as conn:
            conn.execute(ddl)
    except Exception as exc:
        logger.error("could not create conversation tables: %s", exc)

        return False

    logger.info("runtime tables ready (conversation, system_audit_log, escalation_queue, request_latency)")

    return True


def load_history(thread_id: str, limit: int | None = None) -> list[dict]:
    limit = limit or settings.conversation_window_turns

    try:
        with read_only_connection() as conn:
            rows = conn.execute(RECENT_TURNS, {"thread_id": thread_id, "limit": limit}).fetchall()
    except Exception as exc:
        logger.warning("conversation history unavailable for thread=%s: %s", thread_id, exc)

        return []

    return [{"role": row["speaker"], "content": row["content"]} for row in reversed(rows)]


def load_summary(thread_id: str) -> str:
    try:
        with read_only_connection() as conn:
            row = conn.execute(READ_SUMMARY, {"thread_id": thread_id}).fetchone()
    except Exception as exc:
        logger.warning("thread summary unavailable for thread=%s: %s", thread_id, exc)

        return ""

    if row is None:
        return ""

    return row["summary"] or ""


def load_thread(thread_id: str) -> tuple[list[dict], str]:
    return load_history(thread_id), load_summary(thread_id)


def append_turn(thread_id: str, user_id: str, request_id: str, speaker: str, content: str) -> None:
    if not content:
        return

    payload = {
        "thread_id": thread_id,
        "user_id": user_id,
        "request_id": request_id,
        "speaker": speaker,
        "content": content,
    }

    try:
        with writable_connection() as conn:
            conn.execute(APPEND_TURN, payload)
    except Exception as exc:
        logger.error("could not append %s turn for thread=%s: %s", speaker, thread_id, exc)


def record_user_turn(thread_id: str, user_id: str, request_id: str, question: str) -> None:
    """Persist the user's question as soon as the request is accepted.

    Called for every outcome except a refusal, so that an escalated or
    clarification-required request still leaves the question in the thread
    and a follow-up in the same thread has history to rewrite against.
    """
    append_turn(thread_id, user_id, request_id, "user", question)


def record_assistant_turn(thread_id: str, user_id: str, request_id: str, answer: str) -> None:
    """Persist a released answer; only ever called once an answer is certified."""
    append_turn(thread_id, user_id, request_id, "assistant", answer)

    refresh_summary(thread_id)


def record_exchange(thread_id: str, user_id: str, request_id: str, question: str, answer: str) -> None:
    record_user_turn(thread_id, user_id, request_id, question)
    record_assistant_turn(thread_id, user_id, request_id, answer)


def _turn_count(thread_id: str) -> int:
    with read_only_connection() as conn:
        row = conn.execute(TURN_COUNT, {"thread_id": thread_id}).fetchone()

    return int(row["turns"]) if row else 0


def refresh_summary(thread_id: str) -> None:
    try:
        turns = _turn_count(thread_id)
    except Exception as exc:
        logger.warning("could not count turns for thread=%s: %s", thread_id, exc)

        return

    if turns < settings.conversation_summary_after_turns:
        return

    history = load_history(thread_id, limit=settings.conversation_summary_input_turns)

    if not history:
        return

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

    prompt = render_prompt("THREAD_SUMMARY", THREAD_SUMMARY, transcript=transcript)

    try:
        response = model_for("thread_summary").invoke(
            [SystemMessage(content=prompt), HumanMessage(content="Summarise the thread.")]
        )
        summary = str(response.content).strip()
    except Exception as exc:
        logger.warning("thread summary generation failed for thread=%s: %s", thread_id, exc)

        return

    if not summary:
        return

    try:
        with writable_connection() as conn:
            conn.execute(UPSERT_SUMMARY, {"thread_id": thread_id, "summary": summary, "turn_count": turns})
    except Exception as exc:
        logger.error("thread summary write failed for thread=%s: %s", thread_id, exc)
