import datetime
import json
import logging
import secrets
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict, List

from configs.settings import settings

logger = logging.getLogger(__name__)

EXPLICIT_HANDOFF_PHRASES = (
    "human agent",
    "human support",
    "real person",
    "speak to a human",
    "talk to a human",
    "speak with a human",
    "talk to a person",
    "escalate to a human",
    "escalate this",
    "compliance officer",
    "legal review",
    "talk to legal",
    "contact support",
)


def generate_reference_id(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)

    return f"ESC-{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"


def requested_human(question: str) -> bool:
    lowered = (question or "").lower()

    return any(phrase in lowered for phrase in EXPLICIT_HANDOFF_PHRASES)


def _missing_smtp_settings() -> List[str]:
    return [
        name
        for name, value in (
            ("SMTP_HOST", settings.smtp_host),
            ("SMTP_USERNAME", settings.smtp_username),
            ("SMTP_PASSWORD", settings.smtp_password),
            ("APPLICATION_EMAIL", settings.application_email),
            ("REVIEWER_EMAIL", settings.reviewer_email),
        )
        if not value
    ]


def send_escalation_email(context: Dict[str, Any]) -> bool:
    if not settings.escalation_email_enabled:
        return False

    missing = _missing_smtp_settings()

    if missing:
        logger.warning(
            "SMTP configuration incomplete (%s); escalation e-mail skipped for ref=%s. "
            "The escalation itself was still queued.",
            ", ".join(missing),
            context.get("reference_id", "unknown"),
        )
        return False

    def dump(value: Any) -> str:
        return json.dumps(value, indent=2, default=str)

    body = f"""A compliance question has been escalated for human review.

Reference ID:   {context.get('reference_id')}
Request ID:     {context.get('request_id')}
Thread ID:      {context.get('thread_id')}
Raised by:      {context.get('user_id')} ({context.get('role')})
Risk level:     {context.get('risk_level')}
Confidence:     {context.get('confidence')}
Evidence path:  {context.get('evidence_path')}
Reason:         {context.get('reason')}

--- Question ---
{context.get('standalone_query')}

--- Draft the system would not release ---
{context.get('draft_answer')}

--- Validation output ---
{dump(context.get('validation_output'))}

--- Retrieved clauses ---
{dump(context.get('citations'))}

--- SQL evidence ---
{dump(context.get('sql_evidence'))}

--- Reasoning trace ---
{dump(context.get('reasoning_trace'))}

Open the reviewer console at {settings.reviewer_console_url}/{context.get('request_id')}
"""

    message = MIMEText(body, _charset="utf-8")
    message["Subject"] = f"[COMPLIANCE ESCALATION] {context.get('reference_id')} - {context.get('reason')}"
    message["From"] = settings.application_email
    message["To"] = settings.reviewer_email

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            server.starttls()
            server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(message)

        logger.info("escalation e-mail sent (ref=%s) to %s", context.get("reference_id"), settings.reviewer_email)

        return True
    except Exception as exc:
        logger.error("escalation e-mail failed (ref=%s): %s", context.get("reference_id"), exc)

        return False
