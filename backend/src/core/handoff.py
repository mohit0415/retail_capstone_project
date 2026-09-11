import datetime
import json
import logging
import re
import secrets
import smtplib
from email.mime.text import MIMEText
from typing import Any

from configs.settings import settings

logger = logging.getLogger(__name__)

# An explicit request for a human or for legal validation - the brief's own escalation trigger.
#
# The question is read one sentence at a time, lower-cased, with straight apostrophes and single
# spaces, so no pattern can backtrack over a run of whitespace. A sentence that opens as a question
# about the rules ("Do I need legal approval ...", "Should I escalate ...", "How do I contact
# support ...") is a policy question, never a request. Any other sentence is a request when it asks
# for a person: "talk to a human", "I need legal validation", "please contact support", "escalate
# this", "can a human review this". A bare mention of the compliance officer, legal review or
# support is not one - matching the noun held validated Low-risk answers for a reviewer.
MAX_SCANNED_CHARS = 4000

_SENTENCE_BREAK = re.compile(r"[.!?;\n]+")

_FILLER = re.compile(r"^(?:(?:hi|hello|hey|ok|okay|thanks|thank you|also|and|so|then|now)\b[ ,]*)+")

_DESCRIPTIVE_OPENER = re.compile(
    r"^(?:do|does|did|should|shall|must|is|are|was|were|am|when|which|what|who|whom|whose|how|why|where"
    r"|has|had|if|whether)\b"
)

_PERSON = (
    r"(?:a |an |the |someone (?:in|from) |somebody (?:in|from) )?"
    r"(?:human|person|real person|agent|reviewer|compliance officer|legal team|legal|lawyer|attorney"
    r"|solicitor|support|someone)"
)

_LEGAL_ASK = r"(?:validation|sign-off|sign off|signoff|opinion|advice|counsel|approval|review|check|confirmation)"

_REVIEWED = r"(?:validated|checked|reviewed|approved|confirmed|signed off)"

_BY_PERSON = r"(?:a |an |the )?(?:human|legal team|legal|compliance officer|lawyer|reviewer)"

_WANT = r"(?:i|we)(?: would|'d)? (?:want|need|like|require|request)"

_LEAD = r"(?:^|\bplease |\b(?:can|could|would) you (?:please )?)"

HANDOFF_REQUEST = re.compile(
    "|".join(
        (
            rf"\b(?:talk|speak|chat) (?:to|with) {_PERSON}\b",
            rf"\b(?:connect|transfer|put) me (?:with|to|through to) {_PERSON}\b",
            r"^(?:please |kindly )?escalate\b",
            r"\bplease escalate\b",
            r"\bescalate (?:this|it|my (?:question|request|case|query|answer))\b",
            r"\bescalate (?:this |it )?to (?:a |an |the )?(?:human|person|reviewer|compliance officer|compliance"
            r"|legal team|legal|manager)\b",
            r"\b(?:can|could|would|will) (?:you|someone) (?:please )?escalate\b",
            rf"\b{_WANT} (?:to escalate|this escalated|it escalated|an escalation)\b",
            r"^(?:please )?contact (?:support|legal|the legal team|a human|the compliance officer"
            r"|a compliance officer|someone)\b",
            r"\bplease contact (?:support|legal|the legal team|a human|the compliance officer|a compliance officer"
            r"|someone)\b",
            r"\b(?:can|could|would) you (?:please )?contact\b",
            r"\b(?:i|we)(?: would|'d)? (?:want|need|like) to contact (?:support|legal|a human|the compliance officer"
            r"|a compliance officer|someone)\b",
            r"\bcontact (?:support|legal) for me\b",
            rf"\b{_WANT} (?:a |an |some )?legal {_LEGAL_ASK}\b",
            r"\b(?:i|we)(?: would|'d)? (?:want|need|like) (?:legal|the legal team|a lawyer|an attorney) to "
            r"(?:validate|check|review|confirm|approve|look at|sign off)\b",
            rf"\b(?:i|we)(?: would|'d)? (?:want|need|like) (?:this|it|that) {_REVIEWED} by {_BY_PERSON}\b",
            r"\b(?:i|we)(?: would|'d)? like to have (?:the )?legal (?:team )?(?:check|review|validate|confirm|look at)\b",
            rf"\blegal {_LEGAL_ASK},? please\b",
            rf"(?:^|\bplease )(?:get|arrange|request|obtain) (?:a |an )?legal {_LEGAL_ASK}\b",
            rf"\b(?:can|could|may) (?:i|we) (?:please )?(?:get|have|request|ask for) (?:a |an )?legal {_LEGAL_ASK}\b",
            r"\b(?:can|could|would) (?:you|we|i) (?:please )?(?:have|get|ask) (?:the )?legal (?:team )?(?:to )?"
            r"(?:check|review|validate|confirm|approve|look at|sign off)\b",
            r"\b(?:can|could|would) (?:the )?legal (?:team )?(?:please )?(?:validate|check|review|confirm|approve"
            r"|sign off(?: on)?) (?:this|it|that|my)\b",
            r"(?:^|\bplease )(?:have|get|ask|loop in|involve) (?:the )?legal\b",
            r"\b(?:validate|check|confirm|verify|review) (?:this|it|that) with (?:the )?legal\b",
            r"\bsend (?:this|it) (?:to legal|for (?:a )?legal (?:review|validation|check|sign-off|sign off))\b",
            rf"\b(?:this|it|my (?:answer|question|case|request)) (?:needs|requires) (?:a )?legal {_LEGAL_ASK}\b",
            r"\b(?:can|could|would) (?:a human|someone|somebody|a person|a reviewer|a compliance officer"
            r"|someone from legal|legal) (?:please )?(?:review|check|look at|confirm|validate) (?:this|it|that|my)\b",
            rf"{_LEAD}(?:have|get|let) (?:a |an )?(?:human|person|reviewer|compliance officer|lawyer)\b",
            rf"{_LEAD}(?:have|get) (?:this|it|that) {_REVIEWED} by {_BY_PERSON}\b",
            rf"\b{_WANT} (?:a |an |to see a |to have a )?(?:human|real person|human reviewer|reviewer|lawyer"
            r"|attorney|compliance officer|human review)\b",
            r"(?:^|\bplease )ask (?:the |a )?(?:compliance officer|legal team|legal|lawyer|human) to\b",
        )
    )
)


def generate_reference_id(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now(datetime.UTC)

    return f"ESC-{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"


def _sentences(question: str) -> list[str]:
    text = (question or "")[:MAX_SCANNED_CHARS].replace(chr(0x2018), "'").replace(chr(0x2019), "'").lower()
    sentences = []

    for raw in _SENTENCE_BREAK.split(text):
        sentence = _FILLER.sub("", " ".join(raw.split())).strip(" ,:-")

        if sentence:
            sentences.append(sentence)

    return sentences


def requested_human(question: str) -> bool:
    """True when the question explicitly asks for a human reviewer or for legal validation."""
    for sentence in _sentences(question):
        if _DESCRIPTIVE_OPENER.match(sentence) and "please" not in sentence:
            continue

        if HANDOFF_REQUEST.search(sentence):
            return True

    return False


def _missing_smtp_settings() -> list[str]:
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


def send_escalation_email(context: dict[str, Any]) -> bool:
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
