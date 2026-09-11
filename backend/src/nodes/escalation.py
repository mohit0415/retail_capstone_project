import json
import logging
from datetime import UTC, datetime

from configs.database import writable_connection
from configs.settings import settings
from src.core.audit import audit_from_state
from src.core.budget import guard_from_state
from src.core.handoff import generate_reference_id, requested_human, send_escalation_email
from src.graph.routing import repair_attempts_allowed, repair_token_estimate
from src.graph.state import AgentState
from src.observability.tracing import traced_node
from src.schemas.enums import DefectType, RiskLevel, TerminalOutcome

logger = logging.getLogger(__name__)

CONFLICT_DEFECT_TYPES = {DefectType.CLAUSE_CONFLICT, DefectType.POLICY_RECORD_CONFLICT}

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

REPAIR_HEADROOM_SECONDS = 1.0

REPAIR_HEADROOM_TOKENS = 2000

RETRIEVAL_NO_MATCH_CEILING = 0.05


def thread_for_reviewer(state: AgentState) -> list[dict]:
    """The conversation as the reviewer should read it: prior turns, then the question under review.

    ``state["conversation_history"]`` holds only the turns that existed *before*
    this request (that is what the query rewrite works from), and the current
    question is written to the thread in a background task after the response
    goes out. Without this, a first question in a thread - the common case -
    shows an empty history in the review package even though a draft exists.
    """
    prior = [dict(turn) for turn in state.get("conversation_history") or []]
    question = state.get("raw_query") or state.get("standalone_query") or ""

    if not question:
        return prior

    return [
        *prior,
        {"role": "user", "content": question, "request_id": state.get("request_id"), "current": True},
    ]


def build_context_package(state: AgentState) -> dict:
    draft = state.get("draft")
    validation = state.get("validation")
    confidence = state.get("confidence")
    evidence = state.get("sql_evidence")
    panel = state.get("panel_verdict")

    return {
        "original_query": state.get("raw_query"),
        "standalone_query": state.get("standalone_query"),
        "conversation_history": thread_for_reviewer(state),
        "prior_turns": len(state.get("conversation_history") or []),
        "retrieved_documents": [
            {
                "chunk_id": chunk.chunk_id,
                "citation": f"{chunk.document_title} §{chunk.clause_number}",
                "document_title": chunk.document_title,
                "clause_number": chunk.clause_number,
                "doc_type": chunk.doc_type,
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
        "budget_stops": state.get("budget_stops", []),
        "draft_answer": draft.answer if draft else "",
        "risk_level": state["risk"].final_level.value if state.get("risk") else None,
        "evidence_path": state.get("evidence_path") or state.get("routed_path"),
        "confidence": confidence.final_score if confidence else None,
        "reflection_count": state.get("reflection_count", 0),
        "panel_repair_count": state.get("panel_repair_count", 0),
        "degraded": state.get("degraded", False),
    }


def _validation_reason(state: AgentState) -> str:
    attempts = state.get("reflection_count", 0)
    allowed = repair_attempts_allowed(state)

    if attempts >= allowed:
        return f"validation failed and all {allowed} permitted repair attempt(s) were used up"

    guard = guard_from_state(state)
    headroom = settings.repair_headroom_seconds

    if guard.seconds_remaining < headroom:
        return (
            f"validation failed and there was no time left to attempt a repair: "
            f"{guard.seconds_remaining:.2f}s of the deadline remained and a repair needs at least "
            f"{headroom:.0f}s. This is a budget stop, not an exhausted retry count "
            f"({attempts} of {allowed} attempts had been used)"
        )

    needed = repair_token_estimate(state)

    if guard.tokens_remaining < needed:
        return (
            f"validation failed and there was no token budget left to attempt a repair: "
            f"{guard.tokens_remaining} tokens remained and a repair needs at least "
            f"{needed}. This is a budget stop, not an exhausted retry count "
            f"({attempts} of {allowed} attempts had been used)"
        )

    return f"validation failed after {attempts} repair attempt(s)"


def _best_retrieval_score(state: AgentState) -> float | None:
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        return None

    leading = chunks[:5]

    scored = [chunk.rerank_score for chunk in leading if chunk.rerank_score is not None]

    if scored:
        return max(scored)

    dense = [chunk.dense_score for chunk in leading if chunk.dense_score]

    return max(dense) if dense else None


def _no_match_reason(state: AgentState, best: float) -> str:
    from src.auth.rbac import allowed_doc_types

    chunks = state.get("retrieved_chunks") or []
    titles = sorted({chunk.document_title for chunk in chunks})
    readable = sorted(allowed_doc_types(state.get("access_scopes") or []))

    reason = (
        f"the corpus was searched and nothing in it answers this question: the best of "
        f"{len(chunks)} retrieved extract(s) scored {round(best, 4)}, which is a no-match rather "
        f"than a weak match. The documents searched were {titles}. If the document that would "
        f"answer this is not in that list, it is not indexed - check GET /ingest/status for "
        f"unindexed_files and re-ingest before reading anything into the confidence score"
    )

    if readable:
        reason += (
            f". This role can only search {readable}, so a policy outside that list is invisible to it"
        )

    return reason


def _email_worthy(state: AgentState) -> bool:
    """True when the escalation's risk level meets ESCALATION_EMAIL_MIN_RISK.

    Every escalation is queued for the reviewer console; the e-mail is reserved
    for the ones that clear the configured floor (High by default). An unknown
    risk (state carries no assessment) never earns a mail.
    """
    risk = state.get("risk")

    if risk is None:
        return False

    try:
        floor = RiskLevel(settings.escalation_email_min_risk)
    except ValueError:
        logger.warning(
            "ESCALATION_EMAIL_MIN_RISK=%r is not Low/Medium/High; defaulting to High",
            settings.escalation_email_min_risk,
        )
        floor = RiskLevel.HIGH

    return risk.final_level.rank >= floor.rank


def _derive_reason(state: AgentState) -> str:
    """Why this request is with a reviewer: the evidence reason first, the user's own request added to it."""
    reason = _certification_reason(state)

    if requested_human(state.get("raw_query", "")):
        return f"{reason}; the user also asked for a human reviewer" if reason else (
            "the user explicitly asked for a human reviewer"
        )

    return reason or "the system could not certify this answer"


def _certification_reason(state: AgentState) -> str | None:
    """What kept the answer from being certified, read from the state (None when nothing did)."""
    if state.get("escalation_reason"):
        return state["escalation_reason"]

    stops = state.get("budget_stops") or []

    if stops:
        last = stops[-1]

        return (
            f"the {last.get('node')} step was stopped by the budget guard "
            f"({last.get('reason')}), so the answer could not be certified"
        )

    if not state.get("retrieved_chunks") and state.get("sql_evidence") is None:
        return "no grounding evidence was retrieved, so no answer could be supported"

    risk = state.get("risk")

    if risk and risk.final_level is RiskLevel.HIGH:
        if risk.disagreement:
            return (
                "risk level is High, which always requires human review before release "
                "(the risk classifier and the database probe disagreed; the higher level was taken)"
            )

        return "risk level is High, which always requires human review before release"

    best = _best_retrieval_score(state)

    if best is not None and best <= RETRIEVAL_NO_MATCH_CEILING and state.get("sql_evidence") is None:
        return _no_match_reason(state, best)

    validation = state.get("validation")

    if validation and validation.conflict_detected:
        conflicts = [d.description for d in validation.defects if d.defect_type in CONFLICT_DEFECT_TYPES]
        reason = "an unresolved conflict was detected between the cited sources; a human must decide which governs"

        return f"{reason}: {conflicts[0]}" if conflicts else reason

    if validation and not validation.passed:
        return _validation_reason(state)

    panel = state.get("panel_verdict")

    if panel and panel.unresolved_conflict:
        return "the high-risk panel could not reach consensus"

    confidence = state.get("confidence")

    if confidence and confidence.final_score < settings.confidence_threshold:
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

    return None


@traced_node("escalation_manager")
def escalation_node(state: AgentState) -> dict:
    reason = _derive_reason(state)
    already_reviewed = bool(state.get("reviewer_decision"))
    reference_id = state.get("escalation_reference") or generate_reference_id()

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
        "queued_at": datetime.now(UTC),
    }

    try:
        with writable_connection() as conn:
            conn.execute(ENQUEUE, payload)
    except Exception as exc:
        logger.error("escalation enqueue failed request_id=%s error=%s", state["request_id"], exc)

    confidence = state.get("confidence")
    emailed = False
    email_worthy = _email_worthy(state)

    if not already_reviewed and not email_worthy:
        logger.info(
            "escalation ref=%s risk=%s is below ESCALATION_EMAIL_MIN_RISK=%s; queued without e-mail request_id=%s",
            reference_id,
            payload["risk_level"],
            settings.escalation_email_min_risk,
            state["request_id"],
        )

    if not already_reviewed and email_worthy:
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
        detail={
            "reason": reason,
            "reference_id": reference_id,
            "email_sent": emailed,
            "email_skipped_below_min_risk": not email_worthy and not already_reviewed,
            "after_human_review": already_reviewed,
        },
    )

    return result
