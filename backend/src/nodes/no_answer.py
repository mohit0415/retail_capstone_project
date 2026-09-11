"""The honest "I don't know" answer.

When the evidence a role can reach does not answer the question, drafting an answer can only
produce an uncited guess. The validator then rejects it, the repair loop retries the same search,
and the request used to end in the human review queue two minutes later. This node answers straight
away instead: it says the answer is not in the evidence, says which evidence that was, and makes no
policy claim at all, so there is nothing to certify. The reason is in ``state["not_found_reason"]``:

* ``no_match``        - the reranker's best policy extract is a no-match, not a weak match
* ``not_in_extracts`` - the RAG writer read the extracts and none of them answers the question
* ``no_records``      - the records query produced no evidence
* ``no_access``       - the only source that answers the question is outside the role's grant

High-risk questions never reach this node (they run the panel and always go to a human), and
neither does a user who asked for a human reviewer.
"""

import logging

from src.auth.rbac import allowed_doc_types
from src.core.audit import audit_from_state
from src.graph.routing import (
    NOT_FOUND_NO_ACCESS,
    NOT_FOUND_NO_EVIDENCE,
    NOT_FOUND_NO_MATCH,
    NOT_FOUND_NO_RECORDS,
    NOT_FOUND_NOT_IN_EXTRACTS,
    best_retrieval_score,
)
from src.graph.state import AgentState
from src.observability.tracing import traced_node
from src.schemas.models import ConfidenceBreakdown, DraftAnswer

logger = logging.getLogger(__name__)

DOC_LABELS = {
    "privacy_policy": "Privacy",
    "infosec_policy": "InfoSec",
    "anti_bribery_policy": "Anti-bribery",
    "vendor_policy": "Vendor",
    "retention_policy": "Retention",
    "gdpr": "GDPR",
    "iso_27001": "ISO 27001",
}


def _readable_documents(state: AgentState) -> str:
    readable = [DOC_LABELS.get(doc, doc) for doc in sorted(allowed_doc_types(state.get("access_scopes") or []))]

    return ", ".join(readable) if readable else "none"


def not_found_text(state: AgentState) -> str:
    reason = state.get("not_found_reason") or NOT_FOUND_NO_MATCH

    if reason == NOT_FOUND_NO_RECORDS:
        failure = (state.get("sql_failure") or "the query returned nothing usable").rstrip(".")

        return (
            "I don't know. I could not answer this from the compliance records your role can read "
            f"({failure}), so I will not guess at what the records hold.\n\n"
            "If a record should answer this, try naming the vendor, department or record type. If it needs "
            "records your role cannot read, ask a compliance officer."
        )

    if reason == NOT_FOUND_NO_EVIDENCE:
        failure = (state.get("sql_failure") or "no rows came back").rstrip(".")

        return (
            "I don't know. Neither the policy documents your role can search "
            f"({_readable_documents(state)}) nor the compliance records ({failure}) answer this, so I will "
            "not guess.\n\n"
            "Try naming the policy, vendor or record you mean, or ask a compliance officer."
        )

    if reason == NOT_FOUND_NO_ACCESS:
        decision_reason = (state.get("path_decision") or {}).get("reason") or ""
        source = "the compliance records" if "records" in decision_reason else "the policy documents"

        return (
            f"I can't answer this with your access. It can only be answered from {source}, and your role "
            f"({state.get('role') or 'unknown'}) is not granted them, so I will not guess.\n\n"
            "Ask a colleague whose role can read that source, for example a compliance officer."
        )

    return (
        "I don't know. I could not find an answer to this in the policy documents your role can search "
        f"({_readable_documents(state)}), so I will not guess.\n\n"
        "If the answer should be in one of those policies, try naming the policy or rephrasing the question. "
        "If it belongs to a different policy, ask someone whose role can read that policy."
    )


def _uncertainty_note(state: AgentState, reason: str, best: float) -> str:
    if reason == NOT_FOUND_NO_RECORDS:
        return f"The records path produced no evidence: {state.get('sql_failure') or 'no rows'}."

    if reason == NOT_FOUND_NO_EVIDENCE:
        return (
            f"No clause matched (best match score {round(best, 4)}) and the records path produced no "
            f"evidence: {state.get('sql_failure') or 'no rows'}."
        )

    if reason == NOT_FOUND_NO_ACCESS:
        return (state.get("path_decision") or {}).get("reason") or "The role cannot read the source this needs."

    if reason == NOT_FOUND_NOT_IN_EXTRACTS:
        return f"The retrieved extracts do not answer this question (best match score {round(best, 4)})."

    return f"Nothing in the searchable documents matched this question (best match score {round(best, 4)})."


@traced_node("no_answer")
def no_answer_node(state: AgentState) -> dict:
    reason = state.get("not_found_reason") or NOT_FOUND_NO_MATCH
    best = best_retrieval_score(state) or 0.0
    titles = sorted({chunk.document_title for chunk in state.get("retrieved_chunks") or []})

    draft = DraftAnswer(
        answer=not_found_text(state),
        cited_clauses=[],
        uncertainty_note=_uncertainty_note(state, reason, best),
        answer_found=False,
    )

    # the reply makes no claim, so there is nothing to be confident in: the score is 0 (a high
    # retrieval score on a writer's "not in the extracts" used to be shown as 0.9 confidence, and got
    # the reply cached); the retrieval score stays as the diagnostic
    confidence = ConfidenceBreakdown(
        retrieval_score=round(best, 4),
        validation_score=0.0,
        source_agreement=0.0,
        coverage=0.0,
        final_score=0.0,
        degraded=bool(state.get("degraded", False)),
    )

    logger.info(
        "answer NOT FOUND request_id=%s reason=%s best_score=%.4f searched=%s -> honest reply instead of a guess",
        state.get("request_id"),
        reason,
        best,
        titles,
    )

    result = {
        "draft": draft,
        "confidence": confidence,
        "answer_not_found": True,
        "not_found_reason": reason,
    }

    if not state.get("evidence_path"):
        result["evidence_path"] = "none"

    audit_from_state(
        {**state, **result},
        node="no_answer",
        event="not_found",
        detail={"reason": reason, "best_score": round(best, 4), "documents_searched": titles},
    )

    return result
