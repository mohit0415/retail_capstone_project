import asyncio
import logging
import os
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from configs.settings import settings
from src.api import escalation_service
from src.auth import azure_credentials
from src.auth.rbac import allowed_doc_types, permissions_for, screens_for
from src.auth.security import (
    Principal,
    current_principal,
    require_corpus_admin,
    require_reviewer,
)
from src.cache.registry import cache_report, invalidate_corpus_caches
from src.cache.response_cache import get_response_cache, scope_key, should_cache_answer
from src.core import audit, conversation
from src.core.idempotency import build_key, idempotency_store
from src.graph.builder import get_compiled_graph
from src.graph.state import initial_state
from src.guardrails.output_guard import OutputGuardOutcome, canonical_citation, run_output_guardrail
from src.index.models import active_embed_model_name
from src.index.vector_index import (
    check_embed_model_compatibility,
    clear_vector_table,
    indexed_document_summary,
)
from src.ingestion.bootstrap import corpus_dir, corpus_is_indexed, unindexed_files
from src.ingestion.pipeline import ingest_directory, ingest_file
from src.llm_routing.ledger import cost_ledger
from src.llm_routing.router import routing_report
from src.observability import slo
from src.observability.agent_steps import build_agent_steps
from src.observability.langfuse_callback import flush_langfuse_traces, setup_langfuse_callback
from src.observability.logging_config import bind_request_context, reset_request_context
from src.observability.tracing import get_callback_handlers
from src.retrieval.citations import citable_citation_texts, citable_clauses
from src.schemas.api import (
    AnswerResponse,
    AskReleasedResponse,
    AskRequest,
    AzureCredentialsRequest,
    AzureCredentialsResponse,
    CacheInfo,
    CacheInvalidationResponse,
    Citation,
    ClarificationResponse,
    CorpusStatusResponse,
    DecisionTrace,
    IngestResponse,
    MeResponse,
    OptimizationReport,
    PendingReviewResponse,
    QueueItem,
    RebuildResponse,
    RefusalResponse,
    ReviewOutcome,
    ReviewPackage,
    ReviewSubmission,
    SloReport,
    SqlProof,
    StageTimings,
)
from src.schemas.enums import ReviewDecision, RiskLevel, TerminalOutcome
from src.schemas.models import DraftAnswer, RetrievedChunk

router = APIRouter()

logger = logging.getLogger(__name__)

SUPPORTED_UPLOAD_SUFFIXES = {".md", ".pdf", ".docx", ".txt"}


def checkpoint_thread_id(thread_id: str, request_id: str) -> str:
    """The LangGraph checkpoint key for one /ask run.

    The checkpoint used to be keyed by the conversation thread, so every follow-up
    in a chat inherited the previous run's reducer channels: ``tokens_spent`` kept
    adding up (the token budget ran out on the second question), ``budget_stops``
    made every later turn escalate straight away, and ``retrieved_chunks`` carried
    the previous question's extracts into the next answer. Multi-turn context
    comes from ``conversation.load_thread`` and never from the checkpoint, so each
    request now gets its own checkpoint. The reviewer resumes exactly the run it
    is reviewing, even when the user asked something else in the same chat later.
    """
    return f"{thread_id}:{request_id}"


def _validated_document_scope(requested: list[str], principal: Principal) -> list[str]:
    if not requested:
        return []

    grant = allowed_doc_types(principal.access_scopes)
    unknown = [value for value in requested if value not in grant]

    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"document_scope contains {unknown}, which the role "
                f"'{principal.role.value}' cannot read or which is not a document type. "
                f"It accepts any of {sorted(grant)}. Leave document_scope out entirely to "
                f"search everything this role is allowed to see - it only ever narrows."
            ),
        )

    return requested


def _citations_from(final_state: dict) -> list[Citation]:
    if final_state.get("answer_not_found"):
        # the no-answer reply cites nothing; listing the unrelated extracts would look like evidence
        return []

    seen: dict[str, Citation] = {}

    for chunk in final_state.get("retrieved_chunks", []):
        key = canonical_citation(chunk.citation)

        if key in seen:
            continue

        seen[key] = Citation(
            document_title=chunk.document_title,
            clause_number=chunk.clause_number,
            section=chunk.section,
            version=chunk.version,
            excerpt=chunk.content[:400],
        )

    cited = final_state["draft"].cited_clauses if final_state.get("draft") else []

    if cited:
        filtered: list[Citation] = []
        used: set[str] = set()
        citable = citable_clauses(final_state.get("retrieved_chunks", []))

        for reference in cited:
            key = canonical_citation(reference)

            if key in used:
                continue

            if key in seen:
                used.add(key)
                filtered.append(seen[key])
            elif key in citable:
                # a sub-clause cited from inside an extract: show that clause and its own text
                entry = citable[key]
                used.add(key)
                filtered.append(
                    Citation(
                        document_title=entry.chunk.document_title,
                        clause_number=entry.clause_number,
                        section=entry.heading or entry.chunk.section,
                        version=entry.chunk.version,
                        excerpt=entry.chunk.content[entry.offset : entry.offset + 400],
                    )
                )

        if filtered:
            return filtered

    return list(seen.values())


def _sql_proof_from(final_state: dict) -> SqlProof | None:
    evidence = final_state.get("sql_evidence")

    if evidence is None:
        return None

    return SqlProof(
        template_id=evidence.template_id,
        statement=evidence.statement,
        parameters=evidence.parameters,
        row_count=evidence.row_count,
        as_of=str(evidence.as_of),
    )


def _dump(value):
    """model_dump for pydantic objects, pass-through for plain dicts / None."""
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    return value


def _trace_from(final_state: dict) -> DecisionTrace:
    """Project the graph state onto the caller-visible decision trace.

    Every field here is already computed by the graph; this only stops
    ``/ask`` from discarding it. The draft answer is deliberately left out so
    the same object is safe on a 202.
    """
    draft = final_state.get("draft")
    started = final_state.get("started_ts")
    deadline = final_state.get("deadline_ts")

    return DecisionTrace(
        guardrail="refused" if final_state.get("refusal_reason") else "pass",
        standalone_query=final_state.get("standalone_query") or "",
        intent=_dump(final_state.get("intent")),
        resolved_entities=[_dump(e) for e in final_state.get("resolved_entities") or []],
        risk=_dump(final_state.get("risk")),
        plan=_dump(final_state.get("plan")),
        routed_path=final_state.get("routed_path"),
        path_decision=final_state.get("path_decision"),
        evidence_path=final_state.get("evidence_path"),
        retrieved_chunks=list(
            dict.fromkeys(chunk.citation for chunk in final_state.get("retrieved_chunks") or [])
        ),
        sql_evidence=_sql_proof_from(final_state),
        panel_verdict=_dump(final_state.get("panel_verdict")),
        cited_clauses=list(draft.cited_clauses) if draft else [],
        validation=_dump(final_state.get("validation")),
        confidence=_dump(final_state.get("confidence")),
        tokens_spent=int(final_state.get("tokens_spent") or 0),
        token_budget=int(final_state.get("token_budget") or 0),
        deadline_seconds=round(deadline - started, 2) if started and deadline else None,
        degraded=bool(final_state.get("degraded", False)),
        skipped_optional_nodes=list(final_state.get("skipped_optional_nodes") or []),
        budget_stops=list(final_state.get("budget_stops") or []),
        reflection_count=int(final_state.get("reflection_count") or 0),
        panel_repair_count=int(final_state.get("panel_repair_count") or 0),
        terminal_outcome=final_state.get("terminal_outcome"),
        escalation_reason=final_state.get("escalation_reason"),
        escalation_reference=final_state.get("escalation_reference"),
        model_routing=list(final_state.get("model_routing") or []),
        llm_usage=cost_ledger.summary(final_state.get("request_id")),
        agent_steps=build_agent_steps(final_state),
    )


def _timings_from(final_state: dict, total_ms: float) -> StageTimings:
    reached = slo.stage_marks(final_state.get("marks", []))

    return StageTimings(
        t1_ms=reached.get("t1"),
        t2_ms=reached.get("t2"),
        t3_ms=reached.get("t3"),
        t4_ms=reached.get("t4"),
        total_ms=round(total_ms, 2),
    )


# ---------------------------------------------------------------------------
# Auth (Auth0 RBAC). Tokens are minted by Auth0, not here - see src/auth/security.py
# ---------------------------------------------------------------------------


@router.get("/auth/me", response_model=MeResponse, tags=["auth"])
def who_am_i(principal: Principal = Depends(current_principal)) -> MeResponse:
    """The role the Auth0 token resolved to, its scopes and the screens it may open."""
    return MeResponse(
        user_id=principal.user_id,
        role=principal.role,
        departments=principal.departments,
        access_scopes=principal.access_scopes,
        auth0_roles=principal.auth0_roles,
        email=principal.email,
        name=principal.name,
        permissions=permissions_for(principal.role),
        screens=screens_for(principal.role),
        azure_configured=settings.azure_configured,
    )


@router.post("/auth/azure", response_model=AzureCredentialsResponse, tags=["auth"])
def set_azure_credentials(
    payload: AzureCredentialsRequest,
    principal: Principal = Depends(current_principal),
) -> AzureCredentialsResponse:
    """Apply the Azure OpenAI credentials the user typed on the login page.

    Any authenticated role may call it, because every user has to enter the
    credentials on the login page. With ``verify`` (the default) one tiny
    embedding call and one tiny chat call are made first, so a wrong key or
    deployment name is reported here instead of failing inside the graph.
    """
    endpoint = payload.endpoint.strip()

    if not endpoint.startswith("https://"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="endpoint must look like https://<resource>.openai.azure.com/",
        )

    verified = False

    if payload.verify:
        ok, message = azure_credentials.verify_azure_credentials(
            endpoint,
            payload.api_key.strip(),
            payload.api_version.strip(),
            payload.small_deployment.strip(),
            payload.embedding_deployment.strip(),
        )

        if not ok:
            logger.warning("azure credentials rejected user=%s: %s", principal.user_id, message)

            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        verified = True

    azure_credentials.apply_azure_credentials(
        endpoint,
        payload.api_key.strip(),
        payload.api_version.strip(),
        payload.small_deployment.strip(),
        payload.strong_deployment.strip(),
        payload.embedding_deployment.strip(),
    )

    logger.info("azure credentials set by user=%s role=%s verified=%s", principal.user_id, principal.role.value, verified)

    current = azure_credentials.azure_status()

    return AzureCredentialsResponse(
        azure_configured=current["azure_configured"],
        verified=verified,
        endpoint=current["endpoint"],
        api_version=current["api_version"],
        small_deployment=current["small_deployment"],
        strong_deployment=current["strong_deployment"],
        embedding_deployment=current["embedding_deployment"],
        message="credentials applied" + (" and verified" if verified else " without verification"),
    )


ASK_RESPONSES = {
    200: {
        "model": AskReleasedResponse,
        "description": (
            "the request reached a terminal decision the caller can act on. "
            "'answered' carries the answer with its citations, 'refused' means the request was "
            "not accepted, 'clarification_required' means an entity in the question was ambiguous. "
            "Read the 'status' field to tell them apart."
        ),
    },
    202: {
        "model": PendingReviewResponse,
        "description": (
            "the system could not certify an answer and queued it for a human reviewer. "
            "No draft answer is returned, because the reason it escalated is that this answer "
            "could not be certified. Poll the returned poll_url for the outcome."
        ),
    },
}


def _cache_scope(principal: Principal, document_scope: list[str]) -> str:
    return scope_key(
        role=principal.role.value,
        access_scopes=principal.access_scopes,
        departments=principal.departments,
        document_scope=document_scope,
        as_of=str(settings.as_of_date),
    )


def _serve_cached_answer(
    hit,
    *,
    request_id: str,
    thread_id: str,
    payload: AskRequest,
    principal: Principal,
    background: BackgroundTasks,
    started_ts: float,
) -> dict:
    """Re-issue a cached, certified answer under a fresh request id.

    The answer, citations, confidence and decision trace are the ones that
    were certified when the source request ran; only the identifiers, the
    timings and the ``cache`` block are new. The conversation thread still
    records both turns so a follow-up has history to rewrite against, and
    the audit log carries a ``cache_hit`` row naming the source request.
    """
    entry = hit.entry
    total_ms = (time.monotonic() - started_ts) * 1000

    body = dict(entry.body)
    body["request_id"] = request_id
    body["thread_id"] = thread_id
    body["timings"] = StageTimings(total_ms=round(total_ms, 2)).model_dump(mode="json")
    body["cache"] = CacheInfo(
        kind=hit.kind,
        similarity=hit.similarity,
        age_seconds=hit.age_seconds,
        matched_query=hit.matched_query,
        source_request_id=entry.source_request_id,
        saved_ms_estimate=round(max(0.0, entry.produced_in_ms - total_ms), 2),
        saved_tokens_estimate=entry.tokens_spent,
    ).model_dump(mode="json")

    logger.info(
        "ask served from cache request_id=%s kind=%s similarity=%.4f source_request_id=%s "
        "elapsed_ms=%.1f saved_ms=%.0f saved_tokens=%d",
        request_id,
        hit.kind,
        hit.similarity,
        entry.source_request_id,
        total_ms,
        entry.produced_in_ms,
        entry.tokens_spent,
    )

    audit.audit_from_state(
        {
            "request_id": request_id,
            "thread_id": thread_id,
            "user_id": principal.user_id,
            "role": principal.role.value,
            "terminal_outcome": TerminalOutcome.ANSWERED.value,
        },
        node="response_cache",
        event="cache_hit",
        detail={
            "kind": hit.kind,
            "similarity": hit.similarity,
            "source_request_id": entry.source_request_id,
            "age_seconds": hit.age_seconds,
            "saved_ms_estimate": body["cache"]["saved_ms_estimate"],
            "saved_tokens_estimate": entry.tokens_spent,
        },
    )

    slo.record_latency(
        {
            "request_id": request_id,
            "thread_id": thread_id,
            "role": principal.role.value,
            "evidence_path": "cache",
            "marks": [],
        },
        total_ms,
        TerminalOutcome.ANSWERED.value,
    )

    background.add_task(conversation.record_user_turn, thread_id, principal.user_id, request_id, payload.query)
    background.add_task(conversation.record_assistant_turn, thread_id, principal.user_id, request_id, body["answer"])

    return body


def _store_answer_in_cache(
    scope: str,
    payload: AskRequest,
    body: dict,
    *,
    request_id: str,
    total_ms: float,
    final_state: dict,
    embedding: list[float] | None,
) -> None:
    ok, reason = should_cache_answer(body)

    if not ok:
        get_response_cache().reject(reason)
        logger.info("answer not cached request_id=%s reason=%s", request_id, reason)

        return

    get_response_cache().put(
        scope,
        payload.query,
        body,
        status.HTTP_200_OK,
        source_request_id=request_id,
        produced_in_ms=total_ms,
        tokens_spent=int(final_state.get("tokens_spent") or 0),
        embedding=embedding,
        evidence_path=final_state.get("evidence_path"),
        confidence=body.get("confidence"),
    )


def idempotency_key_sent(request: Request) -> bool:
    return bool((request.headers.get("Idempotency-Key") or "").strip())


@router.post("/ask", tags=["ask"], responses=ASK_RESPONSES)
def ask(
    payload: AskRequest,
    request: Request,
    response: Response,
    background: BackgroundTasks,
    principal: Principal = Depends(current_principal),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    cache_key = build_key(principal.user_id, idempotency_key, payload.model_dump())
    cached = idempotency_store.get(cache_key)

    if cached is not None:
        response.status_code = cached.status_code

        return cached.payload

    started_ts = time.monotonic()
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    thread_id = payload.thread_id or str(uuid.uuid4())

    context_tokens = bind_request_context(request_id=request_id, thread_id=thread_id, user_id=principal.user_id)

    try:
        return _ask(payload, request, response, background, principal, cache_key, request_id, thread_id, started_ts)
    finally:
        reset_request_context(context_tokens)


def _ask(
    payload: AskRequest,
    request: Request,
    response: Response,
    background: BackgroundTasks,
    principal: Principal,
    cache_key: str,
    request_id: str,
    thread_id: str,
    started_ts: float,
):
    logger.info(
        "ask START request_id=%s user=%s role=%s thread=%s new_thread=%s document_scope=%s use_cache=%s query_chars=%d",
        request_id,
        principal.user_id,
        principal.role.value,
        thread_id,
        payload.thread_id is None,
        payload.document_scope or "all",
        payload.use_cache,
        len(payload.query),
    )

    document_scope = _validated_document_scope(payload.document_scope, principal)

    history, summary = conversation.load_thread(thread_id) if payload.thread_id else ([], "")

    scope = _cache_scope(principal, document_scope)
    query_embedding: list[float] | None = None
    cache_eligible = settings.enable_response_cache and payload.use_cache

    if settings.enable_response_cache and not payload.use_cache:
        get_response_cache().skip("caller_opted_out")
    elif settings.enable_response_cache and history:
        cache_eligible = False
        get_response_cache().skip("conversation_history")

    if cache_eligible:
        lookup = get_response_cache().lookup(scope, payload.query)
        query_embedding = lookup.embedding

        if lookup.hit is not None:
            body = _serve_cached_answer(
                lookup.hit,
                request_id=request_id,
                thread_id=thread_id,
                payload=payload,
                principal=principal,
                background=background,
                started_ts=started_ts,
            )
            response.status_code = status.HTTP_200_OK
            idempotency_store.put(cache_key, body, status.HTTP_200_OK)

            return body

    state = initial_state(
        request_id=request_id,
        thread_id=thread_id,
        user_id=principal.user_id,
        role=principal.role.value,
        access_scopes=principal.access_scopes,
        raw_query=payload.query,
        started_ts=started_ts,
        deadline_ts=started_ts + settings.deadline_seconds_standard,
        token_budget=settings.default_token_budget,
        departments=principal.departments,
        document_scope_request=document_scope,
        conversation_history=history,
        thread_summary=summary,
    )

    graph = get_compiled_graph()
    config = {
        "configurable": {"thread_id": checkpoint_thread_id(thread_id, request_id)},
        "recursion_limit": 40,
        "callbacks": get_callback_handlers(),
        "metadata": {
            "request_id": request_id,
            "thread_id": thread_id,
            "user_id": principal.user_id,
            "role": principal.role.value,
        },
        "run_name": "policy_graph",
    }

    config, langfuse_handler = setup_langfuse_callback(
        config,
        session_id=thread_id,
        user_id=principal.user_id,
        trace_name="policy_graph",
        tags=["ask", principal.role.value],
    )

    try:
        final_state = graph.invoke(state, config=config)
    except Exception:
        logger.error(
            "ask FAILED request_id=%s elapsed_ms=%.0f: graph raised",
            request_id,
            (time.monotonic() - started_ts) * 1000,
            exc_info=True,
        )

        raise
    finally:
        if langfuse_handler is not None:
            background.add_task(flush_langfuse_traces, langfuse_handler)

    total_ms = (time.monotonic() - started_ts) * 1000
    outcome = final_state.get("terminal_outcome") or TerminalOutcome.ESCALATED.value
    timings = _timings_from(final_state, total_ms)
    usage = cost_ledger.summary(request_id)

    slo.record_latency(final_state, total_ms, outcome)

    logger.info(
        "ask END request_id=%s outcome=%s path=%s risk=%s confidence=%s degraded=%s total_ms=%.0f "
        "tokens_est=%s llm_calls=%d llm_tokens=%d usd=%.6f cache_hits=%d routing=%s",
        request_id,
        outcome,
        final_state.get("evidence_path") or final_state.get("routed_path"),
        final_state["risk"].final_level.value if final_state.get("risk") else None,
        final_state["confidence"].final_score if final_state.get("confidence") else None,
        final_state.get("degraded", False),
        total_ms,
        final_state.get("tokens_spent"),
        usage.get("calls", 0),
        usage.get("total_tokens", 0),
        usage.get("usd", 0.0),
        usage.get("cache_hits", 0),
        [f"{d.get('node')}={d.get('tier')}" for d in final_state.get("model_routing") or []],
    )

    if outcome == TerminalOutcome.REFUSED.value:
        body = RefusalResponse(
            request_id=request_id,
            reason=final_state.get("refusal_reason") or "the request could not be accepted",
        ).model_dump(mode="json")

        response.status_code = status.HTTP_200_OK
        idempotency_store.put(cache_key, body, status.HTTP_200_OK)

        return body

    background.add_task(
        conversation.record_user_turn,
        thread_id,
        principal.user_id,
        request_id,
        payload.query,
    )

    if outcome == TerminalOutcome.CLARIFICATION_REQUIRED.value:
        candidates = []

        for entity in final_state.get("resolved_entities", []):
            candidates.extend(entity.candidates)

        body = ClarificationResponse(
            request_id=request_id,
            thread_id=thread_id,
            question=final_state.get("clarification_question") or "Please clarify the entity you mean.",
            candidates=candidates,
        ).model_dump(mode="json")

        idempotency_store.put(cache_key, body, status.HTTP_200_OK)

        return body

    if outcome == TerminalOutcome.ESCALATED.value:
        risk = final_state.get("risk")

        body = PendingReviewResponse(
            request_id=request_id,
            thread_id=thread_id,
            risk_level=risk.final_level.value if risk else RiskLevel.HIGH.value,
            reason=final_state.get("escalation_reason") or "human review required",
            escalation_reference=final_state.get("escalation_reference"),
            queued_at=datetime.now(UTC),
            poll_url=str(request.url_for("get_request_status", request_id=request_id)),
            timings=timings,
            trace=_trace_from(final_state),
        ).model_dump(mode="json")

        response.status_code = status.HTTP_202_ACCEPTED

        if idempotency_key_sent(request):
            # a retry with the same Idempotency-Key gets the same 202. Without a key the body hash
            # used to replay this 202 for 15 minutes, so asking the same question again - even after
            # a reviewer had released the answer - came straight back as pending_review.
            idempotency_store.put(cache_key, body, status.HTTP_202_ACCEPTED)

        return body

    draft = final_state.get("draft")
    confidence = final_state.get("confidence")
    risk = final_state.get("risk")

    if draft is None:
        raise HTTPException(status_code=500, detail="the graph terminated without an answer")

    body = AnswerResponse(
        request_id=request_id,
        thread_id=thread_id,
        answer=draft.answer,
        citations=_citations_from(final_state),
        sql_evidence=_sql_proof_from(final_state),
        confidence=confidence.final_score if confidence else 0.0,
        risk_level=risk.final_level.value if risk else RiskLevel.LOW.value,
        uncertainty_note=draft.uncertainty_note,
        evidence_path=final_state.get("evidence_path") or "rag",
        degraded=final_state.get("degraded", False),
        timings=timings,
        history_turns_used=len(final_state.get("conversation_history") or []),
        thread_summary_used=bool(final_state.get("thread_summary")),
        standalone_query=final_state.get("standalone_query") or "",
        not_found=bool(final_state.get("answer_not_found", False)),
        not_found_reason=(final_state.get("not_found_reason") or "") if final_state.get("answer_not_found") else "",
        trace=_trace_from(final_state),
    ).model_dump(mode="json")

    background.add_task(
        conversation.record_assistant_turn,
        thread_id,
        principal.user_id,
        request_id,
        draft.answer,
    )

    idempotency_store.put(cache_key, body, status.HTTP_200_OK)

    if cache_eligible:
        _store_answer_in_cache(
            scope,
            payload,
            body,
            request_id=request_id,
            total_ms=total_ms,
            final_state=final_state,
            embedding=query_embedding,
        )

    return body


@router.get("/requests/{request_id}", name="get_request_status", tags=["ask"])
def get_request_status(request_id: str, principal: Principal = Depends(current_principal)):
    review, row = escalation_service.fetch_package(request_id)

    if row is None:
        raise HTTPException(status_code=404, detail="unknown request id")

    if row["status"] == "pending":
        return {
            "status": "pending_review",
            "request_id": request_id,
            "thread_id": row["thread_id"],
            "risk_level": row["risk_level"],
            "reason": row["reason"],
            "escalation_reference": review.reference_id if review else None,
            "queued_at": row.get("queued_at"),
        }

    if row["reviewer_decision"] == ReviewDecision.REJECT.value:
        # the rejected draft used to come back here as "answered", so the chat showed it as released
        return {
            "status": "rejected",
            "request_id": request_id,
            "thread_id": row["thread_id"],
            "escalation_reference": review.reference_id if review else None,
            "reviewer_decision": row["reviewer_decision"],
            "risk_level": row["risk_level"],
            "reviewed_at": row["reviewed_at"],
            "note": "a reviewer rejected this answer, so nothing was released",
        }

    return {
        "status": "answered",
        "request_id": request_id,
        "thread_id": row["thread_id"],
        "escalation_reference": review.reference_id if review else None,
        "answer": row["reviewed_answer"] or (review.draft_answer if review else ""),
        "reviewer_decision": row["reviewer_decision"],
        "risk_level": row["risk_level"],
        "reviewed_at": row["reviewed_at"],
    }


@router.get("/review/queue", response_model=list[QueueItem], tags=["review"])
def review_queue(principal: Principal = Depends(require_reviewer)) -> list[QueueItem]:
    return escalation_service.list_pending()


@router.get("/review/{request_id}", response_model=ReviewPackage, tags=["review"])
def review_package(request_id: str, principal: Principal = Depends(require_reviewer)) -> ReviewPackage:
    review, _ = escalation_service.fetch_package(request_id)

    if review is None:
        raise HTTPException(status_code=404, detail="unknown request id")

    return review


def _review_snapshot(graph, config: dict, thread_id: str, request_id: str | None):
    """Find the checkpoint of the run under review.

    New runs are keyed per request (``checkpoint_thread_id``); escalations queued
    before that change were keyed by the plain conversation thread, so that key
    is tried second.
    """
    keys = [checkpoint_thread_id(thread_id, request_id)] if request_id else []
    keys.append(thread_id)

    for key in keys:
        candidate = {**config, "configurable": {"thread_id": key}}

        try:
            snapshot = graph.get_state(candidate)
        except Exception:
            logger.warning("review resume: could not read graph state for checkpoint=%s", key, exc_info=True)
            continue

        if snapshot is None or not snapshot.values:
            continue

        owner = snapshot.values.get("request_id")

        if request_id and owner and owner != request_id:
            # an escalation queued before checkpoints were keyed per request: the chat key now holds a
            # later question's run, and resuming it would release this answer through that run's state
            logger.warning(
                "review resume: checkpoint %s belongs to request %s, not %s; it is not resumed",
                key,
                owner,
                request_id,
            )
            continue

        return candidate, snapshot

    return config, None


def _package_chunks(review: ReviewPackage) -> list[RetrievedChunk]:
    """The extracts the escalated answer was written from, rebuilt from the review package."""
    chunks: list[RetrievedChunk] = []

    for index, document in enumerate(review.retrieved_documents or []):
        title = str(document.get("document_title") or "")
        clause = str(document.get("clause_number") or "")

        if not title:
            # packages written before the title and clause were stored separately
            title, _, clause = str(document.get("citation") or "").rpartition(" §")

        if not title:
            continue

        chunks.append(
            RetrievedChunk(
                chunk_id=str(document.get("chunk_id") or f"package-{index}"),
                doc_type=str(document.get("doc_type") or "unknown"),
                document_title=title,
                section=str(document.get("section") or ""),
                clause_number=clause,
                version=str(document.get("version") or "1.0"),
                content=str(document.get("content") or ""),
            )
        )

    return chunks


def release_check(review: ReviewPackage, answer: str) -> OutputGuardOutcome:
    """The output guardrail a reviewed answer must pass, run before the decision is recorded.

    Same PII scrub and the same "only cite what was retrieved" rule as the graph's output
    guardrail node; a reviewer's answer is not required to carry clause citations. Running it
    first means a refused answer comes back as a 422 the reviewer can fix, instead of a request
    that is marked reviewed with nothing released.
    """
    allowed = citable_citation_texts(_package_chunks(review))

    return run_output_guardrail(answer, allowed, require_citations=False)


def _apply_review_to_checkpoint(
    thread_id: str,
    decision: ReviewDecision,
    answer: str,
    reviewer_id: str,
    notes: str,
    request_id: str | None = None,
) -> tuple[bool, dict | None]:
    graph = get_compiled_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
        "callbacks": get_callback_handlers(),
        "metadata": {"thread_id": thread_id, "reviewer_id": reviewer_id, "stage": "review_resume"},
        "run_name": "policy_graph_resume",
    }

    config, langfuse_handler = setup_langfuse_callback(
        config,
        session_id=thread_id,
        user_id=reviewer_id,
        trace_name="policy_graph_resume",
        tags=["review_resume"],
    )

    config, snapshot = _review_snapshot(graph, config, thread_id, request_id)

    if snapshot is None:
        return False, None

    draft = snapshot.values.get("draft")

    if draft is None:
        revised = DraftAnswer(answer=answer)
    else:
        revised = draft.model_copy(update={"answer": answer})

    update = {
        "draft": revised,
        "reviewer_decision": decision.value,
        "reviewer_id": reviewer_id,
        "reviewer_notes": notes,
        "escalation_reason": None,
        "terminal_outcome": None,
    }

    try:
        graph.update_state(config, update)
    except Exception:
        logger.warning(
            "review resume: could not update graph state for thread_id=%s", thread_id, exc_info=True
        )
        return False, None

    if decision is ReviewDecision.REJECT:
        return True, None

    try:
        return True, graph.invoke(None, config=config)
    except Exception:
        logger.error(
            "review resume: graph resume invoke failed for thread_id=%s", thread_id, exc_info=True
        )
        return True, None
    finally:
        flush_langfuse_traces(langfuse_handler)


@router.post("/review/{request_id}", response_model=ReviewOutcome, tags=["review"])
def submit_review(
    request_id: str,
    submission: ReviewSubmission,
    background: BackgroundTasks,
    principal: Principal = Depends(require_reviewer),
) -> ReviewOutcome:
    review, row = escalation_service.fetch_package(request_id)

    if row is None:
        raise HTTPException(status_code=404, detail="unknown request id")

    if row["status"] != "pending":
        raise HTTPException(status_code=409, detail="this request has already been reviewed")

    edited = (submission.edited_answer or "").strip()

    if submission.decision is ReviewDecision.EDIT and not edited:
        raise HTTPException(
            status_code=422,
            detail="decision 'edit' requires a non-empty edited_answer; use 'accept' to release the draft as-is",
        )

    if submission.decision is ReviewDecision.ACCEPT:
        if not (review.draft_answer or "").strip():
            raise HTTPException(
                status_code=422,
                detail="this request has no draft to accept; use 'edit' to supply an answer or 'reject'",
            )

        answer = review.draft_answer
        answer_source = "panel_consensus" if review.evidence_path == "high_risk_panel" else "system_draft"
    elif submission.decision is ReviewDecision.EDIT:
        answer = edited
        answer_source = "reviewer_edit"
    else:
        answer = ""
        answer_source = None

    if submission.decision is ReviewDecision.ACCEPT and edited and edited != review.draft_answer:
        logger.info(
            "review accept: ignoring edited_answer for request_id=%s (use decision='edit' to override the draft)",
            request_id,
        )

    gate: OutputGuardOutcome | None = None

    if submission.decision is not ReviewDecision.REJECT:
        gate = release_check(review, answer)

        if gate.enforcement_failed:
            logger.warning(
                "review refused before recording request_id=%s decision=%s reason=%s",
                request_id,
                submission.decision.value,
                gate.failure_reason,
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    f"the reviewed answer cannot be released: {gate.failure_reason}. Nothing was recorded "
                    "and the request is still in the queue - correct the answer with decision 'edit' and submit again"
                ),
            )

    updated = escalation_service.record_decision(
        request_id=request_id,
        reviewer_id=principal.user_id,
        decision=submission.decision,
        edited_answer=answer or None,
        notes=submission.reviewer_notes,
    )

    if not updated:
        raise HTTPException(status_code=409, detail="the request was reviewed by someone else first")

    logger.info(
        "review submitted request_id=%s reviewer=%s decision=%s answer_source=%s",
        request_id,
        principal.user_id,
        submission.decision.value,
        answer_source,
    )

    resumed, final_state = _apply_review_to_checkpoint(
        thread_id=row["thread_id"],
        decision=submission.decision,
        answer=answer or (review.draft_answer or ""),
        reviewer_id=principal.user_id,
        notes=submission.reviewer_notes,
        request_id=request_id,
    )

    if submission.decision is ReviewDecision.REJECT:
        return ReviewOutcome(
            request_id=request_id,
            decision=submission.decision.value,
            resumed=resumed,
            released=False,
            outcome=TerminalOutcome.ESCALATED.value,
            note="the answer was rejected, so nothing was released to the caller",
        )

    if resumed and final_state is not None:
        outcome = final_state.get("terminal_outcome")

        if outcome != TerminalOutcome.ANSWERED.value:
            # the graph's guardrail refused what the check above passed (for example the PII check
            # could not run): put the request back so the reviewer can try again
            reason = final_state.get("escalation_reason") or "the reviewed answer did not pass the output guardrail"
            escalation_service.reopen(request_id)

            logger.warning(
                "review resume did not release request_id=%s outcome=%s reason=%s; request reopened",
                request_id,
                outcome,
                reason,
            )

            return ReviewOutcome(
                request_id=request_id,
                decision=submission.decision.value,
                resumed=True,
                released=False,
                outcome=outcome,
                evidence_path=review.evidence_path,
                note=f"{reason}. The request is back in the review queue so it can be reviewed again",
            )

        released_draft = final_state.get("draft")
        released_answer = released_draft.answer if released_draft else gate.answer
        note = "the reviewed answer passed the output guardrail and was released"
    else:
        # no checkpoint of this request to resume (queued before per-request checkpoints, or the
        # in-memory checkpointer lost it). The answer has already passed the output guardrail check
        # above, so it is released from there instead of leaving the asker with nothing.
        released_answer = gate.answer
        note = (
            "the graph checkpoint for this request could not be resumed, so the reviewed answer was "
            "released after passing the output guardrail check directly"
        )

        audit.write_audit(
            audit.AuditRecord(
                request_id=request_id,
                thread_id=row["thread_id"],
                user_id=row["user_id"],
                role=principal.role.value,
                node="output_guardrail",
                event="answered",
                risk_level=row["risk_level"],
                outcome=TerminalOutcome.ANSWERED.value,
                detail={
                    "after_human_review": True,
                    "released_without_checkpoint": True,
                    "reviewer_id": principal.user_id,
                    "pii_removed": gate.pii_removed,
                },
            )
        )

    escalation_service.store_released_answer(request_id, released_answer)

    logger.info(
        "review released request_id=%s decision=%s resumed=%s answer_chars=%d",
        request_id,
        submission.decision.value,
        resumed and final_state is not None,
        len(released_answer or ""),
    )

    background.add_task(
        conversation.record_assistant_turn,
        row["thread_id"],
        row["user_id"],
        request_id,
        released_answer,
    )

    return ReviewOutcome(
        request_id=request_id,
        decision=submission.decision.value,
        resumed=bool(resumed and final_state is not None),
        released=True,
        outcome=TerminalOutcome.ANSWERED.value,
        answer=released_answer,
        answer_source=answer_source,
        evidence_path=review.evidence_path,
        note=note,
    )


@router.post("/ingest", response_model=IngestResponse, tags=["corpus"])
async def ingest_document(
    file: UploadFile = File(...),
    force: bool = Query(default=False),
    principal: Principal = Depends(require_corpus_admin),
) -> IngestResponse:
    original_name = Path(file.filename or "").name

    if not original_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="the upload has no file name")

    suffix = Path(original_name).suffix.lower()

    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{suffix or 'this file type'} cannot be ingested, "
            f"expected one of {sorted(SUPPORTED_UPLOAD_SUFFIXES)}",
        )

    payload = await file.read()

    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="the uploaded file is empty")

    if len(payload) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"the upload is {len(payload)} bytes, the limit is {settings.upload_max_bytes}",
        )

    compatible, message = check_embed_model_compatibility()

    if not compatible:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message)

    handle, temp_path = tempfile.mkstemp(suffix=suffix, prefix="policy_upload_")

    try:
        with os.fdopen(handle, "wb") as spool:
            spool.write(payload)

        result = await asyncio.to_thread(ingest_file, temp_path, original_name, force)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ingestion failed for {original_name}: {exc}",
        ) from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if result.skipped:
        logger.info("ingest skipped file=%s reason=%s by=%s", result.file_name, result.reason, principal.user_id)

        return IngestResponse(status="skipped", file_name=result.file_name, reason=result.reason)

    logger.info(
        "ingest indexed file=%s doc_type=%s version=%s nodes=%d by=%s",
        result.file_name,
        result.doc_type,
        result.version,
        result.total_nodes,
        principal.user_id,
    )

    invalidate_corpus_caches(f"ingest:{result.file_name}")

    return IngestResponse(
        status="indexed",
        file_name=result.file_name,
        doc_type=result.doc_type,
        version=result.version,
        parsed_with=result.parsed_with,
        total_nodes=result.total_nodes,
        text_nodes=result.text_nodes,
        table_nodes=result.table_nodes,
        image_nodes=result.image_nodes,
        superseded_nodes=result.superseded,
    )


@router.get("/ingest/status", response_model=CorpusStatusResponse, tags=["corpus"])
def corpus_status(principal: Principal = Depends(require_corpus_admin)) -> CorpusStatusResponse:
    compatible, message = check_embed_model_compatibility()

    return CorpusStatusResponse(
        indexed=corpus_is_indexed(),
        corpus_dir=str(corpus_dir()),
        embed_model=active_embed_model_name(),
        embed_model_compatible=compatible,
        message=message,
        documents=indexed_document_summary(),
        unindexed_files=unindexed_files(),
    )


@router.post("/ingest/rebuild", response_model=RebuildResponse, tags=["corpus"])
async def rebuild_corpus(
    confirm: bool = Query(default=False),
    principal: Principal = Depends(require_corpus_admin),
) -> RebuildResponse:
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "this empties the vector table and re-ingests every document in the corpus "
                "directory, which costs one embedding pass over the whole corpus. "
                "Call it again with confirm=true if that is what you want."
            ),
        )

    directory = corpus_dir()

    if not directory.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"the corpus directory does not exist: {directory}",
        )

    try:
        cleared = await asyncio.to_thread(clear_vector_table)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"the vector table could not be cleared: {exc}",
        ) from exc

    logger.warning("corpus rebuild started by=%s cleared_chunks=%d dir=%s", principal.user_id, cleared, directory)
    invalidate_corpus_caches("rebuild:cleared")

    try:
        report = await asyncio.to_thread(ingest_directory, str(directory))
    except Exception as exc:
        logger.error("corpus rebuild failed after clearing the vector table: %s", exc, exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"the vector table was cleared but re-ingestion failed: {exc}",
        ) from exc

    invalidate_corpus_caches("rebuild:indexed")
    logger.info("corpus rebuild finished total_nodes=%d files=%d", report.total_nodes, len(report.results))

    return RebuildResponse(
        cleared_chunks=cleared,
        corpus_dir=str(directory),
        embed_model=active_embed_model_name(),
        indexed_files=[r.file_name for r in report.results if not r.skipped],
        skipped_files=[
            {"file_name": r.file_name, "reason": r.reason} for r in report.results if r.skipped
        ],
        total_nodes=report.total_nodes,
        documents=indexed_document_summary(),
        unindexed_files=unindexed_files(),
    )


@router.get("/metrics/slo", response_model=SloReport, tags=["ops"])
def slo_metrics(
    hours: int = Query(default=None, ge=1, le=720),
    principal: Principal = Depends(require_reviewer),
) -> SloReport:
    try:
        return SloReport(**slo.latency_report(hours or settings.slo_window_hours))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"latency history is unavailable: {exc}") from exc


@router.get("/metrics/optimization", response_model=OptimizationReport, tags=["ops"])
def optimization_metrics(principal: Principal = Depends(require_reviewer)) -> OptimizationReport:
    """Cost and latency optimisation evidence.

    ``caches`` - hit rates and the milliseconds / tokens each cache returned;
    ``routing`` - how many routable calls went to the small tier and why;
    ``cost`` - real token usage and USD spend, per model and per tier, with
    the per-request mean and peak against the configured budget.
    """
    return OptimizationReport(
        generated_at=datetime.now(UTC),
        caches=cache_report(),
        routing=routing_report(),
        cost=cost_ledger.report(),
    )


@router.post("/cache/invalidate", response_model=CacheInvalidationResponse, tags=["ops"])
def invalidate_caches(
    reason: str = Query(default="manual", max_length=120),
    principal: Principal = Depends(require_corpus_admin),
) -> CacheInvalidationResponse:
    """Drop every cached answer, chunk list and completion (admin only)."""
    cleared = invalidate_corpus_caches(f"manual:{principal.user_id}:{reason}")

    return CacheInvalidationResponse(reason=reason, cleared=cleared)


@router.get("/health", tags=["ops"])
def health():
    try:
        depth = escalation_service.queue_depth()
        database = "up"
    except Exception:
        logger.debug("health check: database probe failed", exc_info=True)
        depth = None
        database = "down"

    caches = cache_report()
    cost = cost_ledger.report()

    return {
        "status": "ok" if database == "up" else "degraded",
        "database": database,
        "audit_log": audit.circuit.status(),
        "escalation_queue_depth": depth,
        "as_of_date": str(settings.as_of_date),
        "confidence_threshold": settings.confidence_threshold,
        "deadlines_seconds": {
            "standard": settings.deadline_seconds_standard,
            "hybrid": settings.deadline_seconds_hybrid,
            "agentic": settings.deadline_seconds_agentic,
            "high_risk": settings.deadline_seconds_high_risk,
        },
        "default_token_budget": settings.default_token_budget,
        "azure_configured": settings.azure_configured,
        "optimization": {
            "routing_strategy": settings.model_routing_strategy,
            "llm_gateway": settings.llm_gateway_url or None,
            "caches_enabled": caches["enabled"],
            "cache_hits": caches["totals"]["hits"],
            "cache_saved_ms": caches["totals"]["saved_ms"],
            "usd_spent": cost["process"]["usd"],
            "usd_per_request_mean": cost["usd_per_request_mean"],
            "small_tier_share": cost["small_tier_share"],
        },
    }
