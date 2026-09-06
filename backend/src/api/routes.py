import asyncio
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
from src.auth.rbac import allowed_doc_types
from src.auth.security import (
    Principal,
    current_principal,
    issue_token,
    require_corpus_admin,
    require_reviewer,
)
from src.core import audit, conversation
from src.core.idempotency import build_key, idempotency_store
from src.graph.builder import get_compiled_graph
from src.graph.state import initial_state
from src.guardrails.output_guard import canonical_citation
from src.index.models import active_embed_model_name
from src.index.vector_index import (
    check_embed_model_compatibility,
    clear_vector_table,
    indexed_document_summary,
)
from src.ingestion.bootstrap import corpus_dir, corpus_is_indexed, unindexed_files
from src.ingestion.pipeline import ingest_directory, ingest_file
from src.observability import slo
from src.schemas.api import (
    AnswerResponse,
    AskReleasedResponse,
    AskRequest,
    Citation,
    ClarificationResponse,
    CorpusStatusResponse,
    IngestResponse,
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
    TokenRequest,
    TokenResponse,
)
from src.schemas.enums import ReviewDecision, RiskLevel, TerminalOutcome
from src.schemas.models import DraftAnswer

router = APIRouter()

SUPPORTED_UPLOAD_SUFFIXES = {".md", ".pdf", ".docx", ".txt"}


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

        for reference in cited:
            key = canonical_citation(reference)

            if key in seen and key not in used:
                used.add(key)
                filtered.append(seen[key])

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


def _timings_from(final_state: dict, total_ms: float) -> StageTimings:
    reached = slo.stage_marks(final_state.get("marks", []))

    return StageTimings(
        t1_ms=reached.get("t1"),
        t2_ms=reached.get("t2"),
        t3_ms=reached.get("t3"),
        t4_ms=reached.get("t4"),
        total_ms=round(total_ms, 2),
    )


@router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def create_token(payload: TokenRequest) -> TokenResponse:
    token, expires_in = issue_token(payload.user_id, payload.role, payload.departments)

    return TokenResponse(access_token=token, expires_in=expires_in)


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
    request_id = str(uuid.uuid4())
    thread_id = payload.thread_id or str(uuid.uuid4())

    history, summary = conversation.load_thread(thread_id) if payload.thread_id else ([], "")

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
        document_scope_request=_validated_document_scope(payload.document_scope, principal),
        conversation_history=history,
        thread_summary=summary,
    )

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    final_state = graph.invoke(state, config=config)

    total_ms = (time.monotonic() - started_ts) * 1000
    outcome = final_state.get("terminal_outcome") or TerminalOutcome.ESCALATED.value
    timings = _timings_from(final_state, total_ms)

    slo.record_latency(final_state, total_ms, outcome)

    if outcome == TerminalOutcome.REFUSED.value:
        body = RefusalResponse(
            request_id=request_id,
            reason=final_state.get("refusal_reason") or "the request could not be accepted",
        ).model_dump(mode="json")

        response.status_code = status.HTTP_200_OK
        idempotency_store.put(cache_key, body, status.HTTP_200_OK)

        return body

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
            queued_at=datetime.now(UTC),
            poll_url=str(request.url_for("get_request_status", request_id=request_id)),
            timings=timings,
        ).model_dump(mode="json")

        response.status_code = status.HTTP_202_ACCEPTED
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
    ).model_dump(mode="json")

    background.add_task(
        conversation.record_exchange,
        thread_id,
        principal.user_id,
        request_id,
        payload.query,
        draft.answer,
    )

    idempotency_store.put(cache_key, body, status.HTTP_200_OK)

    return body


@router.get("/requests/{request_id}", name="get_request_status", tags=["ask"])
def get_request_status(request_id: str, principal: Principal = Depends(current_principal)):
    review, row = escalation_service.fetch_package(request_id)

    if row is None:
        raise HTTPException(status_code=404, detail="unknown request id")

    if row["status"] == "pending":
        return {"status": "pending_review", "request_id": request_id, "risk_level": row["risk_level"]}

    return {
        "status": "answered",
        "request_id": request_id,
        "thread_id": row["thread_id"],
        "answer": row["reviewed_answer"] or review.draft_answer,
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


def _apply_review_to_checkpoint(
    thread_id: str,
    decision: ReviewDecision,
    answer: str,
    reviewer_id: str,
    notes: str,
) -> tuple[bool, dict | None]:
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    try:
        snapshot = graph.get_state(config)
    except Exception:
        return False, None

    if snapshot is None or not snapshot.values:
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
        return False, None

    if decision is ReviewDecision.REJECT:
        return True, None

    try:
        return True, graph.invoke(None, config=config)
    except Exception:
        return True, None


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

    answer = submission.edited_answer if submission.edited_answer else review.draft_answer

    updated = escalation_service.record_decision(
        request_id=request_id,
        reviewer_id=principal.user_id,
        decision=submission.decision,
        edited_answer=answer,
        notes=submission.reviewer_notes,
    )

    if not updated:
        raise HTTPException(status_code=409, detail="the request was reviewed by someone else first")

    resumed, final_state = _apply_review_to_checkpoint(
        thread_id=row["thread_id"],
        decision=submission.decision,
        answer=answer,
        reviewer_id=principal.user_id,
        notes=submission.reviewer_notes,
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

    if not resumed or final_state is None:
        return ReviewOutcome(
            request_id=request_id,
            decision=submission.decision.value,
            resumed=resumed,
            released=False,
            answer=answer,
            note=(
                "the decision was recorded, but the graph checkpoint for this thread could not be "
                "resumed, so the reviewed answer did not pass back through the output guardrail"
            ),
        )

    outcome = final_state.get("terminal_outcome")
    released = outcome == TerminalOutcome.ANSWERED.value
    released_draft = final_state.get("draft")
    released_answer = released_draft.answer if released_draft else answer

    if released:
        background.add_task(
            conversation.record_exchange,
            row["thread_id"],
            row["user_id"],
            request_id,
            review.standalone_query or review.original_query,
            released_answer,
        )

        note = "the reviewed answer passed the output guardrail and was released"
    else:
        note = final_state.get("escalation_reason") or "the reviewed answer did not pass the output guardrail"

    return ReviewOutcome(
        request_id=request_id,
        decision=submission.decision.value,
        resumed=True,
        released=released,
        outcome=outcome,
        answer=released_answer if released else None,
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
        return IngestResponse(status="skipped", file_name=result.file_name, reason=result.reason)

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

    try:
        report = await asyncio.to_thread(ingest_directory, str(directory))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"the vector table was cleared but re-ingestion failed: {exc}",
        ) from exc

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


@router.get("/health", tags=["ops"])
def health():
    try:
        depth = escalation_service.queue_depth()
        database = "up"
    except Exception:
        depth = None
        database = "down"

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
    }
