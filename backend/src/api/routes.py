import asyncio
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
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
from src.auth.security import (
    Principal,
    current_principal,
    issue_token,
    require_corpus_admin,
    require_reviewer,
)
from src.core.idempotency import build_key, idempotency_store
from src.graph.builder import get_compiled_graph
from src.graph.state import initial_state
from src.index.models import active_embed_model_name
from src.index.vector_index import (
    check_embed_model_compatibility,
    indexed_document_summary,
)
from src.ingestion.bootstrap import corpus_dir, corpus_is_indexed
from src.ingestion.pipeline import ingest_file
from src.schemas.api import (
    AnswerResponse,
    AskRequest,
    Citation,
    ClarificationResponse,
    CorpusStatusResponse,
    IngestResponse,
    PendingReviewResponse,
    QueueItem,
    RefusalResponse,
    ReviewPackage,
    ReviewSubmission,
    SqlProof,
    TokenRequest,
    TokenResponse,
)
from src.schemas.enums import RiskLevel, TerminalOutcome

router = APIRouter()

SUPPORTED_UPLOAD_SUFFIXES = {".md", ".pdf", ".docx", ".txt"}


def _deadline_for(query: str) -> float:
    lowered = query.lower()

    high_risk_markers = ("breach", "erasure", "terminate", "bribe", "penalty", "sanction")

    if any(marker in lowered for marker in high_risk_markers):
        return time.monotonic() + settings.deadline_seconds_high_risk

    compare_markers = ("compliant", "overdue", "expired", "allowed to", "may we", "can we")

    if any(marker in lowered for marker in compare_markers):
        return time.monotonic() + settings.deadline_seconds_hybrid

    return time.monotonic() + settings.deadline_seconds_standard


def _citations_from(final_state: dict) -> list[Citation]:
    seen: dict[str, Citation] = {}

    for chunk in final_state.get("retrieved_chunks", []):
        key = f"{chunk.document_title} §{chunk.clause_number}"

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
        filtered = [seen[key] for key in cited if key in seen]

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


@router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def create_token(payload: TokenRequest) -> TokenResponse:
    token, expires_in = issue_token(payload.user_id, payload.role, payload.departments)

    return TokenResponse(access_token=token, expires_in=expires_in)


@router.post("/ask", tags=["ask"])
def ask(
    payload: AskRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(current_principal),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    cache_key = build_key(principal.user_id, idempotency_key, payload.model_dump())
    cached = idempotency_store.get(cache_key)

    if cached is not None:
        response.status_code = cached.status_code

        return cached.payload

    request_id = str(uuid.uuid4())
    thread_id = payload.thread_id or str(uuid.uuid4())

    state = initial_state(
        request_id=request_id,
        thread_id=thread_id,
        user_id=principal.user_id,
        role=principal.role.value,
        access_scopes=principal.access_scopes,
        raw_query=payload.query,
        deadline_ts=_deadline_for(payload.query),
        token_budget=settings.default_token_budget,
    )

    state["departments"] = principal.departments

    if payload.document_scope:
        state["document_scope_request"] = payload.document_scope

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    final_state = graph.invoke(state, config=config)

    outcome = final_state.get("terminal_outcome")

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
            queued_at=datetime.now(timezone.utc),
            poll_url=str(request.url_for("get_request_status", request_id=request_id)),
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
    ).model_dump(mode="json")

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
    review, row = escalation_service.fetch_package(request_id)

    if review is None:
        raise HTTPException(status_code=404, detail="unknown request id")

    return review


@router.post("/review/{request_id}", tags=["review"])
def submit_review(
    request_id: str,
    submission: ReviewSubmission,
    principal: Principal = Depends(require_reviewer),
):
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

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": row["thread_id"]}}

    try:
        graph.update_state(
            config,
            {
                "terminal_outcome": TerminalOutcome.ANSWERED.value,
                "escalation_reason": None,
            },
        )
    except Exception:
        pass

    return {"status": "recorded", "request_id": request_id, "decision": submission.decision.value}


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
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ingestion failed for {original_name}: {exc}",
        )
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
    )


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
        "escalation_queue_depth": depth,
        "as_of_date": str(settings.as_of_date),
        "confidence_threshold": settings.confidence_threshold,
        "azure_configured": settings.azure_configured,
    }
