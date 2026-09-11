import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from configs.database import close_pool, open_pool
from configs.settings import settings
from src.api.routes import router
from src.cache.llm_cache import install_llm_cache
from src.core.conversation import ensure_conversation_tables
from src.ingestion.bootstrap import run_startup_bootstrap
from src.observability.logging_config import (
    configure_logging,
    log_startup_summary,
    request_context,
)
from src.observability.tracing import flush_tracing
from src.prompts.langfuse_prompts import warm_prompt_cache

configure_logging(settings.log_level)

logger = logging.getLogger("rpids")
access_logger = logging.getLogger("rpids.access")

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_standard])

REQUEST_ID_HEADER = "X-Request-ID"

QUIET_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """One access-log line per request, and a request id on every log line in between.

    The id is taken from an incoming ``X-Request-ID`` header (so a gateway or
    a client retry can correlate) or generated here, bound to the logging
    context for the duration of the request, exposed to handlers as
    ``request.state.request_id`` and returned in the response header. The
    ``/ask`` handler reuses it as the graph's ``request_id``, so the access
    line, every node line, the audit rows and the Langfuse trace share one id.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = (request.headers.get(REQUEST_ID_HEADER) or "").strip()[:64] or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        client = request.client.host if request.client else "-"
        path = request.url.path

        with request_context(request_id=request_id):
            access_logger.debug(
                "http_request_start method=%s path=%s client=%s", request.method, path, client
            )

            try:
                response = await call_next(request)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                access_logger.error(
                    "http_request method=%s path=%s status=500 elapsed_ms=%.1f client=%s error=%s",
                    request.method,
                    path,
                    elapsed_ms,
                    client,
                    exc,
                )

                raise

            elapsed_ms = (time.perf_counter() - started) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id

            level = logging.DEBUG if path in QUIET_PATHS else logging.INFO

            if response.status_code >= 500:
                level = logging.ERROR
            elif response.status_code >= 400:
                level = logging.WARNING

            access_logger.log(
                level,
                "http_request method=%s path=%s status=%s elapsed_ms=%.1f client=%s",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
                client,
            )

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_startup_summary(logger)

    try:
        open_pool()
        logger.info("database pool opened")
        await asyncio.to_thread(ensure_conversation_tables)
    except Exception as exc:
        logger.error("database pool could not be opened: %s", exc)

    try:
        install_llm_cache()
    except Exception as exc:
        logger.warning("llm cache could not be installed: %s", exc)

    started = time.perf_counter()
    await asyncio.to_thread(run_startup_bootstrap)
    logger.info("startup bootstrap finished in %.1fs", time.perf_counter() - started)

    started = time.perf_counter()

    try:
        from src.guardrails.pii import warm_up as warm_up_pii_analyzer

        presidio_ready = await asyncio.to_thread(warm_up_pii_analyzer)
        logger.info(
            "PII analyzer warm-up finished in %.1fs presidio=%s", time.perf_counter() - started, presidio_ready
        )
    except Exception as exc:
        logger.warning("PII analyzer warm-up skipped: %s", exc)

    try:
        await asyncio.to_thread(warm_prompt_cache)
    except Exception as exc:
        logger.warning("prompt cache warm-up skipped: %s", exc)

    logger.info("application ready")

    yield

    logger.info("application shutting down")
    close_pool()
    logger.info("database pool closed")
    flush_tracing()


app = FastAPI(
    title="Retail Policy Intelligence & Decision Support System",
    version="4.1.0",
    description="Agentic compliance question answering over retail policy documents and operational records",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        "rate limit exceeded path=%s client=%s limit=%s",
        request.url.path,
        request.client.host if request.client else "-",
        exc.detail,
    )

    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded", "limit": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the traceback once, with the request id, and answer a clean 500.

    Without this, uvicorn prints the traceback without the request id and the
    client gets a bare "Internal Server Error" with no way to quote it back.
    """
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        "unhandled exception path=%s request_id=%s error=%s",
        request.url.path,
        request_id,
        exc,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id} if request_id else None,
    )


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
