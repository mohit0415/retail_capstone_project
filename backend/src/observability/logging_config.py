"""Central logging configuration for the Retail Policy Intelligence backend.

Application modules never call ``logging.basicConfig`` themselves; they simply
do ``logger = logging.getLogger(__name__)`` and log. The process entry point
(``main.py``, CLI scripts, evals, test fixtures) calls :func:`configure_logging`
exactly once to install the handlers, format and levels for the whole app.

Two handlers are installed on the root logger:

* a ``StreamHandler`` on stderr, so uvicorn / the terminal still shows every line
* a ``RotatingFileHandler`` on ``settings.log_file`` (default ``logs/rpids.log``),
  rotated at ``settings.log_file_max_bytes`` and keeping
  ``settings.log_file_backup_count`` old files, so a long-running server never
  fills the disk

Set ``LOG_FILE=`` (empty) in ``.env`` to disable the file handler.

Request correlation
-------------------
Every record is stamped with the *request id* of the HTTP request being served
(``[req=<uuid>]`` in the line), taken from a :mod:`contextvars` variable that
the request-logging middleware in ``main.py`` and the ``/ask`` handler bind at
the start of a request. A line logged outside any request shows ``req=-``.
Because the value lives in a context variable it follows the request through
LangGraph node functions without every call site threading ``request_id``
through by hand; worker threads started with ``ThreadPoolExecutor`` do *not*
inherit it automatically, so code that fans out should submit
``with_request_context(fn)`` instead of ``fn`` - see :func:`with_request_context`.
"""

import contextvars
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from configs.settings import settings

LOG_FORMAT = "%(asctime)s %(levelname)-8s [req=%(request_id)s] %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "azure",
    "azure.core.pipeline.policies.http_logging_policy",
    "sqlalchemy.engine",
    "langfuse",
    "llama_index.core",
    "unstructured",
    "pdfminer",
    "fsspec",
    "watchfiles",
    "watchfiles.main",
    "presidio-analyzer",
    "presidio-anonymizer",
    "guardrails-ai",
)

_configured = False

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("rpids_request_id", default="-")
_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar("rpids_thread_id", default="-")
_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("rpids_user_id", default="-")


def bind_request_context(
    request_id: str | None = None,
    thread_id: str | None = None,
    user_id: str | None = None,
) -> list[contextvars.Token]:
    """Attach identifiers to the current context so every log line carries them.

    Returns the reset tokens so a caller can undo the binding with
    :func:`reset_request_context`; the :func:`request_context` context manager
    does both for you.
    """
    tokens: list[contextvars.Token] = []

    if request_id is not None:
        tokens.append(_request_id.set(request_id))

    if thread_id is not None:
        tokens.append(_thread_id.set(thread_id))

    if user_id is not None:
        tokens.append(_user_id.set(user_id))

    return tokens


def reset_request_context(tokens: list[contextvars.Token]) -> None:
    for token in reversed(tokens):
        try:
            token.var.reset(token)
        except ValueError:
            continue


def clear_request_context() -> None:
    _request_id.set("-")
    _thread_id.set("-")
    _user_id.set("-")


def current_request_id() -> str | None:
    value = _request_id.get()

    return None if value == "-" else value


def current_request_context() -> dict[str, str]:
    return {"request_id": _request_id.get(), "thread_id": _thread_id.get(), "user_id": _user_id.get()}


@contextmanager
def request_context(
    request_id: str | None = None,
    thread_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[None]:
    tokens = bind_request_context(request_id=request_id, thread_id=thread_id, user_id=user_id)

    try:
        yield
    finally:
        reset_request_context(tokens)


def with_request_context(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``func`` so it runs in a copy of the *caller's* context.

    Hand the wrapper to ``ThreadPoolExecutor.submit`` so the worker thread
    inherits the request id (and any other context variables) of the thread
    that submitted the job::

        pool.submit(with_request_context(gather_policy_evidence), state)

    The context is captured when the wrapper is built - in the submitting
    thread - and copied again per call, so one wrapper can safely be invoked
    more than once or concurrently.
    """
    captured = contextvars.copy_context()

    def _runner(*args: Any, **kwargs: Any) -> Any:
        return captured.copy().run(func, *args, **kwargs)

    _runner.__name__ = getattr(func, "__name__", "runner")
    _runner.__doc__ = getattr(func, "__doc__", None)

    return _runner


def run_with_context(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` under a copy of the current context (same thread).

    For hand-off to another thread use :func:`with_request_context`, which
    captures the context on the submitting side.
    """
    return contextvars.copy_context().run(func, *args, **kwargs)


class RequestContextFilter(logging.Filter):
    """Stamp ``request_id`` / ``thread_id`` / ``user_id`` onto every record.

    Installed on each handler (not on a logger) so records propagated from
    child loggers are stamped too. A record that already carries a
    ``request_id`` attribute (``logger.info(..., extra={"request_id": ...})``)
    keeps it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id.get()

        if not hasattr(record, "thread_id"):
            record.thread_id = _thread_id.get()

        if not hasattr(record, "user_id"):
            record.user_id = _user_id.get()

        return True


def _numeric_level(level: str | None) -> int:
    resolved = (level or settings.log_level or "INFO").upper()
    value = logging.getLevelName(resolved)

    return value if isinstance(value, int) else logging.INFO


def _build_stream_handler(formatter: logging.Formatter) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())

    return handler


def _build_file_handler(formatter: logging.Formatter) -> logging.Handler | None:
    """Return a rotating file handler, or ``None`` when LOG_FILE is blank or unwritable.

    A failure here must never take the server down: logging to the terminal still
    works, so we warn on stderr and carry on without the file.
    """
    target = (settings.log_file or "").strip()

    if not target:
        return None

    path = Path(target)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            path,
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        logging.getLogger("rpids").warning(
            "log file %s could not be opened (%s); file logging disabled", path, exc
        )

        return None

    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())

    return handler


def configure_logging(level: str | None = None, *, force: bool = False) -> logging.Logger:
    """Install the app-wide logging configuration.

    Idempotent: repeated calls only re-apply the level unless ``force`` is set,
    so importing this from several entry points is safe. ``level`` overrides
    ``settings.log_level`` when provided. Returns the application logger
    ("rpids").
    """
    global _configured

    numeric = _numeric_level(level)

    if _configured and not force:
        logging.getLogger().setLevel(numeric)
        logging.getLogger("rpids").setLevel(numeric)

        return logging.getLogger("rpids")

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()

    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()

    root.addHandler(_build_stream_handler(formatter))

    file_handler = _build_file_handler(formatter)

    if file_handler is not None:
        root.addHandler(file_handler)

    root.setLevel(numeric)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    app_logger = logging.getLogger("rpids")
    app_logger.setLevel(numeric)

    _configured = True
    app_logger.debug("logging configured at level %s", logging.getLevelName(numeric))

    if file_handler is not None:
        app_logger.info(
            "logging to file %s (rotate at %d bytes, keep %d)",
            file_handler.baseFilename,
            settings.log_file_max_bytes,
            settings.log_file_backup_count,
        )

    return app_logger


def log_startup_summary(logger: logging.Logger | None = None) -> None:
    """One INFO line per operator-relevant setting, emitted at process start.

    Secrets are never printed; only whether they are configured.
    """
    target = logger or logging.getLogger("rpids")

    target.info(
        "startup env=%s log_level=%s azure_configured=%s small_model=%s strong_model=%s "
        "embedding=%s llm_gateway=%s",
        settings.app_env,
        settings.log_level,
        settings.azure_configured,
        settings.azure_openai_small_deployment,
        settings.azure_openai_strong_deployment,
        settings.azure_openai_embedding_deployment,
        settings.llm_gateway_url or "direct",
    )
    target.info(
        "startup routing_strategy=%s response_cache=%s semantic_threshold=%s llm_cache=%s "
        "retrieval_cache=%s cache_ttl_s=%s",
        settings.model_routing_strategy,
        settings.enable_response_cache,
        settings.semantic_cache_threshold,
        settings.enable_llm_cache,
        settings.enable_retrieval_cache,
        settings.response_cache_ttl_seconds,
    )
    target.info(
        "startup deadlines_s standard=%s hybrid=%s agentic=%s high_risk=%s token_budget=%s "
        "confidence_threshold=%s tracing=%s",
        settings.deadline_seconds_standard,
        settings.deadline_seconds_hybrid,
        settings.deadline_seconds_agentic,
        settings.deadline_seconds_high_risk,
        settings.default_token_budget,
        settings.confidence_threshold,
        settings.tracing_enabled,
    )
