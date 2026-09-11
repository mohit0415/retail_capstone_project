"""
Langfuse Callback Handler Module

This module provides a centralized Langfuse callback handler for tracing
LangChain / LangGraph executions in the Langfuse dashboard.

Ported from the Course-2 Sprint-9 ``callbacks/langfuse_callback.py`` +
``setup_langfuse_callback`` / ``flush_langfuse_traces`` in ``main.py``, adapted
for the langfuse 4.x SDK this project pins:

* credentials live on one ``Langfuse(...)`` client, ``CallbackHandler`` only
  takes ``public_key`` and resolves that client;
* ``session_id`` / ``user_id`` / ``trace_name`` / tags are no longer handler
  arguments - they are read from the root run's ``metadata`` under the
  ``langfuse_session_id`` / ``langfuse_user_id`` / ``langfuse_trace_name`` /
  ``langfuse_tags`` keys, which is exactly what ``setup_langfuse_callback``
  writes below.

One client and one handler are created per process. The graph run and every
node-level model call share that handler, so a request shows up as a single
trace tree in Langfuse rather than one tree per node.
"""

import logging
import os
from typing import Any

from configs.settings import settings

logger = logging.getLogger(__name__)


class LangfuseCallbackManager:
    """Owns the Langfuse client and the LangChain CallbackHandler for this process."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        self.public_key = public_key or settings.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or settings.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = (
            host
            or settings.langfuse_host
            or os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or "https://cloud.langfuse.com"
        )
        self.session_id = session_id
        self.user_id = user_id
        self.trace_name: str | None = None

        self._client = None
        self._handler = None
        self._ready = False

    @property
    def enabled(self) -> bool:
        return bool(settings.tracing_enabled and self.public_key and self.secret_key)

    @property
    def client(self):
        return self._client

    def _init(self) -> None:
        """Create the client + handler once; failures degrade to untraced operation."""
        if self._ready:
            return

        self._ready = True

        if not self.enabled:
            logger.info("langfuse tracing disabled (TRACING_ENABLED=false or keys missing)")
            return

        os.environ["LANGFUSE_PUBLIC_KEY"] = self.public_key
        os.environ["LANGFUSE_SECRET_KEY"] = self.secret_key
        os.environ["LANGFUSE_HOST"] = self.host
        os.environ["LANGFUSE_BASE_URL"] = self.host

        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
                environment=settings.app_env,
                tracing_enabled=True,
            )

            try:
                self._client.auth_check()
            except Exception as exc:
                logger.warning("langfuse auth_check failed (continuing): %s", exc)

            self._handler = CallbackHandler(public_key=self.public_key)

            logger.info(
                "langfuse tracing enabled (host=%s environment=%s)", self.host, settings.app_env
            )
        except Exception as exc:
            logger.warning("langfuse tracing unavailable, continuing untraced: %s", exc)
            self._client = None
            self._handler = None

    def get_callback_handler(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        trace_name: str | None = None,
    ):
        """Return the shared CallbackHandler, or None when tracing is off.

        ``session_id`` / ``user_id`` / ``trace_name`` keep the Sprint-9
        signature; on langfuse 4.x they cannot be handler arguments, so they are
        remembered as instance defaults and :func:`setup_langfuse_callback`
        writes them into the run ``metadata``.
        """
        self.session_id = session_id or self.session_id
        self.user_id = user_id or self.user_id
        self.trace_name = trace_name or getattr(self, "trace_name", None)

        self._init()

        return self._handler

    def flush(self) -> None:
        """Flush any pending traces to Langfuse (call on shutdown / end of request)."""
        if self._client is None:
            return

        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("langfuse flush failed: %s", exc)


_langfuse_manager: LangfuseCallbackManager | None = None


def get_langfuse_manager() -> LangfuseCallbackManager:
    global _langfuse_manager

    if _langfuse_manager is None:
        _langfuse_manager = LangfuseCallbackManager()

    return _langfuse_manager


def setup_langfuse_callback(
    config: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str = "policy_graph",
    tags: list[str] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Attach the Langfuse handler + trace attributes to a runnable config.

    Args:
        config: an existing LangChain/LangGraph config (the ``configurable``,
            ``callbacks``, ``metadata`` keys are preserved and extended).
        session_id: groups traces in Langfuse - the conversation ``thread_id``.
        user_id: the principal making the request.
        trace_name: name shown for the root trace in the dashboard.
        tags: optional Langfuse tags for the trace.

    Returns:
        ``(config, callback_handler)`` - config is ready for ``graph.invoke``;
        the handler is ``None`` when tracing is disabled.
    """
    config = dict(config or {})
    manager = get_langfuse_manager()

    callback_handler = manager.get_callback_handler(
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
    )

    if callback_handler is None:
        return config, None

    session_id = session_id or manager.session_id
    user_id = user_id or manager.user_id

    callbacks = list(config.get("callbacks") or [])

    if callback_handler not in callbacks:
        callbacks.append(callback_handler)

    config["callbacks"] = callbacks

    metadata = dict(config.get("metadata") or {})

    if session_id:
        metadata["session_id"] = session_id
        metadata["langfuse_session_id"] = session_id

    if user_id:
        metadata["user_id"] = user_id
        metadata["langfuse_user_id"] = user_id

    metadata["trace_name"] = trace_name
    metadata["langfuse_trace_name"] = trace_name
    metadata["langfuse_tags"] = list(tags or [trace_name])

    config["metadata"] = metadata
    config.setdefault("run_name", trace_name)

    return config, callback_handler


def flush_langfuse_traces(callback_handler=None) -> None:
    """Flush buffered spans so a finished request is visible in Langfuse promptly.

    Sync on purpose - the ``/ask`` route is a plain ``def``; schedule it on a
    ``BackgroundTasks`` so the HTTP response is not held back by the flush.
    """
    if callback_handler is None:
        return

    client = None

    for attr in ("_explicit_langfuse_client", "_langfuse_client", "client", "langfuse", "_client"):
        potential = getattr(callback_handler, attr, None)

        if potential is not None and hasattr(potential, "flush"):
            client = potential
            break

    if client is None:
        client = get_langfuse_manager().client

    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:
        logger.debug("langfuse flush failed: %s", exc)


__all__ = [
    "LangfuseCallbackManager",
    "flush_langfuse_traces",
    "get_langfuse_manager",
    "setup_langfuse_callback",
]
