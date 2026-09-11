"""A lightweight LangChain callback handler that logs LLM activity.

This makes every model call visible in the application log - the model used,
the node it belongs to, wall-clock latency and token usage - with no external
tracing backend required. It is always attached (see
:func:`src.observability.tracing.get_callback_handlers`); the Langfuse handler
is layered on top of it only when tracing is configured.

It is also where the cost ledger (:mod:`src.llm_routing.ledger`) learns what
each call actually cost: the token counts the provider returns are priced
and attributed to the request id carried in the run metadata (or, failing
that, the request id bound to the logging context).
"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from configs.settings import settings
from src.observability.logging_config import current_request_id

logger = logging.getLogger("rpids.llm")


def _token_usage(response: LLMResult) -> dict[str, int]:
    """Best-effort token counts across langchain-core response shapes."""
    output = response.llm_output or {}
    usage = output.get("token_usage") or output.get("usage") or {}

    if usage:
        return {
            "prompt": int(usage.get("prompt_tokens", 0) or 0),
            "completion": int(usage.get("completion_tokens", 0) or 0),
            "total": int(usage.get("total_tokens", 0) or 0),
        }

    prompt = completion = total = 0

    for batch in response.generations:
        for generation in batch:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None)

            if meta:
                prompt += int(meta.get("input_tokens", 0) or 0)
                completion += int(meta.get("output_tokens", 0) or 0)
                total += int(meta.get("total_tokens", 0) or 0)

    return {"prompt": prompt, "completion": completion, "total": total}


def _model_name(response: LLMResult, hint: str) -> str:
    output = response.llm_output or {}

    for key in ("model_name", "model", "deployment_name"):
        if output.get(key):
            return str(output[key])

    for batch in response.generations:
        for generation in batch:
            message = getattr(generation, "message", None)
            meta = getattr(message, "response_metadata", None) or {}

            if meta.get("model_name"):
                return str(meta["model_name"])

    return hint or "unknown"


def _model_hint(serialized: dict[str, Any] | None) -> str:
    kwargs = (serialized or {}).get("kwargs") or {}

    for key in ("azure_deployment", "deployment_name", "model_name", "model"):
        if kwargs.get(key):
            return str(kwargs[key])

    return ""


class LoggingCallbackHandler(BaseCallbackHandler):
    """Logs LLM start/end/error with model, node, latency and token counts.

    A single shared instance is attached to every model call, so it keeps only
    per-run bookkeeping (start time, node name, request id, model hint), keyed
    by run id and cleared on completion or error.
    """

    raise_error = False

    def __init__(self) -> None:
        self._runs: dict[UUID, tuple[float, str, str | None, str]] = {}

    @staticmethod
    def _run_name(metadata: dict | None, kwargs: dict) -> str:
        if kwargs.get("name"):
            return str(kwargs["name"])

        if metadata and metadata.get("node"):
            return str(metadata["node"])

        return "llm"

    def _begin(self, run_id: UUID, serialized: dict | None, metadata: dict | None, kwargs: dict) -> None:
        request_id = (metadata or {}).get("request_id") or current_request_id()
        hint = _model_hint(serialized)
        name = self._run_name(metadata, kwargs)

        self._runs[run_id] = (time.monotonic(), name, request_id, hint)

        logger.debug(
            "llm_start node=%s model=%s request_id=%s run_id=%s", name, hint or "?", request_id, run_id
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._begin(run_id, serialized, metadata, kwargs)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._begin(run_id, serialized, metadata, kwargs)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            started, name, request_id, hint = self._runs.pop(run_id, (None, "llm", None, ""))
            elapsed_ms = round((time.monotonic() - started) * 1000, 1) if started else None
            usage = _token_usage(response)
            model = _model_name(response, hint)

            cached = settings.enable_llm_cache and response.llm_output is None

            if cached:
                logger.info(
                    "llm_call node=%s model=%s cached=true elapsed_ms=%s tokens_saved=%s run_id=%s",
                    name,
                    model,
                    elapsed_ms,
                    usage["total"],
                    run_id,
                )

                return

            usd = 0.0

            try:
                from src.llm_routing.ledger import cost_ledger

                usd = cost_ledger.record_call(
                    request_id,
                    node=name,
                    model=model,
                    prompt_tokens=usage["prompt"],
                    completion_tokens=usage["completion"],
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:
                logger.debug("cost ledger update failed: %s", exc)

            logger.info(
                "llm_call node=%s model=%s elapsed_ms=%s tokens[p/c/t]=%s/%s/%s usd=%.6f run_id=%s",
                name,
                model,
                elapsed_ms,
                usage["prompt"],
                usage["completion"],
                usage["total"],
                usd,
                run_id,
            )
        except Exception as exc:
            logger.debug("logging callback on_llm_end failed: %s", exc)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        started, name, request_id, hint = self._runs.pop(run_id, (None, "llm", None, ""))
        elapsed_ms = round((time.monotonic() - started) * 1000, 1) if started else None

        logger.warning(
            "llm_error node=%s error=%s model=%s elapsed_ms=%s request_id=%s run_id=%s",
            name,
            error,
            hint or "?",
            elapsed_ms,
            request_id,
            run_id,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        logger.warning("tool_error error=%s run_id=%s", error, run_id)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        logger.warning("retriever_error error=%s run_id=%s", error, run_id)
