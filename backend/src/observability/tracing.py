import functools
import logging
import time
from collections.abc import Callable

from src.core.audit import audit_from_state
from src.core.budget import finishing_grace_seconds, guard_from_state
from src.observability.agent_steps import describe_step
from src.observability.callbacks import LoggingCallbackHandler
from src.observability.langfuse_callback import get_langfuse_manager
from src.observability.slo import stage_for

logger = logging.getLogger(__name__)

_logging_handler = LoggingCallbackHandler()


def _init_langfuse():
    """Return the process-wide Langfuse CallbackHandler, or ``None``.

    The client/handler lifecycle now lives in
    :mod:`src.observability.langfuse_callback` (``LangfuseCallbackManager``);
    this keeps the old entry point so node-level ``runnable_config`` calls and
    the graph-level ``setup_langfuse_callback`` share one handler instance -
    LangChain de-duplicates identical handler objects, so a request renders as
    a single trace tree.
    """
    return get_langfuse_manager().get_callback_handler()


def get_callback_handlers() -> list:
    """Callbacks attached to every LLM call and to the graph run.

    Always includes the logging handler, so model calls are observable in the
    application log; adds the Langfuse handler on top when tracing is
    configured.
    """
    handlers: list = [_logging_handler]

    langfuse_handler = _init_langfuse()

    if langfuse_handler is not None:
        handlers.append(langfuse_handler)

    return handlers


def flush_tracing() -> None:
    """Flush buffered spans to Langfuse. Call on application shutdown."""
    get_langfuse_manager().flush()


def runnable_config(state: dict, node_name: str) -> dict:
    return {
        "callbacks": get_callback_handlers(),
        "metadata": {
            "request_id": state.get("request_id"),
            "thread_id": state.get("thread_id"),
            "node": node_name,
            "risk_level": state["risk"].final_level.value if state.get("risk") else None,
        },
        "run_name": node_name,
    }


def _elapsed_since_start(state: dict, fallback_started: float) -> float:
    started = state.get("started_ts") or fallback_started

    return round((time.monotonic() - started) * 1000, 2)


def _span(
    node_name: str,
    state: dict,
    elapsed_ms: float,
    since_start_ms: float,
    status: str,
    result: dict | None = None,
) -> dict:
    span = {
        "node": node_name,
        "stage": stage_for(node_name),
        "status": status,
        "elapsed_ms": elapsed_ms,
        "since_start_ms": since_start_ms,
        "request_id": state.get("request_id"),
    }

    # readable agent name, one-line summary and model tier for the workflow view
    span.update(describe_step(node_name, state, result, status))

    decision = state.get("path_decision")

    if decision:
        span["routed_path"] = decision.get("path")
        span["route_reason"] = decision.get("reason")
        span["route_clamped"] = decision.get("clamped")

    return span


def _mark(node_name: str, since_start_ms: float) -> dict:
    return {
        "node": node_name,
        "stage": stage_for(node_name),
        "elapsed_ms": since_start_ms,
    }


def traced_node(node_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: dict) -> dict:
            started = time.monotonic()
            guard = guard_from_state(state)
            grace = finishing_grace_seconds(node_name, state)
            verdict = guard.check(node_name, grace_seconds=grace)

            if verdict.allowed and grace and guard.seconds_remaining <= 0:
                logger.info(
                    "node=%s started %.1fs after the deadline inside its %.0fs grace window request_id=%s",
                    node_name,
                    -guard.seconds_remaining,
                    grace,
                    state.get("request_id"),
                )

            if not verdict.allowed:
                since_start = _elapsed_since_start(state, started)

                stop = {
                    "node": node_name,
                    "reason": verdict.reason,
                    "seconds_remaining": round(guard.seconds_remaining, 3),
                    "tokens_remaining": guard.tokens_remaining,
                }

                result = {
                    "escalation_reason": f"the {node_name} step was stopped by the budget guard ({verdict.reason})",
                    "budget_stops": [stop],
                    "degraded": True,
                    "marks": [_mark(node_name, since_start)],
                }
                result["trace"] = [_span(node_name, state, 0.0, since_start, "budget_stopped", result)]

                logger.warning(
                    "node=%s budget_stopped reason=%s request_id=%s",
                    node_name,
                    verdict.reason,
                    state.get("request_id"),
                )

                audit_from_state(
                    {**state, **result},
                    node=node_name,
                    event="budget_stopped",
                    detail=stop,
                )

                return result

            result = func(state)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            since_start = _elapsed_since_start(state, started)

            logger.info(
                "node=%s elapsed_ms=%s since_start_ms=%s request_id=%s",
                node_name,
                elapsed_ms,
                since_start,
                state.get("request_id"),
            )

            result["trace"] = [
                *result.get("trace", []),
                _span(node_name, {**state, **result}, elapsed_ms, since_start, "completed", result),
            ]
            result["marks"] = [*result.get("marks", []), _mark(node_name, since_start)]

            audit_from_state(
                {**state, **result},
                node=node_name,
                event="node_completed",
                detail={
                    "elapsed_ms": elapsed_ms,
                    "since_start_ms": since_start,
                    "tokens_spent_here": result.get("tokens_spent", 0),
                    "seconds_remaining": round(guard.seconds_remaining, 3),
                },
            )

            return result

        return wrapper

    return decorator
