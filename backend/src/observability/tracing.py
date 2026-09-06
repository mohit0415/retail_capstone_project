import functools
import logging
import time
from collections.abc import Callable

from configs.settings import settings
from src.core.audit import audit_from_state
from src.core.budget import guard_from_state
from src.observability.slo import stage_for

logger = logging.getLogger(__name__)

_langfuse_handler = None


def get_callback_handlers() -> list:
    global _langfuse_handler

    if not settings.tracing_enabled or not settings.langfuse_public_key:
        return []

    if _langfuse_handler is None:
        try:
            from langfuse.callback import CallbackHandler

            _langfuse_handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception as exc:
            logger.warning("langfuse handler unavailable, continuing untraced: %s", exc)
            return []

    return [_langfuse_handler]


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


def _span(node_name: str, state: dict, elapsed_ms: float, since_start_ms: float, status: str) -> dict:
    span = {
        "node": node_name,
        "stage": stage_for(node_name),
        "status": status,
        "elapsed_ms": elapsed_ms,
        "since_start_ms": since_start_ms,
        "request_id": state.get("request_id"),
    }

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
            verdict = guard.check(node_name)

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
                    "trace": [_span(node_name, state, 0.0, since_start, "budget_stopped")],
                    "marks": [_mark(node_name, since_start)],
                }

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
                _span(node_name, {**state, **result}, elapsed_ms, since_start, "completed"),
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
