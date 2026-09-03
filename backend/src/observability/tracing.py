import functools
import logging
import time
from typing import Callable

from configs.settings import settings

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


def traced_node(node_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: dict) -> dict:
            started = time.monotonic()
            result = func(state)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)

            span = {
                "node": node_name,
                "elapsed_ms": elapsed_ms,
                "request_id": state.get("request_id"),
            }

            logger.info("node=%s elapsed_ms=%s request_id=%s", node_name, elapsed_ms, state.get("request_id"))

            existing_trace = result.get("trace", [])
            result["trace"] = existing_trace + [span]

            return result

        return wrapper

    return decorator
