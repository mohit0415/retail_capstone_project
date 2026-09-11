"""LangChain-level completion cache.

LangChain lets a process register one global ``BaseCache``; every chat model
whose ``cache`` attribute is unset consults it before calling the provider.
The key is the serialised prompt plus the model's ``llm_string`` (deployment,
temperature, bound tools / structured-output schema), so a cached completion
is only ever reused for an identical call.

``langchain_core.caches.InMemoryCache`` has no expiry and no bound, which is
wrong for a long-running server: prompts embed retrieved policy text, so a
stale entry would keep answering from a superseded clause. This cache expires
entries and evicts the least recently used one when full.

Model calls that should never be cached (temperature > 0, the panel
challenger) are built with ``cache=False`` in :mod:`configs.llms`.

Every hit is attributed to the request being served (via the logging context
variable) so the cost ledger can count the call as *saved* rather than spent.
"""

import hashlib
import logging
import threading
from collections.abc import Sequence
from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache

from configs.settings import settings
from src.cache.base import TTLCache

logger = logging.getLogger(__name__)


class TTLLLMCache(BaseCache):
    """Bounded, expiring in-process cache for LangChain chat completions."""

    def __init__(self, max_entries: int | None = None, ttl_seconds: float | None = None):
        self._store: TTLCache[str, RETURN_VAL_TYPE] = TTLCache(
            "llm",
            max_entries=max_entries or settings.llm_cache_max_entries,
            ttl_seconds=ttl_seconds or settings.llm_cache_ttl_seconds,
        )
        self._lock = threading.Lock()

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{llm_string}\n\x00\n{prompt}".encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        value = self._store.get(self._key(prompt, llm_string))

        if value is None:
            return None

        try:
            from src.llm_routing.ledger import cost_ledger
            from src.observability.logging_config import current_request_id

            cost_ledger.record_cache_hit(current_request_id(), model=_model_from(llm_string))
        except Exception:
            logger.debug("llm cache: could not attribute hit to a request", exc_info=True)

        logger.info("llm cache HIT model=%s prompt_chars=%d", _model_from(llm_string), len(prompt))

        return value

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        if not return_val:
            return

        self._store.put(self._key(prompt, llm_string), list(return_val))
        logger.debug("llm cache STORE model=%s prompt_chars=%d", _model_from(llm_string), len(prompt))

    def clear(self, **kwargs: Any) -> None:
        self._store.clear(str(kwargs.get("reason", "")))

    def report(self) -> dict[str, Any]:
        return self._store.report()

    def size(self) -> int:
        return len(self._store)

    def __bool__(self) -> bool:
        return True


_MODEL_KEYS = ("deployment_name", "azure_deployment", "model_name", "model")


def _model_from(llm_string: str) -> str:
    """Best-effort deployment / model name out of LangChain's llm_string.

    The string is the model's JSON serialisation followed by the call params,
    so the name sits behind ``"deployment_name": "..."`` (Azure) or
    ``"model_name": "..."`` (the gateway's ChatOpenAI). Both quote styles are
    accepted because the params tail uses Python repr.
    """
    for key in _MODEL_KEYS:
        for quote in ('"', "'"):
            marker = f"{quote}{key}{quote}: {quote}"
            start = llm_string.find(marker)

            if start == -1:
                continue

            start += len(marker)
            end = llm_string.find(quote, start)

            if end != -1 and end > start:
                return llm_string[start:end]

    return "unknown"


_installed: TTLLLMCache | None = None
_install_lock = threading.Lock()


def install_llm_cache(force: bool = False) -> TTLLLMCache | None:
    """Register the cache with LangChain. Idempotent; returns the cache or ``None`` when disabled."""
    global _installed

    if not settings.enable_llm_cache:
        logger.info("llm cache disabled (ENABLE_LLM_CACHE=false)")

        return None

    with _install_lock:
        if _installed is not None and not force:
            return _installed

        from langchain_core.globals import set_llm_cache

        _installed = TTLLLMCache()
        set_llm_cache(_installed)

        logger.info(
            "llm cache installed max_entries=%d ttl_s=%.0f",
            _installed._store.max_entries,
            _installed._store.ttl_seconds,
        )

        return _installed


def get_llm_cache() -> TTLLLMCache | None:
    return _installed


def llm_cache_report() -> dict[str, Any]:
    if _installed is None:
        return {"name": "llm", "enabled": False, "entries": 0}

    return {"enabled": True, **_installed.report()}


def clear_llm_cache(reason: str = "") -> int:
    if _installed is None:
        return 0

    count = _installed.size()
    _installed.clear(reason=reason)

    return count


__all__: Sequence[str] = (
    "TTLLLMCache",
    "clear_llm_cache",
    "get_llm_cache",
    "install_llm_cache",
    "llm_cache_report",
)
