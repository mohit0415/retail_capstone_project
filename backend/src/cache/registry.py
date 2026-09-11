"""One place to report on, and to invalidate, every cache.

``GET /metrics/optimization`` renders :func:`cache_report`; the ingestion
endpoints call :func:`invalidate_corpus_caches` after the vector table
changes, because a cached answer or chunk list built on the old corpus would
otherwise outlive the document that replaced it.
"""

import logging
from typing import Any

from configs.settings import settings
from src.cache.llm_cache import clear_llm_cache, llm_cache_report
from src.cache.response_cache import get_response_cache
from src.cache.retrieval_cache import retrieval_cache

logger = logging.getLogger(__name__)


def cache_report() -> dict[str, Any]:
    response = get_response_cache().report()
    retrieval = retrieval_cache.report()
    llm = llm_cache_report()

    return {
        "enabled": {
            "response": settings.enable_response_cache,
            "semantic": bool(settings.enable_semantic_cache and settings.semantic_cache_threshold < 1.0),
            "retrieval": settings.enable_retrieval_cache,
            "llm": settings.enable_llm_cache,
        },
        "response": response,
        "retrieval": retrieval,
        "llm": llm,
        "totals": {
            "hits": int(response.get("hits", 0)) + int(retrieval.get("hits", 0)) + int(llm.get("hits", 0)),
            "saved_ms": round(
                float(response.get("saved_ms_total", 0.0)) + float(retrieval.get("saved_ms_total", 0.0)), 2
            ),
            "saved_tokens": int(response.get("saved_tokens_total", 0)),
        },
    }


def invalidate_corpus_caches(reason: str) -> dict[str, int]:
    """Drop every entry that depends on the policy corpus. Returns counts per cache."""
    cleared = {
        "response": get_response_cache().invalidate(reason),
        "retrieval": retrieval_cache.invalidate(reason),
        "llm": clear_llm_cache(reason),
    }

    logger.info("caches invalidated reason=%s cleared=%s", reason, cleared)

    return cleared


def invalidate_all(reason: str) -> dict[str, int]:
    return invalidate_corpus_caches(reason)
