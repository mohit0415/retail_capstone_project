
import logging
import re
import threading

from src.observability.langfuse_callback import get_langfuse_manager

logger = logging.getLogger(__name__)

PROMPT_REGISTRY: dict[str, str] = {
    "QUERY_REWRITE": "QUERY_REWRITE",
    "INTENT_CLASSIFICATION": "INTENT_CLASSIFICATION",
    "RISK_CLASSIFIER": "RISK_CLASSIFIER",
    "PLANNER": "PLANNER",
    "RAG_ANSWER": "RAG_ANSWER",
    "NL2SQL_INTENT": "NL2SQL_INTENT",
    "SQL_NARRATION": "SQL_NARRATION",
    "HYBRID_ANSWER": "HYBRID_ANSWER",
    "PANEL_POLICY_INTERPRETER": "PANEL_POLICY_INTERPRETER",
    "PANEL_DATA_VERIFIER": "PANEL_DATA_VERIFIER",
    "PANEL_CHALLENGER": "PANEL_CHALLENGER",
    "PANEL_CONSENSUS": "PANEL_CONSENSUS",
    "COMPLIANCE_VALIDATION": "COMPLIANCE_VALIDATION",
    "REFLECTION": "REFLECTION",
    "THREAD_SUMMARY": "THREAD_SUMMARY",
    "SQL_TEMPLATE_SELECTOR": "SQL_TEMPLATE_SELECTOR",
    "PANEL_REPAIR": "PANEL_REPAIR",
    "NL2SQL_GENERATION": "NL2SQL_GENERATION",
}

PROMPT_LABEL = "production"

PROMPT_CACHE_TTL_SECONDS = 300

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

_source: dict[str, str] = {}
_source_lock = threading.Lock()


def placeholders(template: str) -> set[str]:
    """Names of the ``{{...}}`` variables a template expects."""
    return set(_PLACEHOLDER.findall(template))


def compile_template(template: str, **variables) -> str:
    """Substitute ``{{name}}`` placeholders the same way Langfuse does."""

    def _sub(match: re.Match) -> str:
        key = match.group(1)

        if key in variables:
            return str(variables[key])

        return match.group(0)

    return _PLACEHOLDER.sub(_sub, template)


def _client():
    manager = get_langfuse_manager()

    if not manager.enabled:
        return None

    manager.get_callback_handler()

    return manager.client


def _record_source(name: str, source: str) -> None:
    with _source_lock:
        previous = _source.get(name)
        _source[name] = source

    if previous != source:
        logger.info("prompt '%s' resolved from %s", name, source)


def get_prompt(name: str, fallback: str):
    """Fetch a text prompt from Langfuse, or ``None`` when it is not available.

    ``None`` also covers the case where the SDK could only hand back the
    fallback template, so the caller's local compile path is the single
    fallback implementation.
    """
    client = _client()

    if client is None:
        return None

    try:
        prompt = client.get_prompt(
            name,
            label=PROMPT_LABEL,
            fallback=fallback,
            cache_ttl_seconds=PROMPT_CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("langfuse prompt '%s' unavailable, using local template: %s", name, exc)

        return None

    if getattr(prompt, "is_fallback", False):
        return None

    return prompt


def render_prompt(name: str, fallback: str, **variables) -> str:
    """Return the compiled prompt text for ``name``.

    ``fallback`` is the local ``{{...}}`` template used when Langfuse is
    disabled, unreachable, or has no ``production`` prompt by that name.
    """
    prompt = get_prompt(name, fallback)

    if prompt is not None:
        try:
            text = prompt.compile(**variables)
        except Exception as exc:
            logger.warning("langfuse prompt '%s' compile failed, using local template: %s", name, exc)
        else:
            _record_source(name, "langfuse")

            return text

    _record_source(name, "fallback")

    return compile_template(fallback, **variables)


def prompt_sources() -> dict[str, str]:
    """Snapshot of which source each prompt last resolved to."""
    with _source_lock:
        return dict(_source)


def warm_prompt_cache() -> dict[str, str]:
    """Fetch every registered prompt once and log a summary.

    Called at application startup so the first user request does not pay the
    fetch latency, and so the log shows immediately which prompts are served
    from the dashboard and which fall back to ``src.prompts.library``.
    """
    from src.prompts import library

    for name, attr in PROMPT_REGISTRY.items():
        template = getattr(library, attr)
        prompt = get_prompt(name, template)
        _record_source(name, "langfuse" if prompt is not None else "fallback")

    sources = prompt_sources()
    managed = sorted(n for n, s in sources.items() if s == "langfuse")
    local = sorted(n for n, s in sources.items() if s == "fallback")

    logger.info(
        "prompt cache warmed: %d from langfuse, %d from local library%s",
        len(managed),
        len(local),
        f" (local: {', '.join(local)})" if local else "",
    )

    return sources


__all__ = [
    "PROMPT_CACHE_TTL_SECONDS",
    "PROMPT_LABEL",
    "PROMPT_REGISTRY",
    "compile_template",
    "get_prompt",
    "placeholders",
    "prompt_sources",
    "render_prompt",
    "warm_prompt_cache",
]
