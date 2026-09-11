

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI

from configs.settings import settings

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    SMALL = "small"
    STRONG = "strong"


TIER_FOR_NODE = {
    "query_rewrite": ModelTier.SMALL,
    "thread_summary": ModelTier.SMALL,
    "intent_classification": ModelTier.SMALL,
    "entity_resolution": ModelTier.SMALL,
    "risk_l2_classifier": ModelTier.SMALL,
    "planner": ModelTier.SMALL,
    "rag_generate": ModelTier.STRONG,
    "sql_template_selector": ModelTier.SMALL,
    "nl2sql_intent": ModelTier.SMALL,
    "sql_narration": ModelTier.SMALL,
    "hybrid_generate": ModelTier.STRONG,
    "agentic_rag": ModelTier.STRONG,
    "panel_policy_interpreter": ModelTier.STRONG,
    "panel_data_verifier": ModelTier.STRONG,
    "panel_challenger": ModelTier.STRONG,
    "panel_consensus": ModelTier.STRONG,
    "compliance_validation": ModelTier.STRONG,
    "reflection": ModelTier.SMALL,
}

TEMPERATURE_FOR_NODE = {
    "panel_challenger": 0.4,
    "rag_generate": 0.1,
    "hybrid_generate": 0.1,
}

CACHEABLE_MAX_TEMPERATURE = 0.0


@dataclass(frozen=True)
class ProviderSpec:
    """Where one tier is served from. Hashable so it can key the model cache."""

    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_version: str = ""
    timeout: int = 30
    max_retries: int = 2


_VALID_PROVIDERS = {"azure", "openai", "ollama"}


def _resolve_env(value) -> str:
    """``env:VAR`` in the YAML reads ``VAR`` from the environment."""
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[4:], "")

    return "" if value is None else str(value)


def _assert_ollama_model_available(spec: "ProviderSpec", where: str) -> None:
    """Fail fast if the Ollama tag in the YAML is not pulled locally.

    Ollama otherwise returns a 404 on the first chat call, deep inside a graph
    node, which is a confusing place to learn about a typo in the YAML. Only
    runs on the local YAML path, never in deployment.
    """
    import httpx

    tags_url = spec.base_url.rstrip("/").removesuffix("/v1") + "/api/tags"

    try:
        response = httpx.get(tags_url, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"model_routing.yaml {where}: cannot reach Ollama at {tags_url} ({exc}). "
            "Start it with `ollama serve`, or set USE_MODEL_ROUTING_YAML=false to use Azure."
        ) from exc

    available = {m.get("name", "") for m in response.json().get("models", [])}
    wanted = spec.model if ":" in spec.model else f"{spec.model}:latest"

    if wanted not in available:
        raise RuntimeError(
            f"model_routing.yaml {where}: Ollama model {spec.model!r} is not pulled. "
            f"Available: {sorted(available) or 'none'}. Run `ollama pull {spec.model}` or fix the tag."
        )


@lru_cache
def _yaml_tier_specs() -> dict[ModelTier, ProviderSpec]:
    """Parse configs/model_routing.yaml once. Only called when the flag is on."""
    import yaml

    path = Path(settings.model_routing_yaml_path)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    tiers = raw.get("tiers") or {}
    specs: dict[ModelTier, ProviderSpec] = {}

    for tier in ModelTier:
        entry = tiers.get(tier.value)

        if not entry:
            raise ValueError(f"{path}: missing 'tiers.{tier.value}'")

        provider = str(entry.get("provider", "azure")).lower()

        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"{path}: tiers.{tier.value}.provider={provider!r} must be one of {sorted(_VALID_PROVIDERS)}"
            )

        model = _resolve_env(entry.get("model"))

        if not model:
            raise ValueError(f"{path}: tiers.{tier.value}.model is required")

        if provider == "azure":
            spec = ProviderSpec(
                provider="azure",
                model=model,
                base_url=_resolve_env(entry.get("endpoint")) or settings.azure_openai_endpoint,
                api_key=_resolve_env(entry.get("api_key")) or settings.azure_openai_api_key,
                api_version=_resolve_env(entry.get("api_version")) or settings.azure_openai_api_version,
                timeout=int(entry.get("timeout", 30)),
                max_retries=int(entry.get("max_retries", 2)),
            )
        else:
            base_url = _resolve_env(entry.get("base_url"))

            if provider == "ollama" and not base_url:
                base_url = "http://localhost:11434/v1"

            spec = ProviderSpec(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=_resolve_env(entry.get("api_key")) or "ollama",
                timeout=int(entry.get("timeout", 60)),
                max_retries=int(entry.get("max_retries", 1)),
            )

        if spec.provider == "ollama":
            _assert_ollama_model_available(spec, f"tiers.{tier.value}")

        specs[tier] = spec
        logger.info(
            "model_routing.yaml tier=%s provider=%s model=%s base_url=%s",
            tier.value,
            spec.provider,
            spec.model,
            spec.base_url or "-",
        )

    return specs


def provider_spec_for(tier: ModelTier | str) -> ProviderSpec:
    tier = ModelTier(tier)

    if settings.use_model_routing_yaml:
        return _yaml_tier_specs()[tier]

    # a hung call used to wait the full 30s before its retry (seen as 33-39s calls in the log),
    # so the timeout and retry count now come from settings
    timeout = int(max(5, settings.llm_timeout_seconds))
    retries = max(0, settings.llm_max_retries)

    if settings.llm_gateway_url:
        return ProviderSpec(
            provider="openai",
            model=(
                settings.llm_gateway_strong_model
                if tier is ModelTier.STRONG
                else settings.llm_gateway_small_model
            ),
            base_url=settings.llm_gateway_url,
            api_key=settings.llm_gateway_api_key or "dummy-key",
            timeout=timeout,
            max_retries=retries,
        )

    return ProviderSpec(
        provider="azure",
        model=(
            settings.azure_openai_strong_deployment
            if tier is ModelTier.STRONG
            else settings.azure_openai_small_deployment
        ),
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        timeout=timeout,
        max_retries=retries,
    )


def provider_source() -> str:
    """Which of the three resolution paths is active: yaml | gateway | azure."""
    if settings.use_model_routing_yaml:
        return "yaml"

    if settings.llm_gateway_url:
        return "gateway"

    return "azure"


def deployment_for(tier: ModelTier | str) -> str:
    """Model / deployment name for ``tier`` under the active provider."""
    return provider_spec_for(tier).model


def tier_for(node_name: str) -> ModelTier:
    return TIER_FOR_NODE.get(node_name, ModelTier.SMALL)


def _cache_flag(temperature: float):
    """``None`` lets the model use the global LLM cache; ``False`` opts out."""
    if not settings.enable_llm_cache:
        return False

    return None if temperature <= CACHEABLE_MAX_TEMPERATURE else False


@lru_cache
def _build_chat_model(spec: ProviderSpec, temperature: float) -> BaseChatModel:
    cache = _cache_flag(temperature)

    logger.info(
        "chat model built provider=%s model=%s base_url=%s temperature=%s cache=%s",
        spec.provider,
        spec.model,
        spec.base_url or "-",
        temperature,
        "global" if cache is None else "off",
    )

    if spec.provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=spec.base_url,
            api_key=spec.api_key,
            api_version=spec.api_version,
            azure_deployment=spec.model,
            temperature=temperature,
            timeout=spec.timeout,
            max_retries=spec.max_retries,
            cache=cache,
        )

    return ChatOpenAI(
        base_url=spec.base_url or None,
        api_key=spec.api_key or "dummy-key",
        model=spec.model,
        temperature=temperature,
        timeout=spec.timeout,
        max_retries=spec.max_retries,
        cache=cache,
    )


def model_for_tier(tier: ModelTier | str, temperature: float = 0.0) -> BaseChatModel:
    return _build_chat_model(provider_spec_for(tier), temperature)


def model_for(node_name: str) -> BaseChatModel:
    """The fixed-tier model for ``node_name`` (the TIER_FOR_NODE table)."""
    tier = tier_for(node_name)
    temperature = TEMPERATURE_FOR_NODE.get(node_name, 0.0)
    spec = provider_spec_for(tier)

    logger.info(
        "model selected node=%s tier=%s provider=%s model=%s source=%s",
        node_name,
        tier.value,
        spec.provider,
        spec.model,
        provider_source(),
    )

    return _build_chat_model(spec, temperature)


def routed_model(node_name: str, state: dict | None = None):
    """Cost/latency-aware model for a routable node.

    Returns ``(model, decision)`` where ``decision`` is a
    :class:`src.llm_routing.router.RoutingDecision`; the node should append
    ``decision.as_dict()`` to the ``model_routing`` state channel. Nodes that
    are not routable get their fixed tier, so this is safe to call anywhere.
    """
    from src.llm_routing.router import route_tier

    decision = route_tier(node_name, state)
    temperature = TEMPERATURE_FOR_NODE.get(node_name, 0.0)
    spec = provider_spec_for(decision.tier)

    logger.info(
        "model selected node=%s tier=%s provider=%s model=%s source=%s strategy=%s reason=%s",
        node_name,
        decision.tier,
        spec.provider,
        spec.model,
        provider_source(),
        decision.strategy,
        decision.reason,
    )

    model = _build_chat_model(spec, temperature)

    return model, decision


@lru_cache
def get_embedding_model() -> AzureOpenAIEmbeddings:
    logger.info(
        "embedding model built deployment=%s dimensions=%d",
        settings.azure_openai_embedding_deployment,
        settings.embedding_dimensions,
    )

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
        dimensions=settings.embedding_dimensions,
    )
