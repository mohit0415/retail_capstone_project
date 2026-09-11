import logging
from functools import lru_cache

from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from configs.settings import settings

logger = logging.getLogger(__name__)

SMALL_TIER_NODES = {
    "query_rewrite",
    "intent_classification",
    "risk_l2_classifier",
    "planner",
    "reflection",
    "sql_intent_guard",
    "nl2sql_generate",
    "table_summary",
    "metadata_extraction",
}


@lru_cache
def _build_llm(deployment: str, temperature: float) -> AzureOpenAI:
    logger.info("llama-index llm built deployment=%s temperature=%s", deployment, temperature)

    return AzureOpenAI(
        model=deployment,
        deployment_name=deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=temperature,
        timeout=60,
    )


@lru_cache
def get_embed_model() -> AzureOpenAIEmbedding:
    logger.info(
        "llama-index embedding model built deployment=%s dimensions=%d",
        settings.azure_openai_embedding_deployment,
        settings.embedding_dimensions,
    )

    return AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment,
        deployment_name=settings.azure_openai_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        dimensions=settings.embedding_dimensions,
    )


def deployment_for_tier(tier: str) -> str:
    return (
        settings.azure_openai_small_deployment if tier == "small" else settings.azure_openai_strong_deployment
    )


def get_llm(node_name: str = "default", temperature: float = 0.0, tier: str | None = None) -> AzureOpenAI:
    """The llama-index LLM for ``node_name``.

    ``tier`` (``"small"`` / ``"strong"``) overrides the fixed table when the
    caller has already routed the call (see ``src.llm_routing.router``).
    """
    if tier is not None:
        deployment = deployment_for_tier(tier)
    else:
        deployment = (
            settings.azure_openai_small_deployment
            if node_name in SMALL_TIER_NODES
            else settings.azure_openai_strong_deployment
        )

    logger.debug("get_llm node=%s tier=%s deployment=%s", node_name, tier or "fixed", deployment)

    return _build_llm(deployment, temperature)


def get_vision_llm() -> AzureOpenAI:
    return _build_llm(settings.azure_openai_vision_deployment, 0.0)


_configured = False


def configure_llama_settings() -> None:
    global _configured

    if _configured:
        return

    Settings.llm = get_llm("default")
    Settings.embed_model = get_embed_model()
    Settings.chunk_size = settings.max_chunk_tokens
    Settings.chunk_overlap = settings.chunk_overlap

    _configured = True

    logger.info(
        "llama-index Settings configured chunk_size=%d chunk_overlap=%d",
        settings.max_chunk_tokens,
        settings.chunk_overlap,
    )


def active_embed_model_name() -> str:
    return settings.azure_openai_embedding_deployment
