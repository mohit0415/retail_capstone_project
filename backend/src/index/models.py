from functools import lru_cache

from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from configs.settings import settings

SMALL_TIER_NODES = {
    "query_rewrite",
    "intent_classification",
    "risk_l2_classifier",
    "planner",
    "reflection",
    "sql_intent_guard",
    "table_summary",
    "metadata_extraction",
}


@lru_cache
def _build_llm(deployment: str, temperature: float) -> AzureOpenAI:
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
    return AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment,
        deployment_name=settings.azure_openai_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        dimensions=settings.embedding_dimensions,
    )


def get_llm(node_name: str = "default", temperature: float = 0.0) -> AzureOpenAI:
    deployment = (
        settings.azure_openai_small_deployment
        if node_name in SMALL_TIER_NODES
        else settings.azure_openai_strong_deployment
    )

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


def active_embed_model_name() -> str:
    return settings.azure_openai_embedding_deployment
