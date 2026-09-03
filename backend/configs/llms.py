from enum import Enum
from functools import lru_cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from configs.settings import settings


class ModelTier(str, Enum):
    SMALL = "small"
    STRONG = "strong"


TIER_FOR_NODE = {
    "query_rewrite": ModelTier.SMALL,
    "intent_classification": ModelTier.SMALL,
    "entity_resolution": ModelTier.SMALL,
    "risk_l2_classifier": ModelTier.SMALL,
    "planner": ModelTier.SMALL,
    "rag_generate": ModelTier.STRONG,
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


@lru_cache
def _build_chat_model(deployment: str, temperature: float) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=deployment,
        temperature=temperature,
        timeout=30,
        max_retries=2,
    )


def model_for(node_name: str) -> AzureChatOpenAI:
    tier = TIER_FOR_NODE.get(node_name, ModelTier.SMALL)
    deployment = (
        settings.azure_openai_strong_deployment
        if tier is ModelTier.STRONG
        else settings.azure_openai_small_deployment
    )
    temperature = TEMPERATURE_FOR_NODE.get(node_name, 0.0)

    return _build_chat_model(deployment, temperature)


@lru_cache
def get_embedding_model() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
        dimensions=settings.embedding_dimensions,
    )
