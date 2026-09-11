"""Azure OpenAI credentials typed on the frontend login page.

The frontend never keeps the Azure key in its .env: the user types the
endpoint / key / deployments on the login page, logs in with Auth0, and the
frontend then POSTs them to ``/auth/azure`` with the Auth0 bearer token.

This module swaps those values into the running ``settings`` and clears every
cached model client, so the next graph run uses the new credentials. The
rest of the backend (configs/llms.py, src/index/models.py ...) is untouched -
it keeps reading ``settings.azure_openai_*`` exactly as before.
"""

import logging
from urllib.parse import urlparse

from configs.settings import settings

logger = logging.getLogger(__name__)


def _host(endpoint: str) -> str:
    try:
        return urlparse(endpoint).netloc or endpoint
    except Exception:
        return endpoint


def azure_status() -> dict:
    return {
        "azure_configured": settings.azure_configured,
        "endpoint": _host(settings.azure_openai_endpoint),
        "api_version": settings.azure_openai_api_version,
        "small_deployment": settings.azure_openai_small_deployment,
        "strong_deployment": settings.azure_openai_strong_deployment,
        "embedding_deployment": settings.azure_openai_embedding_deployment,
    }


def verify_azure_credentials(
    endpoint: str,
    api_key: str,
    api_version: str,
    small_deployment: str,
    embedding_deployment: str,
) -> tuple[bool, str]:
    """One tiny embedding call + one tiny chat call, so a typo is caught on the login page."""
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=embedding_deployment,
            dimensions=settings.embedding_dimensions,
            timeout=20,
            max_retries=0,
        )
        vector = embeddings.embed_query("retail policy credential check")

        if not vector:
            return False, f"embedding deployment '{embedding_deployment}' returned an empty vector"
    except Exception as exc:
        return False, f"embedding deployment '{embedding_deployment}' was rejected: {exc}"

    try:
        chat = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=small_deployment,
            temperature=0.0,
            max_tokens=1,
            timeout=20,
            max_retries=0,
        )
        chat.invoke("ping")
    except Exception as exc:
        return False, f"chat deployment '{small_deployment}' was rejected: {exc}"

    return True, "embedding and chat deployments answered"


def apply_azure_credentials(
    endpoint: str,
    api_key: str,
    api_version: str,
    small_deployment: str,
    strong_deployment: str,
    embedding_deployment: str,
) -> None:
    """Put the credentials into the live settings and drop every cached model client."""
    settings.azure_openai_endpoint = endpoint
    settings.azure_openai_api_key = api_key
    settings.azure_openai_api_version = api_version
    settings.azure_openai_small_deployment = small_deployment
    settings.azure_openai_strong_deployment = strong_deployment
    settings.azure_openai_embedding_deployment = embedding_deployment

    _clear_model_caches()

    logger.info(
        "azure credentials applied endpoint=%s small=%s strong=%s embedding=%s api_version=%s",
        _host(endpoint),
        small_deployment,
        strong_deployment,
        embedding_deployment,
        api_version,
    )


def _clear_model_caches() -> None:
    """Every place that memoised a client built from the old settings."""
    # LangChain models (configs/llms.py)
    try:
        from configs import llms

        llms._build_chat_model.cache_clear()
        llms.get_embedding_model.cache_clear()
    except Exception:
        logger.debug("langchain model cache could not be cleared", exc_info=True)

    # llama-index models + global Settings (src/index/models.py)
    try:
        from src.index import models as index_models

        index_models._build_llm.cache_clear()
        index_models.get_embed_model.cache_clear()
        index_models._configured = False
        index_models.configure_llama_settings()
    except Exception:
        logger.debug("llama-index model cache could not be cleared", exc_info=True)

    # the vector index holds the embed model it was opened with
    try:
        from src.index.vector_index import reset_index

        reset_index()
    except Exception:
        logger.debug("vector index could not be reset", exc_info=True)

    # the semantic answer cache resolved its embedder lazily from the old key
    try:
        from src.cache.response_cache import get_response_cache

        cache = get_response_cache()
        cache._embedder = None
        cache._embedder_resolved = False
    except Exception:
        logger.debug("response cache embedder could not be reset", exc_info=True)
