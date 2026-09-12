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
from src.index.embedding_dims import dimensions_for_deployment

logger = logging.getLogger(__name__)


def _host(endpoint: str) -> str:
    try:
        return urlparse(endpoint).netloc or endpoint
    except Exception:
        return endpoint


# where the LlamaParse key in the live settings came from
_llamaparse_source = "env" if settings.llamaparse_api_key else "none"


def azure_status() -> dict:
    return {
        "azure_configured": settings.azure_configured,
        "endpoint": _host(settings.azure_openai_endpoint),
        "api_version": settings.azure_openai_api_version,
        "small_deployment": settings.azure_openai_small_deployment,
        "strong_deployment": settings.azure_openai_strong_deployment,
        "embedding_deployment": settings.azure_openai_embedding_deployment,
        "embedding_dimensions": settings.embedding_dimensions,
        "llamaparse_configured": bool(settings.llamaparse_api_key),
        "llamaparse_source": _llamaparse_source,
    }


def verify_azure_credentials(
    endpoint: str,
    api_key: str,
    api_version: str,
    small_deployment: str,
    embedding_deployment: str,
    embedding_dimensions: int | None = None,
) -> tuple[bool, str]:
    """One tiny embedding call + one tiny chat call, so a typo is caught on the login page."""
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=embedding_deployment,
            dimensions=embedding_dimensions or settings.embedding_dimensions,
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
    embedding_dimensions: int | None = None,
    llamaparse_api_key: str = "",
) -> None:
    """Put the credentials into the live settings and drop every cached model client.

    ``embedding_dimensions`` is derived from the embedding deployment when it is
    not given, so picking 3-small or 3-large on the login page re-dimensions the
    embedder in the same step. ``llamaparse_api_key`` is only overwritten when
    the caller supplies one - blank leaves whatever the .env had.
    """
    global _llamaparse_source

    settings.azure_openai_endpoint = endpoint
    settings.azure_openai_api_key = api_key
    settings.azure_openai_api_version = api_version
    settings.azure_openai_small_deployment = small_deployment
    settings.azure_openai_strong_deployment = strong_deployment
    settings.azure_openai_embedding_deployment = embedding_deployment
    settings.embedding_dimensions = resolve_embedding_dimensions(embedding_deployment, embedding_dimensions)

    if llamaparse_api_key:
        settings.llamaparse_api_key = llamaparse_api_key
        _llamaparse_source = "login"

    _clear_model_caches()

    logger.info(
        "azure credentials applied endpoint=%s small=%s strong=%s embedding=%s dimensions=%d "
        "api_version=%s llamaparse=%s",
        _host(endpoint),
        small_deployment,
        strong_deployment,
        embedding_deployment,
        settings.embedding_dimensions,
        api_version,
        _llamaparse_source,
    )


def resolve_embedding_dimensions(embedding_deployment: str, explicit: int | None = None) -> int:
    """An explicit width from the caller wins; otherwise derive it from the name."""
    if explicit and explicit > 0:
        return int(explicit)

    return dimensions_for_deployment(embedding_deployment, settings.embedding_dimensions)


def corpus_dimension_warning(embedding_dimensions: int) -> str:
    """Message to show when the chosen embedding does not fit the ingested corpus.

    The pgvector column is created at a fixed width, so switching the embedding
    model under an existing corpus makes every search fail on a dimension error.
    Reported on the login response rather than blocked: an empty corpus, or one
    about to be re-ingested, is a legitimate reason to switch.
    """
    from src.index.vector_index import table_embedding_dimensions

    existing = table_embedding_dimensions()

    if existing is None or existing == embedding_dimensions:
        return ""

    return (
        f"the ingested corpus holds {existing}-wide vectors but this embedding deployment "
        f"produces {embedding_dimensions}. Retrieval will fail until the corpus is re-ingested "
        f"with this model (or the previous embedding deployment is used again)."
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
