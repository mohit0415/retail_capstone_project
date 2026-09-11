
import logging

from configs.settings import settings
from src.retrieval.postprocessors import with_heading
from src.schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
    top_n = top_n or settings.rerank_top_n

    if not chunks:
        return []

    if not settings.enable_flashrank_rerank:
        return chunks[:top_n]

    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker(model_name=settings.flashrank_model, cache_dir=settings.flashrank_cache_dir or None)

        passages = [
            {"id": position, "text": with_heading(chunk.section, chunk.content)} for position, chunk in enumerate(chunks)
        ]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))

        ordered = []

        for entry in ranked[:top_n]:
            chunk = chunks[int(entry["id"])]
            chunk.rerank_score = float(entry["score"])
            ordered.append(chunk)

        return ordered or chunks[:top_n]
    except Exception as exc:
        logger.warning("flashrank rerank failed (%s); keeping fusion order", exc)

        return chunks[:top_n]


def truncate_without_rerank(chunks: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
    return chunks[: (top_n or settings.rerank_top_n)]
