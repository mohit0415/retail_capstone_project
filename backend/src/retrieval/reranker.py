from typing import List, Optional

from configs.settings import settings
from src.schemas.models import RetrievedChunk


def rerank(query: str, chunks: List[RetrievedChunk], top_n: Optional[int] = None) -> List[RetrievedChunk]:
    top_n = top_n or settings.rerank_top_n

    if not chunks:
        return []

    if not settings.enable_flashrank_rerank:
        return chunks[:top_n]

    try:
        from flashrank import RerankRequest, Ranker

        ranker = Ranker(model_name=settings.flashrank_model, cache_dir=settings.flashrank_cache_dir or None)

        passages = [{"id": position, "text": chunk.content} for position, chunk in enumerate(chunks)]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))

        ordered = []

        for entry in ranked[:top_n]:
            chunk = chunks[int(entry["id"])]
            chunk.rerank_score = float(entry["score"])
            ordered.append(chunk)

        return ordered or chunks[:top_n]
    except Exception:
        return chunks[:top_n]


def truncate_without_rerank(chunks: List[RetrievedChunk], top_n: Optional[int] = None) -> List[RetrievedChunk]:
    return chunks[: (top_n or settings.rerank_top_n)]
