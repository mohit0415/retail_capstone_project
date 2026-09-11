import contextlib
import logging
from typing import Any

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from configs.settings import settings

logger = logging.getLogger(__name__)

MEDIA_CONTENT_TYPES = {"image_caption", "table_summary"}

# headings that name a kind of extract rather than its subject; they tell the reranker nothing
UNINFORMATIVE_HEADINGS = {"", "general", "figures", "untitled"}


def with_heading(heading: str | None, content: str) -> str:
    """The text the cross-encoder reads for one extract: its section heading, then its text.

    The embeddings carry the section in their metadata, so fusion ranks "§7.1 Investigation
    Principles" fourth for "tell me about the investigation principles"; its text alone is the bullet
    list "- Confidentiality - Fairness and neutrality - ...", and the reranker - which read only that
    - scored it 0.00 and dropped it from the six extracts the writer sees. The writer then said the
    policy does not detail investigation principles, and the answer escalated.
    """
    heading = (heading or "").strip()
    content = content or ""

    if heading.lower() in UNINFORMATIVE_HEADINGS or heading.lower() in content[:200].lower():
        return content

    return f"{heading}\n{content}"


class KeepTopN(BaseNodePostprocessor):
    top_n: int = 6
    max_media: int = 3

    @classmethod
    def class_name(cls) -> str:
        return "KeepTopN"

    @staticmethod
    def _is_media(scored: NodeWithScore) -> bool:
        node = getattr(scored, "node", scored)
        metadata = getattr(node, "metadata", {}) or {}

        return metadata.get("content_type") in MEDIA_CONTENT_TYPES or bool(metadata.get("image_path"))

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return nodes

        kept = nodes[: self.top_n]
        kept_ids = {id(item) for item in kept}

        media_tail = [
            item for item in nodes[self.top_n :] if self._is_media(item) and id(item) not in kept_ids
        ][: self.max_media]

        return kept + media_tail


class FlashRankRerank(BaseNodePostprocessor):
    ranker: Any = None
    top_n: int = 6

    @classmethod
    def class_name(cls) -> str:
        return "FlashRankRerank"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or self.ranker is None:
            return nodes[: self.top_n]

        query = query_bundle.query_str if query_bundle is not None else ""

        try:
            from flashrank import RerankRequest

            passages = []

            for position, scored in enumerate(nodes):
                node = getattr(scored, "node", scored)

                try:
                    content = node.get_content()
                except Exception:
                    logger.debug("node.get_content() failed, falling back to str(node)", exc_info=True)
                    content = str(node)

                metadata = getattr(node, "metadata", {}) or {}

                passages.append({"id": position, "text": with_heading(metadata.get("section"), content)})

            ranked = self.ranker.rerank(RerankRequest(query=query, passages=passages))

            ordered = []

            for entry in ranked[: self.top_n]:
                scored = nodes[int(entry["id"])]

                with contextlib.suppress(Exception):
                    scored.score = float(entry["score"])

                ordered.append(scored)

            return ordered or nodes[: self.top_n]
        except Exception as exc:
            logger.warning("FlashRank rerank failed (%s); keeping fusion order", exc)
            return nodes[: self.top_n]


class CurrentVersionFilter(BaseNodePostprocessor):
    as_of: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "CurrentVersionFilter"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        cutoff = self.as_of or str(settings.as_of_date)
        surviving = []

        for scored in nodes:
            node = getattr(scored, "node", scored)
            metadata = getattr(node, "metadata", {}) or {}

            if metadata.get("is_current") is False:
                continue

            effective = metadata.get("effective_date")

            if effective and str(effective) > cutoff:
                continue

            surviving.append(scored)

        return surviving


_flash_ranker = None


def build_reranker(top_n: int) -> BaseNodePostprocessor:
    global _flash_ranker

    if not settings.enable_flashrank_rerank:
        return KeepTopN(top_n=top_n)

    if _flash_ranker is None:
        try:
            from flashrank import Ranker

            cache_dir = settings.flashrank_cache_dir or None
            _flash_ranker = Ranker(model_name=settings.flashrank_model, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("FlashRank unavailable (%s); using deterministic KeepTopN", exc)
            _flash_ranker = False

    if not _flash_ranker:
        return KeepTopN(top_n=top_n)

    return FlashRankRerank(ranker=_flash_ranker, top_n=top_n)
