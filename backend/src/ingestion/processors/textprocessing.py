import logging
from typing import List

from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode

from configs.settings import settings
from src.ingestion.elements import ClauseSection
from src.ingestion.metadata.stamping import apply_exclusions, text_node_metadata
from src.ingestion.processors.splitters import build_recursive_splitter, build_semantic_splitter

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self):
        self.semantic = None
        self.recursive = build_recursive_splitter()

    def _semantic(self):
        if self.semantic is None:
            self.semantic = build_semantic_splitter()

        return self.semantic

    def _split(self, text: str) -> List[str]:
        document = Document(text=text)

        try:
            nodes = self._semantic().get_nodes_from_documents([document])
            pieces = [node.get_content() for node in nodes]
        except Exception as exc:
            logger.warning("semantic splitting failed, using the recursive splitter: %s", exc)
            pieces = self.recursive.split_text(text)

        bounded: List[str] = []

        for piece in pieces:
            if len(piece) > settings.max_chunk_chars:
                bounded.extend(self.recursive.split_text(piece))
            else:
                bounded.append(piece)

        return [piece for piece in bounded if piece.strip()]

    def process(self, sections: List[ClauseSection], document_metadata: dict) -> List[BaseNode]:
        nodes: List[BaseNode] = []

        for section in sections:
            if not section.body.strip():
                continue

            for chunk_index, piece in enumerate(self._split(section.body)):
                metadata = text_node_metadata(document_metadata, section, chunk_index)

                nodes.append(apply_exclusions(TextNode(text=piece, metadata=metadata)))

        logger.info("built %s text node(s) from %s clause section(s)", len(nodes), len(sections))

        return nodes
