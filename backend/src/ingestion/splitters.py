import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, TextNode

from configs.settings import settings
from src.index.models import get_embed_model

STRUCTURAL_KEYS = [
    "file_name",
    "original_file_name",
    "file_type",
    "file_size_kb",
    "ingested_at",
    "file_hash",
    "parsed_with",
]

CLAUSE_HEADING = re.compile(
    r"^(#{1,4})\s*(?:§\s*)?([0-9]+(?:\.[0-9]+)*|[A-Z]\.[0-9]+(?:\.[0-9]+)*)\s+(.+)$",
    re.M,
)

RECURSIVE_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def build_recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.max_chunk_chars,
        chunk_overlap=settings.chunk_overlap_chars,
        separators=RECURSIVE_SEPARATORS,
        keep_separator=True,
        length_function=len,
    )


def build_semantic_splitter() -> SemanticSplitterNodeParser:
    return SemanticSplitterNodeParser(
        buffer_size=settings.semantic_buffer_size,
        breakpoint_percentile_threshold=settings.semantic_breakpoint_percentile,
        embed_model=get_embed_model(),
    )


def split_by_clause(markdown: str) -> List[tuple[str, str, str]]:
    matches = list(CLAUSE_HEADING.finditer(markdown))

    if not matches:
        return [("general", "1", markdown.strip())]

    sections: List[tuple[str, str, str]] = []

    for position, match in enumerate(matches):
        clause_number = match.group(2)
        heading = match.group(3).strip()
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(markdown)

        body = markdown[start:end].strip()

        if body:
            sections.append((heading, clause_number, body))

    return sections


class ClauseAwareSplitter:
    def __init__(self):
        self.semantic = None
        self.recursive = build_recursive_splitter()

    def _semantic_splitter(self) -> SemanticSplitterNodeParser:
        if self.semantic is None:
            self.semantic = build_semantic_splitter()

        return self.semantic

    def _recursive_fallback(self, text: str, metadata: dict) -> List[BaseNode]:
        pieces = self.recursive.split_text(text)

        return [TextNode(text=piece, metadata=dict(metadata)) for piece in pieces if piece.strip()]

    def split_section(self, text: str, metadata: dict) -> List[BaseNode]:
        if not text or not text.strip():
            return []

        document = Document(text=text, metadata=dict(metadata))

        try:
            nodes = self._semantic_splitter().get_nodes_from_documents([document])
        except Exception:
            nodes = self._recursive_fallback(text, metadata)

        safe: List[BaseNode] = []

        for node in nodes:
            content = node.get_content()

            if len(content) > settings.max_chunk_chars:
                safe.extend(self._recursive_fallback(content, node.metadata))
            else:
                safe.append(node)

        for node in safe:
            node.metadata = {**metadata, **(node.metadata or {})}
            node.metadata.setdefault("content_type", "text")
            node.metadata.setdefault("modality", "text")
            node.excluded_embed_metadata_keys = STRUCTURAL_KEYS
            node.excluded_llm_metadata_keys = STRUCTURAL_KEYS

        return safe

    def split_document(self, markdown: str, base_metadata: dict) -> List[BaseNode]:
        nodes: List[BaseNode] = []

        for section, clause_number, body in split_by_clause(markdown):
            metadata = {
                **base_metadata,
                "section": section,
                "clause_number": clause_number,
                "citation": f"{base_metadata.get('document_title', 'Document')} §{clause_number}",
            }

            nodes.extend(self.split_section(body, metadata))

        return nodes
