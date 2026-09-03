from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser

from configs.settings import settings
from src.index.models import get_embed_model

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
