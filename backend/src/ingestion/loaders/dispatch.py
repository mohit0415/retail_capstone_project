import logging
from pathlib import Path

from src.ingestion.elements import ParsedDocument
from src.ingestion.loaders.llamaindex_loader import load_with_llamaindex
from src.ingestion.loaders.llamaparse_loader import load_with_llamaparse
from src.ingestion.router import choose_route

logger = logging.getLogger(__name__)


def load_document(file_path: str, file_ext: str | None = None) -> ParsedDocument:
    route = choose_route(file_path, file_ext)

    logger.info("%s -> %s: %s", Path(file_path).name, route.parser, route.reason)

    if route.is_multimodal:
        return ParsedDocument(text=load_with_llamaparse(file_path), route=route)

    return ParsedDocument(text=load_with_llamaindex(file_path), route=route)
