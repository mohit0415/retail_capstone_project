import logging
from pathlib import Path

from src.ingestion.loaders.cleaning import assert_not_raw_bytes, clean_extracted_text

logger = logging.getLogger(__name__)

DIRECT_READ_EXTENSIONS = {".md", ".txt"}


def load_with_llamaindex(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() in DIRECT_READ_EXTENSIONS:
        return clean_extracted_text(path.read_text(encoding="utf-8"))

    from llama_index.core import SimpleDirectoryReader

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    text = "\n\n".join(clean_extracted_text(document.text) for document in documents if document.text)

    assert_not_raw_bytes(text, path.name)

    if not text.strip():
        raise ValueError(
            f"the text reader found no readable text in {path.name}. A scanned or image-only "
            f"document has no text layer, so it needs OCR before it can be indexed."
        )

    logger.info("the text reader returned %s character(s) for %s", len(text), path.name)

    return text
