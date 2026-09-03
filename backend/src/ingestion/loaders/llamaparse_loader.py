import logging
from pathlib import Path

from configs.settings import settings
from src.ingestion.loaders.cleaning import clean_extracted_text

logger = logging.getLogger(__name__)

PARSING_INSTRUCTION = (
    "This is a corporate policy or regulatory document. Preserve every clause number and "
    "section heading exactly as printed, including numbering such as 4.1, A.8.15 or Article 17. "
    "Render tables as markdown tables. Do not summarise, reorder or renumber anything."
)


def load_with_llamaparse(file_path: str) -> str:
    if not settings.llamaparse_api_key:
        raise ValueError(
            "this document carries tables or diagrams and needs LlamaParse, but "
            "LLAMAPARSE_API_KEY is not set. Add a key from https://cloud.llamaindex.ai/ or move "
            "the document to a text-only source. Falling back to the plain text reader would "
            "flatten the tables into prose and every figure in them would be lost silently."
        )

    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=settings.llamaparse_api_key,
        result_type="markdown",
        parsing_instruction=PARSING_INSTRUCTION,
        verbose=False,
        num_workers=1,
    )

    documents = parser.load_data(file_path)

    text = "\n\n".join(clean_extracted_text(document.text) for document in documents if document.text)

    if not text.strip():
        raise ValueError(
            f"LlamaParse returned no extractable content for {Path(file_path).name}. "
            f"The file may be empty, image-only without OCR, or corrupted."
        )

    logger.info("LlamaParse returned %s character(s) for %s", len(text), Path(file_path).name)

    return text
