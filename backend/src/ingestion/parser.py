import logging
from pathlib import Path

from configs.settings import settings

logger = logging.getLogger(__name__)

MULTIMODAL_EXTENSIONS = {".pdf", ".docx"}

LLAMAPARSE_INSTRUCTION = (
    "This is a corporate policy or regulatory document. Preserve every clause number and "
    "section heading exactly as printed, including numbering such as 4.1, A.8.15 or Article 17. "
    "Render tables as markdown tables. Do not summarise, reorder or renumber anything."
)


def requires_multimodal_parsing(file_path: str, file_ext: str) -> bool:
    extension = (file_ext or Path(file_path).suffix).lower()

    if extension not in MULTIMODAL_EXTENSIONS:
        return False

    try:
        if extension == ".pdf":
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    if page.extract_tables():
                        return True

                    if getattr(page, "images", None):
                        return True

            return False

        from docx import Document as DocxDocument

        document = DocxDocument(file_path)

        if document.tables:
            return True

        shapes = getattr(document, "inline_shapes", None)

        return bool(shapes and len(shapes) > 0)
    except Exception as exc:
        logger.warning("multimodal detection failed for %s, assuming text only: %s", file_path, exc)
        return False


def parse_with_llamaparse(file_path: str) -> str:
    if not settings.llamaparse_api_key:
        raise ValueError(
            "This document carries tables or diagrams and needs LlamaParse, but "
            "LLAMAPARSE_API_KEY is not set. Add a key from https://cloud.llamaindex.ai/ "
            "or move the document to a text-only source."
        )

    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=settings.llamaparse_api_key,
        result_type="markdown",
        parsing_instruction=LLAMAPARSE_INSTRUCTION,
        verbose=False,
        num_workers=1,
    )

    documents = parser.load_data(file_path)

    if not documents or not (documents[0].text or "").strip():
        raise ValueError(
            f"LlamaParse returned no extractable content for {Path(file_path).name}. "
            "The file may be empty, image-only without OCR, or corrupted."
        )

    return "\n\n".join(document.text for document in documents if document.text)


def parse_plain_text(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension in (".md", ".txt"):
        return Path(file_path).read_text(encoding="utf-8")

    from llama_index.core import SimpleDirectoryReader

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    return "\n\n".join(document.text for document in documents if document.text)


def parse_document(file_path: str) -> tuple[str, str]:
    extension = Path(file_path).suffix.lower()

    if requires_multimodal_parsing(file_path, extension):
        logger.info("routing %s through LlamaParse (tables or images detected)", Path(file_path).name)

        return parse_with_llamaparse(file_path), "llamaparse"

    logger.info("parsing %s as plain text (no tables or images detected)", Path(file_path).name)

    return parse_plain_text(file_path), "llamaindex"
