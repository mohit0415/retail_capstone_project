import logging
from pathlib import Path

from src.ingestion.elements import ParseRoute
from src.ingestion.extractors.figure_detection import count_figure_pages
from src.ingestion.metadata.schema import PARSER_LLAMAINDEX, PARSER_LLAMAPARSE

logger = logging.getLogger(__name__)

LAYOUT_BEARING_EXTENSIONS = {".pdf", ".docx"}

PLAIN_TEXT_EXTENSIONS = {".md", ".txt"}


def _probe_pdf(file_path: str) -> tuple[int, int, int]:
    import pdfplumber

    pages_with_tables = 0
    pages_with_rasters = 0

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        for page in pdf.pages:
            if page.extract_tables():
                pages_with_tables += 1

            if getattr(page, "images", None):
                pages_with_rasters += 1

    pages_with_figures = count_figure_pages(file_path)

    return total_pages, pages_with_tables, max(pages_with_rasters, pages_with_figures)


def _probe_docx(file_path: str) -> tuple[int, int, int]:
    from docx import Document as DocxDocument

    document = DocxDocument(file_path)

    shapes = getattr(document, "inline_shapes", None)

    return 1, len(document.tables), len(shapes) if shapes else 0


def choose_route(file_path: str, file_ext: str | None = None) -> ParseRoute:
    extension = (file_ext or Path(file_path).suffix).lower()

    if extension in PLAIN_TEXT_EXTENSIONS:
        return ParseRoute(
            parser=PARSER_LLAMAINDEX,
            reason=f"{extension} is already structured text, a layout parser would add cost and nothing else",
        )

    if extension not in LAYOUT_BEARING_EXTENSIONS:
        return ParseRoute(
            parser=PARSER_LLAMAINDEX,
            reason=f"{extension or 'this file type'} carries no page layout to preserve",
        )

    try:
        units, with_tables, with_images = (
            _probe_pdf(file_path) if extension == ".pdf" else _probe_docx(file_path)
        )
    except Exception as exc:
        logger.warning("layout probe failed for %s, routing to the text reader: %s", file_path, exc)

        return ParseRoute(
            parser=PARSER_LLAMAINDEX,
            reason=f"the layout probe could not read the file ({exc}), so the text reader was used",
        )

    if not with_tables and not with_images:
        return ParseRoute(
            parser=PARSER_LLAMAINDEX,
            reason=f"no table, image or drawn figure found across {units} page(s), the text reader is enough",
        )

    evidence = []

    if with_tables:
        evidence.append(f"{with_tables} page(s) carry a table")

    if with_images:
        evidence.append(f"{with_images} page(s) carry an image or a drawn figure")

    return ParseRoute(
        parser=PARSER_LLAMAPARSE,
        reason=" and ".join(evidence) + ", so the layout has to survive the parse",
        has_tables=bool(with_tables),
        has_images=bool(with_images),
    )


def requires_multimodal_parsing(file_path: str, file_ext: str | None = None) -> bool:
    return choose_route(file_path, file_ext).is_multimodal
