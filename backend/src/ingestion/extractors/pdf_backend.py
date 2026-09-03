import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_pymupdf():
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        try:
            import fitz

            return fitz
        except ImportError:
            logger.warning("PyMuPDF is not installed, so no image or figure node can be built")

            return None


def open_pdf(file_path: str):
    pymupdf = load_pymupdf()

    if pymupdf is None:
        return None, None

    try:
        return pymupdf, pymupdf.open(file_path)
    except Exception as exc:
        logger.warning("could not open %s with PyMuPDF: %s", Path(file_path).name, exc)

        return pymupdf, None
