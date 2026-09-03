import hashlib
import logging
from pathlib import Path
from typing import List

from src.ingestion.elements import RawImage
from src.ingestion.extractors.figure_detection import RENDER_PADDING, detect_figures
from src.ingestion.extractors.pdf_backend import open_pdf

logger = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 4096

FIGURE_ZOOM = 2.0

KIND_RASTER = "raster"

KIND_FIGURE = "figure"


def _embedded_rasters(pymupdf, document, page, page_number, output_dir, seen) -> List[RawImage]:
    found: List[RawImage] = []

    for image_index, image in enumerate(page.get_images(full=True)):
        try:
            pixmap = pymupdf.Pixmap(document, image[0])

            if pixmap.n - pixmap.alpha >= 4:
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)

            data = pixmap.tobytes("png")
        except Exception as exc:
            logger.debug("skipped an embedded image on page %s: %s", page_number, exc)
            continue

        if len(data) < MIN_IMAGE_BYTES:
            continue

        fingerprint = hashlib.sha256(data).hexdigest()

        if fingerprint in seen:
            continue

        seen.add(fingerprint)

        target = Path(output_dir) / f"page{page_number}_img{image_index}.png"
        target.write_bytes(data)

        found.append(
            RawImage(path=str(target), page=page_number, image_index=image_index, kind=KIND_RASTER)
        )

    return found


def _rendered_figures(pymupdf, page, page_number, output_dir) -> List[RawImage]:
    found: List[RawImage] = []

    for figure_index, box in enumerate(detect_figures(page)):
        region = (box + (-RENDER_PADDING, -RENDER_PADDING, RENDER_PADDING, RENDER_PADDING)) & page.rect

        try:
            pixmap = page.get_pixmap(clip=region, matrix=pymupdf.Matrix(FIGURE_ZOOM, FIGURE_ZOOM))
            data = pixmap.tobytes("png")
        except Exception as exc:
            logger.warning("could not render a figure on page %s: %s", page_number, exc)
            continue

        target = Path(output_dir) / f"page{page_number}_fig{figure_index}.png"
        target.write_bytes(data)

        found.append(
            RawImage(path=str(target), page=page_number, image_index=figure_index, kind=KIND_FIGURE)
        )

    return found


def extract_images_from_pdf(file_path: str, output_dir: str) -> List[RawImage]:
    pymupdf, document = open_pdf(file_path)

    if document is None:
        return []

    extracted: List[RawImage] = []
    seen: set[str] = set()

    try:
        for number in range(len(document)):
            page = document[number]

            extracted.extend(
                _embedded_rasters(pymupdf, document, page, number + 1, output_dir, seen)
            )
            extracted.extend(_rendered_figures(pymupdf, page, number + 1, output_dir))
    finally:
        document.close()

    rasters = sum(1 for image in extracted if image.kind == KIND_RASTER)
    figures = len(extracted) - rasters

    logger.info(
        "%s yielded %s embedded image(s) and %s drawn figure(s)",
        Path(file_path).name,
        rasters,
        figures,
    )

    return extracted
