import logging

from src.ingestion.extractors.pdf_backend import load_pymupdf, open_pdf

logger = logging.getLogger(__name__)

MERGE_GAP = 44.0

MIN_SHAPES = 5

MIN_SOLID_SHAPES = 3

MIN_WIDTH = 120.0

MIN_HEIGHT = 26.0

MIN_AREA = 9000.0

PAGE_COVERAGE_LIMIT = 0.6

TABLE_SLACK = 3.0

SOLID_SIDE = 4.0

LABEL_MAX_WIDTH = 200.0

LABEL_MAX_HEIGHT = 44.0

RENDER_PADDING = 12.0


def _table_boxes(pymupdf, page) -> list:
    try:
        return [pymupdf.Rect(table.bbox) for table in page.find_tables().tables]
    except Exception:
        logger.debug("table-box detection failed on a page", exc_info=True)

        return []


def _sits_in_a_table(pymupdf, rect, tables) -> bool:
    for table in tables:
        grown = pymupdf.Rect(table) + (-TABLE_SLACK, -TABLE_SLACK, TABLE_SLACK, TABLE_SLACK)

        if grown.contains(rect):
            return True

    return False


def _shape_rects(pymupdf, page) -> list:
    tables = _table_boxes(pymupdf, page)
    page_area = page.rect.get_area()

    rects = []

    for drawing in page.get_drawings():
        rect = pymupdf.Rect(drawing["rect"])

        if rect.width > 0 and rect.height > 0 and rect.get_area() > PAGE_COVERAGE_LIMIT * page_area:
            continue

        if _sits_in_a_table(pymupdf, rect, tables):
            continue

        rects.append(rect)

    return rects


def _bounding_box(pymupdf, group):
    box = pymupdf.Rect(group[0])

    for rect in group[1:]:
        box |= rect

    return box


def _merge_neighbours(pymupdf, rects: list, gap: float) -> list:
    groups = [[rect] for rect in rects]
    merging = True

    while merging:
        merging = False
        merged: list = []

        for group in groups:
            probe = _bounding_box(pymupdf, group) + (-gap, -gap, gap, gap)
            placed = False

            for existing in merged:
                if probe.intersects(_bounding_box(pymupdf, existing)):
                    existing.extend(group)
                    placed = True
                    merging = True
                    break

            if not placed:
                merged.append(list(group))

        groups = merged

    return groups


def _absorb_labels(pymupdf, page, box):
    grown = pymupdf.Rect(box)
    reach = pymupdf.Rect(box) + (-RENDER_PADDING, -RENDER_PADDING, RENDER_PADDING, RENDER_PADDING)

    for block in page.get_text("blocks"):
        rect = pymupdf.Rect(block[:4])

        if rect.width > LABEL_MAX_WIDTH or rect.height > LABEL_MAX_HEIGHT:
            continue

        if rect.intersects(reach):
            grown |= rect

    return grown


def detect_figures(page) -> list:
    pymupdf = load_pymupdf()

    if pymupdf is None:
        return []

    boxes = []

    for group in _merge_neighbours(pymupdf, _shape_rects(pymupdf, page), MERGE_GAP):
        box = _bounding_box(pymupdf, group)
        solid = [rect for rect in group if rect.width > SOLID_SIDE and rect.height > SOLID_SIDE]

        if len(group) < MIN_SHAPES or len(solid) < MIN_SOLID_SHAPES:
            continue

        if box.width < MIN_WIDTH or box.height < MIN_HEIGHT or box.get_area() < MIN_AREA:
            continue

        boxes.append(_absorb_labels(pymupdf, page, box))

    return boxes


def count_figure_pages(file_path: str) -> int:
    _pymupdf, document = open_pdf(file_path)

    if document is None:
        return 0

    pages = 0

    try:
        for number in range(len(document)):
            if detect_figures(document[number]):
                pages += 1
    finally:
        document.close()

    return pages
