import logging
import re

from src.ingestion.elements import ClauseSection, RawTable

logger = logging.getLogger(__name__)

PIPE_TABLE = re.compile(
    r"^[ \t]{0,3}\|?[^\n|]*\|.*\r?\n"
    r"^[ \t]{0,3}(?=[^\n]*-{2,})(?=[^\n]*\|)[-:|\t ]+\r?\n"
    r"(?:^[ \t]{0,3}\|?[^\n|]*\|.*(?:\r?\n|$))+",
    re.M,
)

HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)

PLACEHOLDER = "[table {index} from this clause is indexed as its own node]"


def _spans(text: str) -> list[tuple[int, int, str]]:
    found = [(match.start(), match.end(), match.group(0).strip()) for match in PIPE_TABLE.finditer(text)]

    found.extend((match.start(), match.end(), match.group(0).strip()) for match in HTML_TABLE.finditer(text))

    found.sort(key=lambda span: span[0])

    kept: list[tuple[int, int, str]] = []
    cursor = 0

    for start, end, block in found:
        if start < cursor:
            continue

        kept.append((start, end, block))
        cursor = end

    return kept


def find_markdown_tables(text: str) -> list[str]:
    return [block for _, _, block in _spans(text or "")]


def has_data_rows(table_markdown: str) -> bool:
    rows = [line for line in table_markdown.splitlines() if line.strip().startswith("|")]

    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]

        if any(cells):
            return True

    return False


def normalize_markdown_table(table_markdown: str) -> str:
    lines = [line for line in table_markdown.splitlines() if line.strip()]

    if len(lines) < 2 or not lines[0].strip().startswith("|"):
        return table_markdown

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    width = len(header)

    if width == 0:
        return table_markdown

    def row(values: list[str]) -> str:
        if len(values) < width:
            values = values + [""] * (width - len(values))
        elif len(values) > width:
            values = values[: width - 1] + [" ".join(values[width - 1 :])]

        return "| " + " | ".join(values) + " |"

    rebuilt = [row(header), "| " + " | ".join(["---"] * width) + " |"]

    for line in lines[2:]:
        rebuilt.append(row(cells(line)))

    return "\n".join(rebuilt)


def extract_tables(sections: list[ClauseSection]) -> tuple[list[RawTable], list[ClauseSection]]:
    tables: list[RawTable] = []
    reduced: list[ClauseSection] = []
    dropped = 0

    for section in sections:
        spans = _spans(section.body)

        if not spans:
            reduced.append(section)
            continue

        pieces = []
        cursor = 0

        for start, end, block in spans:
            pieces.append(section.body[cursor:start])
            cursor = end

            if not has_data_rows(block):
                dropped += 1
                continue

            index = len(tables)

            tables.append(
                RawTable(
                    markdown=normalize_markdown_table(block),
                    table_index=index,
                    heading=section.heading,
                    clause_number=section.clause_number,
                )
            )

            pieces.append("\n" + PLACEHOLDER.format(index=index) + "\n")

        pieces.append(section.body[cursor:])

        reduced.append(
            ClauseSection(
                heading=section.heading,
                clause_number=section.clause_number,
                body="".join(pieces).strip(),
                order=section.order,
            )
        )

    if dropped:
        logger.info("dropped %s header-only table(s), most likely mis-parsed diagrams", dropped)

    if not tables:
        logger.info("no table with data rows was found in the parsed markdown")

    return tables, reduced
