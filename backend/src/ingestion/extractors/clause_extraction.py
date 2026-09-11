import html
import re

from src.ingestion.elements import ClauseSection

CLAUSE_HEADING = re.compile(
    r"^(#{1,4})\s*(?:(?:§|Section|Clause|Article)\s*)?"
    r"([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)+)"
    r"\.?\s+(.+)$",
    re.M,
)

# plain-text PDFs (no tables or figures) are read by the text reader, which returns
# "4. Retention Period Standards" with no markdown '#'. Without this the whole
# document became one clause "General §1" and every citation pointed at §1.
PLAIN_NUMBERED_HEADING = re.compile(
    r"^[ \t]*(?:(?:§|Section|Clause|Article)\s*)?([0-9]{1,2}(?:\.[0-9]{1,2})*)\.?[ \t]+([A-Z][^\n]{1,80}?)[ \t]*$",
    re.M,
)

MAX_PLAIN_HEADING_WORDS = 9

MIN_PLAIN_HEADINGS = 2

UNNUMBERED_HEADING = "General"

UNNUMBERED_CLAUSE = "1"


def clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _looks_like_heading(title: str) -> bool:
    title = title.strip()

    if not title or title.endswith((".", ":", ";", ",")):
        return False

    return len(title.split()) <= MAX_PLAIN_HEADING_WORDS


def promote_plain_headings(text: str) -> str:
    """Turn numbered plain-text headings into markdown headings.

    Only used when the text carries no markdown heading at all, and only when at
    least two headings are found, so LlamaParse output and ordinary prose are
    left exactly as they were.
    """
    promoted = 0
    lines = []

    for line in (text or "").splitlines():
        match = PLAIN_NUMBERED_HEADING.match(line)

        if match and _looks_like_heading(match.group(2)):
            depth = min(4, match.group(1).count(".") + 2)
            lines.append(f"{'#' * depth} {match.group(1)} {match.group(2).strip()}")
            promoted += 1
        else:
            lines.append(line)

    if promoted < MIN_PLAIN_HEADINGS:
        return text or ""

    return "\n".join(lines)


def extract_clauses(markdown: str) -> list[ClauseSection]:
    text = markdown or ""

    matches = list(CLAUSE_HEADING.finditer(text))

    if not matches:
        text = promote_plain_headings(text)
        matches = list(CLAUSE_HEADING.finditer(text))

    if not matches:
        body = text.strip()

        if not body:
            return []

        return [ClauseSection(heading=UNNUMBERED_HEADING, clause_number=UNNUMBERED_CLAUSE, body=body, order=0)]

    sections: list[ClauseSection] = []

    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)

        body = text[start:end].strip()

        if not body:
            continue

        sections.append(
            ClauseSection(
                heading=clean_heading(match.group(3)),
                clause_number=match.group(2),
                body=body,
                order=len(sections),
            )
        )

    return sections
