import html
import re

from src.ingestion.elements import ClauseSection

CLAUSE_HEADING = re.compile(
    r"^(#{1,4})\s*(?:(?:§|Section|Clause|Article)\s*)?"
    r"([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)+)"
    r"\.?\s+(.+)$",
    re.M,
)

UNNUMBERED_HEADING = "General"

UNNUMBERED_CLAUSE = "1"


def clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def extract_clauses(markdown: str) -> list[ClauseSection]:
    text = markdown or ""

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
