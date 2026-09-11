"""Deterministic grounding of a records answer in the rows it was written from.

The rows are structured evidence, so the figures an answer states can be checked mechanically:
every number in the answer must be the row count, a value in a returned row, a count of one
column's values (12 Low, 10 Medium), a number the question itself used, or a date part of the
as-of date. Anything else is a figure the rows do not support, and it is named exactly, so the
rewrite pass knows what to take out. The LLM validator kept judging correct records answers
"stated more firmly than the rows support" without saying which figure; this check says which.
"""

import re
from collections import Counter
from datetime import date, datetime

from configs.settings import settings
from src.schemas.models import SqlEvidence

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")

ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

CITATION = re.compile(r"\[[^\]]*\]")

LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+", re.M)

MAX_REPORTED = 6


def _canonical(raw: str) -> str:
    cleaned = raw.replace(",", "").strip().rstrip(".")

    try:
        value = float(cleaned)
    except ValueError:
        return cleaned

    if value.is_integer():
        return str(int(value))

    return str(value)


def figures_in(text: str) -> set[str]:
    """Every number written in the text, in canonical form (dates and citation markers excluded)."""
    cleaned = CITATION.sub(" ", text or "")
    cleaned = ISO_DATE.sub(" ", cleaned)
    cleaned = LIST_MARKER.sub(" ", cleaned)

    return {_canonical(match.group(0)) for match in NUMBER.finditer(cleaned) if _canonical(match.group(0))}


def _date_parts(value) -> set[str]:
    if isinstance(value, datetime | date):
        return {str(value.year), str(value.month), str(value.day)}

    text = str(value)
    match = ISO_DATE.search(text)

    if match:
        year, month, day = match.group(0).split("-")

        return {year, str(int(month)), str(int(day))}

    return set()


def supported_figures(evidence: SqlEvidence, question: str = "") -> set[str]:
    """Every number a records answer may state without going beyond the rows."""
    allowed: set[str] = {str(evidence.row_count), str(len(evidence.rows)), str(settings.sql_row_limit)}
    allowed |= _date_parts(evidence.as_of)
    allowed |= figures_in(question)

    for value in (evidence.parameters or {}).values():
        allowed |= figures_in(str(value))
        allowed |= _date_parts(value)

    columns: dict[str, Counter] = {}

    for row in evidence.rows:
        for key, value in row.items():
            if isinstance(value, bool) or value is None:
                continue

            if isinstance(value, int | float):
                allowed.add(_canonical(str(value)))
            elif isinstance(value, date | datetime):
                # a row's own date column supports its year (an answer may say "onboarded in
                # 2023") but not its bare month/day: those are 1-31 and 1-12, so nearly any
                # small number in an answer would coincidentally match one row's day-of-month,
                # defeating the check. A full date is written as an ISO date and figures_in()
                # already excludes those from what an answer states as a number.
                allowed.add(str(value.year))
            else:
                allowed |= figures_in(str(value))
                allowed |= _date_parts(value)

            columns.setdefault(key, Counter())[str(value)] += 1

    # "12 Low and 10 Medium" - a count of one column's values is a figure the rows support
    for counter in columns.values():
        allowed |= {str(count) for count in counter.values()}

    return allowed


def unsupported_figures(evidence: SqlEvidence, answer: str, question: str = "") -> list[str]:
    stated = figures_in(answer)
    allowed = supported_figures(evidence, question)

    return sorted(stated - allowed, key=lambda item: (len(item), item))


def figure_defect_text(evidence: SqlEvidence, unsupported: list[str]) -> str:
    shown = ", ".join(unsupported[:MAX_REPORTED]) + (" ..." if len(unsupported) > MAX_REPORTED else "")
    plural = "s" if len(unsupported) > 1 else ""

    return (
        f"the answer states the figure{plural} {shown}, which {'are' if plural else 'is'} neither the row count "
        f"({evidence.row_count}) nor a value or a per-value count in the returned rows"
    )
