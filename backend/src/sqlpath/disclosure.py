import re
from dataclasses import dataclass
from datetime import date

from configs.settings import settings
from src.schemas.models import SqlEvidence

EMPTY_RESULT = "empty_result"

ROW_CAP = "row_cap"

SCOPE_FILTER = "scope_filter"

GENERATED_QUERY = "generated_query"

EMPTY_RESULT_STATED = re.compile(
    r"\bno (?:rows|records|matching|results|vendors|findings|entries|reviews)\b|\bfound nothing\b|"
    r"\breturned nothing\b|\bzero (?:rows|records|results)\b|\bempty result\b|"
    r"\bdid not (?:return|find|match)\b|\bnone (?:were|was) (?:found|returned)\b",
    re.I,
)

ROW_CAP_STATED = re.compile(
    r"\brow cap\b|\bcapped\b|\btruncat\w*\b|\bat least\b|\bis a floor\b|\bfirst \d+ rows\b",
    re.I,
)

SCOPE_FILTER_STATED = re.compile(
    r"\bscope\b|\bfiltered\b|\bpartial\b|\bat least\b|\bis a floor\b|\brestricted\b|"
    r"\bnot visible to (?:this|your) role\b",
    re.I,
)

GENERATED_QUERY_STATED = re.compile(
    r"\bgenerated\b|\bnot (?:a )?(?:reviewed|vetted)\b|\bad[- ]hoc\b|\bwritten for this question\b|"
    r"\bno reviewed query\b|\bunreviewed\b",
    re.I,
)


@dataclass(frozen=True)
class Disclosure:
    key: str
    text: str
    stated_when: re.Pattern


def disclosures(evidence: SqlEvidence) -> list[Disclosure]:
    required: list[Disclosure] = []

    if evidence.generated:
        required.append(
            Disclosure(
                key=GENERATED_QUERY,
                text=(
                    "no reviewed query in the catalogue answered this question, so the SQL was "
                    "generated for it; the result is safe but it has not been reviewed for "
                    "correctness the way a vetted query has"
                ),
                stated_when=GENERATED_QUERY_STATED,
            )
        )

    if evidence.row_count == 0:
        required.append(
            Disclosure(
                key=EMPTY_RESULT,
                text="the query returned no rows, and an empty result is not evidence of compliance",
                stated_when=EMPTY_RESULT_STATED,
            )
        )

    if evidence.truncated:
        required.append(
            Disclosure(
                key=ROW_CAP,
                text=(
                    f"the result hit the {settings.sql_row_limit}-row cap, so any count stated from "
                    "it is a floor"
                ),
                stated_when=ROW_CAP_STATED,
            )
        )

    if evidence.rows_filtered_by_scope:
        required.append(
            Disclosure(
                key=SCOPE_FILTER,
                text=(
                    f"{evidence.rows_filtered_by_scope} row(s) were removed by this role's department "
                    "or risk-category scope, so the result is partial and any count from it is a floor"
                ),
                stated_when=SCOPE_FILTER_STATED,
            )
        )

    return required


OBSERVED_DATE_COLUMNS = frozenset(
    {
        "onboarding_date",
        "last_audit_date",
        "last_review_date",
        "review_date",
        "issue_identified_date",
        "resolution_date",
    }
)


def data_defects(evidence: SqlEvidence) -> list[str]:
    problems: list[str] = []

    for row in evidence.rows[:20]:
        for key, value in row.items():
            if key not in OBSERVED_DATE_COLUMNS:
                continue

            if isinstance(value, date) and value > evidence.as_of:
                problems.append(
                    f"row records {key}={value}, which is after the pinned as_of date "
                    f"{evidence.as_of}; a thing that already happened cannot be dated in the future"
                )

    return problems


def undisclosed(evidence: SqlEvidence, answer: str) -> list[Disclosure]:
    text = answer or ""

    return [item for item in disclosures(evidence) if not item.stated_when.search(text)]


def caveats(evidence: SqlEvidence) -> list[str]:
    return [item.text for item in disclosures(evidence)] + data_defects(evidence)
