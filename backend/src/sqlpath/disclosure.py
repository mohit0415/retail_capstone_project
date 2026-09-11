import json
import re
from dataclasses import dataclass
from datetime import date

from configs.settings import settings
from src.schemas.models import SqlEvidence

EMPTY_RESULT = "empty_result"

ROW_CAP = "row_cap"

SCOPE_FILTER = "scope_filter"

SCOPE_LIMIT = "scope_limit"

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

SCOPE_LIMIT_STATED = re.compile(
    r"\bscope\w*\b|\b(?:your|this|the user's) role\b|\bvisible to\b|\bpermitted to see\b|"
    r"\b(?:low|medium|high|critical)(?:,? (?:and|or|&|/) (?:low|medium|high|critical))+\b",
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

    if evidence.scope_note:
        required.append(
            Disclosure(
                key=SCOPE_LIMIT,
                text=(
                    f"the query only read the rows this role may see ({evidence.scope_note}), so a count "
                    "covers that scope, not every record in the database"
                ),
                stated_when=SCOPE_LIMIT_STATED,
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


def with_disclosures(evidence: SqlEvidence, answer: str) -> tuple[str, list[str]]:
    """The answer with every caveat it left out appended as a note, and the keys that were added.

    The narration prompt asks the model to state each caveat, but a model that forgets one used
    to fail validation with sql_sanity_failure and send a correct answer round a repair loop. The
    caveat text is fixed, so it is added here instead of asking the model again.
    """
    missing = undisclosed(evidence, answer)

    if not missing:
        return answer, []

    lines = [f"- {item.text[0].upper()}{item.text[1:]}." for item in missing]
    note = "Notes on this result:\n" + "\n".join(lines)

    return f"{(answer or '').rstrip()}\n\n{note}", [item.key for item in missing]


def rows_for_prompt(evidence: SqlEvidence, limit: int, with_query: bool = False) -> str:
    """The rows a model reads, headed by how many there are and whether all of them are shown.

    The validator used to get the first 10 rows with no count, saw an answer that named all 31
    vendors, and reported the extra names and the count as unsupported (sql_sanity_failure).

    ``with_query`` also names the SQL that produced the rows. The validator needs it: a query for
    "vendors with open reviews" filters on review_status in its WHERE clause and often does not
    select that column, so from the rows alone "these vendors have open reviews" looks ungrounded.
    """
    shown = evidence.rows[:limit]

    if evidence.row_count <= len(shown):
        header = f"row_count={evidence.row_count}; all {evidence.row_count} rows are listed below"
    else:
        header = (
            f"row_count={evidence.row_count}; only the first {len(shown)} rows are listed below, so count "
            "from row_count and do not treat a row missing from this list as absent"
        )

    if evidence.truncated:
        header += f"; the query hit the {settings.sql_row_limit}-row cap, so row_count is a floor"

    if evidence.scope_note:
        header += f"; scope: {evidence.scope_note}"

    body = json.dumps(shown, default=str, separators=(",", ":"))

    if not with_query:
        return f"{header}\n{body}"

    source = "SQL generated for this question" if evidence.generated else f"vetted query {evidence.template_id}"
    parameters = ", ".join(
        f"{key}={value}" for key, value in (evidence.parameters or {}).items() if key not in ("row_limit",)
    )
    query = f"query: {source}\nSQL: {evidence.statement}"

    if parameters:
        query += f"\nparameters: {parameters}"

    return f"{query}\n{header}\n{body}"
