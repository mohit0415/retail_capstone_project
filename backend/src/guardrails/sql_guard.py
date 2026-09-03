import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

SQL_WRITE_STATEMENT = re.compile(
    r"\b(insert\s+into|delete\s+from|drop\s+(table|schema|database)|truncate\s+table|"
    r"update\s+\S+\s+set|alter\s+table|create\s+(table|schema|index)|grant\s+|revoke\s+|"
    r"copy\s+\S+\s+(from|to))\b",
    re.I,
)

MUTATION_VERBS = r"(insert|delete|remove|drop|update|modify|alter|overwrite|truncate|purge|wipe|erase|add|create)"

DB_NOUNS = r"(record|records|row|rows|entry|entries|table|tables|database|databases|db|column|columns|schema)"

MUTATION_INTENT = re.compile(
    rf"\b{MUTATION_VERBS}\b(?:\s+\S+){{0,3}}\s+\b{DB_NOUNS}\b"
    rf"|\b{DB_NOUNS}\b(?:\s+\S+){{0,3}}\s+\b{MUTATION_VERBS}\b",
    re.I,
)

STATUS_MUTATION_VERBS = r"(mark|set|flag|close|reopen|approve|reject|assign|update|change|override|waive)"

DOMAIN_NOUNS = (
    r"(vendor|vendors|finding|findings|issue|issues|review|reviews|status|remediation|"
    r"approval|hold|score|category)"
)

STATUS_MUTATION_INTENT = re.compile(
    rf"\b{STATUS_MUTATION_VERBS}\b(?:\s+\S+){{0,4}}\s+\b{DOMAIN_NOUNS}\b"
    rf"|\bmark\b(?:\s+\S+){{0,3}}\s+\bas\b",
    re.I,
)

INTERROGATIVE_OPENER = re.compile(
    r"^\W*(what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|"
    r"can|could|should|would|will|has|have|list|show|find|give|tell|explain|summarise|summarize)\b",
    re.I,
)

STACKED_STATEMENT = re.compile(r";\s*\S|--|/\*|\bunion\s+all\s+select\b|\bunion\s+select\b", re.I)

CLOCK_READ = re.compile(r"\b(CURRENT_DATE|CURRENT_TIMESTAMP|NOW\s*\(\s*\)|LOCALTIMESTAMP)\b", re.I)

BLOCKED_INTENTS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
}

INTENT_PROMPT = """You are a strict SQL-intent classifier sitting in front of a retail compliance database.

Read the natural-language question and decide which single SQL operation it is really asking the
database to perform, however it is phrased.

Respond with EXACTLY ONE WORD from this list and nothing else:
SELECT, INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, OTHER

Guidance:
- Looking up, retrieving, counting, comparing or reading data -> SELECT
- Adding rows however phrased ("log a new review", "record that we disposed of it") -> INSERT
- Changing existing data however phrased ("mark it compliant", "fix the end date") -> UPDATE
- Removing rows however phrased ("purge", "get rid of", "clear out") -> DELETE
- Removing a table or schema -> DROP
- Changing table structure -> ALTER
- Emptying a table -> TRUNCATE
- Creating a table or schema -> CREATE
- Changing permissions -> GRANT or REVOKE
- Anything else or unclear -> OTHER

Question: {query}

One-word classification:"""


def validate_question_intent(question: str) -> Tuple[bool, str]:
    if not question or not question.strip():
        return False, "the question is empty"

    if SQL_WRITE_STATEMENT.search(question):
        return False, (
            "Blocked: the question contains a database write statement. "
            "This tool answers read-only questions about compliance records."
        )

    if MUTATION_INTENT.search(question):
        return False, (
            "Blocked: the question asks to add, change or remove data. "
            "This tool answers read-only questions about compliance records."
        )

    if not INTERROGATIVE_OPENER.match(question) and STATUS_MUTATION_INTENT.search(question):
        return False, (
            "Blocked: this is an instruction to change a record's status rather than a question "
            "about it. This tool answers read-only questions about compliance records."
        )

    return True, ""


def classify_intent_with_llm(llm, question: str) -> Tuple[bool, str]:
    if not question or not question.strip():
        return True, ""

    try:
        raw = str(llm.complete(INTENT_PROMPT.format(query=question))).strip().upper()
    except Exception as exc:
        logger.warning("SQL intent classification failed, blocking the call: %s", exc)
        return False, f"Blocked: the safety classifier could not verify this question ({exc})."

    for candidate in BLOCKED_INTENTS:
        if re.search(rf"\b{candidate}\b", raw):
            return False, (
                f"Blocked: the question was classified as a {candidate} operation. "
                "Only read-only lookups are permitted."
            )

    return True, ""


def validate_generated_sql(statement: str) -> Tuple[bool, str]:
    if not statement or not statement.strip():
        return False, "the engine produced no SQL"

    normalised = " ".join(statement.split())

    if not re.match(r"^\s*(SELECT|WITH)\b", normalised, re.I):
        return False, "generated SQL does not begin with SELECT or WITH"

    if SQL_WRITE_STATEMENT.search(normalised):
        return False, "generated SQL contains a write verb"

    if STACKED_STATEMENT.search(normalised.rstrip(";")):
        return False, "generated SQL contains a stacked statement, comment or UNION injection"

    if CLOCK_READ.search(normalised):
        return False, (
            "generated SQL reads the wall clock; every probe must compare against the pinned "
            "as_of date, otherwise the answer is not reproducible"
        )

    return True, ""


def assert_tables_in_scope(statement: str, allowed_tables: set[str]) -> Tuple[bool, str]:
    referenced = set(re.findall(r"\b(?:FROM|JOIN)\s+\"?([a-z_][a-z0-9_]*)\"?", statement, re.I))

    out_of_scope = {table.lower() for table in referenced} - {table.lower() for table in allowed_tables}

    if out_of_scope:
        return False, f"generated SQL touches out-of-scope tables: {sorted(out_of_scope)}"

    return True, ""
