import logging
import re

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

CLOCK_READ = re.compile(
    r"\b(?:CURRENT_DATE|CURRENT_TIME|CURRENT_TIMESTAMP|LOCALTIME|LOCALTIMESTAMP)\b"
    r"|\b(?:NOW|TRANSACTION_TIMESTAMP|STATEMENT_TIMESTAMP|CLOCK_TIMESTAMP|TIMEOFDAY)\s*\(\s*\)",
    re.I,
)

# functions that run SQL written inside a string, read files or change settings: the SQL in the
# string never passes the table check ("SELECT query_to_xml('select * from audit_logs', ...)")
FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(?:query_to_xml\w*|cursor_to_xml\w*|table_to_xml\w*|schema_to_xml\w*|database_to_xml\w*|"
    r"dblink\w*|pg_\w+|lo_\w+|set_config|current_setting|xpath\w*)\s*\(",
    re.I,
)

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


def validate_question_intent(question: str) -> tuple[bool, str]:
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


def classify_intent_with_llm(llm, question: str) -> tuple[bool, str]:
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


def validate_generated_sql(statement: str) -> tuple[bool, str]:
    if not statement or not statement.strip():
        return False, "the engine produced no SQL"

    normalised = " ".join(statement.split())

    if not re.match(r"^\s*(SELECT|WITH)\b", normalised, re.I):
        return False, "generated SQL does not begin with SELECT or WITH"

    if SQL_WRITE_STATEMENT.search(normalised):
        return False, "generated SQL contains a write verb"

    if STACKED_STATEMENT.search(normalised.rstrip(";")):
        return False, "generated SQL contains a stacked statement, comment or UNION injection"

    if FORBIDDEN_FUNCTIONS.search(normalised):
        return False, "generated SQL calls a function that can run other SQL, read files or change settings"

    if CLOCK_READ.search(normalised):
        return False, (
            "generated SQL reads the wall clock; every probe must compare against the pinned "
            "as_of date, otherwise the answer is not reproducible"
        )

    return True, ""


STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")

# "EXTRACT(YEAR FROM onboarding_date)" and "a IS DISTINCT FROM b" carry a FROM that is not a
# table reference; without this the column after it was read as a table and the query refused
FROM_INSIDE_FUNCTION = re.compile(r"\b(EXTRACT|SUBSTRING|TRIM|OVERLAY)\s*\(([^()]*?)\bFROM\b", re.I)

DISTINCT_FROM = re.compile(r"\bDISTINCT\s+FROM\b", re.I)

SQL_TOKEN = re.compile(r'"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*|[(),.]|\S')

FROM_LIST_STOP_WORDS = frozenset(
    {
        "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "UNION", "INTERSECT", "EXCEPT", "ON", "USING",
        "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL", "OUTER", "WINDOW", "OFFSET",
        "FETCH", "FOR", "RETURNING", "SELECT", "FROM", "AS", "LATERAL", "TABLESAMPLE",
    }
)


def _is_identifier(token: str) -> bool:
    return bool(token) and (token[0].isalpha() or token[0] in '_"')


def _unquote(token: str) -> str:
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('""', '"')

    return token


def _tokens(statement: str) -> list[str]:
    text = STRING_LITERAL.sub("''", statement or "")
    previous = None

    while previous != text:
        previous = text
        text = FROM_INSIDE_FUNCTION.sub(lambda match: f"{match.group(1)}({match.group(2)} _from_", text)

    text = DISTINCT_FROM.sub("DISTINCT _from_", text)

    return SQL_TOKEN.findall(text)


def referenced_tables(statement: str) -> set[str]:
    """Every relation a statement reads, including the ones after a comma in a FROM list.

    The old check only looked at the name straight after FROM or JOIN, so
    "FROM vendors v, audit_logs a" read audit_logs without the scope check seeing it.
    A schema-qualified name keeps its schema ("public.vendors"), and a function in the
    FROM list is reported as "name()", so neither can pass as a granted table.
    """
    return _scan_relations(_tokens(statement))


def _scan_relations(tokens: list[str]) -> set[str]:
    found: set[str] = set()
    index = 0

    while index < len(tokens):
        if tokens[index].upper() not in ("FROM", "JOIN"):
            index += 1
            continue

        index += 1

        while index < len(tokens):
            if tokens[index].upper() in ("LATERAL", "ONLY"):
                index += 1
                continue

            if tokens[index] == "(":
                # a subquery in the FROM list: read what it references, then carry on with
                # the list after it ("FROM (SELECT ...) sub, retention_records r")
                end = _skip_parentheses(tokens, index)
                found |= _scan_relations(tokens[index + 1 : end - 1])
                index = end
            elif _is_identifier(tokens[index]):
                name = _unquote(tokens[index])
                index += 1

                while index + 1 < len(tokens) and tokens[index] == "." and _is_identifier(tokens[index + 1]):
                    name = f"{name}.{_unquote(tokens[index + 1])}"
                    index += 2

                if index < len(tokens) and tokens[index] == "(":
                    found.add(f"{name.lower()}()")
                    break

                found.add(name.lower())
            else:
                break

            if index < len(tokens) and tokens[index].upper() == "AS":
                index += 1

            if (
                index < len(tokens)
                and _is_identifier(tokens[index])
                and tokens[index].upper() not in FROM_LIST_STOP_WORDS
            ):
                index += 1

            if index < len(tokens) and tokens[index] == ",":
                index += 1
                continue

            break

    return found


def defined_cte_names(statement: str) -> set[str]:
    """The names a leading WITH clause defines (they are not tables, so they need no grant)."""
    tokens = _tokens(statement)

    if not tokens or tokens[0].upper() != "WITH":
        return set()

    names: set[str] = set()
    index = 1

    if index < len(tokens) and tokens[index].upper() == "RECURSIVE":
        index += 1

    while index < len(tokens) and _is_identifier(tokens[index]):
        name = _unquote(tokens[index]).lower()
        index += 1

        if index < len(tokens) and tokens[index] == "(":
            index = _skip_parentheses(tokens, index)

        if index >= len(tokens) or tokens[index].upper() != "AS":
            break

        index += 1

        while index < len(tokens) and tokens[index].upper() in ("NOT", "MATERIALIZED"):
            index += 1

        if index >= len(tokens) or tokens[index] != "(":
            break

        names.add(name)
        index = _skip_parentheses(tokens, index)

        if index < len(tokens) and tokens[index] == ",":
            index += 1
            continue

        break

    return names


def _skip_parentheses(tokens: list[str], index: int) -> int:
    depth = 0

    while index < len(tokens):
        if tokens[index] == "(":
            depth += 1
        elif tokens[index] == ")":
            depth -= 1

            if depth == 0:
                return index + 1

        index += 1

    return index


def assert_tables_in_scope(statement: str, allowed_tables: set[str]) -> tuple[bool, str]:
    allowed = {table.lower() for table in allowed_tables} | defined_cte_names(statement)

    out_of_scope = {name for name in referenced_tables(statement) if name not in allowed}

    if out_of_scope:
        return False, f"generated SQL touches out-of-scope tables: {sorted(out_of_scope)}"

    return True, ""
