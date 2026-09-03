import logging
import re
from datetime import date
from functools import lru_cache
from typing import List, Optional

from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine

from configs.settings import settings
from src.guardrails.sql_guard import (
    assert_tables_in_scope,
    classify_intent_with_llm,
    validate_generated_sql,
    validate_question_intent,
)
from src.index.models import get_llm
from src.schemas.models import SqlEvidence
from src.sqlpath.templates import schema_notes_for, tables_visible_to

logger = logging.getLogger(__name__)

LIMIT_CLAUSE = re.compile(r"\blimit\s+(\d+)", re.I)


class SqlPolicyError(RuntimeError):
    pass


@lru_cache
def _engine():
    return create_engine(
        settings.database_url,
        connect_args={"options": f"-c statement_timeout={settings.sql_statement_timeout_ms}"},
        pool_pre_ping=True,
    )


@lru_cache
def _sql_database(tables: tuple[str, ...]) -> SQLDatabase:
    return SQLDatabase(engine=_engine(), include_tables=list(tables), sample_rows_in_table_info=2)


def build_query_engine(allowed_tables: set[str]) -> tuple[NLSQLTableQueryEngine, tuple[str, ...]]:
    tables = tuple(tables_visible_to(allowed_tables))

    if not tables:
        raise SqlPolicyError("this role is not scoped to any compliance table")

    engine = NLSQLTableQueryEngine(
        sql_database=_sql_database(tables),
        tables=list(tables),
        llm=get_llm("nl2sql"),
        context_str_prefix=schema_notes_for(set(tables)),
        synthesize_response=False,
        verbose=False,
    )

    return engine, tables


def _enforce_row_limit(statement: str) -> str:
    match = LIMIT_CLAUSE.search(statement)
    cleaned = statement.rstrip().rstrip(";")

    if match is None:
        return f"{cleaned} LIMIT {settings.sql_row_limit}"

    if int(match.group(1)) > settings.sql_row_limit:
        return LIMIT_CLAUSE.sub(f"LIMIT {settings.sql_row_limit}", cleaned, count=1)

    return cleaned


def _scope_rows(
    rows: List[dict],
    departments: List[str],
    risk_categories: set[str] | None,
) -> tuple[List[dict], int]:
    allowed_departments = {value.lower() for value in departments} if departments else None
    allowed_risk = {value.lower() for value in risk_categories} if risk_categories else None

    if allowed_departments is None and allowed_risk is None:
        return rows, 0

    kept = []
    removed = 0

    for row in rows:
        department = row.get("department")
        risk_category = row.get("risk_category")

        if allowed_departments is not None and department is not None:
            if str(department).lower() not in allowed_departments:
                removed += 1
                continue

        if allowed_risk is not None and risk_category is not None:
            if str(risk_category).lower() not in allowed_risk:
                removed += 1
                continue

        kept.append(row)

    return kept, removed


def run_nl2sql(
    question: str,
    allowed_tables: set[str],
    departments: Optional[List[str]] = None,
    as_of: Optional[date] = None,
    risk_categories: Optional[set[str]] = None,
) -> SqlEvidence:
    as_of = as_of or settings.as_of_date
    departments = departments or []

    ok, reason = validate_question_intent(question)

    if not ok:
        raise SqlPolicyError(reason)

    ok, reason = classify_intent_with_llm(get_llm("sql_intent_guard"), question)

    if not ok:
        raise SqlPolicyError(reason)

    engine, tables = build_query_engine(allowed_tables)

    pinned_question = (
        f"{question}\n\n"
        f"Treat today's date as {as_of}. Compare every date against that literal, "
        f"never against the database clock."
    )

    try:
        response = engine.query(pinned_question)
    except Exception as exc:
        logger.error("NL2SQL engine failed: %s", exc)
        raise SqlPolicyError(f"the database probe could not be completed: {exc}")

    metadata = response.metadata or {}
    statement = str(metadata.get("sql_query") or "").strip()

    ok, reason = validate_generated_sql(statement)

    if not ok:
        logger.warning("rejected generated SQL: %s | %s", reason, statement)
        raise SqlPolicyError(f"the generated query was rejected: {reason}")

    ok, reason = assert_tables_in_scope(statement, set(tables))

    if not ok:
        logger.warning("rejected generated SQL: %s | %s", reason, statement)
        raise SqlPolicyError(f"the generated query was rejected: {reason}")

    raw_rows = metadata.get("result") or []
    columns = metadata.get("col_keys") or []

    rows: List[dict] = []

    for row in raw_rows:
        if isinstance(row, dict):
            rows.append(row)
        elif columns:
            rows.append(dict(zip(columns, row)))
        else:
            rows.append({"value": row})

    rows, filtered_out = _scope_rows(rows, departments, risk_categories)

    return SqlEvidence(
        template_id="nl2sql_engine",
        statement=" ".join(_enforce_row_limit(statement).split()),
        parameters={"as_of": str(as_of), "tables": ", ".join(tables)},
        row_count=len(rows),
        rows=rows[: settings.sql_row_limit],
        as_of=as_of,
        truncated=len(rows) >= settings.sql_row_limit,
        rows_filtered_by_scope=filtered_out,
    )


def sanity_check(evidence: SqlEvidence) -> list[str]:
    problems: list[str] = []

    if evidence.truncated:
        problems.append(
            f"result hit the {settings.sql_row_limit}-row cap, so any count stated from it is a floor"
        )

    if evidence.row_count == 0:
        problems.append("query returned no rows; an empty result is not evidence of compliance")

    if evidence.rows_filtered_by_scope:
        problems.append(
            f"{evidence.rows_filtered_by_scope} row(s) were removed by this role's department or "
            "risk-category scope, so this result is partial and any count from it is a floor"
        )

    for row in evidence.rows[:20]:
        for key, value in row.items():
            if key.endswith("_date") and isinstance(value, date) and value > evidence.as_of:
                problems.append(f"row contains {key}={value} which is later than the pinned as_of date")

    return problems
