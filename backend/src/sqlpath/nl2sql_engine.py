import logging
import re
from datetime import date
from functools import lru_cache

from sqlalchemy import create_engine

from configs.database import read_only_connection
from configs.settings import settings
from src.guardrails.sql_guard import assert_tables_in_scope, validate_generated_sql
from src.index.models import get_llm
from src.prompts.library import NL2SQL_GENERATION
from src.schemas.models import SqlEvidence
from src.sqlpath.templates import SCHEMA_NOTES, tables_visible_to

logger = logging.getLogger(__name__)

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)

LEADING_LABEL = re.compile(r"^\s*(sql\s*(query)?\s*[:=]|query\s*[:=])\s*", re.I)

PLACEHOLDER = re.compile(r"%\(?\w*\)?[sd]|%s|:\w+")

REFUSAL = re.compile(r"^\s*(cannot|unanswerable|no\s+query|none)\b", re.I)


class GeneratedSqlError(RuntimeError):
    pass


@lru_cache
def _sqlalchemy_engine():
    return create_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def _schema_text(tables: tuple[str, ...]) -> str:
    from llama_index.core.utilities.sql_wrapper import SQLDatabase

    database = SQLDatabase(_sqlalchemy_engine(), include_tables=list(tables))

    return "\n\n".join(database.get_single_table_info(name) for name in sorted(tables))


def role_schema(allowed_tables: set[str]) -> str:
    visible = tuple(sorted(tables_visible_to(allowed_tables)))

    if not visible:
        raise GeneratedSqlError("this role is not scoped to any compliance table")

    try:
        return _schema_text(visible)
    except Exception as exc:
        logger.error("schema introspection failed: %s", exc)

        raise GeneratedSqlError(f"the database schema could not be read ({exc})") from exc


def _strip(raw: str) -> str:
    text = (raw or "").strip()

    fenced = FENCE.search(text)

    if fenced:
        text = fenced.group(1)

    text = LEADING_LABEL.sub("", text.strip())

    return text.strip().rstrip(";").strip()


def generate_sql(question: str, allowed_tables: set[str], as_of: date) -> str:
    prompt = NL2SQL_GENERATION.format(
        schema=role_schema(allowed_tables),
        schema_notes=SCHEMA_NOTES,
        as_of=as_of,
        row_limit=settings.sql_row_limit,
        tables=", ".join(sorted(tables_visible_to(allowed_tables))),
        query=question,
    )

    try:
        raw = str(get_llm("nl2sql_generate").complete(prompt))
    except Exception as exc:
        logger.error("sql generation failed: %s", exc)

        raise GeneratedSqlError(f"the query generator could not be reached ({exc})") from exc

    statement = _strip(raw)

    if not statement or REFUSAL.match(statement):
        raise GeneratedSqlError(
            "the generator judged this question unanswerable from the tables this role can read"
        )

    return statement


def _gate(statement: str, visible_tables: set[str]) -> None:
    ok, reason = validate_generated_sql(statement)

    if not ok:
        logger.warning("generated sql rejected by the shape check: %s", reason)

        raise GeneratedSqlError(f"the generated query failed its safety check: {reason}")

    ok, reason = assert_tables_in_scope(statement, visible_tables)

    if not ok:
        logger.warning("generated sql rejected by the scope check: %s", reason)

        raise GeneratedSqlError("the generated query reads tables this role may not see")

    if PLACEHOLDER.search(statement):
        raise GeneratedSqlError(
            "the generated query still carries an unbound placeholder, so it was not run"
        )


def run_generated_sql(
    question: str,
    allowed_tables: set[str],
    as_of: date | None = None,
) -> SqlEvidence:
    as_of = as_of or settings.as_of_date
    visible = set(tables_visible_to(allowed_tables))

    if not visible:
        raise GeneratedSqlError("this role is not scoped to any compliance table")

    statement = generate_sql(question, allowed_tables, as_of)

    _gate(statement, visible)

    try:
        with read_only_connection() as conn:
            raw_rows = conn.execute(statement).fetchall()
    except Exception as exc:
        logger.error("generated sql failed to execute: %s", exc)

        raise GeneratedSqlError(f"the generated query could not be executed: {exc}") from exc

    rows = [dict(row) for row in raw_rows]

    return SqlEvidence(
        template_id="generated",
        statement=statement,
        parameters={"as_of": str(as_of)},
        row_count=len(rows),
        rows=rows[: settings.sql_row_limit],
        as_of=as_of,
        truncated=len(raw_rows) >= settings.sql_row_limit,
        generated=True,
    )
