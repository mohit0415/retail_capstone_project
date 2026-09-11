import logging
from datetime import date

from configs.database import read_only_connection
from configs.settings import settings
from src.guardrails.sql_guard import (
    assert_tables_in_scope,
    classify_intent_with_llm,
    validate_generated_sql,
    validate_question_intent,
)
from src.index.models import get_llm
from src.schemas.models import SqlEvidence
from src.sqlpath.disclosure import caveats
from src.sqlpath.nl2sql_engine import GeneratedSqlError, run_generated_sql
from src.sqlpath.scoping import ScopeError, scope_statement
from src.sqlpath.selector import TemplateSelectionError, match_simple_template, select_template
from src.sqlpath.templates import (
    TemplateBindingError,
    bind_parameters,
    flatten,
    tables_visible_to,
    templates_visible_to,
)

logger = logging.getLogger(__name__)


class SqlPolicyError(RuntimeError):
    pass


def _scope_rows(
    rows: list[dict],
    departments: list[str],
    risk_categories: set[str] | None,
) -> tuple[list[dict], int]:
    """Second line of defence: the statement is already scoped (src/sqlpath/scoping.py)."""
    allowed_departments = {value.lower() for value in departments} if departments else None
    allowed_risk = {value.lower() for value in risk_categories} if risk_categories else None

    if allowed_departments is None and allowed_risk is None:
        return rows, 0

    kept = []
    removed = 0

    for row in rows:
        department = row.get("department")
        risk_category = row.get("risk_category")

        if department is not None and allowed_departments is not None and str(department).lower() not in allowed_departments:
            removed += 1
            continue

        if risk_category is not None and allowed_risk is not None and str(risk_category).lower() not in allowed_risk:
            removed += 1
            continue

        kept.append(row)

    return kept, removed


def _reportable_parameters(bound: dict) -> dict:
    return {key: str(value) for key, value in bound.items()}


def run_vetted_sql(
    question: str,
    allowed_tables: set[str],
    departments: list[str] | None = None,
    as_of: date | None = None,
    risk_categories: set[str] | None = None,
    presets: dict | None = None,
    entity_hints: str = "",
    config: dict | None = None,
) -> SqlEvidence:
    as_of = as_of or settings.as_of_date
    departments = departments or []
    presets = presets or {}

    ok, reason = validate_question_intent(question)

    if not ok:
        raise SqlPolicyError(reason)

    visible_tables = set(tables_visible_to(allowed_tables))

    if not visible_tables:
        raise SqlPolicyError("this role is not scoped to any compliance table")

    catalogue = templates_visible_to(allowed_tables)

    matched = match_simple_template(question, catalogue, named_vendor=presets.get("vendor_id") is not None)

    if matched is None:
        # the matcher only accepts a vendor noun, one status value and filler words, so a
        # paraphrased write request can never reach it; everything else gets the model check
        ok, reason = classify_intent_with_llm(get_llm("sql_intent_guard"), question)

        if not ok:
            raise SqlPolicyError(reason)

    if matched is not None:
        template, supplied = matched
        selection = "matched"

        logger.info(
            "vetted template matched without the selector model template=%s parameters=%s",
            template.template_id,
            supplied,
        )
    else:
        try:
            template, supplied = select_template(
                question=question,
                templates=catalogue,
                allowed_tables=visible_tables,
                entity_hints=entity_hints,
                config=config,
            )
        except TemplateSelectionError as exc:
            logger.debug("template selection failed (%s), using generated SQL fallback", exc)

            return _generated_fallback(
                question=question,
                reason=str(exc),
                allowed_tables=allowed_tables,
                departments=departments,
                risk_categories=risk_categories,
                as_of=as_of,
            )

        selection = "selector"

    for name, value in presets.items():
        if any(parameter.name == name for parameter in template.parameters):
            supplied[name] = value

    try:
        bound = bind_parameters(template, supplied, as_of=as_of)
    except TemplateBindingError as exc:
        raise SqlPolicyError(
            f"the vetted query '{template.template_id}' could not be bound: {exc}"
        ) from exc

    statement = flatten(template.statement)

    ok, reason = validate_generated_sql(statement)

    if not ok:
        logger.error("vetted template %s failed the shape check: %s", template.template_id, reason)

        raise SqlPolicyError(f"the vetted query '{template.template_id}' failed its safety check: {reason}")

    ok, reason = assert_tables_in_scope(statement, visible_tables)

    if not ok:
        logger.error("vetted template %s is out of scope: %s", template.template_id, reason)

        raise SqlPolicyError(f"the vetted query '{template.template_id}' is out of scope for this role")

    try:
        scoped = scope_statement(statement, visible_tables, risk_categories, departments)
    except ScopeError as exc:
        raise SqlPolicyError(str(exc)) from exc

    try:
        with read_only_connection() as conn:
            raw_rows = conn.execute(scoped.statement, bound).fetchall()
    except Exception as exc:
        logger.error("vetted query %s failed: %s", template.template_id, exc)

        raise SqlPolicyError(f"the database probe could not be completed: {exc}") from exc

    rows = [dict(row) for row in raw_rows]

    rows, filtered_out = _scope_rows(rows, departments, risk_categories)

    return SqlEvidence(
        template_id=template.template_id,
        statement=scoped.statement,
        parameters=_reportable_parameters(bound),
        row_count=len(rows),
        rows=rows[: settings.sql_row_limit],
        as_of=as_of,
        truncated=len(raw_rows) >= settings.sql_row_limit,
        rows_filtered_by_scope=filtered_out,
        scope_note=scoped.note,
        selection=selection,
    )


def _generated_fallback(
    question: str,
    reason: str,
    allowed_tables: set[str],
    departments: list[str],
    risk_categories: set[str] | None,
    as_of: date,
) -> SqlEvidence:
    if not settings.enable_generated_sql_fallback:
        raise SqlPolicyError(reason)

    logger.info("no vetted template fits (%s); falling back to the generated NL2SQL engine", reason)

    try:
        evidence = run_generated_sql(
            question=question,
            allowed_tables=allowed_tables,
            as_of=as_of,
            risk_categories=risk_categories,
            departments=departments,
        )
    except GeneratedSqlError as exc:
        raise SqlPolicyError(
            f"no reviewed query answers this question ({reason}), and the generated query "
            f"could not be used either: {exc}"
        ) from exc

    kept, filtered_out = _scope_rows(evidence.rows, departments, risk_categories)

    return evidence.model_copy(
        update={
            "rows": kept,
            "row_count": evidence.row_count - filtered_out,
            "rows_filtered_by_scope": filtered_out,
            "selection": "generated",
        }
    )


def sanity_check(evidence: SqlEvidence) -> list[str]:
    return caveats(evidence)
