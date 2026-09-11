"""Row-level scope written into every SQL statement a role runs.

The executor used to drop out-of-scope rows after the query came back, and only when a row
carried a ``risk_category`` or ``department`` column. A generated query that selected just
vendor names, or a COUNT, came back with High and Critical vendors for a Store Manager and
nothing was filtered (the log showed ``scope_filtered=0`` on exactly that query).

The scope now lives inside the statement. Every table the statement mentions is shadowed by a
common table expression of the same name that keeps only the rows this role may see:

    WITH compliance_reviews AS (SELECT * FROM compliance_reviews
                                WHERE vendor_id IN (SELECT vendor_id FROM vendors
                                                    WHERE risk_category IN ('Low', 'Medium'))),
         vendors AS (SELECT * FROM vendors WHERE risk_category IN ('Low', 'Medium'))
    SELECT ... the original statement, unchanged ...

PostgreSQL resolves an unqualified table name to a CTE of that name before the real table, so a
join, a subquery and a COUNT all read the scoped rows. Inside a non-recursive WITH list a CTE only
sees the siblings defined before it, which is why the child tables come first: their
``FROM vendors`` still reads the real table. A schema-qualified name ("public.vendors") would skip
the CTE, and ``assert_tables_in_scope`` already refuses those.
"""

import re
from dataclasses import dataclass

from src.guardrails.sql_guard import defined_cte_names
from src.sqlpath.templates import DEPARTMENTS, RISK_CATEGORIES, TABLE_DESCRIPTIONS

# tables that hang off vendors through vendor_id: a role that may not see a High-risk vendor may
# not see that vendor's reviews, findings or retention records either
VENDOR_CHILD_TABLES = ("audit_logs", "compliance_reviews", "retention_records")

DEPARTMENT_SCOPED_TABLES = ("retention_records",)

LEADING_WITH = re.compile(r"^\s*WITH\s+(RECURSIVE\s+)?", re.I)

CHILD_LABELS = {
    "audit_logs": "findings",
    "compliance_reviews": "reviews",
    "retention_records": "retention records",
}


class ScopeError(ValueError):
    pass


@dataclass(frozen=True)
class ScopedStatement:
    statement: str
    note: str = ""
    shadowed: tuple[str, ...] = ()


def risk_filter(risk_categories) -> list[str] | None:
    """The risk bands to keep, or None when the role sees every band (no filter needed)."""
    if not risk_categories:
        return None

    wanted = {str(value).strip().casefold() for value in risk_categories}
    kept = [band for band in RISK_CATEGORIES if band.casefold() in wanted]

    if len(kept) == len(RISK_CATEGORIES):
        return None

    return kept


def department_filter(departments) -> list[str] | None:
    if not departments:
        return None

    wanted = {str(value).strip().casefold() for value in departments}

    return [department for department in DEPARTMENTS if department.casefold() in wanted]


def _in_list(column: str, values: list[str]) -> str:
    if not values:
        return "FALSE"

    quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in values)

    return f"{column} IN ({quoted})"


def _mentions(statement: str, table: str) -> bool:
    return re.search(rf"\b{re.escape(table)}\b", statement, re.I) is not None


def _words(values: list[str]) -> str:
    if len(values) <= 1:
        return "".join(values)

    return ", ".join(values[:-1]) + " and " + values[-1]


def scope_statement(
    statement: str,
    visible_tables: set[str],
    risk_categories=None,
    departments=None,
) -> ScopedStatement:
    body = (statement or "").strip().rstrip(";").strip()
    visible = {table.lower() for table in visible_tables}
    risk = risk_filter(risk_categories)
    owned_departments = department_filter(departments)

    ctes: list[tuple[str, str]] = []
    scoped_children: list[str] = []
    department_scoped = False

    for table in VENDOR_CHILD_TABLES:
        if table not in visible or not _mentions(body, table):
            continue

        conditions = []

        if risk is not None:
            conditions.append(
                f"vendor_id IN (SELECT vendor_id FROM vendors WHERE {_in_list('risk_category', risk)})"
            )

        if table in DEPARTMENT_SCOPED_TABLES and owned_departments is not None:
            conditions.append(_in_list("department", owned_departments))
            department_scoped = True

        if conditions:
            ctes.append((table, f"SELECT * FROM {table} WHERE {' AND '.join(conditions)}"))

            if risk is not None:
                scoped_children.append(table)

    vendors_scoped = risk is not None and "vendors" in visible and _mentions(body, "vendors")

    if vendors_scoped:
        ctes.append(("vendors", f"SELECT * FROM vendors WHERE {_in_list('risk_category', risk)}"))

    # a table this role was never granted reads as empty, even if a comma join slipped it past
    # the table check
    for table in TABLE_DESCRIPTIONS:
        if table not in visible and _mentions(body, table):
            ctes.append((table, f"SELECT * FROM {table} WHERE FALSE"))

    if not ctes:
        return ScopedStatement(statement=body)

    names = [name for name, _ in ctes]
    prefix = ", ".join(f"{name} AS ({query})" for name, query in ctes)
    leading = LEADING_WITH.match(body)

    if leading:
        if leading.group(1):
            raise ScopeError("a recursive query cannot be limited to this role's scope, so it was not run")

        clash = sorted(set(names) & defined_cte_names(body))

        if clash:
            raise ScopeError(
                f"the query defines its own '{clash[0]}' expression, which would bypass this role's scope"
            )

        merged = f"WITH {prefix}, {body[leading.end():]}"
    else:
        merged = f"WITH {prefix} {body}"

    notes = []

    if risk is not None and (vendors_scoped or scoped_children):
        linked = _words([CHILD_LABELS[table] for table in scoped_children])
        notes.append(
            f"{_words(risk) or 'no'} risk vendors only"
            + (f" (and the {linked} linked to them)" if linked else "")
        )

    if department_scoped:
        notes.append(f"retention records of the {_words(owned_departments) or 'no'} department only")

    return ScopedStatement(statement=merged, note="; ".join(notes), shadowed=tuple(names))
