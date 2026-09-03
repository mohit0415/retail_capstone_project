from configs.database import read_only_connection
from src.auth.rbac import allowed_tables
from src.graph.state import AgentState
from src.observability.tracing import traced_node
from src.schemas.enums import EntityStatus, TerminalOutcome
from src.schemas.models import ResolvedEntity

EXACT_VENDOR = """
SELECT vendor_id, vendor_name, risk_category, compliance_status
FROM vendors
WHERE lower(vendor_name) = lower(%(name)s)
LIMIT 5
"""

PREFIX_VENDOR = """
SELECT vendor_id, vendor_name, risk_category, compliance_status
FROM vendors
WHERE lower(vendor_name) LIKE lower(%(name)s) || '%%'
ORDER BY vendor_name
LIMIT 6
"""

FUZZY_VENDOR = """
SELECT vendor_id, vendor_name, risk_category, compliance_status,
       similarity(lower(vendor_name), lower(%(name)s)) AS score
FROM vendors
WHERE lower(vendor_name) %% lower(%(name)s)
ORDER BY score DESC
LIMIT 5
"""

DEPARTMENT_LOOKUP = """
SELECT DISTINCT department
FROM retention_records
WHERE lower(department) = lower(%(name)s)
LIMIT 1
"""

FUZZY_THRESHOLD = 0.45


def _resolve_vendor(surface: str) -> ResolvedEntity:
    with read_only_connection() as conn:
        exact = conn.execute(EXACT_VENDOR, {"name": surface}).fetchall()

        if len(exact) == 1:
            row = exact[0]

            return ResolvedEntity(
                surface_form=surface,
                entity_type="vendor",
                resolved_id=row["vendor_id"],
                canonical_name=row["vendor_name"],
                status=EntityStatus.EXACT,
            )

        if len(exact) > 1:
            return ResolvedEntity(
                surface_form=surface,
                entity_type="vendor",
                status=EntityStatus.AMBIGUOUS,
                candidates=[dict(row) for row in exact],
            )

        prefix = conn.execute(PREFIX_VENDOR, {"name": surface}).fetchall()

        if len(prefix) == 1:
            row = prefix[0]

            return ResolvedEntity(
                surface_form=surface,
                entity_type="vendor",
                resolved_id=row["vendor_id"],
                canonical_name=row["vendor_name"],
                status=EntityStatus.FUZZY,
                candidates=[dict(row)],
            )

        if len(prefix) > 1:
            return ResolvedEntity(
                surface_form=surface,
                entity_type="vendor",
                status=EntityStatus.AMBIGUOUS,
                candidates=[dict(row) for row in prefix],
            )

        fuzzy = conn.execute(FUZZY_VENDOR, {"name": surface}).fetchall()

    strong = [row for row in fuzzy if float(row["score"]) >= FUZZY_THRESHOLD]

    if len(strong) == 1:
        row = strong[0]

        return ResolvedEntity(
            surface_form=surface,
            entity_type="vendor",
            resolved_id=row["vendor_id"],
            canonical_name=row["vendor_name"],
            status=EntityStatus.FUZZY,
            candidates=[dict(r) for r in strong],
        )

    if len(strong) > 1:
        return ResolvedEntity(
            surface_form=surface,
            entity_type="vendor",
            status=EntityStatus.AMBIGUOUS,
            candidates=[dict(r) for r in strong],
        )

    return ResolvedEntity(surface_form=surface, entity_type="vendor", status=EntityStatus.UNRESOLVED)


def _resolve_department(surface: str) -> ResolvedEntity:
    with read_only_connection() as conn:
        row = conn.execute(DEPARTMENT_LOOKUP, {"name": surface}).fetchone()

    if row:
        return ResolvedEntity(
            surface_form=surface,
            entity_type="department",
            canonical_name=row["department"],
            status=EntityStatus.EXACT,
        )

    return ResolvedEntity(surface_form=surface, entity_type="department", status=EntityStatus.UNRESOLVED)


def _clarification_for(entity: ResolvedEntity) -> str:
    if entity.status is EntityStatus.AMBIGUOUS:
        names = ", ".join(str(c.get("vendor_name") or c.get("department")) for c in entity.candidates)

        return f"\"{entity.surface_form}\" matches more than one record: {names}. Which one do you mean?"

    return (
        f"No record matches \"{entity.surface_form}\". "
        "Please give the exact registered name, or the vendor id."
    )


@traced_node("entity_resolution")
def entity_resolution_node(state: AgentState) -> dict:
    intent = state.get("intent")

    if intent is None:
        return {"resolved_entities": []}

    tables = allowed_tables(state.get("access_scopes", []))
    resolved: list[ResolvedEntity] = []

    for span in intent.entities:
        if span.entity_type == "vendor" and "vendors" in tables:
            resolved.append(_resolve_vendor(span.text))
        elif span.entity_type == "department" and "retention_records" in tables:
            resolved.append(_resolve_department(span.text))

    blocking = [
        entity
        for entity in resolved
        if entity.status in (EntityStatus.UNRESOLVED, EntityStatus.AMBIGUOUS)
    ]

    if blocking:
        return {
            "resolved_entities": resolved,
            "clarification_question": _clarification_for(blocking[0]),
            "terminal_outcome": TerminalOutcome.CLARIFICATION_REQUIRED.value,
        }

    return {"resolved_entities": resolved}
