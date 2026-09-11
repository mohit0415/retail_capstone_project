import logging
import re

from configs.database import read_only_connection
from src.auth.rbac import allowed_tables
from src.graph.state import AgentState
from src.observability.tracing import traced_node
from src.schemas.enums import EntityStatus, TerminalOutcome
from src.schemas.models import ResolvedEntity

logger = logging.getLogger(__name__)

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

GENERIC_ENTITY_SURFACES = {
    "vendor",
    "vendors",
    "supplier",
    "suppliers",
    "the vendor",
    "the vendors",
    "all vendors",
    "any vendor",
    "each vendor",
    "third party",
    "third parties",
    "department",
    "departments",
    "all departments",
    "store",
    "stores",
    "all stores",
}

ATTRIBUTE_VALUES = {
    "low",
    "medium",
    "high",
    "critical",
    "low risk",
    "medium risk",
    "high risk",
    "critical risk",
    "compliant",
    "non-compliant",
    "noncompliant",
    "under review",
    "pending",
    "approved",
    "rejected",
}

GENERIC_NOUNS = (
    "third parties",
    "third party",
    "vendors",
    "vendor",
    "suppliers",
    "supplier",
    "departments",
    "department",
    "stores",
    "store",
)

DETERMINERS = ("all ", "any ", "each ", "every ", "the ", "our ", "these ", "those ")

# "vendors with open findings", "vendors overdue for review": a description, not a name
DESCRIPTION_WORDS = frozenset(
    {
        "with", "without", "that", "who", "which", "whose", "whom", "under", "having", "marked", "flagged",
        "in", "on", "overdue", "pending", "due", "awaiting", "rated", "not", "missing", "are", "is",
        "requiring", "needing", "for",
    }
)

# a state or kind of vendor ("overdue vendors", "third-party suppliers"), never a vendor's name
MODIFIERS = frozenset(
    {
        "overdue", "expired", "suspended", "unapproved", "pending", "due", "awaiting", "rated", "missing",
        "active", "inactive", "new", "existing", "flagged", "blocked", "terminated", "third party",
        "external", "current", "former", "open", "closed", "risky", "risk",
    }
)


def _normalise(text: str) -> str:
    return re.sub(r"[\s\-]+", " ", (text or "").strip().lower())


_GENERIC_SURFACES = {_normalise(surface) for surface in GENERIC_ENTITY_SURFACES}

_ATTRIBUTE_SURFACES = {_normalise(value) for value in ATTRIBUTE_VALUES}


def _is_attribute_phrase(prefix: str) -> bool:
    """"high and critical risk", "overdue", "third party": every part is a band, a status or a state."""
    parts = [part.strip() for part in re.split(r",| and | or | & ", prefix) if part.strip()]

    if not parts:
        return False

    for part in parts:
        if part in _ATTRIBUTE_SURFACES or part in MODIFIERS:
            continue

        if not all(word in _ATTRIBUTE_SURFACES or word in MODIFIERS for word in part.split()):
            return False

    return True


def is_description(surface: str) -> bool:
    """A category or a description of records ("high risk vendors", "vendors with open findings").

    It is a filter for the records query, not the name of one vendor: looking it up as a name found
    no vendor called "high risk vendors" and stopped the app's own suggested question with a
    clarification.
    """
    text = _normalise(surface)

    for determiner in DETERMINERS:
        if text.startswith(determiner):
            text = text[len(determiner):]
            break

    if text in _GENERIC_SURFACES or text in _ATTRIBUTE_SURFACES:
        return True

    for noun in GENERIC_NOUNS:
        if text.endswith(f" {noun}") and _is_attribute_phrase(text[: -len(noun) - 1]):
            return True

        if noun.endswith("s") and text.startswith(f"{noun} "):
            following = text[len(noun) + 1:].split()

            if following and following[0] in DESCRIPTION_WORDS:
                return True

    return False


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
        logger.debug("entity resolution skipped: no intent request_id=%s", state.get("request_id"))

        return {"resolved_entities": []}

    tables = allowed_tables(state.get("access_scopes", []))
    resolved: list[ResolvedEntity] = []
    skipped: list[str] = []

    for span in intent.entities:
        if is_description(span.text):
            skipped.append(f"{span.entity_type}:{span.text} (generic/attribute)")
            continue

        if span.entity_type == "vendor" and "vendors" in tables:
            resolved.append(_resolve_vendor(span.text))
        elif span.entity_type == "department" and "retention_records" in tables:
            resolved.append(_resolve_department(span.text))
        else:
            skipped.append(f"{span.entity_type}:{span.text} (no table in scope)")

    for entity in resolved:
        logger.info(
            "entity %s=%r status=%s resolved_id=%s canonical=%r candidates=%d request_id=%s",
            entity.entity_type,
            entity.surface_form,
            entity.status.value,
            entity.resolved_id,
            entity.canonical_name,
            len(entity.candidates),
            state.get("request_id"),
        )

    if skipped:
        logger.debug("entity spans not resolved: %s request_id=%s", skipped, state.get("request_id"))

    blocking = [
        entity
        for entity in resolved
        if entity.status in (EntityStatus.UNRESOLVED, EntityStatus.AMBIGUOUS)
    ]

    if blocking:
        question = _clarification_for(blocking[0])

        logger.warning(
            "entity resolution needs clarification request_id=%s blocking=%s question=%r",
            state.get("request_id"),
            [f"{e.entity_type}:{e.surface_form}={e.status.value}" for e in blocking],
            question,
        )

        return {
            "resolved_entities": resolved,
            "clarification_question": question,
            "terminal_outcome": TerminalOutcome.CLARIFICATION_REQUIRED.value,
        }

    return {"resolved_entities": resolved}
