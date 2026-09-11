import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import SQL_TEMPLATE_SELECTOR
from src.sqlpath.templates import SqlTemplate, catalogue_for, get_template, schema_notes_for

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# Deterministic match for the plain "vendors with status X" question
#
# "how many vendors approval status is approved and their names" went to the selector model,
# which answered that a count plus names "requires an aggregate not in the catalogue" and sent
# the question to generated SQL. vendors_by_approval_status answers it exactly: the rows are
# the names and row_count is the count. A question built only from a vendor noun, one status
# value and filler words is matched here without a model call. Anything else (a second filter,
# a date, a negation, a named vendor, any word not in the list below) still goes to the model.
# ---------------------------------------------------------------------------------------------

STATUS_VALUE_PHRASES = [
    # (template, parameter, value, phrase) - longer phrases first, so "non-compliant" is not
    # also read as "compliant" and "pending approval" is removed as one phrase
    ("vendors_by_compliance_status", "compliance_status", "Non-Compliant", re.compile(r"\bnon[\s-]?compliant\b")),
    ("vendors_by_compliance_status", "compliance_status", "Under Review", re.compile(r"\bunder\s+review\b")),
    ("vendors_by_compliance_status", "compliance_status", "Compliant", re.compile(r"\bcompliant\b")),
    ("vendors_by_approval_status", "approval_status", "Pending", re.compile(r"\bpending(?:\s+approval)?\b")),
    ("vendors_by_approval_status", "approval_status", "Approved", re.compile(r"\bapproved\b")),
    ("vendors_by_approval_status", "approval_status", "Rejected", re.compile(r"\brejected\b")),
]

# a risk band only counts as a filter when the question also says "risk" ("critical" on its own
# could be a finding severity)
RISK_BAND_PHRASES = [
    ("vendors_by_risk_category", "risk_category", band, re.compile(rf"\b{band.lower()}\b"))
    for band in ("Low", "Medium", "High", "Critical")
]

VENDOR_NOUN = re.compile(r"\b(?:vendors?|suppliers?)\b")

FILLER_WORDS = frozenset(
    {
        "a", "all", "along", "also", "an", "and", "any", "approval", "are", "as", "band", "be",
        "being", "can", "categories", "category", "compliance", "could", "count", "current",
        "currently", "data", "database", "db", "detail", "details", "display", "do", "does", "exist",
        "exists", "get", "give", "has", "have", "having", "how", "i", "in", "info", "information",
        "is", "it", "its", "know", "level", "levels", "like", "list", "lists", "many", "me", "name",
        "names", "now", "number", "of", "our", "please", "rating", "record", "records", "risk", "s",
        "see", "show", "status", "statuses", "supplier", "suppliers", "table", "tables", "tell",
        "that", "the", "their", "them", "there", "these", "they", "this", "those", "to", "total",
        "us", "vendor", "vendors", "want", "was", "we", "were", "what", "which", "who", "whose",
        "with", "would", "you",
    }
)


def match_simple_template(
    question: str,
    templates: list[SqlTemplate],
    named_vendor: bool = False,
) -> tuple[SqlTemplate, dict] | None:
    """The one vetted template a plain single-status vendor question maps to, or None."""
    if named_vendor or not question:
        return None

    text = " ".join(question.casefold().split())

    if re.search(r"\d", text) or not VENDOR_NOUN.search(text):
        return None

    phrases = list(STATUS_VALUE_PHRASES)

    if re.search(r"\brisk\b", text):
        phrases += RISK_BAND_PHRASES

    found: set[tuple[str, str, str]] = set()

    for template_id, parameter, value, pattern in phrases:
        if pattern.search(text):
            found.add((template_id, parameter, value))
            text = pattern.sub(" ", text)

    if len(found) != 1:
        return None

    leftover = [word for word in re.findall(r"[a-z]+", text) if word not in FILLER_WORDS]

    if leftover:
        return None

    template_id, parameter, value = next(iter(found))
    visible = {template.template_id: template for template in templates}

    if template_id not in visible:
        return None

    return visible[template_id], {parameter: value}


class TemplateArgument(BaseModel):
    name: str = Field(description="the parameter name exactly as the catalogue spells it")
    value: str = Field(description="the value to bind, taken from the question or the resolved entities")


class TemplateChoice(BaseModel):
    answerable: bool = Field(
        description="false when no catalogue entry answers the question without inventing a query"
    )
    template_id: str = Field(default="", description="one template_id from the catalogue, or empty")
    arguments: list[TemplateArgument] = Field(default_factory=list)
    reason: str = ""


class TemplateSelectionError(RuntimeError):
    pass


def arguments_as_dict(choice: TemplateChoice) -> dict:
    return {argument.name: argument.value for argument in choice.arguments if argument.name}


def select_template(
    question: str,
    templates: list[SqlTemplate],
    allowed_tables: set[str],
    entity_hints: str,
    config: dict | None = None,
) -> tuple[SqlTemplate, dict]:
    if not templates:
        raise TemplateSelectionError("this role is not scoped to any vetted compliance query")

    prompt = render_prompt("SQL_TEMPLATE_SELECTOR", SQL_TEMPLATE_SELECTOR,
        catalogue=catalogue_for(templates),
        schema_notes=schema_notes_for(allowed_tables),
        entity_hints=entity_hints or "(none resolved)",
        query=question,
    )

    model = model_for("sql_template_selector").with_structured_output(TemplateChoice)

    try:
        choice: TemplateChoice = model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=question)],
            config=config or {},
        )
    except Exception as exc:
        logger.error("template selection failed: %s", exc)

        raise TemplateSelectionError(
            f"the query planner could not choose a vetted query ({exc})"
        ) from exc

    if not choice.answerable or not choice.template_id:
        raise TemplateSelectionError(
            choice.reason or "no vetted query answers this question, so the records were not read"
        )

    template = get_template(choice.template_id)

    if template is None:
        logger.warning("selector returned an unknown template_id: %s", choice.template_id)

        raise TemplateSelectionError(
            f"'{choice.template_id}' is not a vetted query, so nothing was run against the database"
        )

    if template not in templates:
        raise TemplateSelectionError(
            f"the vetted query '{template.template_id}' reads tables this role may not see"
        )

    return template, arguments_as_dict(choice)
