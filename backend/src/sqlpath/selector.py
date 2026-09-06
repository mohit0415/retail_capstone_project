import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from src.prompts.library import SQL_TEMPLATE_SELECTOR
from src.sqlpath.templates import SqlTemplate, catalogue_for, get_template, schema_notes_for

logger = logging.getLogger(__name__)


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

    prompt = SQL_TEMPLATE_SELECTOR.format(
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
