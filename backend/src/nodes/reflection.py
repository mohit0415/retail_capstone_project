from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import REFLECTION
from src.schemas.models import EvidencePlan


class ReflectionOutput(BaseModel):
    revised_plan: EvidencePlan
    repairable: bool = Field(description="false when no re-plan can supply the missing evidence")
    reasoning: str = ""


@traced_node("reflection")
def reflection_node(state: AgentState) -> dict:
    validation = state["validation"]
    previous_plan = state.get("plan")

    defect_text = "\n".join(
        f"- {d.defect_type.value}: {d.description} (repair: {d.suggested_repair})" for d in validation.defects
    )

    prompt = REFLECTION.format(
        previous_plan=previous_plan.model_dump_json(indent=2) if previous_plan else "(none)",
        defects=defect_text or "(none)",
        query=state["standalone_query"],
    )

    model = model_for("reflection").with_structured_output(ReflectionOutput)

    try:
        outcome: ReflectionOutput = model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
            config=runnable_config(state, "reflection"),
        )
    except Exception:
        return {
            "reflection_count": state.get("reflection_count", 0) + 1,
            "escalation_reason": "reflection could not produce a revised plan",
            "tokens_spent": 900,
        }

    if not outcome.repairable:
        return {
            "reflection_count": state.get("reflection_count", 0) + 1,
            "escalation_reason": f"defects are not repairable by re-planning: {outcome.reasoning}",
            "tokens_spent": 900,
        }

    revised = outcome.revised_plan
    revised.revision = (previous_plan.revision + 1) if previous_plan else 1

    directive = (
        "This is a re-plan. The previous attempt produced these defects:\n"
        f"{defect_text}\n"
        f"The reflection step decided: {outcome.reasoning}\n"
        "The new plan must gather the evidence the previous one missed."
    )

    return {
        "plan": revised,
        "plan_from_reflection": True,
        "replan_directive": directive,
        "reflection_count": state.get("reflection_count", 0) + 1,
        "retrieved_chunks": None,
        "sql_evidence": None,
        "draft": None,
        "validation": None,
        "tokens_spent": 900,
    }
