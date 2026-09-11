import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from configs.settings import settings
from src.graph.routing import (
    HYBRID_PATHS,
    HYBRID_REDRAFT,
    RAG_PATHS,
    SQL_PATHS,
    SQL_RENARRATE,
    current_evidence_path,
)
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import REFLECTION
from src.schemas.enums import EvidencePath
from src.schemas.models import EvidencePlan, PlanStep

logger = logging.getLogger(__name__)


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

    if current_evidence_path(state) in RAG_PATHS:
        return _rag_repair(state, previous_plan, defect_text)

    if current_evidence_path(state) in SQL_PATHS and state.get("sql_evidence") is not None:
        return _sql_repair(state, previous_plan, defect_text)

    if current_evidence_path(state) in HYBRID_PATHS and state.get("sql_evidence") is not None:
        return _hybrid_repair(state, previous_plan, defect_text)

    prompt = render_prompt("REFLECTION", REFLECTION,
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
        logger.warning(
            "reflection LLM call failed, escalating request_id=%s",
            state.get("request_id"),
            exc_info=True,
        )

        return {
            "reflection_count": state.get("reflection_count", 0) + 1,
            "escalation_reason": "reflection could not produce a revised plan",
            "tokens_spent": 900,
        }

    logger.info(
        "reflection request_id=%s attempt=%d/%d defects=%d repairable=%s reasoning=%r",
        state.get("request_id"),
        state.get("reflection_count", 0) + 1,
        settings.max_reflection_retries,
        len(validation.defects),
        outcome.repairable,
        (outcome.reasoning or "")[:160],
    )

    if not outcome.repairable:
        logger.warning(
            "reflection judged defects unrepairable -> escalating request_id=%s", state.get("request_id")
        )

        return {
            "reflection_count": state.get("reflection_count", 0) + 1,
            "escalation_reason": f"defects are not repairable by re-planning: {outcome.reasoning}",
            "tokens_spent": 900,
        }

    revised = outcome.revised_plan
    revised.revision = (previous_plan.revision + 1) if previous_plan else 1

    logger.info(
        "reflection re-plan request_id=%s revision=%d path=%s steps=%d",
        state.get("request_id"),
        revised.revision,
        revised.path.value,
        len(revised.steps),
    )

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
        "repair_strategy": "llm_replan",
        "reflection_count": state.get("reflection_count", 0) + 1,
        "retrieved_chunks": None,
        "sql_evidence": None,
        "draft": None,
        "validation": None,
        "tokens_spent": 900,
    }


def _rag_repair(state: AgentState, previous_plan: EvidencePlan | None, defect_text: str) -> dict:
    """Self-correction for the RAG path without another LLM call.

    The RAG path retrieves with the question and the role's documents; it never reads
    the plan. So an LLM re-plan could only ever produce the same retrieval, the same
    draft and the same defects (the log showed two identical repairs per request).
    The repair that can change the outcome is concrete: search wider and hand the
    validator's defects back to the answer writer so it fixes them.
    """
    revised = (
        previous_plan.model_copy(deep=True)
        if previous_plan is not None
        else EvidencePlan(
            path=EvidencePath.RAG,
            steps=[PlanStep(order=1, source="policy_kb", objective="answer from policy text", must_prove="cited clauses")],
        )
    )
    revised.revision = (previous_plan.revision + 1) if previous_plan else 1

    directive = (
        "This is a repair of a draft the compliance validator rejected. Defects found:\n"
        f"{defect_text or '- (none listed)'}\n"
        "Fix every defect: cite only clause identifiers that appear in the extracts, exactly as written; "
        "drop any claim no extract supports; if the extracts do not settle the question, say so plainly."
    )

    logger.info(
        "reflection rag repair request_id=%s attempt=%d mode=widen_retrieval+defect_feedback defects=%d",
        state.get("request_id"),
        state.get("reflection_count", 0) + 1,
        len(state["validation"].defects),
    )

    return {
        "plan": revised,
        "plan_from_reflection": True,
        "replan_directive": directive,
        "repair_strategy": "rag_widen_and_fix",
        "reflection_count": state.get("reflection_count", 0) + 1,
        "retrieved_chunks": None,
        "sql_evidence": None,
        "draft": None,
        "validation": None,
        "tokens_spent": 0,
    }


def _hybrid_repair(state: AgentState, previous_plan: EvidencePlan | None, defect_text: str) -> dict:
    """Self-correction for a hybrid answer without an LLM re-plan.

    Neither evidence stage reads the plan. An LLM re-plan of "how often must a Low risk vendor be
    monitored under the vendor policy, and which of our vendors are Low risk?" sent the same question
    through the same stages twice more: the records query came back with 21 rows, then 28, and the
    validator rejected the same kind of draft each time (93 s, then escalated). What can change the
    verdict is the answer: the rows are kept, the policies are searched wider, and the writer gets
    the validator's defects.
    """
    revised = (
        previous_plan.model_copy(deep=True)
        if previous_plan is not None
        else EvidencePlan(
            path=EvidencePath.HYBRID,
            steps=[
                PlanStep(order=1, source="policy_kb", objective="the policy rule", must_prove="cited clauses"),
                PlanStep(
                    order=2,
                    source="compliance_db",
                    objective="the records the rule is checked against",
                    must_prove="every figure matches a returned row",
                ),
            ],
        )
    )
    revised.revision = (previous_plan.revision + 1) if previous_plan else 1

    directive = (
        "This is a repair of an answer the compliance validator rejected. The rows above are exactly what "
        "the query returned. Defects found:\n"
        f"{defect_text or '- (none listed)'}\n"
        "Rewrite the answer so every defect is fixed: put the [Document Title §clause] of the supporting "
        "extract on every sentence about the policy, report the records with the as-of date and only the "
        "values in the rows, state each caveat listed above, and if the evidence does not settle part of "
        "the question, say which part plainly."
    )

    logger.info(
        "reflection hybrid repair request_id=%s attempt=%d mode=keep_rows+widen_retrieval+defect_feedback defects=%d",
        state.get("request_id"),
        state.get("reflection_count", 0) + 1,
        len(state["validation"].defects),
    )

    return {
        "plan": revised,
        "plan_from_reflection": True,
        "replan_directive": directive,
        "repair_strategy": HYBRID_REDRAFT,
        "reflection_count": state.get("reflection_count", 0) + 1,
        "retrieved_chunks": None,
        "draft": None,
        "validation": None,
        "tokens_spent": 0,
    }


def _sql_repair(state: AgentState, previous_plan: EvidencePlan | None, defect_text: str) -> dict:
    """Self-correction for a records answer without another LLM re-plan.

    The NL2SQL path never reads the plan either: a re-plan ran the same question through the same
    query, got the same rows, took the selector and the narration from the LLM cache and handed the
    validator the identical draft (the log shows it rejected twice more before escalating). What
    can change the verdict is the answer text, so the rows are kept and the answer is rewritten
    with the validator's defects in front of the writer.
    """
    revised = (
        previous_plan.model_copy(deep=True)
        if previous_plan is not None
        else EvidencePlan(
            path=EvidencePath.NL2SQL,
            steps=[
                PlanStep(
                    order=1,
                    source="compliance_db",
                    objective="answer from the compliance records",
                    must_prove="every figure matches a returned row",
                )
            ],
        )
    )
    revised.revision = (previous_plan.revision + 1) if previous_plan else 1

    directive = (
        "This is a repair of an answer the compliance validator rejected. The rows above are exactly what "
        "the query returned. Defects found:\n"
        f"{defect_text or '- (none listed)'}\n"
        "Rewrite the answer so every defect is fixed: take every count from row_count, name only values "
        "that appear in the rows, state each caveat listed above, and if the rows cannot answer part of "
        "the question, say which part plainly."
    )

    logger.info(
        "reflection sql repair request_id=%s attempt=%d mode=keep_rows+rewrite_answer defects=%d",
        state.get("request_id"),
        state.get("reflection_count", 0) + 1,
        len(state["validation"].defects),
    )

    return {
        "plan": revised,
        "plan_from_reflection": True,
        "replan_directive": directive,
        "repair_strategy": SQL_RENARRATE,
        "reflection_count": state.get("reflection_count", 0) + 1,
        "draft": None,
        "validation": None,
        "tokens_spent": 0,
    }
