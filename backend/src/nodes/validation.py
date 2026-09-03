import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from src.graph.state import AgentState
from src.guardrails.output_guard import extract_citations
from src.retrieval.adapter import format_context as build_context
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import COMPLIANCE_VALIDATION
from src.schemas.enums import DefectType
from src.schemas.models import Defect, ValidationReport
from src.sqlpath.executor import sanity_check

CONFLICT_DEFECTS = {DefectType.CLAUSE_CONFLICT, DefectType.POLICY_RECORD_CONFLICT}

BLOCKING_DEFECTS = {
    DefectType.UNGROUNDED_CLAIM,
    DefectType.POLICY_RULE_BREACH,
    DefectType.CLAUSE_CONFLICT,
    DefectType.POLICY_RECORD_CONFLICT,
    DefectType.SQL_SANITY_FAILURE,
    DefectType.MISSING_CITATION,
}


class ValidatorOutput(BaseModel):
    defects: list[Defect] = Field(default_factory=list)
    grounded_claim_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


def _deterministic_defects(state: AgentState) -> list[Defect]:
    defects: list[Defect] = []
    draft = state.get("draft")
    chunks = state.get("retrieved_chunks", [])
    evidence = state.get("sql_evidence")

    if draft is None:
        return [
            Defect(
                defect_type=DefectType.COVERAGE_GAP,
                description="no draft answer was produced by the execution path",
            )
        ]

    known_clauses = {f"{chunk.document_title} §{chunk.clause_number}" for chunk in chunks}

    for citation in extract_citations(draft.answer):
        if citation not in known_clauses:
            defects.append(
                Defect(
                    defect_type=DefectType.UNGROUNDED_CLAIM,
                    description=f"answer cites [{citation}] which is not among the retrieved extracts",
                    offending_claim=citation,
                    suggested_repair="widen retrieval to the document that clause belongs to, or drop the claim",
                )
            )

    if evidence is not None:
        for problem in sanity_check(evidence):
            defects.append(
                Defect(
                    defect_type=DefectType.SQL_SANITY_FAILURE,
                    description=problem,
                    suggested_repair="restate the figure with its as_of date and its row-cap caveat",
                )
            )

    return defects


def _clause_conflicts(state: AgentState) -> list[Defect]:
    chunks = state.get("retrieved_chunks", [])
    by_section: dict[str, list] = {}

    for chunk in chunks:
        by_section.setdefault(chunk.section.lower(), []).append(chunk)

    defects: list[Defect] = []

    for section, group in by_section.items():
        documents = {chunk.doc_type for chunk in group}

        if len(documents) > 1 and len(group) > 1:
            citations = ", ".join(f"{c.document_title} §{c.clause_number}" for c in group)
            defects.append(
                Defect(
                    defect_type=DefectType.CLAUSE_CONFLICT,
                    description=(
                        f"section \"{section}\" is governed by clauses from more than one document "
                        f"({citations}); the answer must say which one governs"
                    ),
                    suggested_repair="state the precedence between the documents, or escalate",
                )
            )

    return defects


@traced_node("compliance_validation")
def compliance_validation_node(state: AgentState) -> dict:
    draft = state.get("draft")
    evidence = state.get("sql_evidence")

    deterministic = _deterministic_defects(state)

    prompt = COMPLIANCE_VALIDATION.format(
        query=state["standalone_query"],
        plan=state["plan"].model_dump_json(indent=2) if state.get("plan") else "(no plan)",
        context=build_context(state.get("retrieved_chunks", [])),
        rows=json.dumps(evidence.rows[:25], default=str, indent=2) if evidence else "(no rows)",
        answer=draft.answer if draft else "(no answer)",
    )

    model = model_for("compliance_validation").with_structured_output(ValidatorOutput)

    try:
        judged: ValidatorOutput = model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
            config=runnable_config(state, "compliance_validation"),
        )
        llm_defects = judged.defects
        ratio = judged.grounded_claim_ratio
    except Exception:
        llm_defects = []
        ratio = 0.0

    all_defects = deterministic + llm_defects

    if not any(d.defect_type is DefectType.CLAUSE_CONFLICT for d in all_defects):
        all_defects.extend(_clause_conflicts(state))

    blocking = [d for d in all_defects if d.defect_type in BLOCKING_DEFECTS]
    conflict = any(d.defect_type in CONFLICT_DEFECTS for d in all_defects)

    report = ValidationReport(
        passed=not blocking,
        defects=all_defects,
        grounded_claim_ratio=ratio,
        conflict_detected=conflict,
    )

    return {"validation": report, "tokens_spent": 2500}
