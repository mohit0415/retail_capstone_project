import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from src.graph.state import AgentState
from src.guardrails.output_guard import canonical_citation, extract_citations
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import COMPLIANCE_VALIDATION
from src.retrieval.adapter import format_context as build_context
from src.schemas.enums import DefectType
from src.schemas.models import Defect, ValidationReport
from src.sqlpath.disclosure import data_defects, undisclosed

CONFLICT_DEFECTS = {DefectType.CLAUSE_CONFLICT, DefectType.POLICY_RECORD_CONFLICT}

RECORD_DEPENDENT_DEFECTS = {DefectType.POLICY_RECORD_CONFLICT, DefectType.SQL_SANITY_FAILURE}

BLOCKING_DEFECTS = {
    DefectType.UNGROUNDED_CLAIM,
    DefectType.POLICY_RULE_BREACH,
    DefectType.CLAUSE_CONFLICT,
    DefectType.POLICY_RECORD_CONFLICT,
    DefectType.SQL_SANITY_FAILURE,
    DefectType.MISSING_CITATION,
}

GENERIC_SECTIONS = {
    "annex",
    "appendix",
    "background",
    "compliance",
    "definitions",
    "enforcement",
    "figures",
    "general",
    "governance",
    "introduction",
    "objective",
    "objectives",
    "overview",
    "policy statement",
    "principles",
    "purpose",
    "responsibilities",
    "review",
    "roles and responsibilities",
    "scope",
    "summary",
}

OBLIGATION_LANGUAGE = re.compile(
    r"\b(must not|must|shall not|shall|may not|is required|are required|required to|"
    r"prohibited|forbidden|obliged|mandatory)\b",
    re.I,
)


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

    known_clauses = {canonical_citation(chunk.citation) for chunk in chunks}

    for citation in extract_citations(draft.answer):
        if canonical_citation(citation) not in known_clauses:
            defects.append(
                Defect(
                    defect_type=DefectType.UNGROUNDED_CLAIM,
                    description=f"answer cites [{citation}] which is not among the retrieved extracts",
                    offending_claim=citation,
                    suggested_repair="widen retrieval to the document that clause belongs to, or drop the claim",
                )
            )

    if evidence is not None:
        for problem in data_defects(evidence):
            defects.append(
                Defect(
                    defect_type=DefectType.SQL_SANITY_FAILURE,
                    description=problem,
                    suggested_repair="re-probe with a template that respects the pinned as_of date",
                )
            )

        for missing in undisclosed(evidence, draft.answer):
            defects.append(
                Defect(
                    defect_type=DefectType.SQL_SANITY_FAILURE,
                    description=f"the answer does not disclose that {missing.text}",
                    offending_claim=missing.key,
                    suggested_repair="state the caveat in the answer; the result itself is sound",
                )
            )

    return defects


def _normalise_section(section: str) -> str:
    return re.sub(r"\s+", " ", (section or "").strip().lower())


def _states_an_obligation(chunk) -> bool:
    return bool(OBLIGATION_LANGUAGE.search(chunk.content or ""))


def _clause_conflicts(state: AgentState) -> list[Defect]:
    chunks = state.get("retrieved_chunks", [])
    by_section: dict[str, list] = {}

    for chunk in chunks:
        heading = _normalise_section(chunk.section)

        if not heading or heading in GENERIC_SECTIONS:
            continue

        by_section.setdefault(heading, []).append(chunk)

    defects: list[Defect] = []

    for section, group in by_section.items():
        binding = [chunk for chunk in group if _states_an_obligation(chunk)]

        if len(binding) < 2:
            continue

        documents = {chunk.doc_type for chunk in binding}
        clauses = {f"{chunk.doc_type}:{chunk.clause_number}" for chunk in binding}

        if len(documents) < 2 or len(clauses) < 2:
            continue

        citations = ", ".join(f"{c.document_title} §{c.clause_number}" for c in binding)

        defects.append(
            Defect(
                defect_type=DefectType.CLAUSE_CONFLICT,
                description=(
                    f"section \"{section}\" carries binding language in more than one document "
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

    if evidence is None:
        all_defects = [d for d in all_defects if d.defect_type not in RECORD_DEPENDENT_DEFECTS]

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
