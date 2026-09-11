import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import routed_model
from src.graph.routing import current_route
from src.graph.state import AgentState
from src.guardrails.output_guard import CITATION_PATTERN, canonical_citation, extract_citations
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import COMPLIANCE_VALIDATION
from src.retrieval.adapter import format_context as build_context
from src.retrieval.citations import (
    citable_clauses,
    citation_keys,
    claim_units,
    is_record_sentence,
    uncited_sentences,
)
from src.schemas.enums import DefectType, EvidencePath
from src.schemas.models import Defect, ValidationReport
from src.sqlpath.disclosure import data_defects, rows_for_prompt, undisclosed
from src.sqlpath.grounding import figure_defect_text, unsupported_figures

logger = logging.getLogger(__name__)

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

# defect types that presuppose policy extracts: on an answer written only from database rows
# there is no clause to cite, breach or conflict with, so the validator model's verdicts of these
# kinds are noise
POLICY_ONLY_DEFECTS = {
    DefectType.MISSING_CITATION,
    DefectType.POLICY_RULE_BREACH,
    DefectType.CLAUSE_CONFLICT,
    DefectType.POLICY_RECORD_CONFLICT,
}

# the validator model's grounding verdicts on a records answer; blocking on the first pass (they
# drive the rewrite), advisory on the rewrite pass when the deterministic row check passes
RECORD_GROUNDING_DEFECTS = {DefectType.SQL_SANITY_FAILURE, DefectType.UNGROUNDED_CLAIM}

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


VALIDATOR_MAX_EXTRACTS = 4

# the validator reads every row the answer was written from (up to this cap) plus the row count;
# with 10 rows and no count it rejected a correct answer that listed all 31 vendors
VALIDATOR_MAX_ROWS = 60

# the uncited sentences named in the defect list; the writer fixes them all from the first few
MAX_UNCITED_REPORTED = 3


def validator_extracts(state: AgentState) -> list:
    """The extracts the validator reads: the cited ones first, topped up to a small cap.

    The validator used to receive every retrieved extract (up to 10 on a repair) plus an
    indented plan; the prompt was 2-3k tokens and on this Azure deployment each call took
    5-12 s. Uncited extracts cannot ground a claim anyway (a claim without a citation is
    already a missing_citation defect), so a few are enough for coverage checks.
    """
    chunks = list(state.get("retrieved_chunks") or [])
    draft = state.get("draft")

    if not chunks or draft is None:
        return chunks[:VALIDATOR_MAX_EXTRACTS]

    cited = {canonical_citation(item) for item in extract_citations(draft.answer)}
    cited.update(canonical_citation(item) for item in draft.cited_clauses or [])

    # a citation of a sub-clause (§4.1) points at the extract whose text carries that heading
    first = [chunk for chunk in chunks if citation_keys(chunk) & cited]
    rest = [chunk for chunk in chunks if not citation_keys(chunk) & cited]

    limit = max(VALIDATOR_MAX_EXTRACTS, len(first))

    return (first + rest)[:limit]


class ValidatorOutput(BaseModel):
    defects: list[Defect] = Field(default_factory=list)
    grounded_claim_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


def _rag_policy_answer(state: AgentState) -> bool:
    """A policy answer written by rag_path - the one writer that places a citation on every claim."""
    return (
        bool(state.get("retrieved_chunks"))
        and state.get("sql_evidence") is None
        and current_route(state) == EvidencePath.RAG.value
    )


def _claim_is_cited(claim: str | None, answer: str, unlocated: bool = True, rows_are_sourced: bool = False) -> bool:
    """Whether the sentence the model's missing-citation claim points at is sourced.

    A sentence is sourced by a [Document §clause] marker - or, in a hybrid answer, when it reports the
    rows, which are sourced by the as-of date. A claim that cannot be found in the answer counts as
    ``unlocated``.
    """
    needle = re.sub(r"\s+", " ", claim or "").strip().casefold()[:60]

    if not needle:
        return unlocated

    for unit in claim_units(answer):
        if needle in re.sub(r"\s+", " ", unit).casefold():
            return bool(CITATION_PATTERN.search(unit)) or (rows_are_sourced and is_record_sentence(unit))

    return unlocated


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

    known_clauses = set(citable_clauses(chunks))

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

    if _rag_policy_answer(state):
        # a policy answer from rag_path, which places a [Document §clause] marker on every claim an
        # extract clearly supports, so a substantive sentence left bare is a claim no extract backs
        for sentence in uncited_sentences(draft.answer)[:MAX_UNCITED_REPORTED]:
            clipped = sentence if len(sentence) <= 160 else f"{sentence[:157]}..."

            defects.append(
                Defect(
                    defect_type=DefectType.MISSING_CITATION,
                    description=f'this sentence makes a claim with no [Document §clause] citation: "{clipped}"',
                    offending_claim=clipped,
                    suggested_repair=(
                        "cite the extract that supports it, or drop the claim; if the extracts do not "
                        "settle it, say so plainly"
                    ),
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

        if not chunks:
            unsupported = unsupported_figures(evidence, draft.answer, state.get("standalone_query") or "")

            if unsupported:
                defects.append(
                    Defect(
                        defect_type=DefectType.SQL_SANITY_FAILURE,
                        description=figure_defect_text(evidence, unsupported),
                        offending_claim=", ".join(unsupported[:6]),
                        suggested_repair=(
                            "take every count from row_count and state only values that appear in the rows; "
                            "drop any figure you computed yourself"
                        ),
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


class _SkippedDecision:
    """Stands in for a routing decision when the model call is skipped entirely.

    routed_model() itself picks a deployment and is not free to call (it may hit a router,
    and tests treat calling it at all as "spending a validator call"), so it must not run
    until after we know the LLM verdict is actually needed.
    """

    tier = "skipped"

    def as_dict(self) -> dict:
        return {"node": "compliance_validation", "tier": self.tier}


@traced_node("compliance_validation")
def compliance_validation_node(state: AgentState) -> dict:
    draft = state.get("draft")
    evidence = state.get("sql_evidence")

    deterministic = _deterministic_defects(state)

    already_blocked = [d for d in deterministic if d.defect_type in BLOCKING_DEFECTS]
    llm_skipped = ""

    if draft is None:
        llm_skipped = "no draft to judge"
    elif already_blocked:
        # the draft cites a clause that was never retrieved (or breaks a SQL sanity rule):
        # it fails whatever the model says, so do not spend a model call confirming it
        llm_skipped = f"deterministic checks already failed the draft ({already_blocked[0].defect_type.value})"

    if llm_skipped:
        logger.info(
            "compliance validation LLM skipped request_id=%s reason=%s", state.get("request_id"), llm_skipped
        )
        llm_defects = []
        ratio = 0.0
        decision = _SkippedDecision()
    else:
        model, decision = routed_model("compliance_validation", state)
        plan = state.get("plan")

        prompt = render_prompt("COMPLIANCE_VALIDATION", COMPLIANCE_VALIDATION,
            query=state["standalone_query"],
            plan=plan.model_dump_json(exclude={"revision"}) if plan else "(no plan)",
            context=build_context(validator_extracts(state)),
            rows=rows_for_prompt(evidence, VALIDATOR_MAX_ROWS, with_query=True) if evidence else "(no rows)",
            answer=draft.answer,
        )

        try:
            judged: ValidatorOutput = model.with_structured_output(ValidatorOutput).invoke(
                [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
                config=runnable_config(state, "compliance_validation"),
            )
            # "advisory" is decided here, never by the model
            llm_defects = [d.model_copy(update={"advisory": False}) for d in judged.defects]
            ratio = judged.grounded_claim_ratio
        except Exception:
            logger.warning(
                "compliance validation LLM call failed, proceeding on deterministic defects only "
                "request_id=%s",
                state.get("request_id"),
                exc_info=True,
            )
            llm_defects = []
            ratio = 0.0

    record_only = evidence is not None and not state.get("retrieved_chunks")
    rewrite_pass = bool(state.get("reflection_count"))
    deterministic_blocking = [d for d in deterministic if d.defect_type in BLOCKING_DEFECTS]
    advisory_notes: list[str] = []

    if evidence is None:
        llm_defects = [d for d in llm_defects if d.defect_type not in RECORD_DEPENDENT_DEFECTS]

    rag_answer = _rag_policy_answer(state)
    hybrid_answer = current_route(state) == EvidencePath.HYBRID.value and evidence is not None

    if draft is not None and (rag_answer or hybrid_answer) and not deterministic_blocking:
        # the model's "missing citation" verdict on a sourced sentence is noise (the anti-bribery trace
        # failed twice on it with every claim cited) and is kept as a note. On a RAG answer the
        # deterministic check above has confirmed every claim carries a citation, so a claim the model
        # does not quote is noise too; a hybrid answer has no such check, so there only a quoted
        # sentence that is cited - or that reports the rows - is downgraded. Whether a citation
        # actually supports its claim is ungrounded_claim, which always blocks.
        llm_defects = [
            d.model_copy(update={"advisory": True})
            if d.defect_type is DefectType.MISSING_CITATION
            and _claim_is_cited(d.offending_claim, draft.answer, unlocated=rag_answer, rows_are_sourced=hybrid_answer)
            else d
            for d in llm_defects
        ]

    if record_only:
        llm_defects = [d for d in llm_defects if d.defect_type not in POLICY_ONLY_DEFECTS]

        if rewrite_pass and not deterministic_blocking:
            # the rows are the evidence and the deterministic check has confirmed every figure,
            # disclosure and date in the rewritten answer; a repeated model objection is kept as a
            # note on the answer instead of sending a checked answer to a human
            downgraded = []

            for defect in llm_defects:
                if defect.defect_type in RECORD_GROUNDING_DEFECTS:
                    defect = defect.model_copy(update={"advisory": True})
                    advisory_notes.append(defect.description)

                downgraded.append(defect)

            llm_defects = downgraded

    all_defects = deterministic + llm_defects

    if not any(d.defect_type is DefectType.CLAUSE_CONFLICT for d in all_defects):
        all_defects.extend(_clause_conflicts(state))

    blocking = [d for d in all_defects if d.defect_type in BLOCKING_DEFECTS and not d.advisory]
    conflict = any(d.defect_type in CONFLICT_DEFECTS and not d.advisory for d in all_defects)

    report = ValidationReport(
        passed=not blocking,
        defects=all_defects,
        grounded_claim_ratio=ratio,
        conflict_detected=conflict,
    )

    log = logger.info if report.passed else logger.warning

    log(
        "validation %s request_id=%s tier=%s deterministic=%d llm=%d blocking=%s conflict=%s "
        "grounded_ratio=%.2f record_only=%s reflection_count=%d",
        "PASSED" if report.passed else "FAILED",
        state.get("request_id"),
        decision.tier,
        len(deterministic),
        len(llm_defects),
        [d.defect_type.value for d in blocking] or "none",
        conflict,
        ratio,
        record_only,
        state.get("reflection_count", 0),
    )

    if blocking:
        # the defect types alone ("sql_sanity_failure") never said what was wrong; the reviewer
        # package has the full text, the log gets a short version
        logger.warning(
            "validation defects request_id=%s %s",
            state.get("request_id"),
            " | ".join(f"{d.defect_type.value}: {(d.description or '')[:180]}" for d in blocking),
        )

    result = {"validation": report, "model_routing": [decision.as_dict()], "tokens_spent": 2500}

    if llm_skipped:
        result["validation_llm_skipped"] = llm_skipped
        result["tokens_spent"] = 0

    if report.passed and advisory_notes and draft is not None:
        note = "Validator note: " + " ".join(advisory_notes)
        existing = (draft.uncertainty_note or "").strip()

        logger.info(
            "validation passed with %d advisory note(s) on a records answer request_id=%s",
            len(advisory_notes),
            state.get("request_id"),
        )

        result["draft"] = draft.model_copy(update={"uncertainty_note": f"{existing} {note}".strip()})

    return result
