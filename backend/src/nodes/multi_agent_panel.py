import logging
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from configs.settings import settings
from src.auth.rbac import allowed_tables
from src.graph.state import AgentState
from src.nodes.nl2sql_path import NARRATION_MAX_ROWS
from src.observability.logging_config import with_request_context
from src.observability.tracing import runnable_config, traced_node
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import (
    PANEL_CHALLENGER,
    PANEL_CONSENSUS,
    PANEL_DATA_VERIFIER,
    PANEL_POLICY_INTERPRETER,
    PANEL_REPAIR,
)
from src.retrieval.adapter import format_context as build_context
from src.retrieval.citations import with_inline_citations
from src.schemas.models import DraftAnswer, PanelOpinion, PanelVerdict
from src.sqlpath.disclosure import rows_for_prompt, with_disclosures
from src.sqlpath.executor import sanity_check

logger = logging.getLogger(__name__)


class InterpreterOutput(BaseModel):
    position: str
    supporting_citations: list[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    position: str
    row_identifiers: list[str] = Field(default_factory=list)
    data_is_silent_on: list[str] = Field(default_factory=list)


class ChallengerOutput(BaseModel):
    objections: list[str] = Field(default_factory=list)
    material: bool = Field(description="true when at least one objection would change the conclusion")
    position: str = ""


class ConsensusOutput(BaseModel):
    answer: str
    cited_clauses: list[str] = Field(default_factory=list)
    dissent: list[str] = Field(default_factory=list)
    unresolved_conflict: bool = False
    uncertainty_note: str = ""


def _interpret(state: AgentState, context: str) -> InterpreterOutput:
    prompt = render_prompt("PANEL_POLICY_INTERPRETER", PANEL_POLICY_INTERPRETER, query=state["standalone_query"], context=context)
    model = model_for("panel_policy_interpreter").with_structured_output(InterpreterOutput)

    return model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "panel_policy_interpreter"),
    )


def _verify(state: AgentState, evidence, rows: str, caveats: str) -> VerifierOutput:
    prompt = render_prompt("PANEL_DATA_VERIFIER", PANEL_DATA_VERIFIER,
        query=state["standalone_query"],
        as_of=evidence.as_of if evidence else "n/a",
        row_count=evidence.row_count if evidence else 0,
        rows=rows,
        caveats=caveats,
    )
    model = model_for("panel_data_verifier").with_structured_output(VerifierOutput)

    return model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "panel_data_verifier"),
    )


def _challenge(state: AgentState, context: str, rows: str, interpreter: str, verifier: str) -> ChallengerOutput:
    prompt = render_prompt("PANEL_CHALLENGER", PANEL_CHALLENGER,
        query=state["standalone_query"],
        interpreter_position=interpreter,
        verifier_position=verifier,
        context=context,
        rows=rows,
    )
    model = model_for("panel_challenger").with_structured_output(ChallengerOutput)

    return model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "panel_challenger"),
    )


def _consensus(state: AgentState, interpreter: str, verifier: str, challenger: str) -> ConsensusOutput:
    prompt = render_prompt("PANEL_CONSENSUS", PANEL_CONSENSUS,
        query=state["standalone_query"],
        interpreter_position=interpreter,
        verifier_position=verifier,
        challenger_position=challenger,
    )
    model = model_for("panel_consensus").with_structured_output(ConsensusOutput)

    return model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "panel_consensus"),
    )


def _repair(state: AgentState, draft: ConsensusOutput, objections: list[str], context: str, rows: str) -> ConsensusOutput:
    prompt = render_prompt("PANEL_REPAIR", PANEL_REPAIR,
        query=state["standalone_query"],
        draft_answer=draft.answer,
        objections="\n".join(f"- {item}" for item in objections) or "- none",
        context=context,
        rows=rows,
    )
    model = model_for("panel_consensus").with_structured_output(ConsensusOutput)

    return model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "panel_repair"),
    )


def _unresolved(consensus: ConsensusOutput, challenger: ChallengerOutput) -> bool:
    return consensus.unresolved_conflict or (challenger.material and not consensus.dissent)


def _records_gap(state: AgentState) -> str:
    """Why the panel has no rows in front of it."""
    if state.get("sql_failure"):
        return state["sql_failure"]

    if not allowed_tables(state.get("access_scopes", [])):
        return "this role has no access to the compliance database"

    return "the records stage returned no evidence"


@traced_node("multi_agent_panel")
def multi_agent_panel_node(state: AgentState) -> dict:
    request_id = state.get("request_id")
    started = time.perf_counter()

    logger.info(
        "panel START request_id=%s risk=%s scenario=%s",
        request_id,
        state["risk"].final_level.value if state.get("risk") else None,
        state["risk"].scenario_id if state.get("risk") else None,
    )

    # the evidence was gathered by the stages that ran before this one, in order - rag_path read the
    # clauses, nl2sql_path ran the records query - and the panel reviews exactly that evidence
    chunks = list(state.get("retrieved_chunks") or [])
    evidence = state.get("sql_evidence")
    failure = "" if evidence is not None else _records_gap(state)

    logger.info(
        "panel evidence request_id=%s chunks=%d sql_rows=%s sql_failure=%s (gathered by rag_path -> nl2sql_path)",
        request_id,
        len(chunks),
        evidence.row_count if evidence else None,
        failure or "none",
    )

    context = build_context(chunks)
    rows = rows_for_prompt(evidence, NARRATION_MAX_ROWS) if evidence else f"(no rows: {failure})"

    listed = sanity_check(evidence) if evidence else [f"the database probe did not run: {failure}"]
    caveats = "\n".join(f"- {item}" for item in listed) or "- none"

    stage = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        interpreter_future = pool.submit(with_request_context(_interpret), state, context)
        verifier_future = pool.submit(with_request_context(_verify), state, evidence, rows, caveats)

        interpreter = interpreter_future.result()
        verifier = verifier_future.result()

    logger.info(
        "panel interpreter+verifier done request_id=%s citations=%d silent_on=%d elapsed_ms=%.0f",
        request_id,
        len(interpreter.supporting_citations),
        len(verifier.data_is_silent_on),
        (time.perf_counter() - stage) * 1000,
    )

    stage = time.perf_counter()
    challenger = _challenge(state, context, rows, interpreter.position, verifier.position)

    logger.info(
        "panel challenger done request_id=%s objections=%d material=%s elapsed_ms=%.0f",
        request_id,
        len(challenger.objections),
        challenger.material,
        (time.perf_counter() - stage) * 1000,
    )

    challenger_summary = (
        "\n".join(challenger.objections) or challenger.position or "no material objection"
    )

    stage = time.perf_counter()
    consensus = _consensus(state, interpreter.position, verifier.position, challenger_summary)

    logger.info(
        "panel consensus done request_id=%s dissent=%d unresolved=%s cited=%d elapsed_ms=%.0f",
        request_id,
        len(consensus.dissent),
        consensus.unresolved_conflict,
        len(consensus.cited_clauses),
        (time.perf_counter() - stage) * 1000,
    )

    repairs_used = state.get("panel_repair_count", 0)
    tokens = 7000

    if _unresolved(consensus, challenger) and repairs_used < settings.panel_repair_passes:
        logger.info(
            "panel repair pass request_id=%s pass=%d/%d objections=%d",
            request_id,
            repairs_used + 1,
            settings.panel_repair_passes,
            len(challenger.objections),
        )

        repaired = _repair(state, consensus, challenger.objections, context, rows)
        repairs_used += 1
        tokens += 2500

        accepted = (
            not repaired.unresolved_conflict and (repaired.dissent or not challenger.material)
        ) or bool(repaired.dissent)

        if accepted:
            consensus = repaired

        logger.info(
            "panel repair %s request_id=%s unresolved_after=%s dissent_after=%d",
            "ACCEPTED" if accepted else "DISCARDED",
            request_id,
            repaired.unresolved_conflict,
            len(repaired.dissent),
        )

    verdict = PanelVerdict(
        consensus=consensus.answer,
        dissent=consensus.dissent,
        unresolved_conflict=_unresolved(consensus, challenger),
        opinions=[
            PanelOpinion(
                agent="policy_interpreter",
                position=interpreter.position,
                supporting_citations=interpreter.supporting_citations,
            ),
            PanelOpinion(
                agent="data_verifier",
                position=verifier.position,
                objections=verifier.data_is_silent_on,
            ),
            PanelOpinion(
                agent="challenger",
                position=challenger.position,
                objections=challenger.objections,
            ),
        ],
    )

    answer = consensus.answer

    if consensus.cited_clauses:
        answer, added_citations = with_inline_citations(answer, consensus.cited_clauses, chunks)

        if added_citations:
            logger.info(
                "panel consensus left out %d inline citation marker(s), appended them request_id=%s citations=%s",
                len(added_citations),
                request_id,
                added_citations,
            )

    if evidence is not None:
        # a caveat about the records (scope, row cap, generated query) is fixed text; adding it
        # here keeps a sound panel answer from failing validation on a missing caveat
        answer, _added = with_disclosures(evidence, answer)

    draft = DraftAnswer(
        answer=answer,
        cited_clauses=consensus.cited_clauses,
        uncertainty_note=consensus.uncertainty_note,
    )

    log = logger.warning if verdict.unresolved_conflict else logger.info

    log(
        "panel END request_id=%s unresolved_conflict=%s dissent=%d repairs=%d tokens_est=%d total_ms=%.0f",
        request_id,
        verdict.unresolved_conflict,
        len(verdict.dissent),
        repairs_used,
        tokens,
        (time.perf_counter() - started) * 1000,
    )

    return {
        "evidence_path": "high_risk_panel",
        "panel_verdict": verdict,
        "panel_repair_count": repairs_used,
        "draft": draft,
        "tokens_spent": tokens,
    }
