import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.llms import model_for
from configs.settings import settings
from src.graph.state import AgentState
from src.nodes.nl2sql_path import run_sql_evidence
from src.nodes.rag_path import gather_policy_evidence
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import (
    PANEL_CHALLENGER,
    PANEL_CONSENSUS,
    PANEL_DATA_VERIFIER,
    PANEL_POLICY_INTERPRETER,
    PANEL_REPAIR,
)
from src.retrieval.adapter import format_context as build_context
from src.schemas.models import DraftAnswer, PanelOpinion, PanelVerdict
from src.sqlpath.executor import sanity_check


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
    prompt = PANEL_POLICY_INTERPRETER.format(query=state["standalone_query"], context=context)
    model = model_for("panel_policy_interpreter").with_structured_output(InterpreterOutput)

    return model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "panel_policy_interpreter"),
    )


def _verify(state: AgentState, evidence, rows: str, caveats: str) -> VerifierOutput:
    prompt = PANEL_DATA_VERIFIER.format(
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
    prompt = PANEL_CHALLENGER.format(
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
    prompt = PANEL_CONSENSUS.format(
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
    prompt = PANEL_REPAIR.format(
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


@traced_node("multi_agent_panel")
def multi_agent_panel_node(state: AgentState) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        policy_future = pool.submit(gather_policy_evidence, state)
        sql_future = pool.submit(run_sql_evidence, state)

        chunks, skipped = policy_future.result()
        evidence, failure = sql_future.result()

    context = build_context(chunks)
    rows = json.dumps(evidence.rows[:25], default=str, indent=2) if evidence else f"(no rows: {failure})"

    listed = sanity_check(evidence) if evidence else [f"the database probe did not run: {failure}"]
    caveats = "\n".join(f"- {item}" for item in listed) or "- none"

    with ThreadPoolExecutor(max_workers=2) as pool:
        interpreter_future = pool.submit(_interpret, state, context)
        verifier_future = pool.submit(_verify, state, evidence, rows, caveats)

        interpreter = interpreter_future.result()
        verifier = verifier_future.result()

    challenger = _challenge(state, context, rows, interpreter.position, verifier.position)

    challenger_summary = (
        "\n".join(challenger.objections) or challenger.position or "no material objection"
    )

    consensus = _consensus(state, interpreter.position, verifier.position, challenger_summary)

    repairs_used = state.get("panel_repair_count", 0)
    tokens = 7000

    if _unresolved(consensus, challenger) and repairs_used < settings.panel_repair_passes:
        repaired = _repair(state, consensus, challenger.objections, context, rows)
        repairs_used += 1
        tokens += 2500

        if (not repaired.unresolved_conflict and (repaired.dissent or not challenger.material)) or repaired.dissent:
            consensus = repaired

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

    draft = DraftAnswer(
        answer=consensus.answer,
        cited_clauses=consensus.cited_clauses,
        uncertainty_note=consensus.uncertainty_note,
    )

    return {
        "evidence_path": "high_risk_panel",
        "retrieved_chunks": chunks,
        "sql_evidence": evidence,
        "panel_verdict": verdict,
        "panel_repair_count": repairs_used,
        "draft": draft,
        "degraded": bool(skipped) or state.get("degraded", False),
        "skipped_optional_nodes": skipped,
        "tokens_spent": tokens,
    }
