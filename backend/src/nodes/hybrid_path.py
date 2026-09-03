import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from src.graph.state import AgentState
from src.nodes.nl2sql_path import run_sql_evidence
from src.nodes.rag_path import gather_policy_evidence
from src.retrieval.adapter import format_context as build_context
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import HYBRID_ANSWER
from src.schemas.models import DraftAnswer


@traced_node("hybrid_path")
def hybrid_path_node(state: AgentState) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        policy_future = pool.submit(gather_policy_evidence, state)
        sql_future = pool.submit(run_sql_evidence, state)

        chunks, skipped = policy_future.result()
        evidence, failure = sql_future.result()

    rows = json.dumps(evidence.rows[:25], default=str, indent=2) if evidence else f"(no rows: {failure})"

    prompt = HYBRID_ANSWER.format(
        query=state["standalone_query"],
        context=build_context(chunks),
        as_of=evidence.as_of if evidence else "n/a",
        row_count=evidence.row_count if evidence else 0,
        rows=rows,
    )

    model = model_for("hybrid_generate").with_structured_output(DraftAnswer)

    draft: DraftAnswer = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "hybrid_generate"),
    )

    return {
        "evidence_path": "hybrid",
        "retrieved_chunks": chunks,
        "sql_evidence": evidence,
        "draft": draft,
        "degraded": bool(skipped) or state.get("degraded", False),
        "skipped_optional_nodes": skipped,
        "tokens_spent": 4000,
    }
