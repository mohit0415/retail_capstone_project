"""The hybrid reconciliation: one answer written from the policy clauses and the records together.

In the v4 workflow a hybrid question runs the evidence stages in their fixed order - rag_path
gathers the clauses, then nl2sql_path runs the records query - and nl2sql_path calls
draft_hybrid_answer() with both in front of it. There is no separate hybrid node: the clauses are
never gathered twice, and the trace shows the order the workflow runs in (policy, then records).
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import routed_model
from src.graph.state import AgentState
from src.guardrails.output_guard import CITATION_PATTERN
from src.observability.tracing import runnable_config
from src.prompts.langfuse_prompts import render_prompt
from src.prompts.library import HYBRID_ANSWER
from src.retrieval.adapter import format_context as build_context
from src.retrieval.citations import cite_uncited_sentences, with_inline_citations
from src.retrieval.sibling_expansion import append_sibling_clauses
from src.schemas.models import DraftAnswer, RetrievedChunk, SqlEvidence
from src.sqlpath.disclosure import rows_for_prompt, with_disclosures
from src.sqlpath.executor import sanity_check

logger = logging.getLogger(__name__)


def draft_hybrid_answer(
    state: AgentState,
    chunks: list[RetrievedChunk],
    evidence: SqlEvidence | None,
    failure: str,
    max_rows: int,
):
    """(draft, routing decision, caveats) for a policy rule reconciled against the records."""
    rows = rows_for_prompt(evidence, max_rows) if evidence else f"(no rows: {failure})"

    caveats = sanity_check(evidence) if evidence else [f"the database probe did not run: {failure}"]

    prompt = render_prompt(
        "HYBRID_ANSWER",
        HYBRID_ANSWER,
        query=state["standalone_query"],
        context=build_context(chunks),
        as_of=evidence.as_of if evidence else "n/a",
        row_count=evidence.row_count if evidence else 0,
        rows=rows,
        caveats="\n".join(f"- {item}" for item in caveats) or "- none",
    )

    model, decision = routed_model("hybrid_generate", state)

    messages = [SystemMessage(content=prompt)]

    if state.get("reflection_count") and state.get("replan_directive"):
        # a repair pass: the validator's defects go back to the writer
        messages.append(SystemMessage(content=state["replan_directive"]))

    messages.append(HumanMessage(content=state["standalone_query"]))

    draft: DraftAnswer = model.with_structured_output(DraftAnswer).invoke(
        messages,
        config=runnable_config(state, "hybrid_generate"),
    )

    # the policy sentences carry the clause that supports them, as in a RAG answer: the validator
    # failed "Low risk vendors must be monitored with an annual confirmation" twice for having no
    # citation. A sentence about the rows is sourced by the as-of date and is left alone.
    cited_answer, placed = cite_uncited_sentences(draft.answer, draft.cited_clauses, chunks, skip_records=True)

    if placed:
        logger.info(
            "hybrid draft left %d policy sentence(s) without a citation, placed the supporting clause on each "
            "request_id=%s citations=%s",
            len(placed),
            state.get("request_id"),
            placed,
        )

    added_citations: list[str] = []

    if draft.cited_clauses and not CITATION_PATTERN.search(cited_answer):
        # no sentence could be matched to a clause: fall back to the clauses the writer said it used
        cited_answer, added_citations = with_inline_citations(cited_answer, draft.cited_clauses, chunks)

        if added_citations:
            logger.info(
                "hybrid draft left out %d inline citation marker(s), appended them request_id=%s citations=%s",
                len(added_citations),
                state.get("request_id"),
                added_citations,
            )

    # section completeness: siblings of a cited clause (6.1 answered, 6.2 supplied)
    # are appended verbatim - the writer model leaves them out even when instructed
    cited_answer, siblings = append_sibling_clauses(cited_answer, [*draft.cited_clauses, *placed], chunks)

    if siblings:
        logger.info(
            "hybrid draft skipped %d sibling clause(s) of a cited section, appended them "
            "request_id=%s citations=%s",
            len(siblings),
            state.get("request_id"),
            siblings,
        )

    if cited_answer != draft.answer:
        draft = draft.model_copy(
            update={
                "answer": cited_answer,
                "cited_clauses": list(dict.fromkeys([*draft.cited_clauses, *placed, *siblings])),
            }
        )

    if evidence is not None:
        answer, added = with_disclosures(evidence, draft.answer)

        if added:
            logger.info(
                "hybrid draft left out %d record caveat(s), appended them request_id=%s caveats=%s",
                len(added),
                state.get("request_id"),
                added,
            )
            draft = draft.model_copy(update={"answer": answer})

    logger.info(
        "hybrid draft request_id=%s tier=%s clauses=%d sql_rows=%s answer_chars=%d cited=%d caveats=%d",
        state.get("request_id"),
        decision.tier,
        len(chunks),
        evidence.row_count if evidence else None,
        len(draft.answer or ""),
        len(draft.cited_clauses),
        len(caveats),
    )

    return draft, decision, caveats
