import logging

from langgraph.graph import END, START, StateGraph

from configs.settings import settings
from src.graph.routing import (
    after_confidence,
    after_entity_resolution,
    after_escalation,
    after_guardrail,
    after_intent,
    after_nl2sql_path,
    after_output_guardrail,
    after_panel,
    after_planner,
    after_rag_path,
    after_risk,
    after_validation,
)
from src.graph.state import AgentState
from src.nodes.confidence import confidence_node
from src.nodes.entity_resolution import entity_resolution_node
from src.nodes.escalation import escalation_node
from src.nodes.guardrail import input_guardrail_node
from src.nodes.intent_classification import intent_classification_node
from src.nodes.multi_agent_panel import multi_agent_panel_node
from src.nodes.nl2sql_path import nl2sql_path_node
from src.nodes.no_answer import no_answer_node
from src.nodes.planner import planner_node
from src.nodes.query_rewrite import query_rewrite_node
from src.nodes.rag_path import rag_path_node
from src.nodes.reflection import reflection_node
from src.nodes.risk_assessment import risk_assessment_node
from src.nodes.terminal import clarification_node, output_guardrail_node, refusal_node
from src.nodes.validation import compliance_validation_node

logger = logging.getLogger(__name__)

# the evidence stages, in the order every route runs them: RAG first, then NL2SQL, then the
# multi-agent panel, then compliance validation (src/graph/routing.py, evidence_stages)
EVIDENCE_NODES = ("rag_path", "nl2sql_path", "multi_agent_panel")

_checkpointer = None
_compiled = None


def build_checkpointer():
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(
            settings.database_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        saver = PostgresSaver(conn)
        saver.setup()
        _checkpointer = saver
        logger.info("postgres checkpointer ready")
    except Exception as exc:
        logger.warning("postgres checkpointer unavailable, using in-memory saver: %s", exc)

        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer = MemorySaver()

    return _checkpointer


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("entity_resolution", entity_resolution_node)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("planner", planner_node)
    graph.add_node("rag_path", rag_path_node)
    graph.add_node("nl2sql_path", nl2sql_path_node)
    graph.add_node("multi_agent_panel", multi_agent_panel_node)
    graph.add_node("compliance_validation", compliance_validation_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("confidence_scoring", confidence_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("escalation_manager", escalation_node)
    graph.add_node("safe_refusal", refusal_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("no_answer", no_answer_node)

    graph.add_edge(START, "input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        after_guardrail,
        {"refuse": "safe_refusal", "escalate": "escalation_manager", "continue": "query_rewrite"},
    )

    graph.add_edge("query_rewrite", "intent_classification")

    graph.add_conditional_edges(
        "intent_classification",
        after_intent,
        {"refuse": "safe_refusal", "escalate": "escalation_manager", "continue": "entity_resolution"},
    )

    graph.add_conditional_edges(
        "entity_resolution",
        after_entity_resolution,
        {"clarify": "clarification", "escalate": "escalation_manager", "continue": "risk_assessment"},
    )

    graph.add_conditional_edges(
        "risk_assessment",
        after_risk,
        {"escalate": "escalation_manager", "continue": "planner"},
    )

    # the planner's route decides which stages run; the first one it needs comes next
    graph.add_conditional_edges(
        "planner",
        after_planner,
        {
            "rag_path": "rag_path",
            "nl2sql_path": "nl2sql_path",
            "multi_agent_panel": "multi_agent_panel",
            "not_found": "no_answer",
            "escalate": "escalation_manager",
        },
    )

    graph.add_conditional_edges(
        "rag_path",
        after_rag_path,
        {
            "nl2sql_path": "nl2sql_path",
            "multi_agent_panel": "multi_agent_panel",
            "compliance_validation": "compliance_validation",
            "not_found": "no_answer",
            "escalate": "escalation_manager",
        },
    )

    graph.add_conditional_edges(
        "nl2sql_path",
        after_nl2sql_path,
        {
            "multi_agent_panel": "multi_agent_panel",
            "compliance_validation": "compliance_validation",
            "not_found": "no_answer",
            "escalate": "escalation_manager",
        },
    )

    graph.add_conditional_edges(
        "multi_agent_panel",
        after_panel,
        {"compliance_validation": "compliance_validation", "escalate": "escalation_manager"},
    )

    graph.add_edge("no_answer", "output_guardrail")

    graph.add_conditional_edges(
        "compliance_validation",
        after_validation,
        {
            "score": "confidence_scoring",
            "reflect": "reflection",
            "not_found": "no_answer",
            "escalate": "escalation_manager",
        },
    )

    graph.add_edge("reflection", "planner")

    graph.add_conditional_edges(
        "confidence_scoring",
        after_confidence,
        {"respond": "output_guardrail", "escalate": "escalation_manager"},
    )

    graph.add_conditional_edges(
        "output_guardrail",
        after_output_guardrail,
        {"escalate": "escalation_manager", "done": END},
    )

    graph.add_conditional_edges(
        "escalation_manager",
        after_escalation,
        {"review": "output_guardrail", "wait": END},
    )

    graph.add_edge("safe_refusal", END)
    graph.add_edge("clarification", END)

    return graph


def get_compiled_graph():
    global _compiled

    if _compiled is None:
        graph = build_graph()
        checkpointer = build_checkpointer()

        _compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_after=["escalation_manager"],
        )

        logger.info(
            "policy graph compiled nodes=%d evidence_stages=%s checkpointer=%s interrupt_after=escalation_manager",
            len(graph.nodes),
            " -> ".join([*EVIDENCE_NODES, "compliance_validation"]),
            type(checkpointer).__name__,
        )

    return _compiled
