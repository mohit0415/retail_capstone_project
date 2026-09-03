import logging

from langgraph.graph import END, START, StateGraph

from configs.settings import settings
from src.graph.routing import (
    after_confidence,
    after_entity_resolution,
    after_guardrail,
    after_intent,
    after_risk,
    after_validation,
    route_evidence_path,
)
from src.graph.state import AgentState
from src.nodes.agentic_rag import agentic_rag_node
from src.nodes.confidence import confidence_node
from src.nodes.entity_resolution import entity_resolution_node
from src.nodes.escalation import escalation_node
from src.nodes.guardrail import input_guardrail_node
from src.nodes.hybrid_path import hybrid_path_node
from src.nodes.intent_classification import intent_classification_node
from src.nodes.multi_agent_panel import multi_agent_panel_node
from src.nodes.nl2sql_path import nl2sql_path_node
from src.nodes.planner import planner_node
from src.nodes.query_rewrite import query_rewrite_node
from src.nodes.rag_path import rag_path_node
from src.nodes.reflection import reflection_node
from src.nodes.risk_assessment import risk_assessment_node
from src.nodes.terminal import clarification_node, output_guardrail_node, refusal_node
from src.nodes.validation import compliance_validation_node

logger = logging.getLogger(__name__)

_checkpointer = None
_compiled = None


def build_checkpointer():
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver.from_conn_string(settings.database_url)
        saver.setup()
        _checkpointer = saver
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
    graph.add_node("hybrid_path", hybrid_path_node)
    graph.add_node("agentic_rag", agentic_rag_node)
    graph.add_node("multi_agent_panel", multi_agent_panel_node)
    graph.add_node("compliance_validation", compliance_validation_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("confidence_scoring", confidence_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("escalation_manager", escalation_node)
    graph.add_node("safe_refusal", refusal_node)
    graph.add_node("clarification", clarification_node)

    graph.add_edge(START, "input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        after_guardrail,
        {"refuse": "safe_refusal", "continue": "query_rewrite"},
    )

    graph.add_edge("query_rewrite", "intent_classification")

    graph.add_conditional_edges(
        "intent_classification",
        after_intent,
        {"refuse": "safe_refusal", "continue": "entity_resolution"},
    )

    graph.add_conditional_edges(
        "entity_resolution",
        after_entity_resolution,
        {"clarify": "clarification", "continue": "risk_assessment"},
    )

    graph.add_conditional_edges(
        "risk_assessment",
        after_risk,
        {"escalate": "escalation_manager", "continue": "planner"},
    )

    graph.add_conditional_edges(
        "planner",
        route_evidence_path,
        {
            "rag": "rag_path",
            "nl2sql": "nl2sql_path",
            "hybrid": "hybrid_path",
            "agentic": "agentic_rag",
            "high_risk_panel": "multi_agent_panel",
        },
    )

    for path_node in ("rag_path", "nl2sql_path", "hybrid_path", "agentic_rag", "multi_agent_panel"):
        graph.add_edge(path_node, "compliance_validation")

    graph.add_conditional_edges(
        "compliance_validation",
        after_validation,
        {
            "score": "confidence_scoring",
            "reflect": "reflection",
            "escalate": "escalation_manager",
        },
    )

    graph.add_edge("reflection", "planner")

    graph.add_conditional_edges(
        "confidence_scoring",
        after_confidence,
        {"respond": "output_guardrail", "escalate": "escalation_manager"},
    )

    graph.add_edge("output_guardrail", END)
    graph.add_edge("escalation_manager", END)
    graph.add_edge("safe_refusal", END)
    graph.add_edge("clarification", END)

    return graph


def get_compiled_graph():
    global _compiled

    if _compiled is None:
        _compiled = build_graph().compile(
            checkpointer=build_checkpointer(),
            interrupt_after=["escalation_manager"],
        )

    return _compiled
