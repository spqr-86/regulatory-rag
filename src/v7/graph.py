"""V7 RAG pipeline — graph assembly.

build_graph(overrides) assembles the full LangGraph StateGraph.
overrides allows replacing any node for testing.

Source spec: docs/feature/migration-v7 (lines 1500-1549).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from src.v7.nodes.abstain import abstain
from src.v7.nodes.evaluate_complex import evaluate_complex, route_after_decision
from src.v7.nodes.evaluate_triage import evaluate_triage
from src.v7.nodes.generate_answer import generate_answer
from src.v7.nodes.intent_gate import intent_gate, route_by_intent
from src.v7.nodes.rag_complex import rag_complex
from src.v7.nodes.rag_simple import rag_simple
from src.v7.nodes.router import clarify_respond, route_after_router, router
from src.v7.state_types import RAGState


def build_graph(
    overrides: Optional[Dict[str, Callable]] = None,
) -> StateGraph:
    """Assemble the v7 graph.

    overrides: dict of {node_name: replacement_function}.
    Allows replacing any node for testing.

    Usage:
        app = build_graph().compile()
        custom = build_graph({"router": my_router}).compile()
    """
    nodes: Dict[str, Any] = {
        "intent_gate": intent_gate,
        "router": router,
        "clarify_respond": clarify_respond,
        "rag_simple": rag_simple,
        "evaluate_triage": evaluate_triage,
        "rag_complex": rag_complex,
        "evaluate_complex": evaluate_complex,
        "generate_answer": generate_answer,
        "abstain": abstain,
    }
    if overrides:
        nodes.update(overrides)

    g = StateGraph(RAGState)
    for name, func in nodes.items():
        g.add_node(name, func)

    g.set_entry_point("intent_gate")

    g.add_conditional_edges(
        "intent_gate",
        route_by_intent,
        {"end": END, "router": "router"},
    )
    g.add_conditional_edges(
        "router",
        route_after_router,
        {"clarify_respond": "clarify_respond", "rag_simple": "rag_simple"},
    )
    g.add_edge("clarify_respond", END)
    g.add_edge("rag_simple", "evaluate_triage")
    g.add_conditional_edges(
        "evaluate_triage",
        route_after_decision,
        {
            "generate": "generate_answer",
            "complex": "rag_complex",
            "abstain": "abstain",
        },
    )
    g.add_edge("rag_complex", "evaluate_complex")
    g.add_conditional_edges(
        "evaluate_complex",
        route_after_decision,
        {"generate": "generate_answer", "abstain": "abstain"},
    )
    g.add_edge("generate_answer", END)
    g.add_edge("abstain", END)

    return g
