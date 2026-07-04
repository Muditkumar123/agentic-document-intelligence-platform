"""LangGraph execution engine for the agent workflow.

The workflow is defined once (the node functions in ``adip.agents.nodes`` and the
node order in ``adip.agents.runner``); this module compiles that definition into a
real LangGraph ``StateGraph``. Each node becomes a graph node wrapped with the
same AgentOps tracing as the sequential runner, and the runner's break-on-failure
loop becomes explicit **conditional edges**: after every node the graph routes to
``END`` when ``state.status == "failed"``, otherwise to the next node.

LangGraph executes locally and adds no model calls, so graph runs stay exactly as
deterministic as the nodes themselves. The dependency lives behind the
``[agents]`` extra and is imported lazily; when it is not installed the runner
falls back to the sequential loop, so a minimal install keeps working.
"""

from __future__ import annotations

import importlib.util
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Callable

from adip.agents.models import AgentState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adip.rag.retriever import RagIndex

AgentNode = Callable[[AgentState, "RagIndex"], AgentState]


def langgraph_available() -> bool:
    return importlib.util.find_spec("langgraph") is not None


def _state_update(state: AgentState) -> dict[str, Any]:
    """Full-state channel update for LangGraph (nodes mutate AgentState in place)."""
    return {field.name: getattr(state, field.name) for field in fields(state)}


def build_agent_graph(
    index: RagIndex,
    nodes: tuple[tuple[str, AgentNode], ...],
    run_node: Callable[[str, AgentNode, AgentState, "RagIndex"], AgentState],
):
    """Compile the node sequence into a LangGraph ``StateGraph``.

    ``run_node`` is the runner's tracing wrapper (timing, input/output summaries,
    error capture), injected so graph runs and sequential runs produce identical
    AgentOps traces.
    """
    if not langgraph_available():
        raise ImportError(
            'langgraph is required for the langgraph engine. Install the extra: pip install -e ".[agents]"'
        )

    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentState)

    def make_graph_node(node_name: str, node: AgentNode):
        def graph_node(state: AgentState) -> dict[str, Any]:
            return _state_update(run_node(node_name, node, state, index))

        return graph_node

    node_names = [name for name, _ in nodes]
    for node_name, node in nodes:
        graph.add_node(node_name, make_graph_node(node_name, node))

    graph.set_entry_point(node_names[0])
    for position, node_name in enumerate(node_names):
        next_name = node_names[position + 1] if position + 1 < len(node_names) else None
        if next_name is None:
            graph.add_edge(node_name, END)
            continue

        def route(state: AgentState, _next_name: str = next_name) -> str:
            return END if state.status == "failed" else _next_name

        graph.add_conditional_edges(node_name, route, {END: END, next_name: next_name})

    return graph.compile()


def run_agent_graph(
    initial_state: AgentState,
    index: RagIndex,
    nodes: tuple[tuple[str, AgentNode], ...],
    run_node: Callable[[str, AgentNode, AgentState, "RagIndex"], AgentState],
) -> AgentState:
    """Execute the workflow through LangGraph and return the final AgentState."""
    compiled = build_agent_graph(index, nodes, run_node)
    result = compiled.invoke(initial_state)
    return AgentState(**result)
