from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from research_assistant_example.src.core.state import GenerateAnalystsState
from research_assistant_example.src.graph.nodes import create_analysts, human_feedback
from research_assistant_example.src.graph.edges import should_continue
from research_assistant_example.src.core.context import ModelContext

def build_workflow():
    # Add nodes and edges
    graph = StateGraph(GenerateAnalystsState, context_schema=ModelContext)
    graph.add_node("create_analysts", create_analysts)
    graph.add_node("human_feedback", human_feedback)
    graph.add_edge(START, "create_analysts")
    graph.add_edge("create_analysts", "human_feedback")
    graph.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

    # Compile
    memory = MemorySaver()
    return graph.compile(interrupt_before=['human_feedback'], checkpointer=memory)