from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from research_assistant_example.src.schema.state import GenerateAnalystsState, InterviewState
from research_assistant_example.src.schema.context import ModelContext
from research_assistant_example.src.graph.nodes import create_analysts, human_feedback
from research_assistant_example.src.graph.nodes import generate_question, search_wikipedia, generate_answer, \
    save_interview, write_section
from research_assistant_example.src.graph.edges import should_continue, route_messages

def build_analyst_workflow():
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


def build_interview_workflow():
    graph = StateGraph(InterviewState)
    graph.add_node("ask_question", generate_question)
    graph.add_node("search_wikipedia", search_wikipedia)
    graph.add_node("answer_question", generate_answer)
    graph.add_node("save_interview", save_interview)
    graph.add_node("write_section", write_section)

    # Flow
    graph.add_edge(START, "ask_question")
    graph.add_edge("ask_question", "search_wikipedia")
    graph.add_edge("search_wikipedia", "answer_question")
    graph.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
    graph.add_edge("save_interview", "write_section")
    graph.add_edge("write_section", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

