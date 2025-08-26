from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combine_output:str

# Node
def call_llm_1(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def call_llm_2(state: State):
    """Second LLM call to generate a story"""

    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}

def call_llm_3(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    """Combine the joke and story into a single output"""

    combined = f"Here's a story, joke and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['topic']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}\n\n"
    return {"combined_output": combined}



# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("call_llm_1", call_llm_1)
workflow.add_node("call_llm_2", call_llm_2)
workflow.add_node("call_llm_3", call_llm_3)
workflow.add_node("aggregator", aggregator)

# Add edges to connect nodes
workflow.add_edge(START, "call_llm_1")
workflow.add_edge(START, "call_llm_2")
workflow.add_edge(START, "call_llm_3")
workflow.add_edge("call_llm_1", "aggregator")
workflow.add_edge("call_llm_2", "aggregator")
workflow.add_edge("call_llm_3", "aggregator")
workflow.add_edge("aggregator", END)

chain = workflow.compile()

display(Image(chain.get_graph().draw_mermaid_png()))
