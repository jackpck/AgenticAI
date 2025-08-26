from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="The next step in the routing process"
    )

# Augment the LLM with scehma for structured output
router = llm.with_structured_output(Route)

# Graph state
class State(TypedDict):
    input: str
    decision: str
    output: str

# Node
def call_llm_1(state: State):
    """First LLM call to generate initial joke"""

    print("write a joke")
    msg = llm.invoke(f"Write a short joke about {state['input']}")
    return {"output": msg.content}

def call_llm_2(state: State):
    """Second LLM call to generate a story"""

    print("write a story")
    msg = llm.invoke(f"Write a story about {state['input']}")
    return {"output": msg.content}

def call_llm_3(state: State):
    """Third LLM call to generate poem"""

    print("write a poem")
    msg = llm.invoke(f"Write a poem about {state['input']}")
    return {"output": msg.content}

def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke or poem based on the user's request."
            ),
            HumanMessage(content=state["input"])
        ]
    )

    return {"decision":decision.step}

def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("llm_call_router", llm_call_router)
workflow.add_node("call_llm_1", call_llm_1)
workflow.add_node("call_llm_2", call_llm_2)
workflow.add_node("call_llm_3", call_llm_3)

# Add edges to connect nodes
workflow.add_edge(START, "llm_call_router")
workflow.add_conditional_edges(
    "llm_call_router", route_decision,
    {"llm_call_1": "llm_call_1",
     "llm_call_2": "llm_call_2",
     "llm_call_3": "llm_call_3"
     }
)
workflow.add_edge("llm_call_1", END)
workflow.add_edge("llm_call_2", END)
workflow.add_edge("llm_call_3", END)

chain = workflow.compile()

display(Image(chain.get_graph().draw_mermaid_png()))
