from typing_extensions import TypedDict, Literal
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

# Schema for structured output to use in planning
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report."
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section."
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report."
    )

# Augment the LLM with schema for structured output
# Bind List[Section] to the LLM (planner), reflect on it and produce a list of sections (# sections not fixed a priori)
planner = llm.with_structured_output(Sections)

# Graph state
class State(TypedDict):
    topic: str
    sections: List[Section]
    completed_sections: Annotated[
        list, operator.add
    ] # All workers write to this key in parallel
    final_report: str # final report

# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add] # overlap key with State so they are in-sync

# Node
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    report_sections = planner.invoke(
        [
            SystemMessage(content="generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}")
        ]
    )

    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    section = llm.invoke(
        [
            SystemMessage(content="Write a report section."),
            HumanMessage(content=f"""Here is the section name: {state['section'].name} and
                                 description: {state['section'].name}""",
                         )
        ]
    )

    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API to spawn worker dynamically
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("llm_call", llm_call)
workflow.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
workflow.add_edge("llm_call", "synthesizer")
workflow.add_edge("synthesizer", END)

chain = workflow.compile()

display(Image(chain.get_graph().draw_mermaid_png()))
