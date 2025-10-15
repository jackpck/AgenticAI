from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from research_assistant_example.src.prompts import prompts
from research_assistant_example.src.core.llm import LLM
from research_assistant_example.src.core.state import Analyst, Perspectives, GenerateAnalystsState

llm = LLM()

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """

    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = prompts.analyst_instructions.format(topic=topic,
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts)

    # Generate question 
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")])

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass


def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise end
    return END