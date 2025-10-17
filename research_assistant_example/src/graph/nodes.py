from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from research_assistant_example.src.prompts import prompts
from research_assistant_example.src.core.state import Perspectives, GenerateAnalystsState
from research_assistant_example.src.core.context import ModelContext
from research_assistant_example.src.core.llm import MODELS


def create_analysts(state: GenerateAnalystsState, runtime: Runtime[ModelContext]):
    """ Create analysts """

    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # Enforce structured output
    llm = MODELS[runtime.context.model_provider]
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = prompts.analyst_instructions.format(topic=topic,
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts)

    # Generate question 
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)] +
        [HumanMessage(content="Generate the set of analysts.")],
    )

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass


