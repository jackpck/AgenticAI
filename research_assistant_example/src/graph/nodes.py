from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string
from langgraph.types import Send


from research_assistant_example.src.prompts import prompts
from research_assistant_example.src.schema.state import Perspectives, GenerateAnalystsState, InterviewState, SearchQuery
from research_assistant_example.src.schema.state import ResearchGraphState
from research_assistant_example.src.schema.context import ModelContext
from research_assistant_example.src.llm.llm import MODELS


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


def generate_question(state: InterviewState, runtime: Runtime[ModelContext]):
    """ Node to generate a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = prompts.question_instructions.format(goals=analyst.persona)
    llm = MODELS[runtime.context.model_provider]
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {"messages": [question]}


def search_wikipedia(state: InterviewState, runtime: Runtime[ModelContext]):

    # Get state
    messages = state["messages"]

    llm = MODELS[runtime.context.model_provider]
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([prompts.search_instructions] +
                                         messages)

    search_docs = WikipediaLoader(query=search_query.search_query,
                                  load_max_docs=2).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def generate_answer(state: InterviewState, runtime: Runtime[ModelContext]):

    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    system_message = prompts.answer_instructions.format(goals=analyst.persona,
                                                        context=context)
    llm = MODELS[runtime.context.model_provider]
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    answer.name = "expert"

    return {"messages": [answer]}


def save_interview(state: InterviewState):

    # Get messages
    messages = state["messages"]

    interview = get_buffer_string(messages)

    return {"interview": interview}


def write_section(state: InterviewState, runtime:Runtime[ModelContext]):

    # Get state
    context = state["context"]
    analyst = state["analyst"]

    system_message = prompts.section_writer_instructions.format(focus=analyst.description)
    llm = MODELS[runtime.context.model_provider]
    section = llm.invoke([SystemMessage(content=system_message)] +
                         [HumanMessage(content=f"Use this source to write your sections: {context}")])

    return {"sections": [section.content]}




def write_report(state: ResearchGraphState, runtime: Runtime[ModelContext]):

    # get states
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = prompts.report_writer_instructions.format(topic=topic,
                                                       context=formatted_str_sections)
    llm = MODELS[runtime.context.model_provider]
    report = llm.invoke([SystemMessage(content=system_message)] +
                        [HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}


def write_introduction(state: ResearchGraphState, runtime: Runtime[ModelContext]):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = prompts.intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    llm = MODELS[runtime.context.model_provider]
    intro = llm.invoke([instructions] + [HumanMessage(content=f"Write the report introduction")])
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState, runtime: Runtime[ModelContext]):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = prompts.intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    llm = MODELS[runtime.context.model_provider]
    conclusion = llm.invoke([instructions] + [HumanMessage(content=f"Write the report conclusion")])
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

