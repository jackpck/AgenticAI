from langgraph.graph import END
from research_assistant_example.src.schema.state import GenerateAnalystsState, InterviewState,\
    ResearchGraphState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send

def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise end
    return END


def route_messages(state: InterviewState, name: str = "expert"):

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    if num_responses >= max_num_turns:
        return "save_interview"

    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_interview"
    return "ask_question"


def initiate_all_interviews(state: ResearchGraphState):
    """
    This is the map step where we run each interview sub-graph using Send API
    """

    human_analyst_feedback = state.get("human_analyst_feedback")
    if human_analyst_feedback:
        return "create_analysts"

    else:
        topic = state["topic"]
        # "conduct_interview will be a node of sub-graph intitated by build_interview_workflow()
        # which takes in the InterviewState schema
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                           ]}) for analyst in state["analysts"]]
