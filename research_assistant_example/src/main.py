from langchain_core.messages import HumanMessage

from research_assistant_example.src.graph.workflow import build_analyst_workflow, build_interview_workflow
from research_assistant_example.src.schema.context import ModelContext

analyst_graph = build_analyst_workflow()

max_analysts = 3
topic = "The circular transaction among OpenAI, Oracle and Nvidia"
thread = {"configurable": {"thread_id":"1"}}

event = analyst_graph.invoke({"topic":topic,
                      "max_analysts":max_analysts},
                     thread,
                     stream_mode="values",
                     context=ModelContext())
analysts = event.get("analysts", "")
if analysts:
    for analyst in analysts:
        print(f"Name: {analyst.name}")
        print(f"Affiliation: {analyst.affiliation}")
        print(f"Role: {analyst.role}")
        print(f"Description: {analyst.description}")
        print("-"*50)

human_analyst_feedback = input(f"Enter user feedback. If none, simply press enter: \n")
analyst_graph.update_state(thread, {"human_analyst_feedback": human_analyst_feedback},
                   as_node="human_feedback")

event = analyst_graph.invoke(None,
                     thread,
                     stream_mode="values",
                     context=ModelContext())
analysts = event.get("analysts", "")
if analysts:
    for analyst in analysts:
        print(f"Name: {analyst.name}")
        print(f"Affiliation: {analyst.affiliation}")
        print(f"Role: {analyst.role}")
        print(f"Description: {analyst.description}")
        print("-"*50)

interview_graph = build_interview_workflow()
analyst_id = int(input("Which analyst do you want to interview the researcher? "))
print("-" * 50)

messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
interview = interview_graph.invoke({"analyst": analysts[analyst_id],
                                    "messages": messages,
                                    "max_num_turns": 2}, thread,
                                   context=ModelContext())
print(f"Interview:\n{interview['sections'][0]}")
