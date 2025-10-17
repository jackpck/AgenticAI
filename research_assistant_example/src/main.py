from research_assistant_example.src.graph.workflow import build_workflow
from research_assistant_example.src.core.context import ModelContext

graph = build_workflow()

max_analysts = 3
topic = "The circular transaction among OpenAI, Oracle and Nvidia"
thread = {"configurable": {"thread_id":"1"}}


for event in graph.stream({"topic":topic,
                           "max_analysts":max_analysts},
                          thread,
                          stream_mode="values",
                          context=ModelContext()):
    analysts = event.get("analysts", "")
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-"*50)

state = graph.get_state(thread)
print(f"next state: {state.next}")