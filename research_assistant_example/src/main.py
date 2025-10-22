from research_assistant_example.src.graph.workflow import build_research_workflow
from research_assistant_example.src.schema.context import ModelContext

graph = build_research_workflow()

max_analysts = 3
topic = "The circular transaction among OpenAI, Oracle and Nvidia"
thread = {"configurable": {"thread_id":"1"}}

event = graph.invoke({"topic":topic,
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

human_analyst_feedback = "_" # dummy string to start the while loop
while human_analyst_feedback:
    human_analyst_feedback = input(f"Enter user feedback. If none, simply press enter: \n")
    graph.update_state(thread, {"human_analyst_feedback": human_analyst_feedback},
                       as_node="human_feedback")

    if human_analyst_feedback:
        event = graph.invoke(None,
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

# Continue
for event in graph.stream(None,
                          thread,
                          stream_mode="updates",
                          context=ModelContext()):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)

final_state = graph.get_state(thread)
report = final_state.values.get('final_report')
print("*"*50)
print(report)