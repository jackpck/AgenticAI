import os
from urllib.error import URLError
import streamlit as st

from research_assistant_example.src.graph.workflow import build_workflow
from research_assistant_example.src.schema.context import ModelContext

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

model = "gemini-2.5-flash"
model_provider = "google_genai"
max_analysts = 3
thread = {"configurable": {"thread_id":"1"}}
topic = "The circular transaction among OpenAI, Oracle and Nvidia"

@st.cache_resource
def get_graph()
    return build_workflow()

graph = get_graph()

try:
    st.header("Agentic Chatbot")
    st.subheader("Research assistant")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat messages from history and rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_input = st.session_state.messages[-1]["content"]
            if message_input == " ":
                message_input = None
            graph.update_state(thread, {"human_analyst_feedback": message_input},
                               as_node="human_feedback")

            async def get_stream_response():
                stream = graph.stream({"topic":topic,
                                        "max_analysts":max_analysts},
                                       thread,
                                       stream_mode="values",
                                       context=ModelContext())
                try:
                    analysts = stream.get("analysts", "")
                    if analysts:
                        for analyst in analysts:
                            yield (f"Name: {analyst.name} \n "
                                   f"Affiliation: {analyst.affiliation} \n"
                                   f"Role: {analyst.role} \n"
                                   f"Description: {analyst.description} \n"
                                   f"-"*50)
                except:
                    pass

            response = st.write_stream(get_stream_response())
            st.session_state.messages.append({"role": "assistant",
                                              "content": response.replace("$", "\$")
                                              })


    if st.button("Reset chat"):
        st.session_state.clear()
    if st.button("Clear agent cache"):
        st.cache_resource.clear()

except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")