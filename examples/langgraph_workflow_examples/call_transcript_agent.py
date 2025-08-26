from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain import hub

import os

LLM_API_CALL_LIST = ["gemini-2.5-flash"]
API_PATH = "../../config/google_ai_studio_api.txt"

if not os.environ.get("GOOGLE_API_KEY"):
    with open(API_PATH, "r", encoding='utf-8') as f:
        api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = api_key
load_dotenv("../../.venv")

def get_earning_call_transcript(transcript_path: str):
    with open(transcript_path, "r", encoding='utf-8') as f:
        earning_call_transcript = f.read()
    return earning_call_transcript

#@tool
def transcript_preprocess(earning_call_transcript: str,
                          model: str,
                          system_prompt: str,
                          model_provider: str = None) -> str:
    """
    :param earning_call_transcript: string of the earning call transcript to be processed according to the
           system prompt
    :param model: llm model to perform the preprocessing
    :param system_prompt: instruction to preprocess the earning call transcript to the desirable format
    :param model_provider: if using API call for LLM, specify the model provider e.g. 'google_genai'
    :return: string of json of the structured earning call transcript
    """
    if model not in LLM_API_CALL_LIST:
        preprocess_llm = ChatOllama(model=model,
                                    validate_model_on_init=True,
                                    temperature=0.3,
                                    top_p=0.9,
                                    top_k=10)
    else:
        preprocess_llm = init_chat_model(model, model_provider=model_provider)

    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = earning_call_transcript)
    ]
    response = preprocess_llm.invoke(messages)
    return response

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class ReActAgent:
    def __init__(self, model: str,
                 tool_list: List,
                 system_message: str):
        self.model = ChatOllama(model=model,
                                validate_model_on_init=True,
                                temperature=0).bind_tools(tool_list)
        self.tools = {t.name:t for t in tool_list}
        self._setup_graph(system_message=system_message)

    def _setup_graph(self, system_message):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("tools", self.call_tools)
        graph.add_conditional_edges(
            "llm",
            self.should_call_tools,
            ["tools", END]
        )
        graph.add_edge("tools", "llm")
        graph.add_edge(START, "llm")
        self.graph = graph.compile()
        self.system_message = system_message

    def call_llm(self, state: AgentState):
        messages = state["messages"]
        print("call_llm")
        print(f"{messages}")
        print("*******************")
        if self.system_message:
            messages = [SystemMessage(content=self.system_message)] + messages
        message = self.model.invoke(messages)
        return {"messages":[message]}

    def call_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        print("call_tools")
        print(f"{tool_calls}")
        print("*******************")
        results = []
        for t in tool_calls:
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"],
                                       name=t["name"],
                                       content=str(result)))
        return {"messages": results}

    def should_call_tools(self, state: AgentState):
        result = state["messages"][-1]
        print("should_call_tools")
        print(f"{result}")
        print("*******************")
        return "tools" if len(result.tool_calls) > 0 else END



if __name__ == "__main__":
    #model = 'gpt-oss:20b'
    model = "gemini-2.5-flash"
    model_provider = "google_genai"
    config = {"configurable": {"thread_id":"1"}}
    TRANSCRIPT_PATH = "./data/earning_call_transcript/nvda_Q1_2026.txt"
    OUTPUT_PATH = "./data/earning_call_transcript/nvda_Q1_2026_preprocessed.txt"
    PREPROCESS_PROMPT = """
    You are a financial analyst. Given a user earning call transcript, structure it into the following json format:

    {
    "company": "string",
     "quarter": "string",
     "participants": {
        "company participants": ["string"],
        "earning call participants": ["string"]
     },
     "sections": [
        {
         "type": "financial results | Q&A"
         "speaker": "string",
         "content": "string"
        }
     ]
    }

    Note:
    1. each section has a type of either "financial results" or "Q&A"
    2. if a section is Q&A, combine the question and answer paragraphs together in the content with the question
       paragraph starts  with "Q: " and the answer paragraph starts with "\nA: "
    """

    earning_call_transcript = get_earning_call_transcript(TRANSCRIPT_PATH)
    response = transcript_preprocess(earning_call_transcript=earning_call_transcript,
                                     model=model,
                                     system_prompt=PREPROCESS_PROMPT,
                                     model_provider=model_provider)

    print(response.content)
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        f.write(response.content)

