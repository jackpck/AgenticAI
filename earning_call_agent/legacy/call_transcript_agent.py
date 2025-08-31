from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

import os

LLM_API_CALL_LIST = ["gemini-2.5-flash"]
API_PATH = "../config/google_ai_studio_api.txt"

if not os.environ.get("GOOGLE_API_KEY"):
    with open(API_PATH, "r", encoding='utf-8') as f:
        api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = api_key

load_dotenv("../venv")


class AgentState(TypedDict):
    input: str
    transcript: str
    transcript_json: str

class ReActAgent:
    def __init__(self, model: str,
                 system_message: dict,
                 model_provider: str = None):
        if model not in LLM_API_CALL_LIST:
            self.model = ChatOllama(model=model,
                                        validate_model_on_init=True,
                                        temperature=0.3,
                                        top_p=0.9,
                                        top_k=10)
        else:
            self.model = init_chat_model(model=model,
                                         model_provider=model_provider)

        self._setup_graph(system_message=system_message)

    def _setup_graph(self, system_message):
        graph = StateGraph(AgentState)
        graph.add_node("call_read_tool", self.call_read_tool)
        graph.add_node("call_preprocess_tool", self.call_preprocess_tool)
        graph.add_node("call_write_tool", self.call_write_tool)

        graph.add_edge(START, "call_read_tool")
        graph.add_edge("call_read_tool", "call_preprocess_tool")
        graph.add_edge("call_preprocess_tool", "call_write_tool")
        graph.add_edge("call_write_tool", END)

        self.graph = graph.compile()
        self.system_message = system_message

    def call_read_tool(self, state: AgentState) -> str:
        """
        Get the earning call transcript from the transcript path specified by the user
        :param state['input']: path to the earning call transcript
        :return: str of earning call transcript for further preprocessing
        """
        message = [SystemMessage(content="Extract the path to the transcript from the text. Return the path only."),
                HumanMessage(content=state["input"])]
        transcript_path = self.model.invoke(message).content
        print(f"transcript path: \n{transcript_path}")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding='utf-8') as f:
                earning_call_transcript = f.read()
            return {"transcript": earning_call_transcript}
        else:
            raise Exception("Transcript path in user query does not exist.")

    def call_preprocess_tool(self, state: AgentState) -> str:
        """
        Structure earning call transcript in a format specified by the system_prompt
        :param state['transcript']: string of the earning call transcript to be processed according to the
               system prompt
        :return: string of json of the structured earning call transcript
        """
        messages = [
            SystemMessage(content = self.system_message),
            HumanMessage(content = state["transcript"])
        ]
        response = self.model.invoke(messages)
        print(f"Preprocessed transcript: \n{response.content}")
        return {"transcript_json": response.content}

    def call_write_tool(self, state: AgentState):
        """
        Write the preprocessed transcript json to file
        :param state['input']: path to the earning call transcript
        :param state['transcript_json']: destination of where the preprocessed script will be saved
        :return:
        """
        message = [SystemMessage(content=
                                 """Extract the path where the preprocessed transcript json file will be saved.
                                 Return the path only."""
                                 ),
                   HumanMessage(content=state["input"])]
        output_path = self.model.invoke(message).content
        print(f"output path: \n{output_path}")
        output_dir_path = os.path.dirname(output_path)
        if os.path.exists(output_dir_path):
            with open(output_path, "w", encoding='utf-8') as f:
                transcript_json = state["transcript_json"]
                transcript_json_clean = transcript_json.strip()\
                                        .removeprefix("```json")\
                                        .removeprefix("```")\
                                        .removesuffix("```")
                f.write(transcript_json_clean)
        else:
            raise Exception("Output directory in user query does not exist.")


if __name__ == "__main__":
    #model = 'gpt-oss:20b'
    model = "gemini-2.5-flash"
    model_provider = "google_genai"
    config = {"configurable": {"thread_id":"2"}}
    TRANSCRIPT_PATH = "../data/raw/nvda_Q1_2026.txt"
    OUTPUT_PATH = "../data/processed/nvda_Q1_2026_preprocessed.json"
    PREPROCESS_PROMPT_PATH = "../system_prompts/earning_call_preprocess_prompt.txt"
    with open(PREPROCESS_PROMPT_PATH, "r", encoding='utf-8') as f:
        PREPROCESS_PROMPT = f.read()

    agent = ReActAgent(model=model,
                       system_message=PREPROCESS_PROMPT,
                       model_provider=model_provider)
    query = f"transcript path: {TRANSCRIPT_PATH}\noutput path: {OUTPUT_PATH}"
    print(f"\nQuery: {query}")
    agent.graph.invoke({"input": query}, config)


