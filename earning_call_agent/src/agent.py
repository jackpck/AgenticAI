from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
import os

from state import AgentState

class EarningCallAgent:
    def __init__(self,
                 model: str,
                 model_provider: str,
                 system_prompt: str):

        self.model = init_chat_model(model=model,
                                     model_provider=model_provider)
        self.system_prompt = system_prompt
        self._setup_graph()

    def check_processed_json(self, state: AgentState) -> int:
        """
        Check if the preprocessed json file already exist. This act as a control to decide if one
        need to run LLM to preprocess or just simply read from the existing json file.
        :param state: state of the agent
        :return:
        """
        output_path = (f"{state.output_folder_path.rstrip('/')}"
        f"/{state.ticker}_Q{state.quarter}_{state.year}_preprocessed.json")
        if os.path.exists(output_path):
            return "read_preprocessed_json"
        return "read_raw_txt"

    def read_preprocessed_json(self, state: AgentState):
        """
        Read directly from the existing preprocessed json.
        :param state: state of the agent
        :return:
        """
        output_path = (f"{state.output_folder_path.rstrip('/')}"
        f"/{state.ticker}_Q{state.quarter}_{state.year}_preprocessed.json")
        with open(output_path, "r", encoding="utf-8") as f:
            transcript_json = f.read()
        return {"transcript_json": transcript_json}

    def read_raw_txt(self, state: AgentState) -> str:
        """
        Get the earning call transcript from the transcript path specified by the user
        :param state: state of the agent
        :return: str of earning call transcript for further preprocessing
        """
        transcript_path = (f"{state.transcript_folder_path.rstrip('/')}"
        f"/{state.ticker}_Q{state.quarter}_{state.year}.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding='utf-8') as f:
                earning_call_transcript = f.read()
            return {"transcript": earning_call_transcript}
        else:
            raise Exception("Transcript folder does not exist.")

    def preprocess_llm(self, state: AgentState) -> str:
        """
        Structure earning call transcript in a format specified by the system_prompt
        :param state: state of the agent
        :return: string of json of the structured earning call transcript
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=state.transcript)
        ]
        response = self.model.invoke(messages)

        return {"transcript_json": response.content}

    def write_preprocessed_json(self, state: AgentState):
        """
        Write the preprocessed transcript json to file to output_path, which is specified in state
        :param state: state of the agent
        :return:
        """
        if os.path.exists(state.output_folder_path):
            output_path = (f"{state.output_folder_path.rstrip('/')}"
            f"/{state.ticker}_Q{state.quarter}_{state.year}_preprocessed.json")
            with open(output_path, "w", encoding='utf-8') as f:
                transcript_json_clean = state.transcript_json.strip() \
                    .removeprefix("```json") \
                    .removeprefix("```") \
                    .removesuffix("```")
                f.write(transcript_json_clean)
        else:
            raise Exception("Output directory does not exist.")

    def WIP_analyze_llm(self, state: AgentState) -> str:
        """
        Analyze the sentiment and possible risk factor in each speech. Instruction
        given by system_analysis_prompt
        :param state: state of the agent
        :return: string of json of the structured earning call transcript with sentiment and risk
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=state.transcript_json)
        ]
        response = self.model.invoke(messages)

        return {"transcript_analyze_json": response.content}

    def _setup_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("read_preprocessed_json", self.read_preprocessed_json)
        graph.add_node("read_raw_txt", self.read_raw_txt)
        graph.add_node("preprocess_llm", self.preprocess_llm)
        graph.add_node("write_preprocessed_json", self.write_preprocessed_json)

        graph.add_conditional_edges(START, self.check_processed_json)
        graph.add_edge("read_preprocessed_json", END)
        graph.add_edge("read_raw_txt", "preprocess_llm")
        graph.add_edge("preprocess_llm", "write_preprocessed_json")
        graph.add_edge("write_preprocessed_json", END)

        self.graph = graph.compile()

