from agent import EarningCallAgent
import os
from dotenv import load_dotenv

import sys
sys.path.append("../")
from system_prompts import prompts

API_PATH = "../../config/google_ai_studio_api.txt"

if not os.environ.get("GOOGLE_API_KEY"):
    with open(API_PATH, "r", encoding='utf-8') as f:
        api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = api_key

load_dotenv("../../venv")


config = {"configurable": {"thread_id": "1"}}
model = "gemini-2.5-flash"
model_provider = "google_genai"
TRANSCRIPT_FOLDER_PATH = "../data/raw"
OUTPUT_FOLDER_PATH = "../data/processed"
agent = EarningCallAgent(model=model,
                         model_provider=model_provider,
                         system_prompt=prompts)

stock = "AAPL"
context = {"ticker": stock.lower(),
           "year": 2025,
           "quarter": 3,
           "transcript_folder_path": TRANSCRIPT_FOLDER_PATH,
           "output_folder_path": OUTPUT_FOLDER_PATH}

final_state = agent.graph.invoke(context, config)
transcript_json = final_state["transcript_json"]

print(transcript_json)

