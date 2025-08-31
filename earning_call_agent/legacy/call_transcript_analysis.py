import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

API_PATH = "../config/google_ai_studio_api.txt"

if not os.environ.get("GOOGLE_API_KEY"):
    with open(API_PATH, "r", encoding='utf-8') as f:
        api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = api_key

load_dotenv("../venv")

TRANSCRIPT_JSON_PATH = "./data/nvda_Q1_2026_preprocessed.json"
ANALYSIS_PROMPT_PATH = "./system_prompts/earning_call_analyze_prompt.txt"

with open(TRANSCRIPT_JSON_PATH, "r", encoding='utf-8') as f:
    transcript_json_str = f.read()

with open(ANALYSIS_PROMPT_PATH, "r", encoding='utf-8') as f:
    analysis_prompt_str = f.read()

model = "gemini-2.5-flash"
model_provider = "google_genai"

topic_model = init_chat_model(model,
                              model_provider=model_provider)

messages = [
    SystemMessage(content = analysis_prompt_str),
    HumanMessage(content = transcript_json_str)
]

response = topic_model.invoke(messages)
response_clean = response.content.strip()\
                .removeprefix("```json")\
                .removeprefix("```")\
                .removesuffix("```")
print(response_clean)
