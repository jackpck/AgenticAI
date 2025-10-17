from dataclasses import dataclass
from langchain.chat_models import init_chat_model
import os

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

@dataclass
class LLMConfig:
    provider: str = "google_genai"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.9

config = LLMConfig()
MODELS = {"google": init_chat_model(model=config.model,
                                    model_provider=config.provider,
                                    temperature=config.temperature,
                                    top_k=config.top_k,
                                    top_p=config.top_p)
          }

