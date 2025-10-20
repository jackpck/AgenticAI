from langchain.chat_models import init_chat_model
import os

from research_assistant_example.config.llm_config import LLMConfig

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

config = LLMConfig()
MODELS = {"google": init_chat_model(model=config.model,
                                    model_provider=config.provider,
                                    temperature=config.temperature,
                                    top_k=config.top_k,
                                    top_p=config.top_p)
          }

