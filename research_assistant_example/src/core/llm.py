from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any, Generator
import os
import time

# =====================================================
# 1. Configuration Dataclass
# =====================================================

@dataclass
class LLMConfig:
    provider: str = "google_genai"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.9
    max_tokens: int = 4096
    stream: bool = False
    timeout: int = 60
    api_key: Optional[str] = None

# =====================================================
# 2. Core Wrapper Class
# =====================================================

class LLM:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = self._init_client()

    def _init_client(self):
        """Initialize provider client based on configuration."""
        try:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OpenAI API key.")
            return init_chat_model(model=self.config.model,
                                   model_provider=self.config.provider,
                                   temperature=self.config.temperature,
                                   top_k=self.config.top_k,
                                   top_p=self.config.top_p)
        except:
            raise ValueError()
