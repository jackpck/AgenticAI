from dataclasses import dataclass

@dataclass
class LLMConfig:
    provider: str = "google_genai"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.9

