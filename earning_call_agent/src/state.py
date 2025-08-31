from pydantic import BaseModel

class AgentState(BaseModel):
    ticker: str
    quarter: int
    year: int
    transcript_folder_path: str
    output_folder_path: str
    transcript: str = None
    transcript_json: str = None

__all__ = [
    "AgentState"
]