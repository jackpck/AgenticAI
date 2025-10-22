from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
import operator

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions

class InterviewState(MessagesState):
    max_num_turns = int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchGraphState(TypedDict):
    topic: str # research topic
    max_analysts: int #number of analysts
    human_analyst_feedback: str #human feedback
    analysts: List[Analyst] #analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # introduction for the final report
    content: str # content for the final report
    conclusion: str #conclusion for the final report
    final_report: str # final report



__all__ = [
    "Analyst",
    "Perspectives",
    "GenerateAnalystsState",
    "SearchQuery",
    "ResearchGraphState"
]