from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict
import yfinance as yf
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

load_dotenv("../.venv")

@tool
def get_stock_price(symbol: str) -> float:
    """
    Get price of a stock

    :param symbol: ticker symbol of a stock
    :return: stock price
    """
    ticker = yf.Ticker(symbol)
    return ticker.info['regularMarketPrice']


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ReActAgent:
    def __init__(self, model: str, tool_list: List):
        self.model = ChatOllama(model=model,
                                validate_model_on_init=True,
                                temperature=0).bind_tools(tool_list)
        self.tools = {t.name:t for t in tool_list}
        self._setup_graph(system_message="")

    def _setup_graph(self, system_message):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("tools", self.call_tools)
        graph.add_conditional_edges(
            "llm",
            self.should_call_tools,
            ["tools", END]
        )
        graph.add_edge("tools", "llm")
        graph.add_edge(START, "llm")
        self.graph = graph.compile()
        self.system_message = system_message

    def call_llm(self, state: AgentState):
        messages = state["messages"]
        if self.system_message:
            messages = [SystemMessage(content=self.system_message)] + messages
        message = self.model.invoke(messages)
        return {"messages":[message]}

    def call_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"],
                                       name=t["name"],
                                       content=str(result)))
        return {"messages": results}

    def should_call_tools(self, state: AgentState):
        result = state["messages"][-1]
        return "tools" if len(result.tool_calls) > 0 else END



if __name__ == "__main__":
    tool_list = [get_stock_price]
    model = 'gpt-oss:20b'
    config = {"configurable": {"thread_id":"1"}}
    agent = ReActAgent(model=model,
                     tool_list=tool_list)
    query = """What is the stock price of AAPL today? Also what is the price of Microsoft?"""
    print(f"\nQuery: {query}")
    result = agent.graph.invoke({"messages": [HumanMessage(content=query)]}, config)
    print(f"\nResponse:")
    for message in result["messages"]:
        message.pretty_print()
