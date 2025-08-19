from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import List
import yfinance as yf
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

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


class AgentClient:
    def __init__(self, model: str, tool_list: List):
        self.model = ChatOllama(model=model,
                                validate_model_on_init=True,
                                temperature=0).bind_tools(tool_list)
        self.tool_dict = {tool.name:tool for tool in tool_list}
        self.messages = []

    def generate_response(self, query: str):
        self.messages.append(HumanMessage(query))
        ai_msg = self.model.invoke(query)
        for tool_call in ai_msg.tool_calls:
            selected_tool = self.tool_dict[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            self.messages.append(tool_msg)
        response = self.model.invoke(self.messages)
        return response


if __name__ == "__main__":
    tool_list = [get_stock_price]
    model = 'gpt-oss:20b'
    AC = AgentClient(model=model,
                     tool_list=tool_list)
    query = """What is the stock price of AAPL today?"""
    print(f"\nQuery: {query}")
    result = AC.generate_response(query)
    print(f"\nResponse: {result}")
