from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain import hub

import sqlite3
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_experimental.tools.python.tool import PythonREPLTool

load_dotenv("../.venv")

def create_sqldb(sql_script_path: str):
    with open(sql_script_path, "r", encoding='utf-8') as f:
        sql_script = f.read()
    connection = sqlite3.connect(":memory:")
    connection.executescript(f"{sql_script}")

    engine = create_engine("sqlite://",
                           creator=lambda: connection)
    db = SQLDatabase(engine)
    return db

def create_sqltool(sql_script_path: str, model: str):
    sql_db = create_sqldb(sql_script_path)
    llm = ChatOllama(model=model,
                     validate_model_on_init=True,
                     temperature=0)
    sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
    tools = sql_toolkit.get_tools()
    tools.append(PythonREPLTool())
    return tools


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ReActAgent:
    def __init__(self, model: str,
                 tool_list: List,
                 system_message: str):
        self.model = ChatOllama(model=model,
                                validate_model_on_init=True,
                                temperature=0).bind_tools(tool_list)
        self.tools = {t.name:t for t in tool_list}
        self._setup_graph(system_message=system_message)

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
        print("call_llm")
        print(f"{messages}")
        print("*******************")
        if self.system_message:
            messages = [SystemMessage(content=self.system_message)] + messages
        message = self.model.invoke(messages)
        return {"messages":[message]}

    def call_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        print("call_tools")
        print(f"{tool_calls}")
        print("*******************")
        results = []
        for t in tool_calls:
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"],
                                       name=t["name"],
                                       content=str(result)))
        return {"messages": results}

    def should_call_tools(self, state: AgentState):
        result = state["messages"][-1]
        print("should_call_tools")
        print(f"{result}")
        print("*******************")
        return "tools" if len(result.tool_calls) > 0 else END



if __name__ == "__main__":
    model = 'gpt-oss:20b'
    config = {"configurable": {"thread_id":"1"}}
    SQL_SCRIPT_PATH = "data/sql_script.txt"
    SYSTEM_PROMPT_PATH = "langchain-ai/sql-agent-system-prompt"

    tool_list = create_sqltool(SQL_SCRIPT_PATH, model)
    chatprompttemplate = hub.pull(SYSTEM_PROMPT_PATH)
    system_message = chatprompttemplate.format(dialect="SQLite", top_k=5)
    agent = ReActAgent(model=model,
                       tool_list=tool_list,
                       system_message=system_message)
    #query = """What is the total sales revenue for the top 5 performing dealerships in 2022?"""
    query = """For each date, what is the running average of the sales revenue of the past 3 sales between 2022 and 2023?"""
    print(f"\nQuery: {query}")
    result = agent.graph.invoke({"messages": [HumanMessage(content=query)]}, config)
    print(f"\nResponse:")
    for message in result["messages"]:
        message.pretty_print()
