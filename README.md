# Agentic AI

## Introduction

### MCP client-server framework 

Server involves three primitives:
- Tools: e.g. calculator, calender
- Resources: e.g. RAG
- Prompts: e.g. resuable prompt templates

Client involves three primitives:
- Sampling: allows servers to request LLM completions
- Elicitation: allows servers to request additional information
- Logging: enables servers to send log messages for debuggin and monitoring

MCP inspector demo: ``examples/mcp_examples/simple_mcp_client_server_example``

An MCP client server framework was setup under the path above.
To run an MCP inspector locally, run ``mcp dev server.py`` in the command line. This will bring
you to the browser where an MCP inspector was opened up for you to inspect the tools for debugging.
In this example, there are two tools: ``add()`` for addition and ``get_stock_price()`` for getting
stock price from yfinance given the ticker symbol.


### Agent-workflow using langchain/langgraph demo: ``examples/langgraph_workflow_examples``

Workflow represents one end of the spectrum where flow of information/instructions are deterministic
while an agent represents another end of the spectrum where flow are decided by agents without 
defining the flow a priori. Langgraph is a tool that enables developer to develop an Agentic AI
that lies somewhere in between. There are three examples under the path above:

- ``stock_price_agent.py``: write a query to ask for price of stock price. Either company name or ticker symbol 
  can be used. The agent should be able to comprehend. 
- ``sql_agent.py``: write a query to ask questions that can be answered by the querying from the three SQL tables
  listed in ``data/sql_script.txt``.
- ``call_transcript_agent.py`` [TODO]

When the Agent is defined, the agent state is defined by the message, which encompasses the message from the
LLM, tool called, tool results etc. 

While langgraph specifies how the agents and the tools are connected, it does not specify explicitly how they 
interact, which will be decided at runtime by the state of the agent. That being said, langchain can also
be used, albeit much less flexible than langgraph. An example of calling the stock price tool using 
langchain can be found ``stock_price_agent_langchain.py``

### Agent design pattern ``pattern_template``

Common agent design patterns can be found in the path above.



## Resources
https://python.langchain.com/docs/concepts/tool_calling/

https://github.com/daveebbelaar/ai-cookbook/tree/main/mcp/crash-course

https://medium.com/@danushidk507/ollama-tool-calling-8e399b2a17a8

https://python.langchain.com/docs/how_to/tool_calling/

https://www.ibm.com/think/tutorials/build-sql-agent-langgraph-mistral-medium-3-watsonx-ai

https://langchain-ai.github.io/langgraph/tutorials/workflows/


