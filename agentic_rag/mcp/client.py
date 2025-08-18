import asyncio
import nest_asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from langchain.chat_models import ChatOllama

nest_asyncio.apply()

class MCPOpenAIClient:
    def __init__(self, model: str):
        self.model = ChatOllama(model=model,
                                temperature=0)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(self, server_script_path: str = "server.py"):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path]
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        tools_result = await self.session.list_tools()
        print("\nConnected to server with tools:")
        for tool in tools_result.tools:
            print(f" -{tool.name}: {tool.description}")

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        return a schema of all the tools to be considered by the LLM
        based on the description of each tool and the query
        """
        tools_result = await self.session.list_tools()
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.inputSchema
                    }
                }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str) -> str:
        tools = await self.get_mcp_tools()
        response = await self.model.bind_tools(tools).invoke({"query":query})

        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                print(f"\nCalling function: {tool.function.name}")
                print(f"\nArguments: {tool.function.arguments}")
                result = await self.session.call_tool(tool.function.name,
                                                      arguments=json.loads(tool.function.arguments))
                messages.append(
                    {"role":"tool",
                     "tool_call_id":tool.id,
                     "content":result.content[0].text
                     }
                )


                return final_response.choices[0].message.content

    async def ceanup(self):
        """clean up resource"""
        await self.exit_stack.aclose()

async def main():
    model = "llama3"
    client = MCPOpenAIClient(model)
    await client.connect_to_server("server.py")

    query = """what's the stock price of AAPL?"""
    print(f"\nQuery: {query}")

    response = await client.process_query(query)
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    asyncio.run(main())