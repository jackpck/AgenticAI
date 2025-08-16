from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv("../.venv")

SERVER_NAME = 'calculator'
HOST = "0.0.0.0"
PORT = 8050
STATELESS_HTTP = True

mcp = FastMCP(
    name=SERVER_NAME,
    host=HOST,
    port=PORT,
    stateless_http=STATELESS_HTTP
)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")