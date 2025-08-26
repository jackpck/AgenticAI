from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import yfinance as yf

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

@mcp.tool()
def get_stock_price(symbol: str) -> float:
    """get price of the stock"""
    ticker = yf.Ticker(symbol)
    return ticker.info['regularMarketPrice']

if __name__ == "__main__":
    mcp.run(transport="stdio")