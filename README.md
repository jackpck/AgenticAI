## Resources
https://python.langchain.com/docs/concepts/tool_calling/

https://github.com/daveebbelaar/ai-cookbook/tree/main/mcp/crash-course

https://medium.com/@danushidk507/ollama-tool-calling-8e399b2a17a8

https://python.langchain.com/docs/how_to/tool_calling/

Anthropic's MCP vs langchain tool

## Debug

If getting NotImplementedError when calling bind_tools(), make sure use LLM that support tool (e.g. llama3 does not)

If still getting NotImplementedError, make sure use ChatOllama from langchain_ollama instead of langchain.model