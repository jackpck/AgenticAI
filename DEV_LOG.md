## Debug

If getting NotImplementedError when calling bind_tools(), make sure use LLM that support tool (e.g. llama3 does not)

If still getting NotImplementedError, make sure use ChatOllama from langchain_ollama instead of langchain.model
