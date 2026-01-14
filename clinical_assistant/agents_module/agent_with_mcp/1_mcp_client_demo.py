"""
Demo of using the fastmcp client to connect to multiple MCP servers.
"""
import json
import asyncio
from fastmcp import Client

# Load the MCP config from a JSON file
with open("mcp_config.json", "r") as f:
    mcp_config = json.load(f)

# Create the MCP client
mcp_client = Client(mcp_config)

async def main():
    # Use a context manager to ensure the client is closed properly
    async with mcp_client:

        print("<TOOLS>\n")
        tools = await mcp_client.list_tools()
        for tool in tools:
            print(tool.model_dump(), "\n\n")

        print("<RESOURCES>\n")
        resources = await mcp_client.list_resources()
        for resource in resources:
            print(resource.model_dump(), "\n\n")

        print("<PROMPTS>\n")
        prompts = await mcp_client.list_prompts()
        for prompt in prompts:
            print(prompt, "\n\n")

        print("<GETTING RESOURCE>\n")
        resource_result = await mcp_client.read_resource("status://retrieval/last_updated")
        print(resource_result, "\n\n")

        print("<CALLING MANAGE_MEMORIES>\n")
        tool_result = await mcp_client.call_tool("memory_manage_memories", {"action": "create", "id": 1, "content": "The user's name is Kenny."})
        print(tool_result, "\n\n")

        print("<CALLING GET_MEMORIES>\n")
        tool_result = await mcp_client.call_tool("memory_get_memories", {})
        print(tool_result, "\n\n")

        print("<GETTING PROMPT>\n")
        prompt_result = await mcp_client.get_prompt("retrieval_analyze_customer", {"user_id": 1})
        print(prompt_result, "\n\n")

        # Get just the prompt text
        print(prompt_result.messages[0].content.text)

def is_interactive():
    """Check if running in an interactive environment like Jupyter or IPython."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

if __name__ == "__main__":
    if is_interactive():
        import nest_asyncio
        nest_asyncio.apply()

    asyncio.run(main())
