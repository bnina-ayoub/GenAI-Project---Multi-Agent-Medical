"""
A simple MCP server for managing long-term memory.

In this example, we will use a simple in-memory store (dictionary) to manage memories. 
In production, you would use a database like Postgres, MongoDB, or Redis for persistence.
"""
from fastmcp import FastMCP
from typing import Literal, Dict

# Create our FastMCP server and give it a name
mcp = FastMCP(name="memory")

# We will store memories in-memory as a dictionary
memories = {}


@mcp.tool()
def manage_memories(
    action: Literal["create", "update", "delete"], 
    id: int, 
    content: str | None = None
) -> Dict[int, str]:
    """Manage memories.

    Args:
        action (str): The memory action to perform. Can be one of "create", "update", or "delete".
        id (int): The id of the memory. Must be unique.
        content (str): The content of the memory. Only required when action is "create" or "update".

    Returns:
        The updated memories.
    """
    global memories
    if action == "create":
        if id in memories:
            raise ValueError(f"Memory with id {id} already exists.")
        if content is None:
            raise ValueError(
                f"Content cannot be None when creating memory with id {id}."
            )
        memories[id] = content

    elif action == "update":
        if id not in memories:
            raise ValueError(f"Memory with id {id} does not exist.")
        if content is None:
            raise ValueError(
                f"Content cannot be None when updating memory with id {id}."
            )
        memories[id] = content

    elif action == "delete":
        if id not in memories:
            raise ValueError(f"Memory with id {id} does not exist.")
        del memories[id]

    return memories

@mcp.tool()
def get_memories() -> Dict[int, str]:
    """Get all memories.

    Returns:
        The memories.
    """
    return memories


if __name__ == "__main__":
    # You can explicitly specify the transport type, defaults to STDIO
    # See the retrieval_mcp.py for an example of HTTP transport
    mcp.run()
