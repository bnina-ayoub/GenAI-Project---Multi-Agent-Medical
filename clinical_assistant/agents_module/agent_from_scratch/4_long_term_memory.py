"""
Long-term memory allows agents to remember important context across different conversations and sessions.

## Memory Management
- In this simplified example, we will use a simple in-memory store (dictionary) to manage memories. 
- In production, you would use a database like Postgres, MongoDB, or Redis for persistence.
- We're also giving the agent control over managing its own memories. This is a design choice. Sometimes it's better to manage memories externally to improve reliability, or use a combination of both.
"""
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

client = OpenAI()


# We will store memories in-memory as a dictionary
memories = {}


class Memory(BaseModel):
    id: int = Field(..., description="The id of the memory")
    content: str = Field(..., description="The content of the memory")


def manage_memories(
    action: Literal["create", "update", "delete"], 
    id: int, 
    content: str | None = None
):
    """Manage memories.

    Args:
        action (str): The action to perform. Can be one of "create", "update", or "delete".
        id (int): The id of the memory.
        content (str): The content of the memory. Only required when action is "create" or "update".

    Returns:
        The updated memories.
    """
    global memories
    if action == "create":
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


def get_memories():
    """Get all memories.

    Returns:
        The memories.
    """
    return memories


manage_memories(action="create", id=1, content="The user's name is Kenny.")

manage_memories(action="update", id=1, content="The user's name is Bob.")

manage_memories(action="delete", id=1)

tools = [
    {
        "type": "function",
        "name": "manage_memories",
        "description": "Create, update, or delete memories.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform. Can be one of 'create', 'update', or 'delete'.",
                },
                "id": {
                    "type": "integer",
                    "description": "The id of the memory.",
                },
                "content": {
                    "type": "string",
                    "description": "The content of the memory. Only required when action is 'create' or 'update'.",
                },
            },
            "required": ["action", "id"],
        },
    },
    {
        "type": "function",
        "name": "get_memories",
        "description": "Get all memories.",
        "parameters": {}
    },
]


messages = [
    {
        "role": "system",
        "content": "Your name is Aura. You are a personal assistant and your job is to help the user with general tasks, questions, and requests. In order to perform your job well, you need to keep track of important information about the user. You have access to a tool called `manage_memories` that allows you to create, update, or delete memories. Use this tool to keep track of important personal information about the user. Examples of important information includes, but is not limited to, personal details, work-related details, personal preferences, relationships, and goals. Every time you learn new information about the user, you should create a new memory. You also have access to a tool called `get_memories` that allows you to retrieve all memories. You should always use this tool to retrieve all memories which may have important context, before responding to the user.",
    },
    {
        "role": "user",
        "content": "Remember that my name is Kenny and my wife's name is Nancy.",
    },
]

response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    tools=tools,
    input=messages,
)

# Just as with retrieval, we get back a tool call and we'd have to follow the tool calling loop from here.

response.output


# Simulate the agent adding the memories
manage_memories(action="create", id=1, content="The user's name is Kenny.")
manage_memories(action="create", id=2, content="Kenny's wife's name is Nancy.")

# Now if the agent were to call get_memories, it would see that Kenny's wife's name is Nancy.
memories

# Let's start a new conversation and ask the agent if it remembers Kenny's wife's name
messages = [
    {
        "role": "system",
        "content": "Your name is Aura. You are a personal assistant and your job is to help the user with general tasks, questions, and requests. In order to perform your job well, you need to keep track of important information about the user. You have access to a tool called `manage_memories` that allows you to create, update, or delete memories. Use this tool to keep track of important personal information about the user. Examples of important information includes, but is not limited to, personal details, work-related details, personal preferences, relationships, and goals. Every time you learn new information about the user, you should create a new memory. You also have access to a tool called `get_memories` that allows you to retrieve all memories. You should always use this tool to retrieve all memories which may have important context, before responding to the user.",
    },
    {
        "role": "user",
        "content": "Do you remember my wife's name?",
    },
]

response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    tools=tools,
    input=messages,
)

# The agent correctly calls the `get_memories` tool and would see the memory with Kenny's wife's name!
response.output
