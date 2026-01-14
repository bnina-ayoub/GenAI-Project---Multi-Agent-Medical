"""
Tools allow agents to take actions on the environment, retrieve additional context, and get feedback from the environment.

## Tool Calling with APIs
https://platform.openai.com/docs/guides/function-calling#page-top

- Tool calling requires 5 total steps: 
    1. Define the tool as a python function and send the function definition to the LLM model.
    2. The LLM calls the tool by returning a function_call response which includes the name of the tool and the arguments to pass to the tool.
    3. Parse the tool call arguments and execute the function.
    4. Send the tool output back to the LLM.
    5. The LLM returns a final response.
    
    See the guide above for a great illustration of this process.

- Some LLM providers like OpenAI are introducing built-in tools which they implement and manage for you. We'll cover both here.
"""
from openai import OpenAI
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import json

load_dotenv()

client = OpenAI()

# Understand the tool calling loop
# https://platform.openai.com/docs/guides/function-calling#page-top

##########################################
# Built-in Tools
##########################################

response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    tools=[{"type": "web_search_preview"}],
    input="What was one positive news story from today?"
)

print(response.output_text)

response.output[1].content[0].annotations

##########################################
# Function (Custom Tool) Calling
##########################################

# Web search with Tavily
# https://github.com/tavily-ai/langchain-tavily


# 1. Define the tool as a python function
# ----------------------------------------
def search_web(query: str):
    """Search the web and get back a list of search results including the page title, url, and the cleaned content of each webpage.

    Args:
        query: The search query.

    Returns:
        A dictionary of the search results.
    """
    tavily_search = TavilySearch(max_results=3, topic="general")
    response = tavily_search.invoke(input={"query": query})

    return response


# Test the tool and inspect the raw output
search_web("how to make a grilled cheese")


# 2. Define the tool schema
# ----------------------------------------
tools = [
    {
        "type": "function",
        "name": "search_web",
        "description": "Search the web and get back a list of search results including the page title, url, and the cleaned content of each webpage.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    },
]


# 3. Invoke the model with tools defined
# ----------------------------------------
messages = [
    {
        "role": "system",
        "content": "Your name is Aura. You are a researcher. You have access to a tool called `search_web` that allows you to search the web. Do not rely on your own knowledge, always use the `search_web` tool to answer the user's questions.",
    },
    {"role": "user", "content": "What is the latest news about AI?"},
]

response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    tools=tools,
    input=messages,
)

print(response.output)

# Add function call to the conversation
messages += response.output


# 4. Parse the function call arguments and execute the function
# ----------------------------------------
function_call = None
function_call_arguments = None

# iterate to capture parallel function calls
for item in response.output:
    if item.type == "function_call":
        function_call = item
        function_call_arguments = json.loads(item.arguments)


result = {"search_results": search_web(**function_call_arguments)}

# Add the function call output to the conversation
# Must include the call_id that matches the function_call
messages.append(
    {
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": json.dumps(result),
    }
)

for message in messages:
    print(message)
    print("\n-----\n")


# 5. Invoke the model again with the function call results
# ----------------------------------------

# This step closes the loop with the assistant seeing the function call output and responding with a final answer
response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    tools=tools,
    input=messages,
)

print(response.output_text)

messages.append({"role": "assistant", "content": response.output_text})

for message in messages:
    print(message)
    print("\n-----\n")
