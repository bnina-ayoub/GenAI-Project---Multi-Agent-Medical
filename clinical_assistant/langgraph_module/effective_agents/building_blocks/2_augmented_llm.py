"""
The augmented LLM is the core building block of AI agents.

## KEY TAKEAWAYS
1. We can enhance the basic LLM with memory and tools to create an augmented LLM, but it's still missing `agency` or the ability to make decisions so not quite an agent yet.
2. Through tools, augmented LLMs have the ability to act on the environment and retrieve additional context.
"""
from dotenv import load_dotenv
import json
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    name="Jude",
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.1,
)

# To augment the LLM, we'll add memory and tool calling


#################################
# Adding Memory or State
#################################

# Basic memory for the LLM is just keeping track of the conversation history.
messages = []

user_input = "Hello Jude! I'm Kenny."
messages.append(user_input)

print(messages)

# Each time we invoke the LLM now, we'll pass in the entire conversation history
# Each response from the LLM is then added to the conversation history
response = llm.invoke(messages)
print(response.content)

messages.append(response.content)
print(messages)

# Now we ask the question again
user_input = "What's my name?"
messages.append(user_input)

# Because Jude now has the entire conversation in context, he knows our name
response = llm.invoke(messages)
print(response.content)

messages.append(response.content)
for msg in messages:
    print(msg + "\n")

# print only the last message
print(messages[-1])


#################################
# State in Langgraph
#################################

# In Langgraph, the state keeps track of the conversation history
class AgentState(BaseModel):
    """The state of the agent."""
    messages: Annotated[list, add_messages] = []

# Build the simplest graph with one node that calls the LLM
def agent_node(state: AgentState):
    response = llm.invoke(state.messages)
    return {"messages": [response]}

builder = StateGraph(AgentState)

builder.add_node(agent_node)
builder.set_entry_point("agent_node")

graph = builder.compile()


# Invoke the graph with the same intro
user_input = "Hello Jude! I'm Kenny."
response = graph.invoke(
    input=AgentState(messages=[user_input]),
    )

# When we run the graph, Langgraph returns the entire state
# Each message is a langchain message type
print(response)

# We can print out just the contents of each message
for msg in response["messages"]:
    print(msg.content + "\n")


# Now ask the question again
user_input = "What's my name?"
response = graph.invoke(
    input=AgentState(messages=[user_input]),
    )

# Jude still doesn't know our name because each graph run is ephemeral
for msg in response["messages"]:
    print(msg.content + "\n")


# We need to configure Langgraph to persist the state. This is done by passing a checkpointer
# In general, the state can be persisted in memory, on disk, or in a database.
# We have to compile the graph with a MemorySaver and pass a config with thread_id
graph = builder.compile(checkpointer=MemorySaver())

config = RunnableConfig(
    configurable={"thread_id": "1"},
)

user_input = "Hello Jude! I'm Kenny."
response = graph.invoke(
    input=AgentState(messages=[user_input]),
    config=config,
    )

for msg in response["messages"]:
    print(msg.content + "\n")

user_input = "What's my name?"
response = graph.invoke(
    input=AgentState(messages=[user_input]),
    config=config,
    )

for msg in response["messages"]:
    print(msg.content + "\n")


#################################
# Tools
#################################

# ... but first defining Langchain tools


# Let's first ask Jude to run a SQL query
response = llm.invoke("Hello Jude! Can you pull the customer data for John Doe?")

# Jude doesn't have any tools so he can't run the query
print(response.content)


# Langgraph is built on Langchain. In Langchain, tools are defined by adding the tool decorator to any Python function
@tool
def get_customer_data(customer_name: str) -> str:
    """Get customer data from the database"""
    return f"<TOOL RESPONSE> \n\nCustomer Data for {customer_name}\n\nEmail: example@gmail.com"

# We can then add the tool to Jude by binding it to the LLM
llm_with_tools = llm.bind_tools([get_customer_data])

# Now let's keep track of our messages like before
messages = []

# Pass in the same message again
user_input ="Hello Jude! Can you pull the customer data for John Doe?"
messages.append(user_input)
response = llm_with_tools.invoke(messages)

# We get a blank 'content' because Jude is not responding with text, but instead deciding to call a tool.
print(response.content)

# We can see the tool call in the additional_kwargs
print(response.additional_kwargs['tool_calls'])

# It's up to us to execute the tool given the arguments generated by the LLM
# We can do it manually by parsing out the tool arguments...
tool_call = response.additional_kwargs['tool_calls'][0]
tool_args = json.loads(tool_call['function']['arguments'])

print(tool_args)

# ...and passing them directly to the function
tool_response = get_customer_data.invoke(tool_args['customer_name'])

print(tool_response)

# Now we add the tool response/output to the conversation history
messages.append(tool_response)

# We then invoke the LLM again with the message history including the tool output, this is how LLMs are able to perceive their effects on the environment
response = llm_with_tools.invoke(messages)

# Jude can then respond with the tool output
print(response.content)


#################################
# Adding Tools to Langgraph Agents
#################################

tools = [get_customer_data]
llm_with_tools = llm.bind_tools(tools)

# We now use our augmented LLM in our agent node
def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state.messages)
    return {"messages": [response]}

# We add a conditional edge to catch tool calls
def agent_router(state: AgentState) -> str:
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(AgentState)

builder.add_node(agent_node)
# We also add a new node with Langgraph's prebuilt ToolNode class
# This class handles calling the tools and returning the output to the agent, basically everything we did manually before
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("agent_node")

builder.add_conditional_edges(
    "agent_node",
    agent_router,
    {
        "tools": "tools",
        END: END,
    }
)
# After a tool is called, we want to always route back to the agent so it can decide what to do next
builder.add_edge("tools", "agent_node")

graph = builder.compile()

# Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())

user_input ="Hello Jude! Can you pull the customer data for John Doe?"
response = graph.invoke(AgentState(messages=[user_input]))


for msg in response["messages"]:
    print(msg.content + "\n")
    