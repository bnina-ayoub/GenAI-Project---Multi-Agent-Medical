from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal, List
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from datetime import datetime

load_dotenv()


# Switch models to see how the post quality improves with a more capable model
llm = ChatOpenAI(
    name="Planner",
    model="gpt-5-mini-2025-08-07",
    # model="gpt-5-2025-08-07",
    temperature=0.1,
)


#################################
# State
#################################

# We're going to use a simple dictionary as an in-memory store
# In production you could use a database like Postgres, MongoDB, or Redis
store = {}

# Using pydantic models improves reliability of structured inputs and outputs with automatic type validation
# I recommend using pydantic models for the state as well as any other structured data that you want to extract from LLMs or to use with tools. This will ensure the data is always in the correct format.
class Task(BaseModel):
    """A task to be completed."""
    task: str = Field(..., description="The task to be completed")
    status: Literal["todo", "in_progress", "done"] = Field(..., description="The status of the task")
    priority: Literal[1, 2, 3, 4, 5] = Field(..., description="The priority of the task, 1 is the highest priority")
    due_date: str | None = Field(None, description="The due date of the task, in the format of YYYY-MM-DD")

class TaskList(BaseModel):
    """A list of tasks to be completed."""
    tasks: List[Task] = Field(..., description="The list of tasks to be completed")

class AgentState(BaseModel):
    """The state of the agent."""
    messages: Annotated[list, add_messages] = []


#################################
# Tools
#################################

# We will create some tools to help the agent plan and track tasks.
# Tasks will be stored in our in-memory store which is just the dictionary we created earlier
@tool
def generate_task_list(task_list: TaskList) -> str:
    """Generate a new task list or update the existing task list by replacing it."""
    store["tasks"] = task_list
    return store["tasks"].model_dump_json()

@tool
def view_task_list() -> str:
    """View the task list"""
    if "tasks" not in store:
        return "No task list found."
    return store["tasks"].model_dump_json()


#################################
# Build the Graph
#################################

tools = [generate_task_list, view_task_list]
llm_with_tools = llm.bind_tools(tools)

def agent(state: AgentState):
    system_prompt = SystemMessage(content=f"""You are a personal assistant. Your job is to help the user manage their tasks. You have a couple of tools at your disposal to help you manage tasks.

    <Tools>
    generate_task_list: Use this tool to both create new task lists and/or make updates to the existing task list by replacing it.
    view_task_list: Use this tool to view the current task list.
    </Tools>
                                  
    <Tasks>
    Tasks include a task description, status, priority, and due date.
    </Tasks>
                                  
    Today's date is {datetime.now().date()}.
    """)
    response = llm_with_tools.invoke([system_prompt] + state.messages)
    return {"messages": [response]}

def agent_router(state: AgentState) -> str:
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(AgentState)

builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("agent")

builder.add_edge("tools", "agent")
builder.add_conditional_edges(
    "agent",
    agent_router,
    {
        "tools": "tools",
        END: END,
    }
)

graph = builder.compile()
