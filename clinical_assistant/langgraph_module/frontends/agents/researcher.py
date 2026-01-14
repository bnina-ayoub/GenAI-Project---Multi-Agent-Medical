from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Annotated
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_tavily import TavilySearch, TavilyExtract
from datetime import datetime

load_dotenv()


# Switch models to see how the post quality improves with a more capable model
llm = ChatOpenAI(
    name="Researcher",
    model="gpt-5-mini-2025-08-07",
    # model="gpt-5-2025-08-07",
    temperature=0.1,
)


#################################
# State
#################################

class AgentState(BaseModel):
    """The state of the agent."""
    messages: Annotated[list, add_messages] = []


#################################
# Tools
#################################

# The agent will also have web search capabilities for content research
@tool
def search_web(query: str, num_results: int = 3):
    """Search the web.
    
    Args:
        query: The search query.
        num_results: The number of results to return, max is 3.
    """
    tavily_search = TavilySearch(max_results=min(num_results, 3), topic="general")
    search_results = tavily_search.invoke(input={"query": query})
    
    processed_results = {
        "query": query,
        "results": []
    }

    for result in search_results["results"]:
        processed_results["results"].append({
            "title": result["title"],
            "url": result["url"],
            "summary": result["content"]
        })

    return processed_results

tavily_extract = TavilyExtract()

@tool
def extract_content_from_webpage(url: str):
    """Extract the content from a webpage.

    Args:
        url: The url of the webpage to extract content from.
    """
    result_contents = tavily_extract.invoke(input={"urls": [url]})
    raw_content = result_contents["results"][0]["raw_content"]
    return raw_content


#################################
# Build the Graph
#################################

tools = [search_web, extract_content_from_webpage]
llm_with_tools = llm.bind_tools(tools)

def agent(state: AgentState):
    system_prompt = SystemMessage(content=f"""You are a research assistant. Your job is to help the user answer questions by performing research. You have a couple of tools at your disposal to help you perform research.

    <Tools>
    search_web: Use this tool to search the web. Returned results include the page title, url, and a short summary of each webpage.
    extract_content_from_webpage: Use this tool to extract the complete contents from a webpage given the url.
    </Tools>
                                  
    The current date and time is {datetime.now()}.
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
