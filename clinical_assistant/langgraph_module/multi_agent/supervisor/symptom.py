import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, List

import torch

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_tavily import TavilyExtract, TavilySearch
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"
researcher_prompt = (PROMPTS_DIR / "symptom.md").read_text(encoding="utf-8")


# Lazy-load BioBERT NER pipeline for symptom/entity extraction
_biobert_pipe = None


def get_biobert_pipe():
    global _biobert_pipe
    if _biobert_pipe is None:
        # Default to an NER-finetuned checkpoint to avoid random classifier head
        model_name = os.getenv("BIOBERT_MODEL", "d4data/biomedical-ner-all")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Cap max length to avoid truncation warnings on large default values
        if tokenizer.model_max_length and tokenizer.model_max_length > 512:
            tokenizer.model_max_length = 512
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        device_env = os.getenv("BIOBERT_DEVICE")
        if device_env is not None:
            try:
                device = int(device_env)
            except ValueError:
                device = -1
        else:
            device = -1 #if torch.cuda.is_available() else -1
        _biobert_pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
    return _biobert_pipe


@tool
async def search_web(
    query: str,
    num_results: int = 3
    ):
    """Search the web and get back a list of search results including the page title, url, and a short summary of each webpage.

    Args:
        query: The search query.
        num_results: The number of results to return, max is 3.

    Returns:
        A dictionary of the search results.
    """
    web_search = TavilySearch(max_results=min(num_results, 3), topic="general")
    search_results = web_search.invoke(input={"query": query})

    processed_results = {"query": query, "results": []}
    results_payload = []
    if isinstance(search_results, dict):
        results_payload = search_results.get("results", [])
    elif isinstance(search_results, list):
        results_payload = search_results

    for result in results_payload:
        if not isinstance(result, dict):
            continue
        title = result.get("title") or ""
        url = result.get("url") or ""
        content_preview = result.get("content") or result.get("snippet") or ""
        processed_results["results"].append({
            "title": title,
            "url": url,
            "content_preview": content_preview,
        })

    return processed_results


@tool
async def extract_content_from_webpage(urls: List[str]):
    """Extract the content from a webpage.

    Args:
        url: The url of the webpage to extract content from.

    Returns:
        A list of dictionaries containing the extracted content from each webpage.
    """
    web_extract = TavilyExtract()
    response = web_extract.invoke(input={"urls": urls})
    if isinstance(response, dict) and "results" in response:
        return response["results"]
    if isinstance(response, list):
        return response
    return []


@tool
async def extract_symptoms_with_biobert(text: str):
    """Use BioBERT NER to extract clinical entities and symptoms from text.

    Args:
        text: Clinical note or user description to analyze.

    Returns:
        List of extracted entities with type and character spans.
    """

    pipe = get_biobert_pipe()
    ner = pipe(text)
    return ner

class ResearchReport(BaseModel):
    topic: str
    report: str

@tool
async def generate_research_report(
    topic: str,
    report: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    ):
    """Generate a research report on a specific topic.

    Args:
        topic: The topic to research.
        report: The research report.
    
    Returns:
        The research report.
    """
    research_report = ResearchReport.model_validate({
        "topic": topic,
        "report": report
        })

    # We use the Command primitive to update the state with the research report and add a tool message to the conversation with the generated report.
    return Command(update={
        "research_reports": [research_report],
        "messages": [ToolMessage(
            name="generate_research_report",
            content=research_report.model_dump_json(),
            tool_call_id=tool_call_id,
            )],
        })


class ResearcherState(BaseModel):
    """State for the symptom/evidence agent.
    
    The research_reports attribute is shared with the supervisor state. This allows the supervisor to access the research reports generated by the symptom agent and share them downstream.
    """
    messages: Annotated[list, add_messages] = []
    research_reports: Annotated[list, operator.add] = []

tools = [
    search_web,
    extract_content_from_webpage,
    extract_symptoms_with_biobert,
    generate_research_report,
    ]

# Prefer a real OpenAI-style endpoint if provided; default to None so we don't try to hit localhost unless explicitly configured.
base_url = os.getenv("OPENAI_BASE_URL")
model_name = (
    os.getenv("OPENAI_MODEL")
    or os.getenv("OLLAMA_MODEL")
    or "gpt-4o-mini"
)
api_key = os.getenv("OPENAI_API_KEY", "ollama")

if base_url is None and api_key == "ollama":
    # Avoid silently calling OpenAI with an invalid key when no local endpoint is running.
    raise RuntimeError("Set OPENAI_API_KEY for OpenAI, or set OPENAI_BASE_URL/OLLAMA_MODEL for a local endpoint.")

llm = ChatOpenAI(
    name="Symptom",
    model=model_name,
    base_url=base_url,  # None -> uses OpenAI default; set to Ollama URL if you run locally.
    api_key=api_key,
)
llm_with_tools = llm.bind_tools(tools)

async def symptom_agent(state: ResearcherState):
    """The main symptom/evidence agent."""
    response = llm_with_tools.invoke([
        SystemMessage(content=researcher_prompt.format(current_datetime=datetime.now()))
        ] + state.messages)
    return {"messages": [response]}

async def symptom_router(state: ResearcherState) -> str:
    """Route to the tools node if the symptom agent makes a tool call."""
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(ResearcherState)

builder.add_node("symptom", symptom_agent)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("symptom")
builder.add_edge("tools", "symptom")
builder.add_conditional_edges(
    "symptom",
    symptom_router,
    {
        "tools": "tools",
        END: END,
    }
)

# Don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
graph = builder.compile()

# graph = builder.compile(checkpointer=MemorySaver())

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())
