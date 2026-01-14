
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyExtract, TavilySearch
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

try:
	from pymedtermino.snomedct import SNOMEDCT
	SNOMED_AVAILABLE = True
except ImportError:  # pragma: no cover
	SNOMED_AVAILABLE = False

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"
validator_prompt = (PROMPTS_DIR / "validator.md").read_text(encoding="utf-8")


class ValidationResult(BaseModel):
	condition: str
	stance: str  # supporting | contradicting | uncertain
	confidence: float = Field(ge=0, le=1)
	evidence: List[str]
	citations: List[str]


class ValidatorState(BaseModel):
	"""State for the clinical validator."""

	messages: Annotated[list, add_messages] = []
	validation_results: Annotated[List[ValidationResult], operator.add] = []


@tool
async def search_web(query: str, num_results: int = 3):
	"""Search the web for clinical evidence."""

	search = TavilySearch(max_results=min(num_results, 3), topic="general")
	results = search.invoke(input={"query": query})
	return results


@tool
async def extract_content_from_webpage(urls: List[str]):
	"""Extract full text from the provided URLs."""

	extractor = TavilyExtract()
	response = extractor.invoke(input={"urls": urls})
	if isinstance(response, dict) and "results" in response:
		return response["results"]
	# Fallback if API returns a list directly
	if isinstance(response, list):
		return response
	return []


@tool
async def record_validation(
	validations: List[ValidationResult],
	tool_call_id: Annotated[str, InjectedToolCallId],
):
	"""Persist validation results for all hypotheses."""

	validated = [ValidationResult.model_validate(v) for v in validations]
	return Command(
		update={
			"validation_results": validated,
			"messages": [
				ToolMessage(
					name="record_validation",
					content="\n".join(v.model_dump_json() for v in validated),
					tool_call_id=tool_call_id,
				)
			],
		}
	)


llm = ChatOpenAI(
	name="Validator",
	model=os.getenv("OLLAMA_MODEL", "llama3.1"),
	base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
	api_key=os.getenv("OPENAI_API_KEY", "ollama"),
)

tools = [search_web, extract_content_from_webpage, record_validation]
def _add_snomed_tool(tools_list: list):
	tools_list.insert(0, validate_with_snomed)


@tool
async def validate_with_snomed(condition: str):
	"""Check a condition against SNOMED CT via PyMedTermino to filter impossible diagnoses."""

	if not SNOMED_AVAILABLE:
		return {
			"ok": False,
			"reason": "PyMedTermino/SNOMED CT not available; install PyMedTermino and configure SNOMED data.",
			"matches": [],
		}

	try:
		matches = [c for c in SNOMEDCT.search(condition)][:5]
		return {
			"ok": len(matches) > 0,
			"reason": "Found matching SNOMED concepts" if matches else "No SNOMED match",
			"matches": [str(m) for m in matches],
		}
	except Exception as exc:  # pragma: no cover
		return {
			"ok": False,
			"reason": f"SNOMED lookup failed: {type(exc).__name__}: {exc}",
			"matches": [],
		}


_add_snomed_tool(tools)
llm_with_tools = llm.bind_tools(tools)


async def validator_agent(state: ValidatorState):
	"""Validate hypotheses against evidence."""

	system_prompt = SystemMessage(
		content=validator_prompt.format(current_datetime=datetime.now())
	)
	response = llm_with_tools.invoke([system_prompt] + state.messages)
	return {"messages": [response]}


async def router(state: ValidatorState) -> str:
	if state.messages[-1].tool_calls:
		return "tools"
	return END


builder = StateGraph(ValidatorState)
builder.add_node(validator_agent)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("validator_agent")
builder.add_conditional_edges(
	"validator_agent",
	router,
	{
		"tools": "tools",
		END: END,
	},
)
builder.add_edge("tools", "validator_agent")

graph = builder.compile()
