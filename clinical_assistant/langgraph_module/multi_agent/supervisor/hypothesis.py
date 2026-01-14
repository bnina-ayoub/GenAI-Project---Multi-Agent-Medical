
import ast
import json
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Union

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"
hypothesis_prompt = (PROMPTS_DIR / "hypothesis.md").read_text(encoding="utf-8")


class Hypothesis(BaseModel):
	condition: str
	likelihood: float = Field(ge=0, le=1)
	rationale: str
	recommended_tests: List[str] = Field(default_factory=list)
	red_flags: List[str] = Field(default_factory=list)


class HypothesisState(BaseModel):
	"""State for the hypothesis generator."""

	messages: Annotated[list, add_messages] = []
	hypotheses: Annotated[List[Hypothesis], operator.add] = []


@tool
async def record_hypotheses(
	hypotheses: Union[List[Hypothesis], str],
	tool_call_id: Annotated[str, InjectedToolCallId],
):
	"""Persist a ranked differential diagnosis list."""

	def coerce_hypotheses(value: Union[List[Hypothesis], str, dict]):
		"""Handle models that emit JSON strings or Python-literal-like strings."""
		if isinstance(value, list):
			return value
		if isinstance(value, dict):
			return [value]
		if isinstance(value, str):
			s = value.strip()
			# Drop surrounding quotes if present
			if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
				s = s[1:-1]
			# Try JSON first
			try:
				return json.loads(s)
			except json.JSONDecodeError:
				pass
			# Try Python literal (e.g., single quotes)
			try:
				return ast.literal_eval(s)
			except Exception:
				pass
			# Last-ditch: replace single quotes with double and retry JSON
			try:
				return json.loads(s.replace("'", '"'))
			except Exception:
				return []  # fall back to empty list instead of crashing
		return []

	coerced = coerce_hypotheses(hypotheses)
	validated = [Hypothesis.model_validate(h) for h in coerced]

	def _safe_dump_all(items: List[Hypothesis]) -> str:
		"""Serialize hypotheses to JSON, tolerating surrogate/invalid code points."""
		dumps: List[str] = []
		for h in items:
			data = h.model_dump()
			text = json.dumps(data, ensure_ascii=False)
			try:
				text.encode("utf-8")
			except UnicodeEncodeError:
				text = text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
			dumps.append(text)
		return "\n".join(dumps)

	return Command(
		update={
			"hypotheses": validated,
			"messages": [
				ToolMessage(
					name="record_hypotheses",
					content=_safe_dump_all(validated),
					tool_call_id=tool_call_id,
				)
			],
		}
	)


llm = ChatOpenAI(
	name="Hypothesis",
	model=os.getenv("OLLAMA_MODEL", "llama3.1"),
	base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
	api_key=os.getenv("OPENAI_API_KEY", "ollama"),
)

tools = [record_hypotheses]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


async def hypothesis_agent(state: HypothesisState):
	"""Generate a differential diagnosis."""

	system_prompt = SystemMessage(
		content=hypothesis_prompt.format(current_datetime=datetime.now())
	)

	response = llm_with_tools.invoke([system_prompt] + state.messages)
	return {"messages": [response]}


async def router(state: HypothesisState) -> str:
	if state.messages[-1].tool_calls:
		return "tools"
	return END


builder = StateGraph(HypothesisState)
builder.add_node(hypothesis_agent)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("hypothesis_agent")
builder.add_conditional_edges(
	"hypothesis_agent",
	router,
	{
		"tools": "tools",
		END: END,
	},
)
builder.add_edge("tools", "hypothesis_agent")

graph = builder.compile()