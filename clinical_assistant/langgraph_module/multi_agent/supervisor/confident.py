import json
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
confident_prompt = (PROMPTS_DIR / "confident.md").read_text(encoding="utf-8")


class RankedHypothesis(BaseModel):
	condition: str
	confidence: float = Field(ge=0, le=1)
	rationale: str
	next_steps: List[str] = Field(default_factory=list)
	red_flags: List[str] = Field(default_factory=list)


class ConfidencePlan(BaseModel):
	decision: str  # proceed | gather_more_data | escalate
	overall_confidence: float = Field(ge=0, le=1)
	escalate: bool
	ranked: List[RankedHypothesis]
	blockers: List[str] = Field(default_factory=list)


class ConfidenceState(BaseModel):
	"""State for the confidence/triage agent."""

	messages: Annotated[list, add_messages] = []
	plan: ConfidencePlan | None = None


@tool
async def assign_confidence(
	decision: str,
	overall_confidence: Union[float, str],
	escalate: Union[bool, str],
	ranked: Union[List[RankedHypothesis], str],
	blockers: Union[List[str], str],
	tool_call_id: Annotated[str, InjectedToolCallId],
):
	"""Submit triage decision, ranking, and next steps."""

	# Some models may serialize arguments as JSON strings; coerce to structured types.
	def coerce_json(value):
		if isinstance(value, str):
			value = value.strip()
			try:
				return json.loads(value)
			except json.JSONDecodeError:
				return value
		return value

	ranked = coerce_json(ranked)
	blockers = coerce_json(blockers)
	if isinstance(blockers, str):
		blockers = [blockers]

	if isinstance(escalate, str):
		escalate = escalate.lower() in {"true", "1", "yes"}

	if isinstance(overall_confidence, str):
		try:
			overall_confidence = float(overall_confidence)
		except ValueError:
			overall_confidence = 0.0

	plan = ConfidencePlan.model_validate(
		{
			"decision": decision,
			"overall_confidence": overall_confidence,
			"escalate": escalate,
			"ranked": ranked,
			"blockers": blockers,
		}
	)

	return Command(
		update={
			"plan": plan,
			"messages": [
				ToolMessage(
					name="assign_confidence",
					content=plan.model_dump_json(),
					tool_call_id=tool_call_id,
				)
			],
		}
	)


llm = ChatOpenAI(
	name="Confidence",
	model=os.getenv("OLLAMA_MODEL", "llama3.1"),
	base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
	api_key=os.getenv("OPENAI_API_KEY", "ollama"),
)

tools = [assign_confidence]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


async def confidence_agent(state: ConfidenceState):
	"""Synthesize validated hypotheses and set triage direction."""

	system_prompt = SystemMessage(
		content=confident_prompt.format(current_datetime=datetime.now())
	)
	response = llm_with_tools.invoke([system_prompt] + state.messages)
	return {"messages": [response]}


async def router(state: ConfidenceState) -> str:
	if state.messages and state.messages[-1].tool_calls:
		return "tools"
	return END


builder = StateGraph(ConfidenceState)
builder.add_node(confidence_agent)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("confidence_agent")
builder.add_conditional_edges(
	"confidence_agent",
	router,
	{
		"tools": "tools",
		END: END,
	},
)
builder.add_edge("tools", "confidence_agent")

graph = builder.compile()
