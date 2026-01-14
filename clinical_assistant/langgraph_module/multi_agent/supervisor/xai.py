import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, List

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
xai_prompt = (PROMPTS_DIR / "xai.md").read_text(encoding="utf-8")


class Explanation(BaseModel):
	clinician_summary: str
	patient_summary: str
	evidence: List[str] = Field(default_factory=list)
	caveats: List[str] = Field(default_factory=list)
	next_steps: List[str] = Field(default_factory=list)
	faq: List[str] = Field(default_factory=list)


class XAIState(BaseModel):
	"""State for the explainer agent."""

	messages: Annotated[list, add_messages] = []
	explanation: Explanation | None = None


@tool
async def generate_explanation(
	clinician_summary: str,
	patient_summary: str,
	evidence: List[str],
	caveats: List[str],
	next_steps: List[str],
	faq: List[str],
	tool_call_id: Annotated[str, InjectedToolCallId],
):
	"""Submit dual-audience explanations."""

	explanation = Explanation.model_validate(
		{
			"clinician_summary": clinician_summary,
			"patient_summary": patient_summary,
			"evidence": evidence,
			"caveats": caveats,
			"next_steps": next_steps,
			"faq": faq,
		}
	)

	return Command(
		update={
			"explanation": explanation,
			"messages": [
				ToolMessage(
					name="generate_explanation",
					content=explanation.model_dump_json(),
					tool_call_id=tool_call_id,
				)
			],
		}
	)


llm = ChatOpenAI(
	name="XAI",
	model=os.getenv("OLLAMA_MODEL", "llama3.1"),
	base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
	api_key=os.getenv("OPENAI_API_KEY", "ollama"),
)
	# reasoning_effort parameter removed to avoid unsupported 'thinking' on Ollama

tools = [generate_explanation]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


async def xai_agent(state: XAIState):
	"""Generate clinician- and patient-facing explanations."""

	system_prompt = SystemMessage(content=xai_prompt.format(current_datetime=datetime.now()))
	response = llm_with_tools.invoke([system_prompt] + state.messages)
	return {"messages": [response]}


async def router(state: XAIState) -> str:
	if state.messages and state.messages[-1].tool_calls:
		return "tools"
	return END


builder = StateGraph(XAIState)
builder.add_node(xai_agent)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("xai_agent")
builder.add_conditional_edges(
	"xai_agent",
	router,
	{
		"tools": "tools",
		END: END,
	},
)
builder.add_edge("tools", "xai_agent")

graph = builder.compile()
