import json
import operator
from datetime import datetime
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import RunnableConfig
from pydantic import BaseModel

from clinical_assistant.langgraph_module.multi_agent.supervisor.symptom import graph as symptom_agent
from clinical_assistant.langgraph_module.multi_agent.supervisor.hypothesis import graph as hypothesis_agent
from clinical_assistant.langgraph_module.multi_agent.supervisor.validator import graph as validator_agent
from clinical_assistant.langgraph_module.multi_agent.supervisor.confident import graph as confident_agent
from clinical_assistant.langgraph_module.multi_agent.supervisor.xai import graph as xai_agent

load_dotenv()

class SupervisorState(BaseModel):
    """State shared across the clinical multi-agent pipeline."""

    messages: Annotated[list, add_messages] = []
    task_description: str | None = None
    research_reports: Annotated[list, operator.add] = []
    hypotheses: Annotated[list, operator.add] = []
    validation_results: Annotated[list, operator.add] = []
    plan: dict | None = None
    explanation: dict | None = None


def _get_task_description(state: SupervisorState) -> str:
    if state.task_description:
        return state.task_description
    # fall back to last human message
    if state.messages:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
    return ""


async def call_symptom(state: SupervisorState, config: RunnableConfig):
    """Gather clinical evidence and citations for the presenting complaint."""

    task_desc = _get_task_description(state)
    symptom_response = await symptom_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=task_desc)],
        },
        config=config,
    )

    ai_message = AIMessage(
        name="symptom",
        content=symptom_response["messages"][-1].content,
    )

    return {
        "task_description": task_desc,
        "research_reports": symptom_response.get("research_reports", []),
        "messages": [ai_message],
    }


async def call_hypothesis(state: SupervisorState, config: RunnableConfig):
    """Generate differential hypotheses using the evidence from symptom agent."""

    task_desc = _get_task_description(state)
    evidence_blob = "\n\n".join(r.report for r in state.research_reports) if state.research_reports else ""
    hyp_input_text = f"Task: {task_desc}\n\nEvidence:\n{evidence_blob}"

    hyp_response = await hypothesis_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=hyp_input_text)],
        },
        config=config,
    )

    ai_message = AIMessage(
        name="hypothesis",
        content=hyp_response["messages"][-1].content,
    )

    return {
        "hypotheses": hyp_response.get("hypotheses", []),
        "messages": [ai_message],
    }


async def call_validator(state: SupervisorState, config: RunnableConfig):
    """Validate hypotheses against external evidence."""

    hyp_blob = "\n".join([h.model_dump_json() for h in state.hypotheses]) if state.hypotheses else ""
    val_input = f"Validate these hypotheses:\n{hyp_blob}"

    val_response = await validator_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=val_input)],
        },
        config=config,
    )

    ai_message = AIMessage(
        name="validator",
        content=val_response["messages"][-1].content,
    )

    return {
        "validation_results": val_response.get("validation_results", []),
        "messages": [ai_message],
    }


async def call_confident(state: SupervisorState, config: RunnableConfig):
    """Synthesize triage decision and next steps."""

    validation_blob = "\n".join([v.model_dump_json() for v in state.validation_results]) if state.validation_results else ""
    conf_input = f"Create triage plan using validation results:\n{validation_blob}"

    conf_response = await confident_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=conf_input)],
        },
        config=config,
    )

    ai_message = AIMessage(
        name="confident",
        content=conf_response["messages"][-1].content,
    )

    plan_obj = conf_response.get("plan")
    plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj

    return {
        "plan": plan_dict,
        "messages": [ai_message],
    }


async def call_xai(state: SupervisorState, config: RunnableConfig):
    """Generate clinician and patient-facing explanations."""

    plan_blob = json.dumps(state.plan, ensure_ascii=False) if isinstance(state.plan, dict) else str(state.plan or "")
    validation_blob = (
        "\n".join([v.model_dump_json() for v in state.validation_results])
        if state.validation_results
        else ""
    )
    hyp_blob = (
        "\n".join([h.model_dump_json() for h in state.hypotheses])
        if state.hypotheses
        else ""
    )

    xai_input = (
        "Explain the plan and findings using the JSON below.\n\n"
        f"Plan:\n{plan_blob}\n\n"
        f"Validated hypotheses:\n{validation_blob}\n\n"
        f"Original hypotheses:\n{hyp_blob}\n"
    )

    xai_response = await xai_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=xai_input)],
        },
        config=config,
    )

    ai_message = AIMessage(
        name="xai",
        content=xai_response["messages"][-1].content,
    )

    return {
        "explanation": xai_response.get("explanation"),
        "messages": [ai_message],
    }

builder = StateGraph(SupervisorState)
builder.add_node(call_symptom)
builder.add_node(call_hypothesis)
builder.add_node(call_validator)
builder.add_node(call_confident)
builder.add_node(call_xai)

builder.set_entry_point("call_symptom")

builder.add_edge("call_symptom", "call_hypothesis")
builder.add_edge("call_hypothesis", "call_validator")
builder.add_edge("call_validator", "call_confident")
builder.add_edge("call_confident", "call_xai")
builder.add_edge("call_xai", END)

graph = builder.compile()
