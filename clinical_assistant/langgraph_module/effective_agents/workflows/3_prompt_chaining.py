"""
Prompt chaining is useful for complex workflows that require multiple steps to complete.

The example we'll build here is a blog post creation workflow with 3 main steps: outline, draft, and SEO optimization.

## KEY TAKEAWAYS
1. Breaking down a complex task into multiple steps allows us to have fine-grained control over the output of each step.
2. We can use the output of one step as the input to the next step. This allows us to build complex workflows by chaining together multiple prompts. Using multiple prompts also separates concerns which minimizes model hallucinations and context distraction. 
3. Task decomposition also allows us to highly optimize each step independently.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Annotated
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatOpenAI(
    name="Jude",
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.1,
)

# For this workflow we'll initiate a state with the message history and also 3 fields for the outline, draft, and final output. These correspond to the 3 main steps of the workflow.
class WorkflowState(BaseModel):
    """The state for our workflow."""
    messages: Annotated[list, add_messages] = []

    outline: str | None = None
    draft: str | None = None
    final: str | None = None

# We'll define a dummy research tool that just gets a summary for a topic from the LLM.
# In a real-world scenario, this tool would query a search engine or perform RAG over your own data, and optimize the returned context
@tool
def research_topic(topic: str) -> str:
    """Search the web for a query"""
    reponse = llm.invoke([f"Give a comprehensive but concise summary of the topic: {topic}"])
    return f"<TOOL RESPONSE> \n\n{reponse.content}"

tools = [research_topic]
llm_with_research_tool = llm.bind_tools(tools)

# The first step is to create an outline for the blog post. Notice that we're specifying some requirements for the outline
def create_outline(state: WorkflowState):
    system_prompt = SystemMessage(content="""You are a world class writer. Your job is to create an outline for a blog post based on the given topic. To create a good outline, you should research the topic by calling the 'research_topic' tool. Ensure the outline includes an introduction, a conclusion which always includes a call to action to my youtube video, a maximum of 3 sections in the body.
    """)
    response = llm_with_research_tool.invoke([system_prompt] + state.messages)
    if isinstance(response, AIMessage) and response.tool_calls:
        return {"messages": [response]}
    return {"outline": response.content}

# We can use a conditional edge as a gate to check that our outline meets the requirements and route based on the result
def check_outline_gate(state: WorkflowState) -> str:
    """This is a conditional edge which allows us to route the graph based on some condition"""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    elif state.outline:
        response = llm.invoke(f"""<Requirements>
        1. Has an introduction section
        2. Has a conclusion section with a call to action to my youtube video
        3. Has a maximum of 3 sections in the body
        4. Uses at least 3 emojis
        </Requirements>
        
        <Outline>
        {state.outline}
        </Outline>
        
        Ensure the outline meets the requirements. If the outline does not meet the requirements, response with FAIL. If it does, response with PASS. 
        """)
        if response.content == "PASS":
            return "write_draft"
    return END

# If the outline meets the requirements, we can write the draft
def write_draft(state: WorkflowState):
    response = llm.invoke(f"You are a world class writer. Write a short blog post based on the outline below: \n\n{state.outline}")
    return {"draft": response.content}

# Finally, we can optimize the draft for SEO
def optimize_seo(state: WorkflowState):
    response = llm.invoke(f"You are a world class writer and SEO expert. Your job is to revise the blog post draft to improve readability, increase engagement, and optimize for SEO. Rewrite the following draft: \n\n{state.outline}")
    return {"final": response.content}

# We build the graph as before with the additional nodes and edges
builder = StateGraph(WorkflowState)

builder.add_node(create_outline)
builder.add_node(write_draft)
builder.add_node(optimize_seo)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("create_outline")

builder.add_conditional_edges(
    "create_outline",
    check_outline_gate,
    {
        "tools": "tools",
        END: END,
        "write_draft": "write_draft",
    }
)

builder.add_edge("tools", "create_outline")
builder.add_edge("write_draft", "optimize_seo")
builder.add_edge("optimize_seo", END)

graph = builder.compile()

# Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())

response = graph.invoke(WorkflowState(messages=["Write a blog post about SEO optimization of Youtube Videos"]))

# Because are returned the complete state after the graph run, we can immediately access the output of each step which we saved to the state
print(response['outline'])
print(response['draft'])
print(response['final'])

# You can compare the output to a normal LLM response
result = llm.invoke("Write a blog post about SEO optimization of Youtube Videos")
print(result.content)

# By using prompt chaining you gain fine-grained control over each step of the process. This is useful when you want a high level of control over the final output, or when you want to break down a complex task into multiple steps.
