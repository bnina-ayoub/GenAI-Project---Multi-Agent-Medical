"""
Orchestration-workers is useful for when you have a complex goal that we want to break down into smaller tasks, but where we don't know the number or details of the sub-tasks in advance.

The example we'll build here is a deep research workflow that breaks down a user's query into multiple sub-tasks and assigns those tasks to workers. The workers perform web search and content extraction to gather information, and then a synthesizer combines the reports from all workers into a single, comprehensive final report.

## KEY TAKEAWAYS
1. The orchestrator is responsible for breaking down the task into smaller sub-tasks and assigning those sub-tasks to the workers. So the orchestrator decides  what the sub-tasks are and how many there are.
2. While orchestrator-workers and parallelization are similar, the key difference is that the orchestrator agent has agency and can decide how to break down the task and assign work to the workers. There is no pre-determined set of parallel tasks or workflows, as we've defined in the parallelization pattern.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, add_messages, END, START
from langgraph.types import Send
from langchain_tavily import TavilySearch, TavilyExtract
import operator

load_dotenv()

llm = ChatOpenAI(
    name="Jude",
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.1,
)

# Deep Research
# orchestrator agent breaks down query into multiple subqueries for workers to research
# for each subquery, a worker agent researches the topic by performing websearch

class ResearchTask(BaseModel):
    """A research task to be performed by a worker agent."""
    topic: str
    search_query: str
    task: str

class ResearchTasks(BaseModel):
    """A list of research tasks to be performed by worker agents."""
    tasks: list[ResearchTask] = []

class CompletedTask(BaseModel):
    """A completed research task."""
    task: ResearchTask
    report: str


#################################
# State
#################################

class WorkflowState(BaseModel):
    messages: Annotated[list, add_messages] = []
    tasks: list[ResearchTask] = []
    completed_tasks: Annotated[list[CompletedTask], operator.add] = []
    final_report: str | None = None


#################################
# Orchestrator
#################################

orchestrator_llm = llm.with_structured_output(schema=ResearchTasks)

def orchestrator(state: WorkflowState):
    """Decompose the user's query into multiple research tasks for individual workers to complete in parallel."""
    system_prompt = SystemMessage(content="""
    You are a world-class research orchestrator managing a team of specialized research agents. Your role is to decompose the user’s query into distinct, self-contained research tasks that collectively address the query in depth. 
    
    Each subtopic will be researched in independently, in parallel, and the results will be combined into a final report. The research arents are not aware of each other, nor the overall objective or how their individual task fits into the overall objective. It is therefore important that each task is self-contained and does not rely on other tasks. It is also important that the task is specific and well-defined so that the agent can perform the task without any additional context. It is up to you to decide how each individual task fits into the overall objective.
                                  
    <Responsibilities>
    1. Understand the user’s objective and identify all key components needed for a comprehensive answer.
    2. Break the query into 3–7 research tasks that cover the topic fully, with minimal overlap and no missing aspects.
    3. Design each task to be self-contained:
        - Assume workers do not know the original query or each other’s tasks.
        - Restate necessary context in the task description itself.
    4. Provide a targeted search query for each worker:
        - Include relevant synonyms and context (e.g., “urban farming” for “vertical farming”).
        - Keep queries concise but broad enough for discovery.
    5. Balance depth and breadth: tasks should collectively form a complete picture of drivers, barriers, enablers, and context.
    6. Output strictly in JSON format as an array of objects with the following fields:
        - worker (string): short descriptive name of the task.
        - search_query (string): suggested search query to find information.
        - task (string): detailed description of what the worker must research.
    </Responsibilities>

    <Example>
        <User_Query>
        "What are the key factors driving the adoption of vertical farming in urban areas, and what are the main barriers preventing faster scaling?"
        </User_Query>
                                    
        <Research_Tasks>
        [
            {
                "topic": "Market Growth Drivers",
                "search_query": "vertical farming adoption drivers urban areas benefits sustainability food security",
                "task": "Research the primary reasons why urban areas and businesses are adopting vertical farming. Include factors such as sustainability, food security, local supply chains, and urban land use efficiency."
            },
            {
                "topic": "Technology Enablers",
                "search_query": "vertical farming technology advances LED automation hydroponics cost reduction",
                "task": "Investigate recent technological innovations that have improved the viability of vertical farming in urban settings, such as LED lighting, hydroponics, AI-driven automation, and cost reductions from scale."
            },
            {
                "topic": "Economic Barriers",
                "search_query": "vertical farming scaling barriers costs investment ROI economic challenges",
                "task": "Identify financial and economic challenges preventing faster scaling of vertical farming, including high capital costs, operational expenses, and investor hesitancy."
            },
            {
                "topic": "Regulatory and Infrastructure Barriers",
                "search_query": "vertical farming regulations urban planning zoning policy obstacles",
                "task": "Explore policy, zoning, and infrastructure challenges that limit vertical farming adoption in cities, such as building codes, land-use regulations, and utility access."
            },
            {
                "topic": "Consumer and Market Barriers",
                "search_query": "vertical farming consumer acceptance distribution scaling challenges market demand",
                "task": "Research consumer and market-related barriers to scaling vertical farming, including public perception, price sensitivity, distribution logistics, and market demand constraints."
            }
        ]
        </Research_Tasks>
    </Example>
    
    Generate the research tasks for the user's query.
    """)
    response = orchestrator_llm.invoke([system_prompt] + state.messages)
    return {"tasks": response.tasks}

def researcher_router(state: WorkflowState) -> str:
    """Route to the researcher node for each task."""
    if state.tasks:
        # We use the Send API from Langgraph to specify the node name and input data for each task.
        # This will spawn a new research node for each task at runtime. So we don't need to specify a fixed number of workers or graph routes.
        return [Send("researcher", {"task": task}) for task in state.tasks]
    

#################################
# Workers
#################################

# We will define a separate worker state. This allows for each worker to have its own state and context that doesn't depend on the overall state or other workers.
class WorkerState(BaseModel):
    task: ResearchTask
    completed_tasks: Annotated[list[CompletedTask], operator.add]

# We will use Tavily for web search and TavilyExtract for content extraction
# In this simplified example, we'll manage the search and content extraction for each worker. In a more complex system, we might use a multi-agent approach where each worker can be given just the topic and then generate their own strategy for search and content extraction.
tavily_search = TavilySearch(max_results=2, topic="general")
tavily_extract = TavilyExtract()

def researcher(state: WorkerState):
    """Perform research for a given task."""
    system_prompt = SystemMessage(content=f"""
    You are a specialized research agent assigned one focused task. You will receive a task description and search results for the given task. Your goal is to synthesize high-quality information to answer your assigned task thoroughly and concisely.
                                  
    <Instructions>
    - Focus only on the provided task — do not address unrelated topics.
    - Present findings as:
        - Key insights / facts (bullet points preferred).
        - Brief explanations or examples where helpful.
        - Cite timeframes or quantitative data when available.
    - Each finding should be supported by a source from the search results: include the source URL and a short quote or reference.
    - Do not speculate — base answers only on the provided search results.
    - Output should be standalone (no need for other tasks or context).
    </Instructions>
    """)

    # The compressed context is what we'll give the LLM in order to generate the research report for the individual task. It will include the raw content from each search result.
    compressed_context = {
        "query": state["task"].search_query,
        "results": []
    }

    # Perform web search. For this example, we're limiting search results to 2 above where we defined the tavily_search tool.
    search_results = tavily_search.invoke(input={
        "query": state["task"].search_query,
        })
    
    # Extract the content from each search result.
    # For each search result, we'll provide the title, URL, and raw content.
    for result in search_results["results"]:
        try:
            result_contents = tavily_extract.invoke(input={"urls": [result["url"]]})
            raw_content = result_contents["results"][0]["raw_content"]

            compressed_context["results"].append({
                "title": result["title"],
                "url": result["url"],
                "content": raw_content
            })
        except:
            continue

    research_context = HumanMessage(content=f"<SEARCH RESULTS>\n{str(compressed_context)}</SEARCH RESULTS>")

    # Now we can pass the system prompt and context of all search results to the LLM.
    response = llm.invoke([system_prompt, research_context])

    # We'll save the original task and report to the completed tasks list.
    # Because this key is shared between our WorkerState and WorkflowState, Langgraph will automatically update the parent WorkflowState. The shared attribute names must be exactly the same.
    completed_task = CompletedTask.model_validate({
        "task": state["task"],
        "report": response.content
    })
    return {"completed_tasks": [completed_task]}


#################################
# Synthesizer
#################################

def synthesizer(state: WorkflowState):
    """Combine the reports from all workers into a single, comprehensive final report."""
    system_prompt = SystemMessage(content=f"""
    You are a senior research analyst responsible for combining the outputs of multiple specialized research agents into a single, comprehensive final report.

    <Role>
    - Integrate findings from all workers into a clear, logically organized report.
    - Resolve overlap and contradictions (prioritize accuracy and consensus).
    - Provide balanced depth — highlight key insights while summarizing details.
    - Ensure the report is standalone: no internal references to workers or tasks.
    </Role>

    <Instructions>
    - Read all worker outputs carefully.
    - Identify common themes and differences; merge them into cohesive sections.
    - Organize the report logically, e.g.:
        - Introduction / context
        - Key drivers or enablers
        - Barriers or challenges
        - Conclusion or recommendations (if relevant)
    - Always include a sources section with all relevant citations. Citations should be in the format: [Source Name] (URL): they must include the URL.
    - Include references to all sources cited within the report.
    - Maintain neutrality — report findings rather than personal opinions.
    - Incorporate data, examples, and sources from workers when available.
    - Eliminate redundancy and resolve conflicting information with clarification.
    - Output format: well-structured markdown for readability.
    - The report should read like a McKinsey research report.
    </Instructions>
    """)

    # We'll merge all of the individual reports into a single string.
    reports = [str(task.model_dump_json()) for task in state.completed_tasks]

    response = llm.invoke([system_prompt] + ["\n".join(reports)])
    return {"final_report": response.content}


#################################
# Build the Graph
#################################

builder = StateGraph(WorkflowState)

builder.add_node(orchestrator)
builder.add_node(researcher)
builder.add_node(synthesizer)

builder.set_entry_point("orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    researcher_router,
    ["researcher"]
)

builder.add_edge("researcher", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()


# Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())


# Let's find out when AI will take over all of our jobs
response = graph.invoke(WorkflowState(messages=["What is the impact of AI on the job market?"]))

# inspect tasks
for task in response["tasks"]:
    print(task.model_dump_json() + "\n")

# inspect completed tasks
for task in response["completed_tasks"]:
    print(task.model_dump_json() + "\n")

response["completed_tasks"][0].task

# final report
print(response["final_report"])
