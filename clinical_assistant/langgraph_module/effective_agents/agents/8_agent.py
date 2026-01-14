"""
Agents are useful when you have a complex task that requires multiple steps to complete but the number, order, and details of the steps are not known in advance.

The defining capabilities of agents are:
1. Agency. The agent can decide what to do next based on the current state.
2. Acting. The agent can use tools to take action on its environment. 
3. Perceiving. It can also perceive the environment.
3. Self-refining. The agent has the capacity to improve its performance over time, e.g. learning from its own mistakes.

## KEY TAKEAWAYS
1. Agents are useful when you have a complex task that requires multiple steps to complete but the number, order, and details of the steps are not known in advance.
2. When put to the right task, they can be extremely powerful and capable.
3. Agents are probabilistic and can therefore be highly creative. But this also means that the agent may be highly unpredictable with varying outputs.
4. If you can clearly define the steps needed to solve a task, then an agent is likely not the best solution. Instead, a workflow where you can optimize every step of the process will likely produce a higher quality output and more reliably.
5. Always explore the simplest solutions first. Agents are complex and harder to debug and optimize. Only use them if you have clearly established that a simpler workflow will not work.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal, List
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, AIMessageChunk
from langgraph.graph import StateGraph, add_messages, END
from langgraph.types import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch, TavilyExtract
from datetime import datetime

load_dotenv()


# Switch models to see how the post quality improves with a more capable model
llm = ChatOpenAI(
    name="Jude",
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

class TaskList(BaseModel):
    """A list of tasks to be completed."""
    tasks: List[Task] = Field(..., description="The list of tasks to be completed")

class AgentState(BaseModel):
    """The state of the agent."""
    messages: Annotated[list, add_messages] = []
    post: str | None = None


#################################
# Tools
#################################

# We will create some tools to help the agent plan and track tasks.
# Tasks will be stored in our in-memory store which is just the dictionary we created earlier
@tool
def generate_task_list(task_list: TaskList):
    """Generate a new task list or update the existing task list by replacing it."""
    store["tasks"] = task_list
    return store["tasks"].model_dump_json()

@tool
def view_task_list():
    """View the task list"""
    if "tasks" not in store:
        return "No task list found."
    return store["tasks"].model_dump_json()

# The agent will also have web search capabilities for content research
@tool
def search_web(query: str, num_results: int = 3):
    """Search the web and get back a list of search results including the page title, url, and a short summary of each webpage.

    Args:
        query: The search query.
        num_results: The number of results to return, max is 3.

    Returns:
        A dictionary of the search results.
    """
    tavily_search = TavilySearch(max_results=max(num_results, 3), topic="general")
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
    """Extract the raw content from a webpage. Use this tool if you need the full context of a webpage.

    Args:
        url: The url of the webpage to extract content from.
    """
    result_contents = tavily_extract.invoke(input={"urls": [url]})
    raw_content = result_contents["results"][0]["raw_content"]
    return raw_content

# Finally, we'll dump a bunch of golden standard posts into this retrieval tool. There are a couple of benefits to putting the examples here rather than the system prompt.
# 1. This simplifies the system prompt so we can focus on the main goals of the agent and optimize for agent performance.
# 2. Because we instruct the agent to review the golden standard posts before writing the final post, the agent will pull this information when it needs it and it will be the newest information in the context window. This avoids common issues with long context windows such as "lost in the middle".
@tool
def view_golden_posts():
    """View examples of gold standard posts."""
    golden_posts = """
    <Examples>
        <Example_1>
        I hit 250,000 followers in 365 days.
        Not by doing more.
        By unlearning everything I was told about LinkedIn.

        Here’s what actually moved the needle: 👇 

        1. I stopped chasing vanity metrics
        146,000 likes? Didn’t change my life.
        One DM: “You saved my career.” That did.

        → Impressions = ego
        → Impact = legacy

        2. I stopped writing to impress
        I wrote what I needed to hear - back when I was 32, burnt out, and breaking down.
        250K didn’t follow polish. They followed truth.

        3. I stopped copying “top creators”
        “Don’t use Singlish.”
        “Write like an American.”
        I said no.

        I wrote like a Singaporean - because that’s who I am lah!
        → When I owned my voice, others saw themselves in it.

        4. I stopped apologising for my voice
        That post on toxic workplaces?
        I was shaking when I hit publish.

        167 million views later, I realised:
        → The post you're scared to share
        → Is the one someone else needs

        5. I stopped writing for the algorithm
        I wrote for the girl crying in the office bathroom.
        Because I was her.

        No gimmicks. No formulas.
        Just truth people could feel.

        This isn’t just content.
        It’s a call.

        For those who are:
        ✔ Done pretending
        ✔ Done performing
        ✔ Done playing small

        Join me.
        I’m not just building a following.
        I’m starting a movement.

        -----

        If this resonated, repost it. 

        And follow me (Shulin Lee), for more!

        P.S. 300K? It’s coming. 🤞
        </Example_1>
                                  
        <Example_2>
        Talk is cheap.

        Your actions say everything.

        One of the most important lessons I've learned as a founder: 

        Not all feedback is created equal. 

        The harsh reality is, people will always criticise.
        So you need the right people around you. 

        Because where the encouragement comes from, 
        Is the most important factor... 

        It must come from people with the right attitude. 
        Who have done what you're trying to do. 

        Those are the ones who can lift you up, 
        And give you the right feedback to keep going. 

        Here’s what it looks like in the real world:

        🚫 Average achievers mock you for working out.
        ✅ Healthy people cheer your first workout.

        🚫 Insecure individuals tear your ideas apart.
        ✅ Secure people support your ideas.

        🚫 Bitter people will mock you.
        ✅ Happy people give you compliments.

        🚫 Unsuccessful folks want you to fail.
        ✅ Successful people root for others.

        🚫 Untalented individuals will judge your business.
        ✅ A successful business person will give you the feedback you need.

        The way we treat others is a reflection of our state of mind. 

        Because most of the time, it's always the people who are going nowhere,
        That have something genuinely negative to say. 

        So next time you give or ask for advice, 
        Ask yourself who or what you want to emulate. 

        If you want to surround yourself with better inputs,
        I share my frameworks, systems and tools every Sunday in Step By Step. 

        Join 200K+ founders and creators who are: 

        - Building and scaling in real-time
        - Want more financial and time freedom
        - Are committed to creating systems that work for them

        Sign up here 👇
        https://bit.ly/4dCkNEf

        You’ll get an additional 30+ free learning resources.

        Who’s in your circle that truly lifts you up?
        Tag them in the comments to show some appreciation. 

        ♻️ Repost to help someone upgrade their environment.
        And follow Chris Donnelly for more personal development advice.
        </Example_2>

        <Example_3>
        I had never heard anyone explain how to manifest like this before…

        I just learned of the passing of renowned neurosurgeon and neuroscientist, Dr. Jim Doty, who had a profound impact on my life. Dr. Doty taught me how to tap into the unseen possibilities and greater forces of life using neuroscience and the scientific approach to manifesting.

        When Dr. Doty appeared on The Mel Robbins Podcast in late 2024, he taught us all how to use manifestation, neuroscience, and magic to unlock the power of your brain, your heart, and your deepest purpose in life.

        That episode went on to become one of our most popular of all time, including becoming our most viewed episode on YouTube ever.

        It is one of the most shared, listened to, and beloved episodes of all time.

        It is my honor to re-release that episode again today, with a new opening that I just recorded 🎧 "#1 Neurosurgeon: How to Manifest Anything You Want & Unlock the Unlimited Power of Your Mind (In Memory of Dr. Jim Doty)"

        I cannot wait for you to experience the magic for yourself.

        I hope you will not only listen to this episode but that you will also share it so Dr. Doty's wisdom, magic, and legacy lives on through us all.

        https://lnkd.in/eBDtG--3
        </Example_3>

        <Example_4>
        11 phrases you should NOT say in a job interview -

        And what TO say instead:

        Interviews are high-pressure.

        Sometimes you have only 20 minutes to:
        ↳Showcase your value
        ↳Express genuine interest
        ↳And prove you're the right fit

        Don't let a poor answer send the wrong signal.

        Use this infographic to avoid common pitfalls,
        And leave a lasting positive impression:

        1. Avoid: "I don't have any questions"
        ↳Because: It makes you seem disinterested or unprepared
        ↳Instead say: "What does success look like in this position over the first few months?"

        2. Avoid: "I'll take any job I can get"
        ↳Because: It comes across as desperate instead of focused
        ↳Instead say: "I'm eager to apply my skills in XYZ, and this role is a great fit"

        3. Avoid: "My last manager was awful"
        ↳Because: Speaking poorly of others is a red flag
        ↳Instead say: "That role taught me a lot - especially about X - but now I'm looking for a better fit for my goals"

        4. Avoid: "I don't know much about your company"
        ↳Because: It signals poor research and low enthusiasm
        ↳Instead say: "I looked into your work on X and found it really compelling - can you share more about it?"

        5. Avoid: "What's the salary?"
        ↳Because: It's fair to ask, just do it in a way that shows it's not your only focus
        ↳Instead say: "I'm really interested in the role - could you share the pay range to make sure we're aligned?"

        6. Avoid: "I don't have any weaknesses"
        ↳Because: It sounds arrogant and out of touch
        ↳Instead say: "One area I've been improving is X, and I've been doing Y to get better at it"

        7. Avoid: "I struggle with time management"
        ↳Because: Weaknesses need to come with solutions
        ↳Instead say: "Time management has been a priority for me, and I now rely on X and Y to stay on top of things"

        8. Avoid: "I know I'm not fully qualified, but..."
        ↳Because: You shouldn't undermine your own candidacy
        ↳Instead say: "I'm excited by the challenge and confident my background in X can bring real value here"

        9. Avoid: "I got fired"
        ↳Because: Without context, you'll be judged - even if that's unfair
        ↳Instead say: "That job ended unexpectedly, but I used the transition to grow and sharpen my skills in X"

        10. Avoid: "I don't really have career plans"
        ↳Because: Employers want to see ambition to grow
        ↳Instead say: "Right now, I'm focused on excelling in this role - over time, I hope to advance in X direction"

        11. Avoid: "When can I get promoted?"
        ↳Because: Despite number 10 above, impatient ambition is a red flag (no one said this was easy!)
        ↳Instead say: "Can you tell me how career growth typically works here and how people are supported?"


        A single bad answer can overshadow your strengths.

        Avoid these mistakes, and let your talent shine through.

        I'm always curious: what's your best interview tip? 

        ---

        ♻️ Repost to help someone in your network who's interviewing. 

        And follow me George Stern for more career growth content.
        </Example_4>
    </Examples>
    """
    return golden_posts


#################################
# Build the Graph
#################################

tools = [generate_task_list, view_task_list, search_web, extract_content_from_webpage, view_golden_posts]
llm_with_tools = llm.bind_tools(tools)

def agent(state: AgentState):
    system_prompt = SystemMessage(content=f"""You are a LinkedIn content creator specializing in AI topics. Your job is to create engaging LinkedIn posts that are informative and create high value for the reader. All posts should meet the requirements below.

    <Post_Requirements>
    - 100-300 words long
    - Use a conversational, professional tone
    - Start with a strong hook - e.g. a surprising statement, statistic, or thought-provoking question
    - Include practical insights or actionable takeaways
    - End with a call-to-action or question, encouraging engagement
    - Focus on business value and real-world applications
    - Do not include a title, just get right into the post
    - Avoid long paragraphs
    - Create content that would resonate with business professionals, entrepreneurs, and tech enthusiasts on LinkedIn that are interested in AI.
    - Include at least 1 emoji
    - Include a call to action to my youtube channel at the end of the post. My youtube channel is https://www.youtube.com/@KennethLiao
    </Post_Requirements>
    
    <Guidelines>
    - Always create a plan before executing. Use the generate_task_list tool to save the plan and track progress.
    - If you need to update the task list, use the generate_task_list tool to replace the existing task list with the updated task list.
    - Always check the status of the plan before executing a new task using the view_task_list tool.
    - Once you've come up with a plan and task list, continue working on the plan until every task is complete. 
    - Use the view_task_list tool and verify that all tasks are complete before providing your final response.
    - When you're ready to write the post, always use the view_golden_posts tool to view examples of gold standard posts. Use the golden standard posts to inform the writing style and format of your post before writing your final post.
    - If you need more information to understand the request, ask the user. Gather any necessary context before proceeding.
    - Bias towards searching the web for the latest information rather than relying on your existing knowledge or making assumptions.
    </Guidelines>
    
    <Tool_Usage>
    You have access to several tools. You can use the tools as needed, in any order, and as many times as needed to accomplish your goal. For example, if you do not find the information you need in the first search results, you can search again using a different query.
    </Tool_Usage>

    <Tools>
    generate_task_list: Use this tool to both create new task lists and/or make updates to the existing task list by replacing it.
    view_task_list: Use this tool to view the current task list.
    search_web: Use this tool to search the web with a natural language query. Returned results include the page title, url, and a short summary of each webpage.
    extract_content_from_webpage: Use this tool to extract the complete contents from a webpage given the url.
    view_golden_posts: Use this tool to view examples of gold standard posts before writing your final post.
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

graph = builder.compile(checkpointer=MemorySaver())
graph_no_checkpointer = builder.compile()

# Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())


#################################
# Add Streaming
#################################

# All agents need streaming. This is a common function I use to process stream chunks from the graph.
async def stream_graph_responses(
        input: AgentState,
        graph: StateGraph,
        **kwargs
        ):
    """Asynchronously stream the result of the graph run.

    Args:
        input: The input to the graph.
        graph: The compiled graph.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The final LLM or tool call response
    """
    async for message_chunk, metadata in graph.astream(
        input=input,
        stream_mode="messages",
        **kwargs
        ):
        if isinstance(message_chunk, AIMessageChunk):
            if message_chunk.response_metadata:
                finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                if finish_reason == "tool_calls":
                    yield "\n\n"

            if message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]

                tool_name = tool_chunk.get("name", "")
                args = tool_chunk.get("args", "")

                if tool_name:
                    tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"
                if args:
                    tool_call_str = args

                yield tool_call_str
            else:
                yield message_chunk.content
            continue


async def main():
    try:
        config=RunnableConfig(configurable={
            "thread_id": "1",
            "recursion_limit": 30,
            })

        while True:

            user_input = input("\n\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("\n\nExit command received. Exiting...\n\n")
                break

            print(f"\n\n ----- 🥷 Human ----- \n\n{user_input}\n")

            graph_input = AgentState(
                messages=[
                    HumanMessage(content=user_input),
                ]
            )

            print(f" ---- 🤖 Jude ---- \n")
            async for response in stream_graph_responses(graph_input, graph, config=config):
                print(response, end="", flush=True)

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Try these example queries
    "Create a post about how people often overcomplicate things by unnecessarily building AI agents where workflows would suffice."
    "Create a post about how AI is disrupting the job market."

    asyncio.run(main())
