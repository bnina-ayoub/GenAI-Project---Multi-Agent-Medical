"""
Routing is useful when you have a complex workflow with multiple paths that depend on some condition.

The example we'll build here is a content creation workflow with a router that classifies the customer's request into one of the content types: LinkedIn, Instagram, and Blog.

## KEY TAKEAWAYS
1. Similar to prompt chaining, having separate workflows that we can route to allows us to separate concerns and optimize each workflow independently. Our requirements, tools, and evaluation criteria for LinkedIn posts can be very different from our requirements for a Blog post.
2. Routing also makes the system modular. In this example you can add a new platform by simply adding a new node and edge to the graph, without having to significantly change the existing code.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END, START
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatOpenAI(
    name="Jude",
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.1,
)


#################################
# State
#################################

# For this routing example we're going to build a workflow for a copywriter where the content requirements are different for each platform: LinkedIn, Instagram, and Blog.
# We'll keep track of the content as well as a flag for whether the content needs to be rewritten. For simplicity in this example we're only going to demonstrate a rewrite process for the LinkedIn flow.
class WorkflowState(BaseModel):
    messages: Annotated[list, add_messages] = []
    content: str | None = None
    rewrite: bool = False


#################################
# Content Type Router
#################################

class ContentChoice(BaseModel):
    """The type of content to generate."""
    content_type: Literal["linkedin", "instagram", "blog"] = Field( description="The type of content to generate", default="linkedin")

# In Langgraph, we can create a structured output version of the LLM by specifying the output schema with a pydantic class.
# This will ensure the output to be an instance of the pydantic class and automatically handle validation errors.
llm_with_content_choice = llm.with_structured_output(schema=ContentChoice)


# This conditional edge is our router, responsible for classifying the customer's request into one of the content types.
# The returned value is equal to the content_type field in the ContentChoice class.
def generate_content_router(state: WorkflowState) -> str:
    """This is a conditional edge which allows us to route the graph based on some condition"""
    system_prompt = SystemMessage(content=f"""You are a copywriter. Your job is to classify the customer's request into one of the following categories. If the customer does not specify a platform, default to 'linkedin'.
    
    1. linkedin - if the customer wants to generate a linkedin post
    2. instagram - if the customer wants to generate an instagram post
    3. blog - if the customer wants to generate a blog post
    """)
    response = llm_with_content_choice.invoke([system_prompt] + state.messages)
    # Handle both pydantic object and dict responses
    if isinstance(response, dict):
        content_type = response["content_type"]
    else:
        content_type = response.content_type
    return content_type

# We will show a more complex case for the LinkedIn flow that includes a review and rewrite step. The other flows will be simplified but still show unique requirements.


#################################
# LinkedIn Flow
#################################

# The first step in the LinkedIn flow is to generate the content
# The last message in the state will either be the original user request or feedback on the previous draft. So in the system prompt, we're specifying how the LLM should handle each case.
def generate_linkedin(state: WorkflowState):
    system_prompt = SystemMessage(content=f"""
    You are a LinkedIn content creator specializing in AI topics. Your job is to create professional, engaging LinkedIn posts that meet the requirements below.

    <Requirements>
    - Are 150-300 words long
    - Include 3-5 relevant hashtags
    - Use a conversational, professional tone
    - Start with a hook or thought-provoking question
    - Include practical insights or actionable takeaways
    - End with a call-to-action encouraging engagement
    - Avoid overly technical jargon
    - Focus on business value and real-world applications
    - Do not include a title, just get right into the post
                                  
    Format: Write the post text, then add hashtags on separate lines at the end.
    
    When given an AI topic, create content that would resonate with business professionals, entrepreneurs, and tech enthusiasts on LinkedIn.
    </Requirements>
                                  
    You will either be given a topic to write about or asked to refine a draft based on feedback. If given feedback, refine the draft to incorporate the feedback and meet the requirements.
    """)
    response = llm.invoke([system_prompt] + state.messages)
    # At the end, we're updating the state with the generated content as well as adding the content as a new message, and finally resetting the rewrite flag
    return {"content": response.content, "messages": [response], "rewrite": False}

# We define a new pydantic class for the review process
class LinkedReview(BaseModel):
    """The results of reviewing the LinkedIn post."""
    is_valid: bool = Field(..., description="Whether the post meets the requirements")
    feedback: str | None = Field(None, description="Feedback on how to improve the post")

# and create another structured output LLM
llm_with_linkedin_format = llm.with_structured_output(schema=LinkedReview)

def review_linkedin(state: WorkflowState):
    system_prompt = SystemMessage(content=f"""
    You are a LinkedIn content creator and copywriter. Your job is to evaluate the quality of LinkedIn posts. The post should meet the following requirements:

    1. Includes 3-5 relevant hashtags
    2. Uses a conversational, professional tone
    3. Starts with a hook or thought-provoking question
    4. Includes practical insights or actionable takeaways
    5. Ends with a call-to-action encouraging engagement
    6. Avoids overly technical jargon
    7. Focuses on business value and real-world applications
    
    If the post meets the requirements, you should respond with is_valid: true. If it does not meet the requirements, respond with is_valid: false and include feedback on how to improve the post.
                                  
    <Post>
    {state.content}
    </Post>
    """)
    response = llm_with_linkedin_format.invoke([system_prompt] + state.messages)
    if isinstance(response, dict):
        is_valid = response["is_valid"]
        feedback = response.get("feedback")
    else:
        is_valid = response.is_valid
        feedback = response.feedback

    # If the post is valid, we're done. Otherwise, we flag it for rewrite and add the feedback as a new message
    if is_valid:
        return {"rewrite": False}
    else:
        return {"rewrite": True, "messages": feedback}

# Finally, we define a router to check if the post was flagged for a rewrite in the previous step, routing to the generate node if it does.
def linkedin_router(state: WorkflowState) -> str:
    if state.rewrite:
        return "generate_linkedin"
    return END


#################################
# Instagram Flow
#################################

def generate_instagram(state: WorkflowState):
    system_prompt = SystemMessage(content=f"""
    You are an Instagram content creator specializing in AI topics. Your job is to create engaging Instagram posts that meet the requirements below.        

    <Requirements>
    - Are 150-250 words long
    - Uses at least 1 emoji
    - Don't include a title, instead start with a strong hook or thought-provoking question
    - Always end with a question to encourage engagement
    """)
    response = llm.invoke([system_prompt] + state.messages)
    return {"content": response.content, "messages": [response]}


#################################
# Blog Flow
#################################

def generate_blog(state: WorkflowState):
    system_prompt = SystemMessage(content=f"""
    You are a blog writer specializing in AI topics. Your job is to create engaging blog posts that meet the requirements below.        

    <Requirements>
    - Has an introduction section
    - Has a conclusion section with a call to action to my youtube video. My youtube channel is https://www.youtube.com/@KennethLiao
    - Has a maximum of 3 sections in the body
    - Uses at least 3 emojis
    - Is SEO optimized
    """)
    response = llm.invoke([system_prompt] + state.messages)
    return {"content": response.content, "messages": [response]}


#################################
# Build the Graph
#################################

builder = StateGraph(WorkflowState)

builder.add_node(generate_linkedin)
builder.add_node(review_linkedin)
builder.add_node(generate_instagram)
builder.add_node(generate_blog)

builder.add_conditional_edges(
    START,
    generate_content_router,
    {
        "linkedin": "generate_linkedin",
        "instagram": "generate_instagram",
        "blog": "generate_blog",
    }
)

builder.add_edge("generate_linkedin", "review_linkedin")
builder.add_conditional_edges(
    "review_linkedin",
    linkedin_router,
    {
        "generate_linkedin": "generate_linkedin",
        END: END,
    }
)

graph = builder.compile()

# Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())


response = graph.invoke(WorkflowState(messages=["How MCPs just unlocked $100B in value for Google - for instagram"]))

for msg in response["messages"]:
    print(msg.content + "\n")
