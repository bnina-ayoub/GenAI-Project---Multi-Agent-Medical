"""
Parallelization is useful when you have multiple independent tasks that can be performed in parallel.

The example we'll build here is a LinkedIn content creation workflow with 3 main tasks: generate post, generate image, and generate hashtags. These will all be ran in parallel and then combined into a final output.

## KEY TAKEAWAYS
1. Similar to routing, having separate workflows that we can run in parallel allows us to separate concerns and optimize each workflow independently. But the key benefit of parallelization is that we can perform multiple tasks simultaneously, which can significantly reduce the overall latency of the workflow.
2. Another benefit of parallelization is breadth. 
    - By running multiple tasks in parallel, we can explore multiple angles and perspectives simultaneously, which can lead to more creative and insightful solutions.
    - Some examples where parallelization and breadth are beneficial are evaluations (run guardrails, evaluate tone, quality, etc.), brainstorming (explore multiple angles), or sectioning as in this example.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END, START
from openai import OpenAI
from PIL import Image as PILImage
from io import BytesIO
import requests

load_dotenv()

llm = ChatOpenAI(
    name="Jude",
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.1,
)

openai_client = OpenAI()


#################################
# State
#################################

class WorkflowState(BaseModel):
    messages: Annotated[list, add_messages] = []
    image_prompt: str | None = None
    image_url: str | None = None
    image_data: bytes | None = None
    image_filename: str | None = None
    post: str | None = None
    hashtags: str | None = None


#################################
# Generate Linkedin Post Flow
#################################

def generate_post(state: WorkflowState):
    system_prompt = SystemMessage(content=f"""
    You are a LinkedIn content creator specializing in AI topics. Your job is to create professional, engaging LinkedIn posts that meet the requirements below.

    <Requirements>
    - Are 150-300 words long
    - Use a conversational, professional tone
    - Start with a hook or thought-provoking question
    - Include practical insights or actionable takeaways
    - End with a call-to-action encouraging engagement
    - Avoid overly technical jargon
    - Focus on business value and real-world applications
    - Do not include a title, just get right into the post
    - Organize the post into shorter 1-2 sentence paragraphs spaced by newlines. Avoid long paragraphs
    </Requirements>
                                  
    When given an AI topic, create content that would resonate with business professionals, entrepreneurs, and tech enthusiasts on LinkedIn. Respond with the post text only.
    """)
    response = llm.invoke([system_prompt] + state.messages)
    return {"post": response.content}


#################################
# Generate Image Flow
#################################

def generate_image_prompt(state: WorkflowState):
    last_message = state.messages[-1]
    if isinstance(last_message, HumanMessage):
        context = last_message.content
    else:
        context = ""
    prompt = HumanMessage(content=f"""
    You are a DALLE-3 prompt optimization specialist for LinkedIn content. Your role is to transform any given topic into a compelling visual prompt that will generate professional, engaging images suitable for LinkedIn posts.

    CORE REQUIREMENTS:
    - Create prompts that produce clean, professional imagery appropriate for business social media
    - Focus on modern, minimalist aesthetics that perform well on LinkedIn
    - Avoid text-heavy images (DALLE-3 text rendering is unreliable)
    - Ensure images work well as small thumbnails in LinkedIn feeds

    PROMPT STRUCTURE:
    1. Visual style: Specify clean, modern, professional aesthetic
    2. Subject matter: Translate the topic into concrete visual elements
    3. Composition: Describe layout, perspective, and focal points
    4. Color palette: Suggest colors that align with professional branding
    5. Context: Add relevant background/setting details

    OUTPUT FORMAT:
    Return only the optimized DALLE-3 prompt text, maximum 400 characters for best results.

    EXAMPLES:
    Topic: "Remote work productivity" 
    → "Clean minimalist home office setup, modern laptop on wooden desk, natural lighting through window, organized workspace, soft neutral colors, professional photography style, shallow depth of field"

    Topic: "Data-driven decision making"
    → "Abstract visualization of data flowing into clear insights, geometric shapes and flowing lines, blue and white color scheme, modern infographic style, clean background"

    Focus on imagery that stops the scroll and communicates the topic's value proposition visually.
    
    <Post_Idea>
    {context}
    </Post_Idea>

    Respond with the prompt text only and ensure the prompt is highly relevant to the post idea.
    """)
    response = llm.invoke([prompt])
    return {"image_prompt": response.content}

def generate_image(state: WorkflowState):
    if state.image_prompt:
        result = openai_client.images.generate(
            prompt=state.image_prompt,
            model="dall-e-3",
            n=1,
            size="1024x1024",
        )

        if result.data:
            image_url = result.data[0].url

            if not image_url:
                raise ValueError("No image URL returned")
            
            image_response = requests.get(image_url)
            image_data = image_response.content

            return {"image_url": image_url, "image_data": image_data}
        else:
            raise ValueError("No image data returned")
        
def save_image(state: WorkflowState):
    if state.image_data:
        image = PILImage.open(BytesIO(state.image_data))
        filename = "generated_image.png"
        image.save(filename)
        return {"image_filename": filename}
        

#################################
# Generate Hashtags Flow
#################################

def generate_hashtags(state: WorkflowState):
    last_message = state.messages[-1]
    if isinstance(last_message, HumanMessage):
        context = last_message.content
    else:
        context = ""
    prompt = HumanMessage(content=f"""
    You are an expert content creator for LinkedIn. Your job is to generate a list of hashtags for the following post. The hashtags should meet the requirements below.
    
    <Requirements>
    The hashtags should:
    1. Be related to the post
    2. Be popular and relevant
    3. Be between 1 and 3 hashtags
    </Requirements>
                                  
    <Post>
    {context}
    </Post>
    """)
    response = llm.invoke([prompt])
    return {"hashtags": response.content}


#################################
# Combine Results Flow
#################################

# This is a perfect example where simpler is better and less AI leads to better results.
# We could have an LLM generate all of this HTML code including injecting the image, text, and hashtags. However, this is a very brittle approach that is likely to break and requires a lot of tuning.
def create_preview(state: WorkflowState):
    """Create post preview - just image, text, hashtags in an html file"""
    try:
        if state.hashtags:
            hashtags = [hashtag for hashtag in state.hashtags.split(" ") if hashtag.startswith("#")]
        else:
            hashtags = []
        
        # Ultra-minimal HTML - just the content
        html_content = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Post Preview</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    max-width: 600px;
                    margin: 20px auto;
                    padding: 20px;
                    line-height: 1.6;
                    background: white;
                }}
                .cover-image {{
                    width: 100%;
                    max-width: 100%;
                    height: auto;
                    margin-bottom: 20px;
                    border-radius: 8px;
                }}
                .post-text {{
                    font-size: 16px;
                    color: #333;
                    margin-bottom: 20px;
                    white-space: pre-wrap;
                }}
                .hashtags {{
                    font-size: 15px;
                    color: #0073b1;
                    line-height: 1.4;
                }}
                .hashtag {{
                    margin-right: 8px;
                }}
            </style>
        </head>
        <body>
            <img src="{state.image_filename}" alt="Cover image" class="cover-image" 
                onerror="this.src='{state.image_url}';">
            
            <div class="post-text">{state.post}</div>
            
            <div class="hashtags">
                {"".join(f'<span class="hashtag">{hashtag}</span>' for hashtag in hashtags)}
            </div>
        </body>
        </html>
        """
        
        # Save the minimal preview
        preview_filename = "post_preview.html"
        
        with open(preview_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "messages": [AIMessage(content=f"Minimal preview created: {preview_filename}")],
        }
        
    except Exception as e:
        print(f"❌ Minimal preview creation failed: {str(e)}")
        return state


builder = StateGraph(WorkflowState)

builder.add_node(generate_post)
builder.add_node(generate_image_prompt)
builder.add_node(generate_image)
builder.add_node(save_image)
builder.add_node(generate_hashtags)
builder.add_node(create_preview)

builder.add_edge(START, "generate_post")
builder.add_edge(START, "generate_image_prompt")
builder.add_edge(START, "generate_hashtags")

builder.add_edge("generate_image_prompt", "generate_image")
builder.add_edge("generate_image", "save_image")
builder.add_edge("save_image", "create_preview")
builder.add_edge("generate_post", "create_preview")
builder.add_edge("generate_hashtags", "create_preview")

builder.add_edge("create_preview", END)

graph = builder.compile()

# Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())

response = graph.invoke(WorkflowState(messages=["Most AI integrations fail because of one reason: overcomplication. In reality, AI workflows can solve most problems, are much easier to optimize and have fine-grained control, and have more reliability"]))

response["image_prompt"]