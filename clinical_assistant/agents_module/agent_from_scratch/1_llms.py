"""
LLMs are the brains or intelligence of AI agents and applications.

## Working with LLMs
- LLMs are stateless. Each time you invoke an LLM (whether through an API or locally), it has no memory of previous interactions.
- To build AI agents from scratch, we have to implement memory, tools, and the agent loop ourselves.

## Working with APIs
- You can call API endpoints directly using the requests or httpx libraries for example.
- However, it's convenient to use an SDK (software development kit) when available. SDKs handle authentication, error handling, and other boilerplate code for you.
- The examples in this tutorial use the OpenAI Python SDK but the same concepts apply to any LLM provider.
"""
from openai import OpenAI
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()

##########################################
# Basic API Call
##########################################

# Generate a response from the model
response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    input="Hello world.",
)

# Access just the model output text
print(response.output_text)

# Adding a system prompt
response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    input=[
        {"role": "system", "content": "Your name is Aura. Always respond like a pirate."},
        {"role": "user", "content": "Hello world."},
    ],
)

# Print the response text
print(response.output_text)

##########################################
# Structured Outputs
##########################################
# https://platform.openai.com/docs/guides/structured-outputs

class SupportTicket(BaseModel):
    """A support ticket."""
    subject: str = Field(..., description="The subject of the support ticket")
    body: str = Field(..., description="A description of the support ticket")

response = client.responses.parse(
    model="gpt-4.1-mini-2025-04-14",
    input="I can't login to my account.",
    text_format=SupportTicket,
    )

response.output_parsed


##########################################
# Streaming API Call
##########################################

# Generate a response and stream back the results
stream = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    input=[
        {"role": "system", "content": "Your name is Aura. Always respond like a pirate."},
        {"role": "user", "content": "Hello world."},
    ],
    stream=True,
)

# Print the raw events
for event in stream:
    print(event)
    print("\n-----\n")


stream = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    input=[
        {"role": "system", "content": "Your name is Aura. Always respond like a pirate."},
        {"role": "user", "content": "Hello world."},
    ],
    stream=True,
)

# Filter for just the text events
for event in stream:
    if isinstance(event, ResponseTextDeltaEvent):
        print(event.delta, end="", flush=True)
