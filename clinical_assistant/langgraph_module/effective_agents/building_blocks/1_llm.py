"""
LLMs are the brains or intelligence of AI agents and applications.

## KEY TAKEAWAYS
1. LLMs are stateless. Each API call is independent. Each time you invoke the LLM or a Langgraph graph, it has no memory of previous interactions.
2. To go from LLMs to agents, we require 3 additional components: memory or state, tools or the ability to act, and agency or the ability to make decisions.
"""
from dotenv import load_dotenv

# For Langgraph applications we can use the OpenAI integration from Langchain (ChatOpenAI)
from langchain_openai import ChatOpenAI
# This gives several benefits including automatic tracing and observability if using LangSmith
# Langchain has similar integrations for almost all LLM providers

load_dotenv()

# You can think of the LLM as the brains or intelligence of the agent
# However this basic LLM is missing several critical components to make it an agent
# 1. Memory or state
# 2. Tools or the ability to act
# 3. Agency or the ability to make decisions
llm = ChatOpenAI(
    name="Jude",
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.1,
)


#################################
# LLMs are Stateless
#################################

# We first introduce ourselves to Jude
response = llm.invoke("Hello Jude! I'm Kenny.")

print(response.content)

# Jude doesn't have any memory so is unaware of our previous message
response = llm.invoke("What's my name?")

print(response.content)
    
# Each invocation is an independen API call and has no memory of previous interactions