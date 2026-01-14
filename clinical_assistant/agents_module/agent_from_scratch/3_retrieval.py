"""
Retrieval is the process of retrieving context to incorporate into an LLM or AI agent conversation.

There are many ways to implement retrieval. In this tutorial, we will cover the most common method which involves using a vector database. This example shows how we can give an agent access to a product database.

## Retrieval with Vector Databases
- Vector databases are a type of database that stores data as vectors.
- Vectors are a numerical representation of data. In the context of AI, vectors are typically used to represent text or images.
- Vector databases are used for similarity search. This means finding the most similar vectors to a given query vector.
- Chroma is a popular open-source vector database.
"""
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = OpenAI()


##########################################
# Create the Knowledgebase
##########################################

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="products")

collection.upsert(
    documents=[
        "SwiftStride Running Shorts: Engineered for peak performance, these lightweight running shorts feature a moisture-wicking fabric to keep you dry and comfortable. The built-in liner provides extra support, while a secure zippered back pocket is perfect for your keys or a small music device.",
        "AuraFlow Yoga Mat: Elevate your practice with the AuraFlow Yoga Mat. Its dual-sided non-slip surface offers superior grip and stability, allowing you to hold even the most challenging poses. Made from eco-friendly, high-density TPE material, it provides optimal cushioning for your joints.",
        "CoreFlex Training Hoodie: Stay warm without sacrificing mobility. The CoreFlex Training Hoodie is designed with a soft, breathable fleece that provides insulation while allowing for a full range of motion. Its athletic fit, thumbholes, and a three-panel hood offer comfort and a sleek look for your gym sessions or outdoor runs."
    ],
    ids=["1", "2", "3"]
)

results = collection.query(
    query_texts=["I just started running and I'm looking for some shorts."],
    n_results=1
)

print(results)


##########################################
# Retrieval
##########################################

def search_products(query: str, num_results: int = 3):
    """Search the product database and get back a list of products.

    Args:
        query: The search query.
        num_results: The number of results to return, max is 3.

    Returns:
        A dictionary of the search results.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(num_results, 3)
    )
    return results

tools = [
    {
        "type": "function",
        "name": "search_products",
        "description": "Search the product database and get back a list of products.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "The number of results to return, max is 3.",
                },
            },
            "required": ["query"],
        },
    },
]

messages = [
    {
        "role": "system",
        "content": "Your name is Aura. You are a sales agent. You have access to a tool called `search_products` that allows you to search a product database. Do not rely on your own knowledge, always use the `search_products` tool to answer the user's questions.",
    },
    {"role": "user", "content": "I just started running and I'm looking for some shorts."},
]

response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    tools=tools,
    input=messages,
)

# Since we've implemented product search as a retreival tool, we get back a tool call.
# We'd have to follow the tool calling loop which we covered in the previous step, `2_tool_calling.py`.

print(response.output)
