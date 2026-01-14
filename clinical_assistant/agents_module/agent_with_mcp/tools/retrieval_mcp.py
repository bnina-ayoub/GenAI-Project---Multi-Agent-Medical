"""
An MCP server for retrieval and analysis.

This server provides the following functionality:
1. Search a product database.
2. Search a FAQ database.
3. Get a user's profile.
4. Analyze a customer and provide insights.
"""
from dotenv import load_dotenv
import json
import chromadb
from typing import Literal, Dict, List, Any
import os
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

load_dotenv()

# Create our FastMCP server and give it a name
mcp = FastMCP(name="retrieval")

# Add a custom route to check the health of the server
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


##########################################
# Retrieval from Knowledgebase
##########################################

# Create the Knowledgebase
# ----------------------------------------

chroma_client = chromadb.Client()

knowledgebase_path = "../../agent_from_scratch/knowledgebase"

for collection in os.listdir(knowledgebase_path):
    collection_name = collection.split(".")[0]

    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)

        collection_data = json.load(open(f"{knowledgebase_path}/{collection_name}.json"))

        for item in collection_data:
            collection.upsert(
                documents=[json.dumps(item)],
                ids=[str(item["id"])],
                metadatas=[item["metadata"]],
            )
    except Exception as e:
        print(f"Error creating {collection_name}: {e}")


# Define the MCP tools
# ----------------------------------------

@mcp.tool()
def search_products(
        query: str, 
        gender: Literal["men", "women"] | None = None, 
        category: Literal["running", "gym", "yoga"] | None = None, 
        num_results: int = 3) -> List[Dict[str, Any]]:
    """Search the product database and get back a list of products.

    Args:
        query: The search query.
        gender: The gender of the product. Can be one of "men" or "women".
        category: The category of the product. Can be one of "running", "gym", or "yoga".
        num_results: The number of results to return, max is 3.

    Returns:
        A dictionary of the search results.
    """
    where = {}
    if gender and category:
        where["$and"] = [
            {"gender": gender},
            {"category": category},
        ]
    elif category:
        where["category"] = category
    elif gender:
        where["gender"] = gender

    collection = chroma_client.get_collection(name="products")

    results = collection.query(
        query_texts=[query],
        n_results=min(num_results, 3),
        where=where
    )
    if not results["ids"][0]:
        return "No matching products found."
    
    return [json.loads(doc) for doc in results["documents"][0]]

@mcp.tool()
def search_faq(
        query: str, 
        category: Literal["returns", "shipping", "discounts", "products"] | None = None, 
        num_results: int = 3) -> List[Dict[str, Any]]:
    """Search the FAQ database and get back a list of answers.

    Args:
        query: The search query.
        category: The category of the question. Can be one of "returns", "shipping", "discounts", or "products".
        num_results: The number of results to return, max is 3.

    Returns:
        A dictionary of the search results.
    """
    where = {}
    if category:
        where["category"] = category

    collection = chroma_client.get_collection(name="faq")

    results = collection.query(
        query_texts=[query],
        n_results=min(num_results, 3),
        where=where
    )
    if not results["ids"][0]:
        return "No matching answers found."
    
    return [json.loads(doc) for doc in results["documents"][0]]


# Create an MCP resource
# ----------------------------------------

# Resources expose read-only data sources (like a GET request)
# Examples: server logs, documents, api version, etc.
@mcp.resource("status://last_updated")
def get_last_updated():
    """Get the last updated date of the knowledgebase."""
    return "Last Updated: 2025-09-02"


##########################################
# User Profiling and Analysis
##########################################

# In this example we query a user's data from our CRM database (JSON file)
# We create a user profile by joining the user data with their past purchases and defining our own customer metrics

def _get_user_profile_data(user_id: int) -> Dict[str, Any]:
    """Internal function to get user profile data."""
    try:
        with open("../crm_db/crm.json", "r") as f:
            crm_data = json.load(f)

        users = crm_data["users"]
        transactions = crm_data["transactions"]

        user = next((user for user in users if user["id"] == user_id), None)
        if user is None:
            raise ValueError(f"User with id {user_id} does not exist.")

        past_purchases = [purchase for purchase in transactions if purchase["user_id"] == user_id][:5]

        return {
            "id": user_id,
            "name": user["name"],
            "age": user["age"],
            "gender": user["gender"],
            "location": user["location"],
            "total_purchases": len(past_purchases),
            "total_amount_spent": sum([purchase["price"] for purchase in past_purchases]),
            "average_purchase_amount": sum([purchase["price"] for purchase in past_purchases]) / len(past_purchases),
            "past_purchases": [
                {"id": purchase["id"], "name": purchase["name"], "price": purchase["price"], "category": purchase["category"]} for purchase in past_purchases
            ]
        }
    except Exception:
        raise ValueError(f"User with id {user_id} does not exist.")


@mcp.resource("user://profile/{user_id}")
def get_user_profile(user_id: int) -> Dict[str, Any]:
    """Get the user profile."""
    return _get_user_profile_data(user_id)


@mcp.prompt()
def analyze_customer(user_id: int) -> str:
    """Analyze the customer and provide insights."""
    try:
        profile = _get_user_profile_data(user_id)
    except ValueError:
        raise ValueError(f"User with id {user_id} does not exist.")

    return f"""
    You are a sales agent for an athletic apparel company called FitFlex. You are analyzing a customer's profile to provide insights to the sales team.

    Here is the customer's profile:
    {json.dumps(profile)}

    Your goal is to provide insights about the customer that can help you provide a highly personalized experience for the customer.

    Insights should include but are not limited to:
    1. Most common purchased categories.
    2. Most common purchased products.
    3. Most likely preferred colors.
    4. Most likely purchase amount range (low, high).
    5. Categories of products the customer might be interested in but has not purchased yet.
    6. Any other insights you can provide.

    Provide insights about the customer.
    """


if __name__ == "__main__":
    # HTTP is the same as streamable-http
    mcp.run(transport="http", host="127.0.0.1", port=8001)
