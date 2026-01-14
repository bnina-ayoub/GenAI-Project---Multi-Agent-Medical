"""
This file contains local tools
"""
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


##########################################
# Tools
##########################################

def search_web(query: str):
    """Search the web and get back a list of search results including the page title, url, and the cleaned content of each webpage.

    Args:
        query: The search query.

    Returns:
        A dictionary of the search results.
    """
    tavily_client = TavilyClient()
    response = tavily_client.search(query, max_results=3)

    return response
