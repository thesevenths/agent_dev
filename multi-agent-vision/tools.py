from langchain_community.tools.tavily_search import TavilySearchResults
from config import Config
from langchain_core.tools import tool

# Initialize Tavily search tool
tavily_search = TavilySearchResults(api_key=Config.TAVILY_API_KEY, max_results=5)

@tool
def database_query(query: str) -> str:
    """
    Placeholder for database query tool.
    In a real implementation, this would query a PostgreSQL database using PG_CONN_STR.
    """
    return f"Database query result for: {query}"