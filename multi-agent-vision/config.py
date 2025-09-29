import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration class to hold environment variables
class Config:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    PG_CONN_STR = os.getenv("PG_CONN_STR")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-system")

    # Validate required environment variables
    @classmethod
    def validate(cls):
        required_vars = ["DASHSCOPE_API_KEY", "TAVILY_API_KEY", "PG_CONN_STR", "LANGSMITH_API_KEY"]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

# Initialize configuration
Config.validate()