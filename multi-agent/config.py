
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Load from environment variables to avoid committing to GitHub
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")   
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PG_CONN_STR = os.getenv("PG_CONN_STR")   
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")   