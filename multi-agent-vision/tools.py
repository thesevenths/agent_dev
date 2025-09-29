import os
import json
import subprocess
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from config import TAVILY_API_KEY, PG_CONN_STR
import psycopg2


@tool
def create_file(file_name: str, file_contents: str):
    """Create a new file with the provided contents."""
    if not file_name or not file_contents:
        return {"error": "file_name and file_contents must be non-empty strings"}
    file_path = os.path.join(os.getcwd(), file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(file_contents)
    return {"message": f"Successfully created file at {file_path}"}


@tool
def str_replace(file_name: str, old_str: str, new_str: str):
    """Replace specific text in a file."""
    if not all([file_name, old_str, new_str]):
        return {"error": "All parameters must be non-empty strings"}
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, "r", encoding='utf-8') as file:
        content = file.read()
    new_content = content.replace(old_str, new_str, 1)
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(new_content)
    return {"message": f"Successfully replaced '{old_str}' with '{new_str}' in {file_path}"}


@tool
def shell_exec(command: str):
    """Execute a shell command."""
    if not command:
        return {"error": "Command must be non-empty"}
    result = subprocess.run(command, shell=True, cwd=os.getcwd(), capture_output=True, text=True, encoding='utf-8')
    return {"stdout": result.stdout, "stderr": result.stderr}


@tool
def crawl_web3_news(query: str, output_file: str):
    """Crawl Web3 cryptocurrency news using Tavily search and save as JSON."""
    if not query or not output_file:
        return {"error": "query and output_file must be non-empty"}
    print(f"Querying Tavily with: {query}")  # Debug output
    tavily = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=10)
    results = tavily.invoke({"query": f"{query} site:cointelegraph.com OR site:cryptoslate.com OR site:techflowpost.com OR site:podcasts.apple.com OR site:www.coinglass.com/zh/news OR weather.com"})
    news_data = []
    for res in results:
        news_data.append({"date": res.get("publish_date", "extracted_date"), "news": res.get("content", "")})
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)
    return {"message": f"Saved to {output_file}"}


@tool
def query_pg_database(sql: str):
    """Query PostgreSQL database."""
    if not PG_CONN_STR:
        return {"error": "PG_CONN_STR not set"}
    if not sql:
        return {"error": "SQL must be non-empty"}
    try:
        conn = psycopg2.connect(PG_CONN_STR)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"results": rows}
    except Exception as e:
        return {"error": str(e)}


@tool
def dummy_tool() -> dict:
    """Placeholder tool to satisfy API requirements."""
    return {"status": "no-op"}
