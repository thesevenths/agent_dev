"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
import logging
import os
from pathlib import Path
from random import random
import smtplib
import ssl
import time
import json
import subprocess

from config import CRYPTO_SENTIMENT_KEY, QQ_APP_PASSWORD, QQ_EMAIL

import requests
from typing_extensions import Annotated
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel
from langchain_core.tools import tool
from sqlalchemy import inspect
from sqlalchemy import text
from config import PG_CONN_STR, TAVILY_API_KEY, TOP_N
from typing import List, Dict, Optional
from functools import wraps
from collections import deque

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# 运行日志：把关键事件（含本工具的联网检索 query）持久化到 multi-agent/log/ 的时间戳文件。
# runlog 仅依赖标准库，无循环依赖（agent.py / tools.py 都可安全导入）。
from runlog import log_event

# 调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 时间工具：显式提供给各 agent，让 LLM 在检测到时间引用时主动调用 ===
# 模型无实时时钟，必须靠此 tool 或 agent.py 中的 _date_context_str 注入获取"今天"。
# 与 agent.py 的 _WEEKDAY_CN 各自定义一份（输出格式不同），避免跨模块耦合。
_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

@tool
def get_current_time() -> str:
    """Return the current local date and time.

    MUST call this FIRST whenever the user request mentions any relative or
    absolute time that needs anchoring to "now", e.g. 昨天/今天/上周/本月/近期/
    最近/去年/today/yesterday/last week/recently. Use the returned EXACT date
    to resolve the expression — never guess the year/month/day from training
    memory.

    Returns a string like:
      "Current local date and time: 2026-09-29 15:02:17 (周二)."
    """
    now = datetime.now()
    return f"Current local date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({_WEEKDAY_CN[now.weekday()]})."


from langchain_tavily import TavilySearch
tavily_search = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5  # 返回前 5 个结果
)

# 创建基类
Base = declarative_base()
# connect_timeout：PG 不可达时快速失败（默认会走 OS 级 TCP 超时，可达 100s+）。
# 本模块是 import 链 tools -> llm -> supervisor -> graph -> agent 的必经节点，
# 顶层一旦阻塞，langgraph dev 的 graph 加载会整体失败（GraphLoadError: startup failed），
# 表现为"服务只打印心跳、2024 端口不监听、浏览器打不开"。故超时必须收敛。
engine = create_engine(
    PG_CONN_STR,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)
try:
    Base.metadata.create_all(engine)
except Exception as _db_err:
    # DB 不可用（PG 挂了/网段不通）不应阻断整个服务启动：只告警，依赖 DB 的工具调用时再暴露。
    logger.warning("[db] create_all 跳过（DB 不可达？）：%s", _db_err)

Session = sessionmaker(bind=engine)
session = Session()

repl = PythonREPL()


# 防呆：缓存最近若干次查询（归一化），用于检测 agent 在工具循环里反复用相同 query 调本工具。
# 命中则直接回吐上次结果并提示停止，避免无限循环 + 重复抓取（create_agent 无 max_iterations 参数）。
_tavily_recent = deque(maxlen=4)


def _norm_query(q: str) -> str:
    return " ".join(str(q).lower().split())


@tool
def resilient_tavily_search(query: str) -> str:
    """
        web search using TavilySearch tool with rate limit handling.
        handle API rate limits by retrying with exponential backoff.

    Returns: search results as a string.
    """
    norm = _norm_query(query)
    logger.info(f"[tavily] query = {query!r}")  # 便于在 langgraph dev 终端观察模型生成的 query
    log_event(f"[tavily] query = {query!r}")     # 持久化到本次运行的 log/ 时间戳文件
    # 同 query 重复调用防呆：agent 在 ReAct 循环里常把相同 query 反复丢给本工具
    for prev_norm, prev_res in reversed(_tavily_recent):
        if prev_norm == norm:
            return (
                prev_res
                + "\n\n[system] 该查询与上一次完全相同，已复用结果。"
                + "请勿再次调用本工具，直接基于以上结果产出最终答案/JSON。"
            )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = tavily_search.run(query)
            _tavily_recent.append((norm, res))
            return res
        except Exception as e:
            if "429" in str(e):
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
                continue
            raise
    raise Exception("Tavily rate limit exceeded")

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str


@tool
def shell_exec(command: str) -> dict:
    """
    Execute a command in the specified shell session.

    Args:
        command (str): The shell command to execute (required, non-empty)

    Returns:
        dict: Contains 'stdout' and 'stderr'
    """
    try:
        if not command:
            return {"error": {"stderr": "Command must be non-empty"}}
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='ignore'
        )
        return {"message": {"stdout": result.stdout, "stderr": result.stderr}}
    except Exception as e:
        return {"error": {"stderr": str(e)}}


# === 统一产物目录 ===
# 所有写入类工具（create_file / str_replace）的相对路径统一落到 <project>/tmp，避免污染仓库根目录；
# 绝对路径原样尊重。读取类工具按「原样 → tmp → 仓库根」依次尝试，保证读写 round-trip 一致，
# 同时不破坏历史产出（散落在根目录的旧文件仍可读）。
def _output_dir() -> Path:
    d = os.environ.get("AGENT_TMP_DIR") or os.path.join(Path(__file__).resolve().parent, "tmp")
    p = Path(d)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _resolve_write_path(file_name: str) -> Path:
    p = Path(file_name)
    return p if p.is_absolute() else _output_dir() / p


def _resolve_read_path(file_name: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    for candidate in (p, _output_dir() / p, Path(os.getcwd()) / p):
        if candidate.exists():
            return candidate
    return p


@tool
def create_file(file_name: str, file_contents: str):
    """
    Create a new file with the provided contents at a given path in the workspace.
    Relative paths are stored under the project's tmp/ directory; absolute paths are used as given.
    
    Args:
        file_name (str): Name of the file to be created (required, non-empty)
        file_contents (str): The content to write to the file (required)
    """
    try:
        if not file_name or not file_contents:
            return {"error": "file_name and file_contents must be non-empty strings"}
        file_path = _resolve_write_path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_contents)
        return {"message": f"Successfully created file at {file_path}"}
    except Exception as e:
        return {"error": str(e)}



@tool
def list_files_metadata(files_path: str = ".\\documents") -> dict:
    """
    List all files in the specified directory and its subdirectories, 
    and extract file name, author, and main content description (if available).

    Returns:
        dict: A list of dicts with 'file_name', 'author', 'description'.
    """
    file_metadata = []
    for root, dirs, files in os.walk(os.path.abspath(files_path)):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), os.getcwd())
            author = ""
            description = ""
            try:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [next(f) for _ in range(10)]  # 读取前10行
                    for line in lines:
                        if "Author:" in line:
                            author = line.strip().replace("#", "").replace("Author:", "").strip()
                        if line.strip().startswith('"""') or line.strip().startswith("'''"):
                            description = line.strip().strip('"""').strip("'''").strip()
                            break
                        if line.strip().startswith("#"):
                            description = line.strip().replace("#", "").strip()
            except Exception:
                pass
            file_metadata.append({
                "file_name": relative_path,
                "author": author,
                "description": description
            })
    return {"files": file_metadata}


@tool
def read_file(file_name: str) -> dict:
    """
    Read the contents of a specified file.
    
    Args:
        file_name (str): Name of the file to read (required, non-empty)
    
    Returns:
        dict: Contains 'file_name' and 'content' or 'error'
    """
    try:
        if not file_name:
            return {"error": "file_name must be a non-empty string"}
        file_path = _resolve_read_path(file_name)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        return {"file_name": str(file_path), "content": content}
    except Exception as e:
        return {"error": f"Error reading {file_name}: {str(e)}"}


@tool
def grep_files(
    pattern: str,
    files_path: str = ".\\tmp",
    glob: str = "*",
    ignore_case: bool = True,
    max_results: int = 50,
) -> dict:
    """Search file CONTENTS under a directory for a keyword / regex — like grep / ripgrep.

    Use this to LOCATE which file(s) contain a given ID / absolute path / keyword / error string
    WITHOUT reading every file whole. Complements read_file (returns one whole file) and
    list_files_metadata (lists files + descriptions). Typical use: upstream steps wrote several
    artifacts to tmp/ and a downstream agent must find the one mentioning "AAPL" / a specific
    order id / a timestamp before reading it.

    Args:
        pattern (str): Python regular expression, matched line by line (required, non-empty).
        files_path (str): directory to search. Relative paths resolve under the project tmp/ dir
            then CWD (same rules as read_file). Default ".\\tmp".
        glob (str): filename glob filter, e.g. "*.md", "*.json", "*step3*" (default "*" = all).
        ignore_case (bool): case-insensitive matching (default True).
        max_results (int): cap on returned matches, to avoid flooding the context (default 50).

    Returns:
        dict: {"pattern", "searched_dir", "files_scanned", "total", "truncated",
               "matches": [{"file", "line", "text"}]}  or  {"error": ...}
    """
    import re
    import fnmatch
    try:
        if not pattern:
            return {"error": "pattern must be a non-empty string"}
        flags = re.IGNORECASE if ignore_case else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"invalid regex '{pattern}': {e}"}
        base = _resolve_read_path(files_path)
        if not base.exists():
            return {"error": f"search path does not exist: {base}"}
        if base.is_file():
            base = base.parent
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 50
        max_results = max(1, min(max_results, 500))
        # 跳过超大文件（默认 5MB），避免拖慢检索 / 撑爆 context。
        max_bytes = int(os.environ.get("AGENT_GREP_MAX_FILE_BYTES", str(5 * 1024 * 1024)))

        matches = []
        files_scanned = 0
        total = 0
        truncated = False
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if not fnmatch.fnmatch(fn, glob):
                    continue
                fp = os.path.join(root, fn)
                try:
                    if os.path.getsize(fp) > max_bytes:
                        continue
                    files_scanned += 1
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if rx.search(line):
                                total += 1
                                if len(matches) < max_results:
                                    snippet = line.strip()
                                    if len(snippet) > 300:
                                        snippet = snippet[:300] + "…"
                                    matches.append({
                                        "file": os.path.relpath(fp, os.getcwd()),
                                        "line": lineno,
                                        "text": snippet,
                                    })
                                else:
                                    truncated = True
                except Exception:
                    continue
        return {
            "pattern": pattern,
            "searched_dir": str(base),
            "files_scanned": files_scanned,
            "total": total,
            "truncated": truncated,
            "matches": matches,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def str_replace(file_name: str, old_str: str, new_str: str):
    """
    Replace specific text in a file.
    
    Args:
        file_name (str): Name of the target file (required, non-empty)
        old_str (str): Text to be replaced (must appear exactly once, required, non-empty)
        new_str (str): Replacement text (required, non-empty)
    """
    try:
        if not all([file_name, old_str, new_str]):
            return {"error": "All parameters must be non-empty strings"}
        file_path = _resolve_read_path(file_name)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        new_content = content.replace(old_str, new_str, 1)
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(new_content)
        return {"message": f"Successfully replaced '{old_str}' with '{new_str}' in {file_path}"}
    except Exception as e:
        return {"error": f"Error replacing '{old_str}' with '{new_str}' in {file_path}: {str(e)}"}

@tool
def send_message(message: str):
    """
    Send a message to the user.
    
    Args:
        message (str): The message to send to the user (required, non-empty)
    """
    if not message:
        return {"error": "Message must be non-empty"}
    return {"message": message}

@tool
def shell_exec(command: str) -> dict:
    """
    Execute a command in the specified shell session.

    Args:
        command (str): The shell command to execute (required, non-empty)

    Returns:
        dict: Contains 'stdout' and 'stderr'
    """
    try:
        if not command:
            return {"error": {"stderr": "Command must be non-empty"}}
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='ignore'
        )
        return {"message": {"stdout": result.stdout, "stderr": result.stderr}}
    except Exception as e:
        return {"error": {"stderr": str(e)}}


@tool
def get_nasdaq_top_gainers(top_n: int = 5) -> List[Dict]:
    """
    fetch NASDAQ top gainers from Yahoo Finance, sorted by percentage gain descending.
    
    Args:
        top_n (int): default 5 No. of top gainers to return.
    
    Returns:
        List[Dict]: top n stock list，each include symbol, name, change_pct
    """
    url = "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
    
    # fetch more to ensure enough NASDAQ stocks
    top_n = TOP_N
    fetch_count = max(top_n * 3, 100)  
    
    params = {
        "scrIds": "day_gainers",
        "count": fetch_count
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/screener/predefined/day_gainers"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        if not quotes:
            return []
        
        nasdaq_stocks = []
        for q in quotes:
            if q.get('exchange') == 'NMS' and q.get('regularMarketChangePercent') is not None:
                nasdaq_stocks.append({
                    'symbol': q.get('symbol', 'N/A'),
                    'name': q.get('shortName', 'N/A'),
                    'change_pct': float(q['regularMarketChangePercent'])
                })
        
        # sort by percentage gain descending
        nasdaq_stocks.sort(key=lambda x: x['change_pct'], reverse=True)
        
        # return top nq
        return nasdaq_stocks[:top_n]
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to fetch data: {e}")
        return []



# 定义模型
class SalesData(Base):
    __tablename__ = 'sales_data'
    sales_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('product_information.product_id'))
    employee_id = Column(Integer)  # 示例简化，未创建员工表
    customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))
    sale_date = Column(String(50))
    quantity = Column(Integer)
    amount = Column(Float)
    discount = Column(Float)


class CustomerInformation(Base):
    __tablename__ = 'customer_information'
    customer_id = Column(Integer, primary_key=True)
    customer_name = Column(String(50))
    contact_info = Column(String(50))
    region = Column(String(50))
    customer_type = Column(String(50))


class ProductInformation(Base):
    __tablename__ = 'product_information'
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(50))
    category = Column(String(50))
    unit_price = Column(Float)
    stock_level = Column(Integer)


class CompetitorAnalysis(Base):
    __tablename__ = 'competitor_analysis'
    competitor_id = Column(Integer, primary_key=True)
    competitor_name = Column(String(50))
    region = Column(String(50))
    market_share = Column(Float)


class AddSaleSchema(BaseModel):
    product_id: int
    employee_id: int
    customer_id: int
    sale_date: str
    quantity: int
    amount: float
    discount: float


class DeleteSaleSchema(BaseModel):
    sales_id: int


class UpdateSaleSchema(BaseModel):
    sales_id: int
    quantity: int
    amount: float


class QuerySalesSchema(BaseModel):
    sales_id: int


from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base

# 创建基类
Base = declarative_base()


# 定义模型
class SalesData(Base):
    __tablename__ = 'sales_data'
    sales_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('product_information.product_id'))
    employee_id = Column(Integer)  # 示例简化，未创建员工表
    customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))
    sale_date = Column(String(50))
    quantity = Column(Integer)
    amount = Column(Float)
    discount = Column(Float)


# 1. 添加销售数据：
@tool(args_schema=AddSaleSchema)
def add_sale(product_id, employee_id, customer_id, sale_date, quantity, amount, discount):
    """Add sale record to the database."""
    session = Session()
    try:
        new_sale = SalesData(
            product_id=product_id,
            employee_id=employee_id,
            customer_id=customer_id,
            sale_date=sale_date,
            quantity=quantity,
            amount=amount,
            discount=discount
        )
        session.add(new_sale)
        session.commit()
        return {"messages": ["销售记录添加成功。"]}
    except Exception as e:
        return {"messages": [f"添加失败，错误原因：{e}"]}
    finally:
        session.close()


# 2. 删除销售数据
@tool(args_schema=DeleteSaleSchema)
def delete_sale(sales_id):
    """Delete sale record from the database."""
    session = Session()
    try:
        sale_to_delete = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_delete:
            session.delete(sale_to_delete)
            session.commit()
            return {"messages": ["销售记录删除成功。"]}
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        return {"messages": [f"删除失败，错误原因：{e}"]}
    finally:
        session.close()


# 3. 修改销售数据
@tool(args_schema=UpdateSaleSchema)
def update_sale(sales_id, quantity, amount):
    """Update sale record in the database."""
    session = Session()
    try:
        sale_to_update = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_update:
            sale_to_update.quantity = quantity
            sale_to_update.amount = amount
            session.commit()
            return {"messages": ["销售记录更新成功。"]}
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        return {"messages": [f"更新失败，错误原因：{e}"]}
    finally:
        session.close()


# 4. 查询销售数据
@tool(args_schema=QuerySalesSchema)
def query_sales(sales_id):
    """Query sales record from the database."""
    session = Session()
    try:
        sale_data = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_data:
            return {
                "sales_id": sale_data.sales_id,
                "product_id": sale_data.product_id,
                "employee_id": sale_data.employee_id,
                "customer_id": sale_data.customer_id,
                "sale_date": sale_data.sale_date,
                "quantity": sale_data.quantity,
                "amount": sale_data.amount,
                "discount": sale_data.discount
            }
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}。"]}
    except Exception as e:
        return {"messages": [f"查询失败，错误原因：{e}"]}
    finally:
        session.close()


@tool
def execute_sql(
        sql_query: Annotated[str, "The SQL query to execute against the database. Use SELECT for reads, INSERT/UPDATE/DELETE for writes."],
        is_read_only: Annotated[bool, "Set to True for read-only queries (SELECT). False allows writes. Default: True."] = True
):
    """Execute an arbitrary SQL query on the database and return the results.

    - For SELECT queries, returns a list of dictionaries (rows).
    - For write queries (INSERT/UPDATE/DELETE), returns the number of affected rows or success message.
    - Always include schema names if needed (e.g., public.sales_data).
    - Do not execute DDL (e.g., CREATE/DROP TABLE) unless explicitly allowed.
    """
    session = Session()
    try:
        # Use text() for raw SQL to prevent injection (though LLM-generated, still safer)
        stmt = text(sql_query)

        if is_read_only:
            # For reads: Fetch all rows as dicts
            result = session.execute(stmt)
            rows = result.fetchall()
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
            session.commit()  # Not needed for reads, but harmless
            return {"results": data, "messages": ["Query executed successfully."]}
        else:
            # For writes: Execute and get affected rows
            result = session.execute(stmt)
            session.commit()
            return {"affected_rows": result.rowcount, "messages": ["Write operation successful."]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"Query failed: {str(e)}"]}
    finally:
        session.close()


@tool
def query_table_schema():
    """
    Query all table names and their column details (name and type) in the database.
    Returns a dictionary mapping table names to list of column info.
    """
    session = Session()
    try:
        inspector = inspect(session.bind)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = [
                {
                    "name": col["name"],
                    "type": str(col["type"])  # Convert SQLAlchemy type to string for readability
                }
                for col in columns
            ]
        return schema
    except Exception as e:
        return {"error": f"Failed to retrieve schema: {str(e)}"}
    finally:
        session.close()


@tool
def save_context_snapshot(name: str, content: str):
    """
    Save a context snapshot under ./contexts/<timestamp>__<name>.json
    """
    try:
        ctx_dir = os.path.join(os.getcwd(), "contexts")
        os.makedirs(ctx_dir, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        fname = f"{ts}__{name}.json"
        path = os.path.join(ctx_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": name, "timestamp": ts, "content": content}, f, ensure_ascii=False, indent=2)
        return {"message": f"Snapshot saved: {path}", "path": path}
    except Exception as e:
        return {"error": str(e)}

@tool
def list_context_snapshots():
    """
    List saved context snapshots in ./contexts
    """
    try:
        dir_path = os.path.join(os.getcwd(), "contexts")
        if not os.path.exists(dir_path):
            return {"snapshots": []}
        files = sorted(os.listdir(dir_path), reverse=True)
        return {"snapshots": files}
    except Exception as e:
        return {"error": str(e)}

@tool
def evaluate_output(criteria: str, output: str) -> dict:
    """
    evaluator for context_engineer verification.

    Supported evaluation dimensions include:
    - Correctness: Whether the output contains the expected key information or values.
    - Completeness: Whether all required points are covered.
    - Format/Protocol: Whether the output adheres to the specified format (e.g., JSON schema, strict keys).
    - No Hallucination: Whether the response uses only verifiable information or explicitly marks missing info as NOT FOUND.
    - Tool Usage: Whether specified tools were correctly invoked or expected tool result identifiers were returned.
    - Reproducibility: Whether the same input consistently produces compliant output.
    - Non-regression: Whether behavior on a set of examples has not degraded after a change.
    - Conciseness/Length: Whether the output is within acceptable length limits (not too long or timeout-prone).
    - Persistence/Commit: Whether the change has been persisted (e.g., snapshot file exists).

    Args:
        criteria (str): The evaluation criterion or dimension (e.g., "Correctness", "Completeness").
        output (str): The model output to be evaluated.

    Returns:
      dict: {"passed": bool, "reason": str, "details": {...}}
    """
    import re
    try:
        crit = (criteria or "").strip()
        out = (output or "")
        if crit == "":
            return {"passed": False, "reason": "No criteria provided.", "details": {}}

        parts = [p.strip() for p in crit.split(";") if p.strip()]
        details = {"checks": []}
        for p in parts:
            if p.lower() == "not empty":
                ok = bool(out.strip())
                details["checks"].append({"check": "not empty", "passed": ok})
                if not ok:
                    return {"passed": False, "reason": "Output is empty but expected non-empty.", "details": details}
            elif p.lower().startswith("contain:"):
                kw = p.split(":", 1)[1].strip().lower()
                ok = kw in out.lower()
                details["checks"].append({"check": f"contain:{kw}", "passed": ok})
                if not ok:
                    return {"passed": False, "reason": f"Missing expected keyword: '{kw}'.", "details": details}
            elif p.lower().startswith("not contain:"):
                kw = p.split(":", 1)[1].strip().lower()
                ok = kw not in out.lower()
                details["checks"].append({"check": f"not contain:{kw}", "passed": ok})
                if not ok:
                    return {"passed": False, "reason": f"Output contains forbidden keyword: '{kw}'.", "details": details}
            elif p.lower().startswith("contains_all:"):
                vals = [v.strip().lower() for v in p.split(":",1)[1].split(",") if v.strip()]
                missing = [v for v in vals if v not in out.lower()]
                ok = len(missing) == 0
                details["checks"].append({"check": f"contains_all:{vals}", "passed": ok, "missing": missing})
                if not ok:
                    return {"passed": False, "reason": f"Missing items: {missing}", "details": details}
            elif p.lower().startswith("regex:"):
                pattern = p.split(":",1)[1].strip()
                ok = bool(re.search(pattern, out, flags=re.MULTILINE))
                details["checks"].append({"check": f"regex:{pattern}", "passed": ok})
                if not ok:
                    return {"passed": False, "reason": f"Regex '{pattern}' did not match output.", "details": details}
            elif p.lower().startswith("equals:"):
                expected = p.split(":",1)[1].strip()
                ok = out.strip() == expected
                details["checks"].append({"check": "equals", "passed": ok})
                if not ok:
                    return {"passed": False, "reason": "Output did not equal expected value.", "details": details}
            elif p.lower().startswith("min_len:"):
                try:
                    n = int(p.split(":",1)[1].strip())
                    ok = len(out) >= n
                    details["checks"].append({"check": f"min_len:{n}", "passed": ok, "len": len(out)})
                    if not ok:
                        return {"passed": False, "reason": f"Output length {len(out)} < min_len {n}.", "details": details}
                except Exception:
                    return {"passed": False, "reason": "Invalid min_len value.", "details": details}
            elif p.lower().startswith("max_len:"):
                try:
                    n = int(p.split(":",1)[1].strip())
                    ok = len(out) <= n
                    details["checks"].append({"check": f"max_len:{n}", "passed": ok, "len": len(out)})
                    if not ok:
                        return {"passed": False, "reason": f"Output length {len(out)} > max_len {n}.", "details": details}
                except Exception:
                    return {"passed": False, "reason": "Invalid max_len value.", "details": details}
            elif p.lower().startswith("json_schema:"):
                schema_str = p.split(":",1)[1].strip()
                try:
                    import jsonschema, json as _json
                    schema = _json.loads(schema_str)
                    data = _json.loads(out) if out.strip() else None
                    if data is None:
                        return {"passed": False, "reason": "Output is empty; cannot validate JSON schema.", "details": details}
                    jsonschema.validate(instance=data, schema=schema)
                    details["checks"].append({"check": "json_schema", "passed": True})
                except ImportError:
                    return {"passed": False, "reason": "jsonschema package not installed.", "details": details}
                except Exception as e:
                    details["checks"].append({"check": "json_schema", "passed": False, "error": str(e)})
                    return {"passed": False, "reason": f"JSON schema validation failed: {e}", "details": details}
            else:
                details["checks"].append({"check": p, "passed": None, "note": "Unknown rule"})
                # Unknown rules are non-fatal; could change to fail if desired

        # All checks passed
        return {"passed": True, "reason": "All checks passed.", "details": details}
    except Exception as e:
        return {"passed": False, "reason": str(e), "details": {}}


@tool
def restore_snapshot(snapshot_id: str):
    """
    回滚到指定上下文快照（与 save_context_snapshot / list_context_snapshots 配套）。

    snapshot_id 可为：快照文件名（如 20260101T120000__myname.json）、不带 .json
    后缀的名称，或 ./contexts 目录下的相对/绝对路径。
    返回该快照保存时的 content，供 agent 在出错时回滚到该状态。
    """
    try:
        ctx_dir = os.path.join(os.getcwd(), "contexts")
        candidate = snapshot_id
        # 1) 已是存在的路径；2) 当作 ./contexts 下的文件名；3) 当作名称补 .json
        if not os.path.isabs(candidate) and not os.path.exists(candidate):
            candidate = os.path.join(ctx_dir, snapshot_id)
            if not os.path.exists(candidate) and not snapshot_id.endswith(".json"):
                candidate = os.path.join(ctx_dir, f"{snapshot_id}.json")
        if not os.path.exists(candidate):
            return {"error": f"Snapshot not found: {snapshot_id}"}
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        content = data.get("content", "")
        return {"message": f"Snapshot restored: {candidate}", "content": content}
    except Exception as e:
        return {"error": str(e)}


# ============ 主 Tool 函数 ============
def retry_smtp(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (smtplib.SMTPServerDisconnected, 
                        smtplib.SMTPAuthenticationError, 
                        smtplib.SMTPConnectError, 
                        ConnectionResetError) as e:
                    last_exc = e
                    wait = (2 ** i) + random.uniform(0, 1)
                    logger.warning(f"SMTP 错误 {type(e).__name__}: {e}, {i+1}/{max_retries} 重试，等待 {wait:.1f}s...")
                    time.sleep(wait)
                except smtplib.SMTPResponseException as e:
                    # 新增：解析 -1 码
                    if e.smtp_code == -1:
                        logger.error(f"无效服务器响应 (可能是 TLS/授权问题): {e.smtp_error}")
                        # 尝试备用端口 587 (STARTTLS)
                        kwargs['use_ssl'] = False  # 临时切换
                        if i == max_retries - 1:
                            raise Exception("备用端口也失败，请检查授权码/网络")
                    raise
                except Exception as e:
                    logger.error(f"发邮件未知错误: {e}")
                    raise
            raise last_exc or Exception("发邮件失败")
        return wrapper
    return decorator

@tool
@retry_smtp(max_retries=3)
def send_qq_email(
    to_email: str,
    subject: str,
    body: str,
    *,
    attachment_paths: Optional[List[str]] = None,
    is_html: bool = True,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    use_ssl: bool = True  #  SSL/STARTTLS
) -> str:
    """
    QQ emial send tool
    Args:
        -to_email: str,
        -subject: str,
        -body: str,
        *,
        -attachment_paths: Optional[List[str]] = None,
        -is_html: bool = True,
        -cc: Optional[str] = None,
        -bcc: Optional[str] = None,
        -use_ssl: bool = True  
    """
    msg = MIMEMultipart("alternative")
    msg["From"] = QQ_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    if cc: msg["Cc"] = cc
    if bcc: msg["Bcc"] = bcc

    # 正文
    msg.attach(MIMEText(body, "plain", "utf-8"))
    if is_html: msg.attach(MIMEText(body, "html", "utf-8"))

    # 附件
    if attachment_paths:
        for file_path in attachment_paths:
            file_path = Path(file_path).expanduser().resolve()
            if not file_path.exists():
                logger.warning(f"附件不存在: {file_path}")
                continue
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={file_path.name}")
            msg.attach(part)

    # 发送逻辑（增强）
    server = None
    try:
        if use_ssl:
            # 首选 SSL 465
            context = ssl.create_default_context()  # 新增：现代 TLS
            server = smtplib.SMTP_SSL("smtp.qq.com", 465, context=context, timeout=30)
        else:
            # 备用 STARTTLS 587
            server = smtplib.SMTP("smtp.qq.com", 587, timeout=30)
            context = ssl.create_default_context()
            server.starttls(context=context)

        logger.info(f"连接 QQ SMTP {'(SSL)' if use_ssl else '(STARTTLS)'}...")
        server.login(QQ_EMAIL, QQ_APP_PASSWORD)
        logger.info("登录成功")

        recipients = [to_email]
        if cc: recipients.extend([x.strip() for x in cc.split(",")])
        if bcc: recipients.extend([x.strip() for x in bcc.split(",")])
        server.sendmail(QQ_EMAIL, recipients, msg.as_string())
        logger.info("邮件发送成功")

        return json.dumps({
            "status": "success",
            "message": f"邮件发送成功 → {to_email}",
            "recipients": len(recipients)
        }, ensure_ascii=False)

    except Exception as e:
        error_msg = f"邮件发送失败: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "status": "error",
            "message": error_msg,
            "tip": "检查授权码、网络，或试用_ssl=False"
        }, ensure_ascii=False)

    finally:
        # 新增：优雅关闭（关键修复！）
        if server:
            try:
                server.quit()
                logger.info("SMTP 连接已关闭")
            except:
                logger.warning("关闭连接时出错（忽略）")


@tool
def get_crypto_sentiment_indicators():
    """
    Agent Tool: Fetch cryptocurrency market sentiment indicators with fault tolerance and extensibility.

    Functionality:
    - Calls the CryptOracle API to retrieve BTC sentiment data from the last 4 hours.
    - Supports core sentiment endpoints: 
        - CO-A-02-01: positive sentiment ratio
        - CO-A-02-02: negative sentiment ratio
    - Computes derived metrics: net sentiment, sentiment strength, and data delay.
    - Includes comprehensive error handling: network failures, empty responses, 
      type conversion errors, and missing fields will NOT crash the program.

    Returns (on success):
    {
        'positive_ratio': float,      # e.g., 65.2
        'negative_ratio': float,      # e.g., 34.8
        'net_sentiment': float,       # = positive_ratio - negative_ratio
        'sentiment_strength': float,  # = positive_ratio + negative_ratio (indicates activity level)
        'ar_ratio': float or None,    # Placeholder for future AR indicator (e.g., CO-A-03-01)
        'br_ratio': float or None,    # Placeholder for future BR indicator (e.g., CO-A-03-02)
        'data_time': str,             # Timestamp in "YYYY-MM-DD HH:MM:SS"
        'data_delay_minutes': int,    # Delay from current time in minutes
        'source': 'cryptoracle'
    }

    Returns (on failure):
        None (no exception is raised)

    Extensibility Notes:
    - AR (Advance-Decline Ratio) and BR (Breadth Ratio) are classic sentiment indicators 
      used in technical analysis [[1]][[5]]. If the API adds support (e.g., via new endpoints),
      they can be integrated by extending the endpoint list and parsing logic.
    """

    try:
        # === Configuration ===
        API_URL = "https://service.cryptoracle.network/openapi/v2/endpoint"
        API_KEY = CRYPTO_SENTIMENT_KEY

        if not API_KEY:
            logger.warning("API_KEY is not configured. Skipping sentiment fetch.")
            return None

        # === Define time window: last 4 hours with 15-minute granularity ===
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        # === Build request payload ===
        request_body = {
            "apiKey": API_KEY,
            "endpoints": ["CO-A-02-01", "CO-A-02-02"],  # Core sentiment indicators
            "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timeType": "15m",
            "token": ["BTC"]
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": API_KEY
        }

        # === Make HTTP POST request ===
        try:
            response = requests.post(
                API_URL,
                json=request_body,
                headers=headers,
                timeout=10  # Prevent hanging on slow networks
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Network request failed: {e}")
            return None

        # === Check HTTP status code ===
        if response.status_code != 200:
            logger.error(f"HTTP request failed with status code: {response.status_code}")
            return None

        # === Parse JSON response safely ===
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None

        # === Validate API business logic response ===
        if not (data.get("code") == 200 and data.get("data")):
            logger.warning(f"API returned non-success code or empty data: code={data.get('code')}")
            return None

        # === Process time periods to find the first valid data point ===
        time_periods = data["data"][0].get("timePeriods", [])
        if not time_periods:
            logger.warning("No 'timePeriods' found in response data")
            return None

        # Iterate from newest to oldest for faster valid data retrieval
        for period in reversed(time_periods):
            period_data = period.get("data", [])

            sentiment = {}
            valid_data_found = False

            for item in period_data:
                endpoint = item.get("endpoint")
                value_str = item.get("value", "").strip()

                # Skip empty values
                if not value_str:
                    continue

                # Safely convert value to float
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    logger.debug(f"Skipping non-numeric value for endpoint {endpoint}: '{value_str}'")
                    continue

                # Only process known sentiment endpoints
                if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                    sentiment[endpoint] = value
                    valid_data_found = True

            # Check if both core indicators are present
            if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                positive = sentiment['CO-A-02-01']
                negative = sentiment['CO-A-02-02']
                net_sentiment = positive - negative
                sentiment_strength = positive + negative  # Reflects overall sentiment activity

                # Calculate data delay in minutes
                try:
                    data_time = datetime.strptime(period['startTime'], '%Y-%m-%d %H:%M:%S')
                    data_delay = int((datetime.now() - data_time).total_seconds() // 60)
                    data_time_str = period['startTime']
                except (ValueError, KeyError):
                    # Fallback if timestamp parsing fails
                    logger.warning("Failed to parse 'startTime'; using placeholder")
                    data_delay = 0
                    data_time_str = period.get('startTime', 'unknown')

                logger.info(f"✅ Using sentiment data from: {data_time_str} (delay: {data_delay} minutes)")

                # === Return standardized, agent-friendly result ===
                return {
                    'positive_ratio': positive,
                    'negative_ratio': negative,
                    'net_sentiment': net_sentiment,
                    'sentiment_strength': sentiment_strength,
                    'ar_ratio': None,  # Reserved for future AR indicator
                    'br_ratio': None,  # Reserved for future BR indicator
                    'data_time': data_time_str,
                    'data_delay_minutes': data_delay,
                    'source': 'cryptoracle'
                }

        logger.warning("❌ No valid sentiment data found in any time period")
        return None

    except Exception as e:
        # Fallback: catch any unexpected errors to prevent crashes
        logger.error(f"Unexpected error in get_sentiment_indicators: {e}")
        return None


# === 工具调用辅助（供 middleware / 节点 / invoke_with_memory 调用工具）===
# @tool 装饰的 StructuredTool 在 langchain-core>=1.6 已不可直接调用（tool(...) 会抛
# TypeError: 'StructuredTool' object is not callable）。这里统一走底层 .func 调用。
# 原定义位于 agent.py，拆分时迁至本模块（tools.py 是工具定义的叶子模块，无内部依赖）。
def _run_tool(tool, *args, **kwargs):
    func = getattr(tool, "func", None)
    if callable(func):
        return func(*args, **kwargs)
    if hasattr(tool, "invoke") and kwargs:
        return tool.invoke(kwargs)
    return tool(*args, **kwargs)

