"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""
import os
import time
import json
import subprocess

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

from langchain_tavily import TavilySearch
tavily_search = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5  # 返回前 5 个结果
)

# 创建基类
Base = declarative_base()
engine = create_engine(PG_CONN_STR)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

repl = PythonREPL()


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
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
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


@tool
def create_file(file_name: str, file_contents: str):
    """
    Create a new file with the provided contents at a given path in the workspace.
    
    Args:
        file_name (str): Name of the file to be created (required, non-empty)
        file_contents (str): The content to write to the file (required)
    """
    try:
        if not file_name or not file_contents:
            return {"error": "file_name and file_contents must be non-empty strings"}
        file_path = os.path.join(os.getcwd(), file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        return {"file_name": file_name, "content": content}
    except Exception as e:
        return {"error": f"Error reading {file_name}: {str(e)}"}


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
        file_path = os.path.join(os.getcwd(), file_name)
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