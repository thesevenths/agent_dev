"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

from typing_extensions import Annotated
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel
from langchain_core.tools import tool
from sqlalchemy import inspect
from sqlalchemy import text
from config import PG_CONN_STR, TAVILY_API_KEY

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


# 用于数据分析师执行代码
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