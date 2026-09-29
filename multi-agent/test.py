# # -*- coding: utf-8 -*-
# """
# Author: MuYu_Cheney
# """
# from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
# from sqlalchemy.orm import sessionmaker, declarative_base
# from faker import Faker
# import random
#
# # 创建基类
# Base = declarative_base()
#
# # 定义模型
# class SalesData(Base):
#     __tablename__ = 'sales_data'
#     sales_id = Column(Integer, primary_key=True)
#     product_id = Column(Integer, ForeignKey('product_information.product_id'))
#     employee_id = Column(Integer)  # 示例简化，未创建员工表
#     customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))
#     sale_date = Column(String(50))
#     quantity = Column(Integer)
#     amount = Column(Float)
#     discount = Column(Float)
#
# class CustomerInformation(Base):
#     __tablename__ = 'customer_information'
#     customer_id = Column(Integer, primary_key=True)
#     customer_name = Column(String(50))
#     contact_info = Column(String(50))
#     region = Column(String(50))
#     customer_type = Column(String(50))
#
# class ProductInformation(Base):
#     __tablename__ = 'product_information'
#     product_id = Column(Integer, primary_key=True)
#     product_name = Column(String(50))
#     category = Column(String(50))
#     unit_price = Column(Float)
#     stock_level = Column(Integer)
#
# class CompetitorAnalysis(Base):
#     __tablename__ = 'competitor_analysis'
#     competitor_id = Column(Integer, primary_key=True)
#     competitor_name = Column(String(50))
#     region = Column(String(50))
#     market_share = Column(Float)
#
# # 数据库连接和表创建
# DATABASE_URI = 'mysql+pymysql://root:snowball2019@localhost/langgraph_agent?charset=utf8mb4'     # 这里要替换成自己的数据库连接串
# engine = create_engine(DATABASE_URI)
# Base.metadata.create_all(engine)
#
# Session = sessionmaker(bind=engine)
# session = Session()

import sys
print(sys.version)

from config import PG_CONN_STR, TAVILY_API_KEY
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
Base = declarative_base()
engine = create_engine(PG_CONN_STR)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
print(session)

from sqlalchemy import inspect

# inspector = inspect(engine)
# tables = inspector.get_table_names()
# print("数据库中的表：", tables)
# for table_name in tables:
#     print(f"\n表：{table_name}")
#     for column in inspector.get_columns(table_name):
#         print(f"  {column['name']} ({column['type']})")

# 用 session 绑定的引擎做 inspect
inspector = inspect(session.bind)

# 所有表名
session = Session()

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
print(schema)