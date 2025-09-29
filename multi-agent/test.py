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