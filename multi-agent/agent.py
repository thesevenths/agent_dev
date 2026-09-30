"""Multi-Agent System with Memory, Rollback, and Visualization —— 组合根（composition root）。

所有实现已拆分到子模块：
  state       Agent 状态 schema（PlanStep / AgentState）唯一真源
  planutil    计划解析 / 成员路由配置（叶子）
  context     当前日期 / 交易时段 / 数据时效上下文
  compress    上下文压缩（防 context 溢出的安全网）
  llm         create_llm + 角色 LLM + 自定义 Middleware
  summary     跨步语义摘要（要点提取）引擎
  plan        计划改写 + 进度标记 + 自适应重规划引擎
  handoff     子 agent 任务下发 + 产物落盘（agent 间 handoff 通道）
  agents      子 agent 工厂 + 节点工厂
  supervisor  Supervisor 节点（规划 + 多轮执行）
  graph       Graph 装配 + 可视化 + 带记忆调用入口

本文件只负责：
  - 统一初始化（KMP 兼容、logging、.env）；
  - 暴露 langgraph.json 契约要求的模块级 `graph` 对象（`"agent:graph"`）；
  - 提供 `invoke_with_memory` 与 `if __name__ == "__main__"` 自测入口。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 由 graph.py 构造并暴露编译后的 graph（满足 langgraph.json 的 "agent:graph" 契约）。
# 该 import 会触发整条依赖链（supervisor → plan/summary → ...）的加载，需在 load_dotenv() 之后。
from graph import graph, invoke_with_memory  # graph = langgraph.json 契约要求的模块级对象；invoke_with_memory 供 __main__ 自测



# === 测试 ===
if __name__ == "__main__":
    # 初始化上下文目录
    os.makedirs("./contexts", exist_ok=True)
    os.makedirs("./snapshots", exist_ok=True)
    os.makedirs("./documents", exist_ok=True)

    # 测试 1：简单对话
    print("=== 测试 1：简单对话 ===")
    result1 = invoke_with_memory("你好，我是金融分析师")
    print(f"Final response: {result1['messages'][-1].content if result1 else 'Failed'}")

    # 测试 2：复杂查询（触发工具 + 错误恢复）
    print("\n=== 测试 2：纳斯达克查询 + 模拟错误 ===")
    try:
        # 模拟一个可能出错的查询
        result2 = invoke_with_memory("分析今天纳斯达克涨幅前3的股票，生成报告。如果出错请自动恢复。")
        print(f"Success: {result2['messages'][-1].content[:100] if result2 else 'Failed'}...")
    except Exception as e:
        print(f"Expected error handled: {e}")

    # 测试 3：加载记忆
    print("\n=== 测试 3：加载记忆继续对话 ===")
    thread_id = "test_thread_123"
    invoke_with_memory("之前我问了纳斯达克，现在帮我查数据库里的销售数据", thread_id=thread_id)

    print("\n🎉 Multi-Agent System with Memory & Recovery is ready!")
