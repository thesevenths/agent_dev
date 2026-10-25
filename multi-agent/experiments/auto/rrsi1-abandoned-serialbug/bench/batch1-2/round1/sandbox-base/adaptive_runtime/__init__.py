"""Adaptive Runtime：Evidence-Driven Harness 自优化的控制平面。

设计依据：docs/adaptive_agent_runtime_final_plan.md（V3 定稿）。

不变量
------
1. 不改现有生产代码：supervisor.py / agents.py / plan.py / graph.py / runlog.py 一行不改。
2. 模型冻结：只搜索 harness（参数 / 提示 / 结构），不动权重。
3. 评分器不可自改：frozen.py 用「符号级 hash + 解析后 env hash」双断言。
4. 证据不完整即失败：任一臂缺失 run，整轮判定 INCOMPLETE_EVIDENCE。

阶段开关（P0 / P1 / P2）由 settings.Stage 控制，默认 P0。
"""
from .settings import CFG, PATHS  # noqa: F401

__all__ = ["CFG", "PATHS"]
__version__ = "3.0.0"
