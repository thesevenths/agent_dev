"""Multi-Agent Research Runtime —— 图装配与运行时入口的包根。

模块职责（按调用顺序）：

- ``agent``            : 组合根。装配 LLM、工具、图与记忆，并暴露 ``graph`` 供
                         ``langgraph.json`` 的 ``agent:graph`` 契约加载；
- ``graph``            : StateGraph 装配、Checkpointer 选择、快照可视化；
- ``supervisor``       : 规划 / 调度 / 自适应再规划 / 收尾对账；
- ``agents``           : 成员 Agent 工厂与每步内部的三级质量门；
- ``state``            : ``AgentState`` schema（全局唯一真源）；
- ``adaptive_runtime`` : 自优化实验层，不修改上述任何生产模块。

设计约束：本文件不得引入对具体子模块的导入。包根保持无副作用，使
``python -c "import <pkg>"`` 与工具链（``langgraph dev``、测试收集器）在
任意工作目录下都不会触发图构建或网络访问。
"""
