multi agent开发！ 经典的supervised + other agent结构！

生成研究报告：

![1759239379759](image/readme/1759239379759.png)

数据库读写：

![1759239410997](image/readme/1759239410997.png)

text2sql：根据用户的prompt自动生成sql查询并返回结果

![1759243044703](image/readme/1759243044703.png)

agentic RAG：plan + self evaluation

![1760107451075](image/readme/1760107451075.png)

自动生成代码，做复杂的数据分析、价格预测等：

![1762735267322](image/readme/1762735267322.png)

自动生成markdown格式的研究报告：

![1762735315100](image/readme/1762735315100.png)

自动发送研报到指定邮箱：

![1762735667940](image/readme/1762735667940.png)

![1762735737951](image/readme/1762735737951.png)

langgraph优势： https://mp.weixin.qq.com/s/LHIfNCovj9eknufCiEEcUA

* 显式的流程建模能力: 有向图（Directed Graph）描述agent行为

  * 流程逻辑清晰，可快速理解系统架构；
  * 支持条件跳转、循环、并行等复杂控制流；
  * 易于做流程变更与功能扩展。
* 状态管理机制

  * **状态持久化** ：通过 Checkpointer（如 PostgreSQL、Redis）保存执行快照，支持断点续跑。
  * **会话隔离** ：每个会话（thread）独立存储，避免状态污染。
  * **状态合并策略** ：通过 Annotated 定义字段更新规则 这使得系统具备了处理长时间运行任务的能力，例如跨天的任务审批、多轮调研报告生成等。
* 内建的生产级能力

  * 全链路追踪（Trace）；
  * 节点级耗时、Token 消耗统计；
  * 错误堆栈与输入输出快照；
  * A/B 测试与评估指标管理。
  * 故障排查、性能优化、合规审计有优势
* 架构开放，易于集成

  * 多模型供应商（OpenAI、Anthropic、本地部署模型）；
  * 自定义工具调用（Tool Calling）；
  * 外部系统集成（数据库、API、消息队列）；
  * MCP（Model Context Protocol）扩展。
* 用中间件做 Context Engineering

  LangChain 1.0 的中间件可以实现**Context Engineering** （上下文工程），因为它允许在 Agent 循环中动态管理上下文（如注入/过滤消息、评估相关性、快照恢复），这与ContextEngineer Agent 高度契合。 中间件可以作为“插件”注入到 ReAct Agent（如 create_react_agent）中，实现更细粒度的控制，而非依赖独立 Agent 节点。

可以做到的场景

* **动态上下文注入** ：在模型调用前，基于查询相关性添加/移除历史消息（e.g., RAG 检索）。
* **上下文评估与压缩** ：**post-model hook 中评估输出质量，压缩冗余上下文（类似 evaluate_output 工具）**。
* **快照管理** ：**pre-tool call 时保存/恢复快照（扩展 save_context_snapshot）**。
* **Human-in-the-loop** ：暂停循环等待人工确认上下文。
* **错误恢复** ：如果上下文导致 hallucination，自动回滚。


代码执行时用sandbox隔离环境：https://aws.amazon.com/cn/blogs/china/agentic-ai-sandbox-practice/

* 代码执行环境: Agent应用需要单独的代码执行环境来执行特定任务。以企业数据分析Agent为例，业务分析师可以直接上传一个1GB的销售数据文件到应用平台，然后通过自然语言告诉Agent：”分析过去一年的销售趋势，找出表现最好的产品类别，并生成可视化报表”。Agent能够自动解析用户意图，并调用大型语言模型生成数据读取、处理及分析代码。它会多次启动沙盒环境执行这些代码，最终生成包含图表和统计分析的完整报告。尽管整个流程可能需要数小时的连续计算，但用户只需通过自然语言描述需求并进行必要的修正即可。所有复杂工作均由Agent、大型语言模型和沙盒工具协同完成，无需用户直接参与技术操作。
* Computer Use和Browser Use：以某社区媒体营销文案生成Agent为例，营销人员只需输入”收集某某竞品在该平台上的营销策略”，Agent就能像真实用户一样操作浏览器：自动打开多个网页标签，浏览不同的产品页面和用户评论，收集关键的市场数据和用户反馈信息，然后基于收集到的数据进行分析，最终实现精准的内容推荐和广告投放策略。整个过程中，Agent通过Browser Use功能模拟人类的点击、滚动、输入等操作，完成复杂的数据收集任务。类似的应用还包括游戏AI测试、软件自动化测试、在线订票等场景。这些应用的共同特点是需要Agent能够精确控制鼠标和键盘操作，与图形界面进行自然交互，处理那些没有API接口、只能通过视觉操作的应用程序。
