
---

### 一、 核心架构设计 (Core Architecture)

这是 Agent 的“大脑”和“四肢”设计，决定了它能不能完成任务。

#### 1. 上下文工程 (Context Engineering)
*   **要点**：不要只盯着 Prompt，要管理整个上下文的生命周期。
*   **实践**：
    *   **动态注入**：根据当前任务状态，动态检索并注入相关的背景知识（RAG）或历史摘要。
    *   **Token 预算控制**：严格计算 System Prompt、历史对话、工具返回结果的 Token 占比，防止 OOM（超出上下文窗口）。
    *   **状态外置**：把复杂的中间计算结果、任务清单写入外部文件（如 `scratchpad.md`），而不是全塞在 LLM 的对话历史里。

#### 2. 工具设计与封装 (Tool Use & Function Calling)
*   **要点**：工具是给 LLM 用的，**工具的“说明书”比代码本身更重要**。
*   **实践**：
    *   **清晰的描述**：工具的名称、描述、参数说明必须极其精准，包含使用场景和反面示例（Few-shot）。
    *   **结果格式化与压缩**：工具返回给 LLM 的结果不能是原始的大 JSON 或长文本。必须在代码层进行**提炼、截断、格式化**（例如：将 500 行日志压缩为“发现 3 个 Error，分别是...”），减少 Token 消耗并降低模型的认知负荷。
    *   **原子化设计**：工具功能要单一，避免一个工具做太多复杂逻辑，让 LLM 难以决定何时调用。

#### 3. 记忆系统 (Memory System)
*   **要点**：让 Agent 具备跨会话和长周期的记忆能力。
*   **实践**：
    *   **短期记忆 (Working Memory)**：当前对话的滑动窗口，结合自动摘要（Auto-summarization）压缩历史。
    *   **长期记忆 (Long-term Memory)**：使用向量数据库（Vector DB）存储用户偏好、历史任务经验；或使用知识图谱（Knowledge Graph）存储实体关系。
    *   **实体记忆**：提取对话中的关键实体（如人名、项目名、特定配置）持久化存储。

#### 4. 规划与反思 (Planning & Reflection)
*   **要点**：让 Agent 学会拆解任务，并在出错时自我纠正。
*   **实践**：
    *   **ReAct 模式**：思考 (Thought) -> 行动 (Action) -> 观察 (Observation) 的循环。
    *   **Plan-and-Solve**：对于复杂任务，先让 LLM 生成一个 Step-by-step 的计划，然后逐步执行。
    *   **自我反思 (Self-Correction)**：引入 Critic（评估者）角色，或者在 Prompt 中要求模型在执行后检查输出是否符合预期，如果报错则自动重试或修改策略。

---

### 二、 工程化与生产落地 (Production Engineering)

这是区分“玩具”和“产品”的关键，决定了 Agent 能不能在生产环境中稳定运行。

#### 5. 鲁棒性与防死循环 (Robustness & Loop Prevention)
*   **要点**：LLM 会幻觉、会犯傻，必须用工程手段兜底。
*   **实践**：
    *   **最大步数限制 (Max Iterations)**：强制设定 Agent 最多执行 N 步（如 15 步），超出则强制终止并返回给用户，**绝对不能让 Agent 陷入无限死循环**（这会产生巨额账单）。
    *   **格式强校验**：使用 Pydantic 或 JSON Schema 严格校验 LLM 的工具调用参数，如果解析失败，将错误信息喂给 LLM 让其重新生成（通常重试 2-3 次）。
    *   **降级策略 (Fallback)**：当主模型 API 超时或报错时，自动切换到备用模型或降级为规则引擎。

#### 6. 可观测性与调试 (Observability & Tracing)
*   **要点**：Agent 是黑盒，没有可观测性就无法调试和优化。
*   **实践**：
    *   **全链路追踪**：接入 LangSmith、Langfuse、Arize 等工具，记录每一次 LLM 调用的 Prompt、Response、Tool Call、Token 消耗和耗时。
    *   **状态快照**：在关键节点保存 Agent 的状态，方便复现 Bug。

#### 7. 评估体系 (Evaluation / Evals)
*   **要点**：不能靠肉眼看效果，必须建立自动化的评估流水线。
*   **实践**：
    *   **构建 Eval 数据集**：收集真实场景的 Query 和期望的 Golden Answer/Action 轨迹。
    *   **多维度指标**：评估任务完成率、步骤数、Token 消耗、幻觉率、工具调用准确率。
    *   **LLM-as-a-Judge**：使用另一个更强的 LLM（或同模型）来对 Agent 的输出结果进行打分和评判。

---

### 三、 安全、权限与人机协同 (Security & Human-in-the-Loop)

Agent 拥有执行能力，一旦失控破坏力极大。

#### 8. 最小权限与沙盒隔离 (Sandbox & Permissions)
*   **要点**：永远不要给 Agent  Root 权限或不受限的网络访问。
*   **实践**：
    *   **代码执行沙盒**：如果 Agent 需要写代码并运行（如 Code Interpreter），必须在 Docker、gVisor 或 E2B 等隔离沙盒中执行，限制 CPU/内存/网络/执行时间。
    *   **API 权限最小化**：Agent 调用的第三方 API（如发邮件、查数据库）必须使用只读或受限权限的 Token。

#### 9. 人机协同确认 (Human-in-the-Loop, HITL)
*   **要点**：对于高风险操作，必须让人类把关。
*   **实践**：
    *   **操作分级**：将工具分为“安全（如查询）”、“警告（如修改配置）”、“危险（如删除数据、发送邮件、转账）”。
    *   **阻断与确认**：当 Agent 准备调用“危险”工具时，暂停执行，将计划展示给用户，等待用户点击“确认”后才真正执行。

#### 10. 防注入攻击 (Prompt Injection Defense)
*   **要点**：防止恶意用户通过输入内容劫持 Agent 的系统指令。
*   **实践**：
    *   使用分隔符（如 `"""` 或 `<user_input>`）严格隔离系统指令和用户输入。
    *   引入输入过滤器，或者使用专门的安全模型（如 Llama Guard）对输入/输出进行安全审查。

---

### 四、 性能、成本与体验 (Performance, Cost & UX)

#### 11. 延迟优化与流式体验 (Latency & Streaming)
*   **要点**：Agent 思考时间长，用户容易失去耐心。
*   **实践**：
    *   **流式输出 (Streaming)**：将 Agent 的思考过程（Thought）、工具调用状态（如“正在搜索...”）实时流式展示给用户，缓解等待焦虑。
    *   **并行工具调用**：如果 LLM 支持（如 Parallel Function Calling），让 Agent 同时调用多个无依赖的工具，减少串行等待时间。

#### 12. 成本控制 (Cost Management)
*   **要点**：Agent 会消耗大量 Token，不控制成本会亏本。
*   **实践**：
    *   **模型路由 (Model Routing)**：简单任务（如意图识别、信息提取）路由给便宜的小模型（如 Haiku/GPT-4o-mini），复杂推理路由给大模型。
    *   **语义缓存 (Semantic Cache)**：对相似的问题和工具调用结果进行缓存，避免重复计算。
    *   **Prompt 瘦身**：定期清理冗余的 System Prompt 和无用的历史对话。

---

### 五、 进阶：多智能体协同 (Multi-Agent Systems)

当单 Agent 无法胜任时，才考虑多 Agent。

#### 13. 角色划分与编排 (Orchestration)
*   **要点**：不要为了多 Agent 而多 Agent，单 Agent 能解决的绝不用多 Agent（会增加延迟和成本）。
*   **实践**：
    *   **明确分工**：如 Router（路由分发）、Planner（任务拆解）、Executor（执行者）、Critic（审查者）。
    *   **通信机制**：设计好 Agent 之间的消息传递格式，避免信息在传递中丢失或失真。
    *   **共享黑板 (Blackboard)**：多个 Agent 通过一个共享的状态空间（如共享的内存或数据库）来交换信息和协同工作，而不是仅仅依靠对话传递。