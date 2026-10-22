# 评审：Adaptive Agent Runtime 完整改造方案

> 被评审对象：`docs/adaptive_agent_runtime_rebuild_plan.md`（34 节 / 1747 行）
> 评审基准：本仓库实际代码 + Harness²（arXiv:2610.14224）/ RRSI（arXiv:2609.24972）/ ModularRSI（arXiv:2609.14857）/ PluginRSI（arXiv:2609.32423）
> 评审日期：2026-10-22

---

## 0. 评审结论

**方向正确、结构完整、文献引用准确；但被评审文档的形态是「研究议程（proposal）」，尚不足以作为施工依据。**

若按当前写法直接开工，会在两处阻塞：**实验设计无法产生可信结论**（单轨迹 n=1），**P0 阶段不可验收**（度量尺被排到 P2）。

| 维度 | 评价 | 说明 |
|---|---|---|
| 方向（是否该做这件事） | 正确 | 诊断 → 假设 → 受控实验 → 迁移，闭环形状正确 |
| 结构（应包含哪些模块） | 较完整 | 组件全谱、Trace、评价层次、研究实验设计均已覆盖 |
| 可执行性（能否照此施工） | 弱 | 0 个样本量、0 处噪声估计、0 处成本预算、0 条阶段验收标准 |
| 与本仓库的贴合度 | 低 | 对现有系统的描述与代码不符，把已有能力当作「从零新建」 |
| 文献准确性 | 高 | 引用的三篇 arXiv 编号均可查、信息准确（见附录） |
| 主要风险 | 中高 | 单轨迹结论不可信；§22 的目录结构实质是重写；缺少评分器冻结 |

---

## 1. 应当保留的六项设计

| # | 内容 | 保留理由 |
|---|---|---|
| 1 | **§6 Trace schema 的任务级统计**：`sources / primary_sources / claims / verified_claims / unsupported_claims / reviewer_findings` | 研报类任务的客观锚点，比纯 LLM 打分稳定，可直接构成 L2 结构契约层。是该文档最实用的部分 |
| 2 | **§5 组件全谱（9 类）+ 每类「解决什么问题」** | 比评审方原清单完整：`tool_guidance`（有工具但不知何时使用）与 `script`（确定性重复任务）两类此前被遗漏，而这正对应 `tools.py` 与脚本产物的现状 |
| 3 | **§21 Git 版本化 + `branch: candidate/<id>`** | 比 JSON 版本文件更实用：`git diff / log / revert` 天然满足可审计、可回滚、可复现。已采纳 |
| 4 | **§11「Behavior Improvement ≠ Proven Task Correctness」** | 表述克制。避免把「行为变化」表述为「质量提升」，这是同类方案最常见的过度声明 |
| 5 | **§28 六组对照实验**（No Diagnosis vs Diagnosis-guided / Unregularized vs Regularized / Behavior-only vs Verifier+Behavior 等） | 把方案做成研究项目、撰写 README 或论文的正确骨架，价值最高的一节 |
| 6 | **§29 定位策略**（不宣称「复现 Harness²」） | 务实。定位为「在三者之上补 Diagnosis + Component-level + Transfer」是站得住的 |

---

## 2. 六个必修问题（按严重度排序）

### 问题 1（阻塞）· 实验设计为 n=1 的单轨迹对比

**原文**：§10「所有 Candidate 在相同条件下运行：same task / same model / same tools / same budget / same seed」；§25 MVP 示例「Base unsupported=8 → A: 5 → B: 2 → 选 B」。

**问题**：这是单任务、单次执行的对比，全文未出现一次「重复 k 次」。单次采样的差异无法与实例噪声区分，据此选择候选等同于用噪声做决策。

**证据（引自该文档自身引用的论文）**：ModularRSI 的第一条动机即为——

> "**single-trajectory updates** can conflate systematic harness deficiencies with **instance-specific** reasoning and solution details, producing modifications that transfer poorly to unseen tasks."

即：该方案的核心实验方法，正是其参考文献明确否定的做法。

**建议**：同一任务两臂配对 × k ≥ 3 次重复，以配对差的正态近似确定噪声带：

```
δ = t_{0.975, n−1} · σ_diff / √n        n = 任务数 × k
```

单任务单次的 `unsupported 8→2` 在统计上不携带信息——它可能仅反映本次采样多写了两段。

---

### 问题 2（阻塞）· 正则化机制均停留在「应当存在」，无一条可执行判据

**原文**：

- §19.4「Base = 70 ± 3，Candidate = 71，不要认为 +1 是 improvement」——±3 的来源未说明。
- §19.5「quality +1% / tokens +100% 应该拒绝」——阈值未给定。
- §19.6「长期没有贡献就删除」——「长期」未量化，「没有贡献」未定义度量方式。
- §19.1「限制一次能修改多少个独立组件」——预算取值与退火方式未给定。

**全文仅有两个定义式**（`Generalization Gap = Evol − Held-out`、`Transfer Gain = 平均 held-out 改进`），没有一处样本量、噪声估计、成本预算或阶段验收标准。

**为何构成阻塞项**：这四项机制正是 RRSI 的技术核心，且均有精确形式：

| 机制 | RRSI 的可执行形式 | 该文档的写法 |
|---|---|---|
| 噪声带 δ | 演化**之前**反复评估未改动的 base harness 完成标定 | 「70 ± 3」，无出处 |
| 编辑预算 | `b_t = ⌈b_min+(b_max−b_min)·½(1+cos(πt/T))⌉` | 「限制数量」 |
| 成本规则 | `ΔS > δ` 时要求 `ΔC ≤ β₀+β₁·ΔS` | 「应该拒绝」 |
| 剪枝 | `B_t = {ℓ ∈ T_t : g_t(ℓ) ≤ 0}`，`g_t` 为窗口内最佳实测增益 | 「长期没贡献」 |

**结论**：按该文档实施，推进至 §19 时无法确定各判据的具体取值，最终只能凭主观选定——即回到该文档 §8 自身反对的做法。

---

### 问题 3（阻塞）· 缺少「评分器不可编辑」这一约束

**原文**：§19.3 Candidate Critic 仅检查四项——benchmark-specific hack、仅针对某个 task、泄漏答案、增加不合理复杂度。

**遗漏的第五项、也是最关键的一项**：**修改评分器本身**。

**为何在本仓库中必然发生**：评分逻辑与优化对象位于同一代码库——

```
agents.py: _rule_gate / _critic_gate / _action_contract_gate / _is_deliverable_step
agents.py: _CRITIC_MIN_LEN / _DEFAULT_ACTION_PROBES
```

§22 又建议将 harness 拆为 `harness/*.yaml` 并交由系统自行修改，其中包含 `config/`。一旦 `_CRITIC_MIN_LEN` 可被改动，闭环最先学到的行为就是把它调至 0。

**建议**：引入 Frozen Core（只读清单 + 变更即拒），并在 Critic 中增加硬规则：`diff 触及 frozen_core_hash → 直接拒绝，不进入评分`。

---

### 问题 4（严重）· P0 的范围与顺序倒置

**原文**：§33 P0 = Trace + Diagnosis + Hypothesis + Candidate + Probe + Comparator + Harness Versioning（7 项）；Edit Budget / Critic / Held-out 全部排至 **P2**。

**问题**：先建流水线、后建闸门。结果是 P0 产出的每一个「有效改动」都没有噪声基线可比，P1 的 Experience Library 会把噪声当作经验存储，P2 引入闸门时需推翻前述全部结论。

**建议顺序**：尺子 → 闸门 → 流水线。即 Trace + **δ 标定** + **最小 Gate** 必须先于 Probe 完成，否则 probe 产出的数字不可用。

**补充**：P0 的 7 项实际对应 2–3 倍工作量，且没有任何「做到什么程度算完成」的判据——该阶段不具备可验收性。

---

### 问题 5（严重）· 声明「不重写」，但 §22 的目录结构即重写

**原文**：§23「不要重写，外面增加 Adaptive Runtime」；§22 同时给出：

```
adaptive_runtime/harness/prompt.yaml
adaptive_runtime/harness/workflow.yaml
adaptive_runtime/harness/tools.yaml
```

**矛盾**：将 `prompt.py` 中 7 个 system prompt 抽为 `prompt.yaml`、将 `plan.py` 的 replan 逻辑抽为 `workflow.yaml`，属于**迁移**，不是「外层封装」。

**尚未覆盖的真实问题**：现有 harness 有一半是**命令式代码**，并非声明式配置：

- `plan.py` 的 8 个 `_REPLAN_*` 环境变量与 `_should_replan` 判定
- `agents.py` 的三道 gate 与 salvage 逻辑
- `graph.py` 的拓扑与 `planutil.members`

YAML 化之后，这些逻辑只有两种结局：**复制一份**（双源漂移，最坏结果）或**修改生产代码**（回归风险）。这是本仓库落地时最大的工程量，而该文档未涉及。

---

### 问题 6（严重）· 对现有系统的描述与代码不符，把已有能力当作「从零新建」

| 文档描述 | 代码事实 |
|---|---|
| §1.1「Planner → Multi-Agent → Tools/Memory → **Reviewer** → Final Answer」 | 不存在独立 Reviewer 节点。实际链路为 `run_start → supervisor`（含 replan / 早停 / 失败仲裁）→ 6 个 member（**每步内部**含 critic + rule + action-contract 三道 gate）→ FINISH；另有 HITL、SqliteSaver checkpoint、snapshot 回滚 |
| §6「第一阶段最重要的是把执行过程记录完整」 | `runlog.py` 已具备 per-run 唯一命名日志、`[step-failure] kind=... agent=... WHY=...`、节点耗时聚合 `timing_summary`；`agents.py` 已有 `_append_step_failure` 记账。此处需要的是「结构化 + 补 token 计量」，而非从零建设 |
| §5 组件全谱 | 漏掉 gate 层（决定「什么叫完成」）与 replan 策略层（8 个可枚举、可回滚、零副作用的阈值——恰恰最适合自动搜索） |

**影响具有连锁性**：未对齐代码 → 组件清单漏项 → §22 的目录结构无法对应真实文件 → 实施阶段才发现要改的是命令式逻辑。

---

## 3. 参考文献中未被吸收的三类机制

以下机制在原文参考文献中存在、但正文未吸收，价值高于原文自行撰写的大部分内容。

| 来源 | 机制 | 对本仓库的直接含义 |
|---|---|---|
| **ModularRSI** | **benchmark-disjoint**：从外部整理 2,000 个可执行任务，与下游评估基准**不相交** | evolve 集不能只从自身历史 run 抽取；OOD 样本须来自**另一种来源或任务族**，否则等同于用旧题考自己。评审方原方案的三分数据需增加一条：OOD 集优先取外部来源 |
| **ModularRSI** | **成功 / 失败轨迹对照**（同一任务的成功与失败对比），而非 base vs candidate | 可显著降低 probe 成本：不必每次重跑基线，只要历史中存在成功与失败样本。本仓库历史 run 中两者齐备 |
| **ModularRSI** | **模块化 + 受限修改范围 + integration stage 解冲突** | 「一次只改一个组件」是正确的，但原文未说明多个组件改完后如何合并；ModularRSI 设有专门集成阶段，是该文档缺失的一环 |
| **PluginRSI** | **插件库 + 重组**：harness = workflow over plugins，每轮仅变异一个 plugin，只保留在 balanced minibatch（成功/失败各半）上优于父代的变体 | 比直接修改 `prompt.py` 字符串更可复用：把「证据校验」「提前返回轮询」「对抗式复核」做成可插拔 skill，按需组装、可单独回滚。balanced minibatch 的采样设计本身即降方差手段 |
| **Harness²** | **每个任务均从 base harness 出发**，lessons 单独存为 experience file，**不允许编辑静默累积至主分支** | 纠正原文 §21 的做法：原文让 accepted candidate `merge → main` 持续累积，累积导致漂移不可控。Harness² 的取舍是「经验复用、harness 不累积」 |

---

## 4. 合并策略：两份方案各自负责的范围

| 职责 | 来源 | 理由 |
|---|---|---|
| Trace schema（字段级） | 原文 §6 + 补 token 埋点 | 任务级统计部分更细致，直接吸收为 L2 层 |
| 组件全谱与「每类解决什么」 | 原文 §5（补 gate / replan 两层） | 完整度更高 |
| 版本化 | 原文 §21（Git branch） | 比 JSON 版本文件更实用 |
| 研究实验设计 | 原文 §28 | 可直接用于 README / 论文 |
| 评分与噪声带 | RRSI + 评审方 §5 | 原文缺可执行判据，是全案最硬的缺口 |
| 闸门与准入规则 | 评审方 §4（Gate） | 原文的 Comparator → Composer 之间没有拒绝权 |
| 评分器冻结 | 评审方 Frozen Core | 原文 §19.3 只检查泄漏，不检查闸门是否被改动 |
| P0 范围与退出条件 | 评审方 P0 | 原文的 P0 不可验收 |
| 任务族与数据划分 | ModularRSI（benchmark-disjoint）+ 评审方三分 | 原文 §18 仅有 evolve / held-out 二分 |
| 经验复用形态 | PluginRSI（插件库）+ Harness²（不累积） | 原文 §15 的 Lesson 卡片无法执行、无法回滚 |

---

## 5. 开工前必须完成的六项修订

- [ ] **1. 将 §10 / §25 的实验改为配对重复**：每任务 k ≥ 3、两臂配对，先完成 δ 标定；MVP 示例中的 `8→5→2` 改为「配对差 + 置信区间」
- [ ] **2. 为 §19 的四项机制补齐可执行判据**：δ 的标定流程、`b_t` 退火式、`β₀/β₁` 选取方式、`g_t(ℓ) ≤ 0` 的窗口定义
- [ ] **3. 引入 Frozen Core**：列出只读清单（`_rule_gate` / `_critic_gate` / `_action_contract_gate` / `_is_deliverable_step` / `_CRITIC_MIN_LEN` / `_DEFAULT_ACTION_PROBES`），Critic 增加「触及 frozen 即拒」
- [ ] **4. 将 δ 标定与最小 Gate 由 P2 提前至 Probe 之前**，并为 P0 增加可验收的退出条件（建议：同一 harness 重复 5 次后 `σ_diff` 已知且 < 8 分）
- [ ] **5. 明确 harness 的载体**：先只改 `prompt.py` 字符串与 `.env` 数值（不动代码结构），YAML 化推至 P2 之后；并说明「命令式逻辑（plan / gate / topology）如何表达为可编辑补丁」
- [ ] **6. 先对齐代码再定 §22 的目录**：至少对齐 `runlog.py` / `agents.py`（三道 gate）/ `plan.py`（8 个 `_REPLAN_*`）/ `graph.py` / `planutil.members`，避免把已有能力当作新建、把代码迁移当作外层封装

---

## 6. 总体判断

**该文档不是「方向错误的方案」，而是「尚不可落地的方案」。**

它准确复述了三篇论文的问题陈述、列全了模块清单、设计了研究实验——这三件事优于多数同类文档，参考文献亦真实准确。但**解法中唯一具备技术含量的部分（如何判定一个改动确实有效）全部停留在口号层面**，且在最关键的位置（实验设计）退化为其自身参考文献所批判的单轨迹做法。

结论：**以原文 §6 / §5 / §21 / §28 为素材，以 RRSI 的判据与 Frozen Core 为骨架，按评审方 P0 顺序实施。**

---

## 附录 · 参考文献核验

| 文档引用 | 核验结果 |
|---|---|
| Harness²：Recursive Agent Harnessing for an Open World | 属实。arXiv:2610.14224，2026-10-09，`github.com/google-research/harness2`（Apache 2.0）。8 个可编辑组件；任务级 + harnessing 级双层递归；无 ground truth 的对比式 Propose–Probe–Compose；WorkBuddy-Bench +11.5 / JobBench +4.5 / LAB partial +11.8 |
| RRSI：Regularized Recursive Self-Improvement of Agent Harnesses | 属实。arXiv:2609.24972v3 |
| ModularRSI：Modular and Generalizable Recursive Harness Self-Improvement | 属实。arXiv:2609.14857（v1 2026-09-14 / v2 10-09）。5 模块；benchmark-disjoint 的 2,000 个演化任务；TB2.0 47.57→52.43，SWE-bench Verified 73.40→76.45 |
| PluginRSI：Reusable Plugin-based Harness Evolution | 属实（标题与正式题略有出入，正式题为 *Recursive Improvement of Agent Harnesses with Reusable Plugins*）。arXiv:2609.32423，2026-09-26，`github.com/syr-cn/PluginRSI` |
| 「Recursive Harness Self-Improvement（RHI）」 | 未检索到对应论文，更像一类方法的泛称。引用时应替换为具体论文，避免评审环节被质疑 |
