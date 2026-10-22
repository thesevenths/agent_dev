# 评审：Adaptive Agent Runtime V2 实施规格（补充意见）

> 评审对象：`docs/adaptive_agent_runtime_v2_implementation_spec.md`（65 节 / 2846 行）
> 前置评审：`docs/reviews/review-01-runtime-rebuild.md`（针对 V1）
> 评审日期：2026-10-22

---

## 0. 评审结论

**V2 已由「研究议程」推进为「可开工的规格」，距可施工仍差一步：可验收性。**

| 维度 | V1 | V2 | 状态 |
|---|---|---|---|
| 实验设计 | 单轨迹 n=1 | 配对 + 重复 + 噪声标定（§16 / §17） | 已修复 |
| 评分器保护 | 无 | Frozen Core + `git diff` + hash（§14 / §15） | 已修复（实现方式存在漏洞，见问题 A） |
| 结构累积 | `merge → main` 无限累积 | Base 不累积、经验单独存储（§29） | 已修复 |
| 改动分级 | 一次只改一个组件 | Parameter → Prompt → Structure 三级（§12） | 新增，且与本仓库高度适配 |
| 多维判定 | 六个维度并列，无裁决机制 | 四层优先级：硬约束 → 主目标 → 次要证据 → 优化（§22） | 新增，为该版最佳设计 |
| 正则化判据 | 仅结论性表述 | 有示例、有 Gate 骨架，但 `C_max / R_max / δ` 仍无定义 | 部分完成 |
| P0 验收 | 无 | 11 条 [✓]，均为「可跑通」，无一条数值判据 | 部分完成 |
| 与本仓库的贴合 | 描述错误 | §5 承认「无法获取源码树」，§6 仍给出不存在的目录树 | 未修复，且错误更具体 |

V1 提出的六个问题，V2 修复三个、部分修复一个、明确承认但未修复两个。当前状态可以开工，但无法判断「是否做对了」。

下文仅讨论尚未修复的部分，以及可直接替换的规格。

---

## 1. 六个剩余的结构性问题（P0 阻塞级）

### A · Frozen Core 按「目录」冻结，在本仓库必然遗漏

**原文**（§15）：

```python
FROZEN_PATHS = {
    "adaptive_runtime/evaluation/",
    "adaptive_runtime/experiments/protocol/",
    "adaptive_runtime/policy/frozen/",
}
```

**为何在本仓库失效**：真正的评分逻辑不在任何独立目录内，而是 `agents.py` 的模块级常量与函数：

| 冻结对象 | 实际位置 | 属于评分而非 harness 的理由 |
|---|---|---|
| `_rule_gate` / `_critic_gate` / `_action_contract_gate` | `agents.py` | 决定「该步是否算完成」 |
| `_is_deliverable_step` + `_DELIVERABLE_VERBS` + `_PATH_EXTS` | `agents.py` | 决定「什么算交付物」 |
| `AGENT_CRITIC_MIN_LEN=4` | `.env` | 判定阈值（当前值等同于无门槛，见问题 J） |
| `AGENT_ACTION_PROBES`（动作契约词表） | `.env` | 决定「发邮件」是否算完成 |
| `AGENT_TRUNCATED_SALVAGE` + `AGENT_SALVAGE_EXTS` | `.env` | 决定「截断是否算完成」 |
| `AGENT_STEP_FAILURES_MAX=20` | `.env` | 决定「何时判整体失败」 |

按目录冻结的结果是上述 6 项全部落入「可编辑」区间，闭环只需改动 `AGENT_CRITIC_MIN_LEN` 即可提高通过率。

**修订建议**：将路径冻结替换为**符号冻结**，并采用双 hash 断言：

```python
FROZEN_SYMBOLS = {          # 以 文件:符号 为单位，而非目录
  "agents.py": ["_rule_gate", "_critic_gate", "_action_contract_gate",
                "_is_deliverable_step", "_DELIVERABLE_VERBS", "_PATH_EXTS",
                "_DEFAULT_ACTION_PROBES", "_TRUNCATED_SALVAGE"],
  "planutil.py": ["members", "options"],       # 拓扑属于评分口径
  "runlog.py": ["log_event", "timing_summary"], # trace 完整性
}
FROZEN_ENV = ["AGENT_CRITIC_MIN_LEN", "AGENT_ACTION_PROBES", "AGENT_CRITIC_RETRIES",
              "AGENT_STEP_FAILURES_MAX", "AGENT_TRUNCATED_SALVAGE", "AGENT_SALVAGE_EXTS"]

# 两道断言缺一不可
assert hash_files(FROZEN_SYMBOL_FILES) == frozen_file_hash      # 防文件被改
assert hash_resolved_env(FROZEN_ENV) == frozen_env_hash         # 防 .env 或环境变量覆盖
```

第二道断言不可省略：仅对文件做 hash 不足以防护——全部参数均可被 `.env` 或环境变量覆盖，修改 `.env` 即可在不改动代码的前提下改变评分口径。

---

### B · §16 的算例与 §17 的原则互相矛盾，且 n=5 不足

**原文**（§16）：

```
Base:     72 69 71 70 73  → mean=71, std=1.58
Candidate:75 76 74 77 75  → mean=75.4
Δ = 4.4  → 再判断是否超过 evaluation noise
```

**三处问题**：

1. **属非配对比较**（两组各 5 次、比较均值），而 §17 明确要求配对。配对设计应计算 `Δ_i` 的均值与标准误，而非两组各自的标准差。
2. **使用了 std 而非 SE**。σ = 1.58 是单次观测的波动，判定应使用 `SE = σ/√n`。
3. **「再判断是否超过」未给出判据**——4.4 与 1.58 如何比较，没有规则。

**正确判据**（配对 t）：

```
Δ_i = Candidate_i − Base_i          # 同一任务两臂
δ   = t_{0.975, n−1} · sd(Δ)/√n
ACCEPT if mean(Δ) > δ
```

**n=5 在本场景通常不足**。给定需检出 +4 分：

| 判分噪声 sd(Δ) | 所需配对样本量 n | 该 n 下的 δ |
|---|---|---|
| 4 分（优质 judge + 结构化锚点） | ≈ 4 | 3.3 |
| 7 分（rubric 化 + 双向呈现） | ≈ 13 | 4.0 |
| 12 分（裸 LLM 打分） | ≈ 36 | 4.0 |

反向固定 n 观察 δ（`t·σ/√n`）：

| sd(Δ) \ n | 5 | 10 | 25 | 50 |
|---|---|---|---|---|
| 4 | 5.0 | 2.9 | 1.7 | 1.1 |
| 7 | 8.7 | 5.0 | 2.9 | 2.0 |
| 12 | 14.9 | 8.6 | 5.0 | 3.4 |

**解读**：`sd(Δ) = 12, n = 5` 时 δ ≈ 14.9，意味着任何小于 15 分的改进都无法与噪声区分。V2 §47 的示例为 `+7.2`——在裸 LLM 打分且仅 5 次重复的条件下，该增益落在噪声带内，应被正确拒绝。

因此 §16 的重点不是「给出示例」，而是**先测 sd(Δ)，再反推 n**。该步骤缺失时，「5 paired runs」这一取值缺乏依据。

---

### C · §21 的「ΔQ **或** ΔB」是 reward hacking 的直接入口，且与 §22 冲突

**原文**（§21）：

```
ACCEPT IF
EvidenceGain >= δ_noise AND CostDelta <= C_max AND Regression <= R_max ...
```

其中 `EvidenceGain` 的定义包含 `ΔB`（行为指标），且为「或」关系：**ΔQ 与 ΔB 任一过门即通过**。

**风险**：研报场景下 `ΔB` 极易被优化，且属加法型指标：

```
sources 4 → 7             # 多列三个来源
verified_claims 8 → 13    # 多写五条「已验证」结论
```

上述数值上升时，报告质量可能毫无变化，甚至下降（堆砌来源、稀释结论）。

**同时，§22 已给出正确处理方式**：四层优先级中 `Primary Objective = Verifier / Task Quality`，`Secondary Evidence = Behavior / Robustness / Transfer`。但 §21 的判定式未落实该层级——两节冲突，实现时会以 §21（伪代码形态）为准，四层优先级设计随之失效。

**修订建议**：将 ΔB 由「通过条件」降级为「归因证据」：

```
硬约束（任一失败即拒，不可补偿）
  ① 触及 Frozen → FROZEN_CORE_VIOLATION
  ② 泄漏检测命中 → BENCHMARK_LEAKAGE
  ③ 存在缺失/超时的 run → INCOMPLETE_EVIDENCE      # 缺失即失败，不丢样本
  ④ A_cand < A_FLOOR(=85) → CONTRACT_FLOOR        # 契约达成率为硬门槛

主目标（唯一通过依据）
  ⑤ mean(ΔQ) > δ_Q  ∧  ΔC ≤ β0 + β1·mean(ΔQ)      # β0=0.05, β1=1.5 起步
  ⑥ regression_rate ≤ R_MAX(=0.15)

若 ⑤ 未过门，仅以下三种情形可准入（ΔB 不在此列）
  ⑦ ΔC < 0（同分更省）／ ⑧ 新增结构组件且从未被接受 ／ ⑨ 消除一类 hard failure

ΔB 的输出位置：仅用于报告中的「为何可能有效」，不参与通过判定
```

---

### D · P0 的 11 条验收标准，无一能证明「尺子准确」

**原文**（§40）：11 个 `[✓]` 全部为「能跑通」——有 Trace、能 replay、能生成 hypothesis、能 accept/reject 等。

**缺失的关键一项**：**阴性对照（negative control）**。

> 将一个「零改动」候选（仅改一行注释或增加一个空行）送入完整流水线，系统必须输出「WITHIN_NOISE，拒绝」。

若输出 ACCEPT，则说明 δ 计算有误、judge 存在偏差，或 base 与 candidate 的执行路径不对称——后续所有实验结论均不可信。这是评估系统上线前唯一必须通过的测试，V2 中完全缺失。

建议将 P0 验收改为「能力 + 数值 + 阴性对照」三段（可替换规格见 §3 第 ③ 项）。

---

### E · `Current Harness` 与 `Base Harness` 的关系未定义

V2 中并存两个概念：

- §2 架构图：`Experience Retrieval → Current Harness → 执行`
- §29：`Base Harness` 不累积，每个任务从 base 出发，经验仅存库

**两者关系未说明**。实现时必然三选一，选错即退回原有问题：

| 选项 | 含义 | 后果 |
|---|---|---|
| ① Current = Base | 始终从 base 出发，经验通过检索现场注入 | 与 Harness² 一致，但「已验证的改进」从不生效，只提建议不落地 |
| ② Current = 最后一个 accepted 版本 | 累积 | 漂移不可控（§29 反对的正是此做法） |
| ③ Current = Base + 当前任务检索到的经验组装 | 每次现场合成「临时 harness」 | 要求 harness 可组合（需 P3 插件化能力） |

**建议明确为「① + 定期人工合并」**：日常执行从 base + 检索经验组装；每 N 轮将由跨任务族验证的 ≤3 条经验人工合并进 base，形成新 base 并冻结旧版。该方案同时避免漂移并保证改进可落地。

---

### F · §6 的目录树对应不存在的文件，照此实施第一步即为目录搬迁

§5 已声明：「公开仓库页面无法稳定提供完整源码树……**不应凭空假设文件名**」。随后 §6 给出了如下目录树：

```
agents/planner.py · agents/executor.py · agents/reviewer.py
runtime/orchestrator.py
adaptive_runtime/...
```

**这些文件均不存在**。真实结构如下：

| 文档假设 | 实际结构 |
|---|---|
| `agents/planner.py` | `plan.py`（replan 判定）+ `supervisor.py`（规划 / 路由 / 仲裁） |
| `agents/executor.py` | `agents.py`（6 个 node 集中于一个文件，入口 `create_nodes()`） |
| `agents/reviewer.py` | 不存在独立节点；评审为 `agents.py` 中每步内部的三道 gate |
| `runtime/orchestrator.py` | `graph.py`（`build_graph_with_memory`）+ `supervisor.py` |
| `config.py`（§45 的候选修改目标） | 存在，但大部分参数位于 `.env` 与模块常量中，`config.py` 仅承载 LLM 与路径配置 |

**这是 V2 最实际的风险**：「不重写」的承诺 + 一份需搬迁的目录树，实施时极易先执行 `mkdir agents/` 再 `git mv`。而这恰是最不应做的事——现有 6 个 node 共享同一套 `_run_tool` / gate / handoff 逻辑，拆分文件会引入大量循环导入风险。

**修订建议**：将 §6 替换为「新增文件位置」的最小清单——仅新增 `adaptive_runtime/` 一个包，**不改动现有文件**：

```
multi-agent/
├── adaptive_runtime/          # 全新独立包，不 import 现有模块的内部实现
│   ├── trace_export.py        # 从 runlog 产物旁路导出结构化 trace（只读）
│   ├── evaluator.py           # 分层打分
│   ├── probe.py               # worktree 执行与采集
│   ├── comparator.py          # 配对统计与 δ
│   ├── gate.py                # 非补偿准入
│   ├── diagnosis.py
│   ├── proposer.py
│   └── experience.py
└── （现有 supervisor.py / agents.py / plan.py / graph.py 一行不改）
```

---

## 2. 四个落地缺口

### G · §35 的「成功 / 失败轨迹对比」在现有数据上不可执行

ModularRSI 的对比前提是**同一任务**同时存在成功与失败轨迹。本仓库的 95 条 run 对应 95 个不同提问，缺少「同一任务多次执行」的数据。

**因此 §35 当前无法执行**。可选两条路径：

- **弱化版（P1 可落地）**：跨任务聚合——将失败 run 按 `(kind, agent, step_no)` 聚类，与成功 run 的对应步骤做**分布对比**（例如「失败的 CodeAgent 平均 42 次工具往返，成功的为 18 次」）。该方式不要求同任务多次执行。
- **完整版（P2）**：先将同一任务执行 5–8 次，取得有成有败的样本，再做轨迹对比。

建议先实现弱化版，并明确标注「与 ModularRSI 的严格定义存在差异」。

### H · 12 类 taxonomy 与现有 `kind=` 无映射，历史 run 无法复用

现有日志中可判定的事实分类仅三类：`kind=recursion (28) / truncated (7) / exception (1)`。V2 的 taxonomy 含 12 类，其中 9 类在当前数据中无法自动判定——缺少对应字段。

若直接实现，Diagnosis 将退化为「LLM 每次自由生成」，正是 §9 试图避免的情形。

**建议**：先仅启用可判定的 5 类，其余输出 `UNKNOWN` 并统计占比。

| 现有 `kind=` | taxonomy | 判定规则（基于现有字段） |
|---|---|---|
| `recursion` | `COMPLETION_FAILURE` | 有产物落盘且过 gate → 预算耗尽；无产物 → `EXECUTION_FAILURE` |
| `truncated` | `COMPLETION_FAILURE` | 同上，按 artifact 是否可救分叉 |
| `exception` | `EXECUTION_FAILURE` | 再按异常类型分 `TOOL_USAGE_FAILURE` / 基础设施 |
| gate 拦截（`[critic+summary]` 有记录） | `REVIEW_FAILURE` | 按 `_critic_gate` 的 reason 子串分类 |
| replan 次数 ≥ `AGENT_REPLAN_NOOP_MAX` | `PLANNING_FAILURE` | 从 supervisor 日志计数 |
| 其余 | `UNKNOWN` | 先统计占比；> 40% 说明 taxonomy 超前于埋点，应先补埋点而非增加类别 |

原则：**taxonomy 的规模应由「可判定字段」约束，而非由设想约束。**

### I · token 计量已有现成钩子但未指明挂载点，且缺少成本预算

§8 要求 trace 记录 `tokens / cost`，§23 的成本规则依赖该数据。当前存在两个缺口：

1. **落点未指定**。`llm.py` 使用 `langchain_openai.ChatOpenAI`，且已具备 `CustomContextMiddleware.wrap_model_call`（`llm.py:173`）这一钩子，token 计量挂载于此可实现零回归（使用 `response.usage_metadata` / `response_metadata`）。这是 Day 1 的隐性前置工作，若未写明，会在 Day 6 才发现成本规则无法计算。
2. **缺少成本预算**。一轮 probe 为双份执行：

```
|D_probe| = 13 配对 × 2 臂 × 单任务 20 万 token ≈ 520 万 token ≈ 2 小时
20 轮 ≈ 1 亿 token
```

该量级必须写入 P0 验收，否则首日即可能超支。建议设定「单轮 ≤ 5 M token，超限中止」。

### J · judge 异构未作要求（`AGENT_CRITIC_LLM` 已存在但未被约束）

§18 设有 `Level 2: Benchmark Judge`，但未要求 judge 不得与被评 / 提议模型同源。`AGENT_CRITIC_LLM` 是现成开关，可要求 proposer 与 judge 使用不同配置。这是抑制 self-preference 的最低成本手段。

---

## 3. 可直接替换的五段规格

### ① 真实的参数搜索空间（替换 §12 Level 1 示例清单）

V2 列出的 `MAX_REPLAN / MAX_RETRIES / REVIEW_THRESHOLD / TIMEOUT / TOOL_BUDGET / MAX_STEPS` 均为虚构名称。真实清单如下：

| 层 | 参数（真实名） | 现值 | 建议搜索空间 | 分级 |
|---|---|---|---|---|
| 可搜索 | `AGENT_MAX_ITERATIONS_CODE` | 60 | {60, 80, 100} | L1（安全，仅更慢 / 更贵） |
| 可搜索 | `AGENT_MAX_ITERATIONS_CRAWLER` | 15（默认） | {15, 25, 40} | L1 |
| 可搜索 | `AGENT_PARSE_TRIES` | 3 | {2, 3, 5} | L1 |
| 可搜索 | `AGENT_REPLAN_ARTIFACT_MAX` | 3 | {2, 3, 5} | L1（弱） |
| 结构层 | `AGENT_REPLAN_NOOP_MAX` / `AGENT_REPLAN_OBS_WINDOW` / `AGENT_PLAN_GROW_MAX` | 2 / 20 / 1 | P2 开启 | L3 |
| 结构层 | `AGENT_CTX_*` / `AGENT_SUMMARIZE_*` / `AGENT_MEMORY_*` | — | P2 开启 | L3 |
| 冻结 | `AGENT_CRITIC_MIN_LEN` | 4 | 不搜索 | — |
| 冻结 | `AGENT_ACTION_PROBES` / `AGENT_CRITIC_RETRIES` / `AGENT_STEP_FAILURES_MAX` / `AGENT_TRUNCATED_SALVAGE` | — | 不搜索 | — |

**两个值得注意的真实现状**：

1. `AGENT_CRITIC_MIN_LEN=4`——该参数看似「质量阈值」，实际取值为 4 个字符，等同于无门槛。这同时说明两件事：其一，它属于**惰性机制（inert machinery）**，任何「调整它」的候选都应被 Critic 直接拒绝；其二，若将其纳入搜索空间，proposer 必然会尝试调整（其命名最接近「质量旋钮」），而收益恒为 0——这构成 Leakage / Inert Critic 的第一个真实测试用例。
2. `AGENT_MAX_ITERATIONS_CODE` 已达全局值 15 的 4 倍，仍产生 28 次 recursion 失败，说明瓶颈不在该数值上。`{60, 80, 100}` 只应作为**证伪候选**进入 probe（用于彻底否定「提高预算」这一假设并写入负证据），不应期待其有效。

### ② Gate 伪代码（替换 §21）

```python
from enum import Enum
class Reason(Enum):        # 拒绝原因必须是枚举，不得为自由文本
    FROZEN_CORE_VIOLATION = 1
    BENCHMARK_LEAKAGE = 2
    INCOMPLETE_EVIDENCE = 3
    CONTRACT_FLOOR = 4
    WITHIN_NOISE = 5
    COST_RULE = 6
    REGRESSION = 7

def admit(cand, ev, cfg):
    if touches_frozen(cand):                    return Reason.FROZEN_CORE_VIOLATION
    if leakage(cand.diff):                      return Reason.BENCHMARK_LEAKAGE
    if ev.missing_or_timeout_runs > 0:          return Reason.INCOMPLETE_EVIDENCE
    if ev.A_cand < cfg.A_FLOOR:                 return Reason.CONTRACT_FLOOR
    dQ, se = paired_delta(ev.Q)
    if dQ <= cfg.t_crit(ev.n) * se:
        if ev.dC < -cfg.C_SAVE_MIN:  return ACCEPT("CHEAPER_EQUAL")
        if cand.adds_novel_component: return ACCEPT("NOVELTY")
        if ev.hard_failure_removed:   return ACCEPT("HARD_FAILURE_REMOVED")
        return Reason.WITHIN_NOISE
    if ev.regression_rate > cfg.R_MAX:          return Reason.REGRESSION
    if ev.dC > cfg.BETA0 + cfg.BETA1 * dQ:      return Reason.COST_RULE
    return ACCEPT("VALIDATED")
```

要点：`ev.dB` 不出现在该函数中。行为指标仅进入报告，不参与 gate 判定。

### ③ P0 验收标准（替换 §40）

```
能力（保留原有 11 条）
数值
  [✓] 同一任务重复 5 次，sd(Δ) 已实测并写入 experiments/calibration.json
  [✓] comparator 能输出 mean(Δ) / SE / t / δ / 是否过门
  [✓] 一次完整 evolve 的 token 成本已记录，且 ≤ 预设上限（建议 5 M/轮）
  [✓] 每个 accepted candidate 具备可复现三元组：base_commit + patch_hash + judge_version
阴性对照（新增，优先级最高）
  [✓] 零改动候选（仅改注释）必须被判 WITHIN_NOISE
  [✓] 已知变差的候选（如 AGENT_MAX_ITERATIONS_CODE 60→15）必须被判 REGRESSION 或 WITHIN_NOISE
原因可统计
  [✓] 所有拒绝输出 Reason 枚举，可计算「10 个候选里 6 个 WITHIN_NOISE」这类分布
```

倒数第二条是**第二个阴性对照**：一个「故意变差」的候选若未被判负，说明比较逻辑符号反转或 base / candidate 接反——该类缺陷在探针系统中高频出现且难以靠人工检视发现。

### ④ 实验目录结构（替换 §63，补齐元数据）

当前版本仅保存 `trace.json / comparison.json / decision.json`，缺少复现所需的指纹：

```
runs/<task_id>/<round_id>/
├── meta.json          # harness_hash, base_commit, judge_model, cfg_snapshot, started_at
├── base/{trace.jsonl, score.json, artifacts/}
├── candidates/<cid>/{patch.diff, worktree_head, trace.jsonl, score.json, artifacts/}
├── comparison.json    # per-task Δ 明细 + mean/SE/t/δ + regression_rate
├── decision.json      # {decision, reason(枚举), gate_trace:[每条判据的布尔]}
└── experience.jsonl
```

`gate_trace` 为关键字段：记录每条判据的逐项布尔值，而非仅记录最终结论。缺少该字段时，事后无法回答「本轮为何被拒」。

### ⑤ 修订版第一周计划（替换 §44）

原版在 Day 2 即要求「冻结 Evaluator」，但此时 Evaluator 尚不存在。正确顺序是先构建度量、再标定、最后冻结。

| Day | 工作项 | 产出 / 完成标志 |
|---|---|---|
| 1 | token 埋点（`llm.py:173` 的 `wrap_model_call`）+ trace 旁路导出 | 任意一次 run 可导出结构化 jsonl，且 token 数非 0 |
| 2 | evaluator v0：L1（gate 返回值 / step-failure / replan / 产物）/ L2（研报结构要素） | 任意一次历史 run 可打出 0–100 的 A |
| 3 | **标定**：base harness 在同一任务执行 5 次，测 sd(Δ) | `calibration.json` 中包含真实数值 |
| 4 | comparator + 两个阴性对照（零改动 / 故意变差） | 两个对照判定正确；否则回退至 Day 2 |
| 5–6 | worktree probe 跑通，仅实现 parameter 层两个候选 | 打通一条完整 `run_adaptive.py` 链路并产出报告 |
| 7 | experience.jsonl + Reason 枚举统计 | 能回答「10 个候选为何只通过 1 个」 |

**冻结 Evaluator 的正确时机是 Day 4 通过阴性对照之后，而非 Day 2。** 冻结一把尚未验证的尺子，等同于将错误固化。

---

## 4. 总体判断

V2 的骨架已具备开工条件。**剩余工作不是「增加模块」，而是「填入数值」**：将 `C_max / R_max / δ / β0 / β1 / n` 由占位符替换为实测值，并以**两个阴性对照**证明该度量至少不会把「零改动」与「故意变差」判为改进。

若只做一件事：**在编写任何 proposer 之前，先完成 Day 3–4 的标定与阴性对照。** 这两步完成后，后续模块才有意义；跳过它们，V2 会产出一批「看似有证据」的结论。
