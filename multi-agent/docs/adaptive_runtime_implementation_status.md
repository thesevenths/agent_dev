# Adaptive Runtime 实施对照与验收记录

> 依据：`docs/adaptive_agent_runtime_final_plan.md`（V3 定稿）
> 代码：`adaptive_runtime/`（新增包，28 个模块）
> 原则：**只新增，不改生产逻辑**（`agents.py` / `plan.py` / `planutil.py` / `llm.py` / `runlog.py` / `graph.py` 无逻辑改动，由 Frozen Core 双 hash 机器校验）
> 说明：另有一轮全仓库措辞整理，仅涉及注释与文档串，不含任何逻辑变更；整理后 `verify` 仍为 `Frozen Core 完整`。

---

## 0. 结论摘要

| 项目 | 状态 |
|---|---|
| 新增包 | `adaptive_runtime/`，28 个模块 |
| 生产文件逻辑改动 | **0 处**（`run_adaptive verify` → `Frozen Core 完整`） |
| 单元 / 集成测试 | **200 个，全部通过** |
| P0 强制验收（自检） | **9/9 通过**（含 2 个阴性对照、1 个阳性对照、1 个敏感性对照） |
| 真实 run 端到端 | 通过；token 计量 `estimated_ratio = 0.0`（全部为真实 usage） |
| 方案章节覆盖 | §2–§18 全覆盖；§13.5 按方案要求实现**弱化版**并显式标注边界 |
| L2 提示层 | 已接通（`PROMPT_PATCHES` 4 条 + `apply_prompt_patch`），`ADAPTIVE_MAX_EDIT_LEVEL=2` 打开即产出 L2 候选 |
| 反刷分 | 报告最小体积 + 结构分 + `plan_undercut`（禁止通过削减计划提分） |
| 多轮 benchmark | `benchmark.py`；输出 ΔQ 序列 + holdout 平稳性 |
| 任务集分层 | `bench_tasks.py`：evolve / ID held-out / OOD 三层，哈希划分，互斥且稳定 |
| 外部分布外基准 | **未接入**（`external_status()` 全部 `available=False`），尚不可对外声称 |

---

## 1. 逐功能点对照

### §3 不变量

| 功能点 | 代码 | 验证 |
|---|---|---|
| 改评分器 / 协议 / 预算 = 越界 | `frozen.py`（`FROZEN_SYMBOL_FILES` 19 个符号 + `FROZEN_ENV` 8 项） | `selftest:frozen_files_intact` |
| 符号级冻结（目录级无效） | `frozen.symbol_hashes()` 以 AST 抽取单个函数体 | `test_frozen.py` |
| 双 hash（文件侧 + 解析后 env 侧） | `frozen.verify()` → `Violation` | `selftest:frozen_env_detects_override` |
| Gate 非补偿 | `gate.gate_a` / `gate.gate_b` 顺序短路 | `test_gate.py` |
| ΔB 不进入 Gate | `gate.py` 全文不含行为指标；ΔB 仅出现在 `comparator` 与报告中 | 代码检索可验 |

### §4 四级版本模型

| 功能点 | 代码 | 产物 |
|---|---|---|
| **Base**（git tag + env 快照，仅人工 promote） | `version.py`：`ensure_initial` / `current` / `promote` / `history` | `experiments/base_state.json`、`base_history.jsonl` |
| promote 必须人工（拒绝 system / auto / 空操作者） | `version.promote(operator=...)` | `test_evolution_version.py` |
| **Task Harness** = Base + ≤3 条经验 | `harness.assemble` / `harness.write` | 每轮 `task_harness.json` |
| **Incumbent**（轮末丢弃） | `version.incarnate` | 每轮 `incumbents.json` |
| **Candidate** | `proposer.Candidate` | 每轮 `candidate.json` |
| 每 10 轮提示人工评审 | `version.promote_due` | `status` 输出 |
| 打 tag 仅提示，不自动执行 | `version.tag_hint` | `base show` |

### §6 观测层

| 功能点 | 代码 | 备注 |
|---|---|---|
| Token 计量，不改 `llm.py` | `token_probe.py` 以 monkeypatch 覆盖 `_generate/_agenerate/_stream/_astream` | 每个方法仅记录「最近实现类」，避免重复计数 |
| 流式 usage（真实数字） | 追加 `_should_stream_usage` 补丁注入 `stream_options.include_usage` | 见 §2 缺陷 1 |
| 后端不支持时降级而非失败 | `_is_stream_options_rejection` + 回退后原样重试一次 | `test_token_probe.py` |
| 按 agent 归因 | `_patch_agent_context`（`on_chat_model_start` 旁路） | 见 §2 缺陷 2 |
| estimated 占比必须上报 | `totals()["estimated_ratio"]`、`stream_usage_injection` | 真实 run：0.0 |
| Trace 结构化 | `trace_export.py`（12 条正则覆盖既有 `runlog` 文本） | 不新增日志字段 |
| Trace 可回放（L0） | `evaluator.score_trace(mode="replay")` + `score --live` | `test_evaluator.py` |
| 产物归属窗口 | `scan_artifacts(since, until, window_max=7200)` + `artifacts_unavailable` | 防止 mtime 被复制后误挂全部产物 |

### §7 评价层

| 功能点 | 代码 |
|---|---|
| `Q = 100×(0.50A + 0.30E + 0.20C)` | `evaluator.score_trace`（权重可配 + 归一化） |
| A = 契约达成率 | `trace.contract_rate` |
| E = 带来源关键数字占比 | `evaluator.evidence_rate`（段落级 + 8 条声明上限） |
| C = 完整性 | `evaluator.completeness`（截断 + 兜底 + 缺失产物） |
| live 无报告 = 失败信号（E=0，不重归一化） | `score_trace` |
| L2 rubric（5 维，双向打分） | `evaluator.score_l2`（`ADAPTIVE_L2=1` 时启用） |
| **判分口径版本** `judge_version` | `evaluator.judge_version()`（权重或口径变更即失效） |

### §8 噪声与统计

| 功能点 | 代码 |
|---|---|
| 配对 t + 噪声调整门限 `δ` | `stats.paired_stats` → `delta_crit` |
| 纯 stdlib（无 scipy 依赖） | `stats.t_pvalue` / `betainc` / `_norm_ppf` 自行实现 |
| 反推 n（80% power） | `stats.required_n`、`calibrate.suggest_n` |
| 标定协议 | `calibrate.py` + `calibrate` 子命令 → `calibration.json` |
| 两级 Racing（Stage1 止损 → Stage2 补齐） | `probe.probe_candidate` + `_stop_loss` |

### §9 判定层

| 功能点 | 代码 |
|---|---|
| Gate A 五项（frozen / leakage / inert / budget / edit） | `gate.gate_a` |
| Gate B 顺序短路（incomplete → contract floor → judge stability → regression rate → paired gain → cost rule） | `gate.gate_b` |
| A_FLOOR 相对化（`min(a_floor, A_base) − a_tol`） | `gate_b`（见 §2 缺陷 3） |
| 「每个任务均略差」→ REGRESSION | `gate_b` |
| 成本规则 `ΔC ≤ β0 + β1·ΔQ` | `gate_b` |
| Reason 枚举 11 项 + 逐项 gate_trace | `gate.explain` |

### §10 演化层

| 功能点 | 代码 |
|---|---|
| 真实参数搜索空间（7 个真实 env 参数） | `proposer.SEARCH_SPACE` + `CURRENT_DEFAULTS` |
| 惰性机制识别（`AGENT_CRITIC_MIN_LEN` 等） | `gate.INERT_TARGETS` |
| 三级演化 + edit budget | `settings.max_edit_level` / `edit_budget_p0` + `gate_a` |
| Hypothesis Ledger（失败假设不可原样重提） | `ledger.py`（`is_duplicate` / `inconclusive_streak` / 连续 3 次提升为 rejected） |
| **Exploration（防搜索坍缩）** | `evolution.py`：`should_explore` / `exploration_pairs` / `evolve_plan`，作用单位为 **(失败模式 × 层)** |
| **Candidate Critic（前置）** | `frozen.verify` + `proposer.leakage_hits` + `INERT_TARGETS` + `gate_a` 成本 / 编辑预算 |
| **Pruning（淘汰无效组件）** | `evolution.py`：连续 10 次 `WITHIN_NOISE` → `dormant` 30 天，并在报告中输出「该组件无效」结论 |

### §11 Frozen Core

| 功能点 | 代码 |
|---|---|
| 2 层冻结（符号 + env） | `frozen.build_manifest` / `freeze` / `verify` |
| 运行时校验（每轮 Gate A 前） | `cmd_round` 调用 `frozen.verify()` |
| 范围仅覆盖判分链，不冻结业务代码 | `FROZEN_SYMBOL_FILES`（14+2+2+1 个符号） |

### §12 诊断层

| 功能点 | 代码 |
|---|---|
| 硬失败 5 类 + UNKNOWN | `diagnosis.Taxonomy.HARD` |
| **质量缺陷 2 类** | `diagnosis.Taxonomy.QUALITY`（`EVIDENCE_FAILURE` / `COMPLETENESS_FAILURE`） |
| 仅由「可判定字段」触发 | `diagnosis.classify(trace, score=...)` |
| 质量判定前提：必须真的读到成品文本 | `quality_findings()` 以 `detail["report"]` 为统一前提 |
| UNKNOWN > 40% 时告警而非增类 | `settings.unknown_max` + `diagnosis.summarize` 的 `hint` |

质量通道的存在理由：只看硬失败时，「每一步都 PASS、正常收尾、报告已落盘」的 run
会被判成「无失败信号」，闭环遂认为无改进空间。而实测一组 run 的 A/C 已接近满分、
27 分差距全部落在 E（268 个关键数字仅 27 个带来源）—— 最大失分项恰恰无法被自检发现。
阈值外置：`ADAPTIVE_EVIDENCE_FLOOR` / `ADAPTIVE_COMPLETENESS_FLOOR` /
`ADAPTIVE_EVIDENCE_MIN_CLAIMS`。

### §13 经验层

| 功能点 | 代码 |
|---|---|
| Experience schema | `experience.Experience` / `Store.create` |
| `confidence = 0.4·样本 + 0.3·稳健 + 0.3·显著` | `experience.confidence` |
| 检索（按 problem fingerprint） | `Store.retrieve` / `lessons_for` |
| promote 候选（≥2 任务族验证） | `Store.promotion_candidates` + `base promote` |
| **Regression 集（P1）** | `regression.py`：`add_from_accepted` / `check` / `select_for_probe` / `coverage`，**不参与假设生成** |
| **成功 / 失败轨迹对比（P1 弱化版）** | `trajectory.py`：`contrast` / `cluster_failures` / `by_agent` + 强制 `CAVEAT` |

### §14 预算治理

| 功能点 | 代码 |
|---|---|
| 单轮 / 单候选硬上限 | `budget.Budget.from_settings`（`can_spend` / `spend`） |
| 超预算即停，并在判定中显式记 COST_CAP | `probe._one` + `BudgetExceeded` |
| 预估与降级 n | `budget.estimate_round_tokens` / `fits` / `degraded_n` |

### §15–§18 路线图 / 验收 / 目录

| 功能点 | 代码 |
|---|---|
| 失败候选可回滚 + quarantine | `quarantine.py`（沙箱删除 = 回滚；`quarantine/*.json` = 证据归档） |
| accepted 候选携带 base_commit + patch_hash + judge_version | `Candidate.patch_hash` + `run_adaptive.cmd_round` + `experience` |
| 拒绝原因可统计 | `ledger.stats()` / `quarantine.summary()['by_reason']` / `decision.json` |
| 单一 CLI 覆盖全流程 | `run_adaptive.py`（15 个顶层子命令，每个子命令含参数说明） |

---

## 2. 本轮修复的实测缺陷

以下缺陷均在真实运行或自检过程中暴露，而非代码审阅阶段发现。

| # | 现象 | 根因 | 修复方式 | 验证 |
|---|---|---|---|---|
| 1 | 流式调用 `estimated_ratio = 1.0`，全部退化为字符估算 | `llm.py` 未设置 `stream_usage`，`_should_stream_usage()` 返回 `False`，请求不带 `stream_options.include_usage`，vLLM 因此不回流式 usage | 补 `_should_stream_usage` 补丁（仍不改 `llm.py`）；后端返回 400 时自动回退注入并原样重试一次 | 4 条路径（`_generate/_agenerate/_stream/_astream`）均取得真实 usage，数值一致（23/9/32），无重复计数 |
| 2 | 全部 token 归因至 `unknown`，修复后又归因至 `model` | ① `langchain_core` 调用 `_stream` 时不传 `run_manager`；② 成员 agent 为子图，子图内部节点名即为 `model` | 通过 `CallbackManager.on_chat_model_start` 旁路捕获 metadata 到 contextvar；取 `langgraph_checkpoint_ns` 的**最外层**段而非 `langgraph_node` | 真实 run 归因为 `['supervisor', 'chat_agent']`；单测覆盖 6 种 metadata 形状 |
| 3 | 绝对阈值 `A_FLOOR=0.85` 会把 `A 0.60→0.80` 这类目标改进判为不合格 | 绝对门槛忽略了 base 自身水平 | `floor = min(a_floor, A_base) − a_tol` | 阳性对照 `good` 可被 ACCEPT |
| 4 | 「每个任务均略差」被归入 `WITHIN_NOISE` | 伪代码先比较噪声、后比较方向 | 先判 `dQ < -thr` → `REGRESSION` | 阴性对照 `worse` → `contract_floor` / `regression` |
| 5 | live 模式「无报告」被重归一化权重，使劣化候选得分偏高 | 将「失败信号」误作「数据缺失」 | live 固定 `E=0.0`；仅 replay + `artifacts_unavailable` 时归一化 | `worse` 得分 38.1 vs base 53.33 |
| 6 | Windows 下沙箱残留，`test_no_sandbox_leak` 持续失败 | ① `rmtree` 遇只读文件失败；② 测试用 `rglob` 扫描**历史全部** selftest 目录 | `_rmtree` 去除只读位 + 4 次重试；测试改为仅检查本次 `report["root"]` | 9/9 自检 + 0 残留 |

---

## 3. 暂未启用的能力（设计取舍）

| 方案位置 | 内容 | 暂不启用的原因 |
|---|---|---|
| §15 P1 | L2 提示层 / L3 结构层演化 | 由 `ADAPTIVE_MAX_EDIT_LEVEL`（默认 1）控制；P0 需先证明度量可靠。L2 代码路径已接通并验证可产出候选（4 条补丁） |
| §15 P1 | L2 rubric 作为准入 | 由 `ADAPTIVE_L2`（默认关）控制；方案明确其仅作**次要证据**，不进入 P0 判定 |
| §13.5 严格版 | 同一任务 5–8 次逐步对照 | 方案定为 P2，且要求有成有败的重复执行数据；当前仅有历史单次日志 |
| §10.5 | 「+80 行仅为 +1 分」复杂度判据 | P0 仅改 env（`patch.kind == "env"`），不存在行数维度；L2 / L3 启用时再引入 |
| §12 | 扩充到 6 类以上 taxonomy | 方案要求「先补埋点字段，再增类别」，否则只会提高 UNKNOWN 占比、掩盖真实问题 |

---

## 4. 复现命令

```bash
python -m adaptive_runtime.run_adaptive selftest      # 9/9，含阴性 / 阳性对照
python -m adaptive_runtime.run_adaptive verify        # 生产文件零改动
python -m adaptive_runtime.run_adaptive status        # 阈值 + Base + 台账 + 演化状态
python -m adaptive_runtime.run_adaptive evolve        # exploration / pruning 判定
python -m adaptive_runtime.run_adaptive base show     # base/vN + env 快照
python -m adaptive_runtime.run_adaptive regression init
python -m adaptive_runtime.run_adaptive trajectory    # 成功 / 失败轨迹分布对比
python -m adaptive_runtime.run_adaptive round --fake --case good   # 模拟轮次，全链路
python -m adaptive_runtime.run_adaptive round --tasks-file experiments/tasks/replay.json  # 真实轮次
python -m unittest discover -s adaptive_runtime/tests -t .          # 200 tests
```

> 注：`.env` 中的 `AGENT_*` 阈值一律不改；演化仅通过**子进程环境变量覆盖**生效，因此实验双臂共享同一份代码。
