# ARR 1.0 — Autonomous Research Runtime 架构与验收记录

> 本文记录 **Package D / E / F** 的实际落地（不是设计提案）。所有数字来自本机
> 真实运行产物，路径与 SHA-256 可在 `experiments/arr_1_0*/` 下复核。
> Git 分支：`feature/arr-0.1-domain-registry`（不 merge main、不开 PR）。

## 一、六层架构与复用关系

| 层 | 现有基础（**未重写**） | ARR 1.0 新增 |
|---|---|---|
| L1 Trust & Evidence | `EvidenceStore`、`HypothesisCritic`、`Ledger`、候选配置 SHA-256 校验 | `research_quality.py`：领域可注入的 `ResearchQualityEvaluator`（E/R/G/D/C + 非补偿硬约束）；`QUALITY_FROZEN_FIELDS` |
| L2 Experiment Infra | `CandidateConfigBuilder`、`CampaignRunner`、`TrialRunner`、`CampaignComparator`、冻结数据集 manifest | `domains/local_engine.py`：`LocalComputeAdapter`（真计算、真 checkpoint、真 result.json）；两个冻结数据集（SHA-256 验证） |
| L3 Domain Registry | `DomainRegistry` / `DomainBundle` / `build_scinet_domain_bundle` | `DomainBundle` **增量**加 `quality_evaluator` / `metric_interpreter`；新增 `embodied` / `quant` 两个真 Bundle |
| L4 Research Engine | `ResearchService`（14 个门控方法，0 改动）、`LLMProposer` / `RuleBasedProposer` | `research_task.py`（统一 `ResearchTask` / `TaskResult`）、`arr_runtime.py`（唯一 `run_research_task` 驱动） |
| L5 Orchestrator | `agent_dev` Multi-Agent、审批 CLI | `ResearchHarness`（Planner / Designer / Reviewer / Workflow / Memory 的真实配置对象） |
| L6 RRSI | `gate_a`/`gate_b`、`Ledger`、`Experience.Store`、`Version`、`probe.Sandbox` | `evolution_spaces.py`（两类优化对象分离）、`research_failure_analyzer.py`、`harness_evolution.py` |

**关键判断**：`ResearchService` / `CampaignRunner` / `CampaignComparator` / `CandidateConfigBuilder`
本来就与引擎无关，因此 E 阶段没有复制或 fork 任何通用层 —— 两个新领域只是各自提供了
一个真实的 `ModelAdapter.compute()` 与一份 `ResearchDomainPolicy`。

## 二、Package D：核心缺陷修复

### D1 — 取消 `quality = accepted/n`

旧实现（`research_ingest.py:561`，本轮之前）：`round(accepted / n, 4)`。
这是典型 Goodhart：证伪一个错误假设得 0 分，不可复现的「通过」反而得高分。

新协议（`research_runtime/research_quality.py`）：

```
Q = (w_E·E + w_R·R + w_G·G + w_D·D − w_C·C) / Σw
```

| 维度 | 真实输入（不接受自声明） |
|---|---|
| E 证据完整性 | `report_sha256`、`evidence_refs`、hypothesis/review/plan/approval 的 SHA 绑定、comparison 的 config/dataset/split hashes |
| R 可复现性 | 配对 seed 是否完整、`missing_pairs`、trial 全成功、selection 未用保留 split、config hash 已核验 |
| G 泛化 | 独立 holdout / OOD 任务、方向一致度、跨 seed 符号一致性、护栏无违规 |
| D 发现价值 | **rejected = 0.75 正向分**（证伪是发现）、`accepted` 且有效应 = 1.0、inconclusive = 0.15；竞争性假设、证据锚定、经验回写 |
| C 成本 | jobs / 耗时 / 失败 trial 相对声明预算 |

- **硬约束不可补偿**：`E < 0.5` 或 `R < 0.5`、领域规则（机器人 `safety_violations`、量化
  `lookahead_leakage`、物理 `cosine@1` 下限）违反时 `Q` 被 cap 到 0.35 且 `gate_eligible=False`。
- **候选改不到尺子**：`QUALITY_FROZEN_FIELDS` + `HARNESS_FORBIDDEN_KEYS` +
  `evaluation_substrate_snapshot()` 的 before/after SHA-256（评价协议、`gate.py`、
  `research_task.py`、`evolution_spaces.py`）三重防护。
- 三个领域各自注入权重与规则：`scinet_quality_evaluator` / `embodied_quality_evaluator` /
  `quant_quality_evaluator`。
- 兼容：`export_research_trace()` 保留 `quality_score` 键但改为协议分，另加 `accepted_rate`
  作为**纯诊断量**；`research_ingest` 的 fake 门控链路单测仍 12/12 通过。

### D2 — 分离两类优化对象

`adaptive_runtime/evolution_spaces.py`：

| | 优化对象 | 配置空间 | 评价目标 |
|---|---|---|---|
| `DOMAIN_MODEL`（**Package C 保留**） | 科学模型 / 因子 / 策略 | `train.lr` / `weight_decay` / `jepa.*` | 配对 seed 的领域指标 delta + holdout |
| `RESEARCH_HARNESS`（**Package F 新增**） | 研究系统自身 | `planner.strategy` / `planner.hypothesis_prompt` / `memory.experience_top_k` / `workflow.*` / `reviewer.*` | 固定 D1 评价协议，同一任务 + 独立 held-out / OOD |

两者共用 `Sandbox / Gate / Ledger / Experience / Version`，但 **ledger 指纹按类型命名空间**
（`research_harness::<fp>`），不可能混淆。

### D3 — 统一任务协议

`ResearchTask(task_id, domain, objective, constraints, dataset_refs, baseline_ref,
budget, evaluation_protocol, heldout_policy)` → 复用既有 `ResearchObjective`；
`TaskResult` 统一封装 `ResearchReport` + `QualityScore` + 真实证据 + SHA-256 provenance。
`run_research_task(task)` 是唯一入口，对三个领域调用**同一个** `ResearchService`。

### D4 — 历史可信度

- 6 个 deselected 测试逐项实测登记：`docs/arr_1_0_deselect_registry.md`（含真实错误行）。
- **移除 1 条 deselect**（`test_landing_roundtrip` 在干净 venv + 完整 CI env 下通过），
  未新增任何 deselect / skip。
- 新增 27 条 ARR 1.0 测试全部落在 CI 默认 scope 内真实运行；两个本地领域跑的是
  **真实数值计算**，不是 mock。

## 三、Package E：三个领域真实接入

| 领域 | 真实执行器 | 冻结数据 | 关键真实指标 |
|---|---|---|---|
| `scinet` | `SciNetAdapter`（外部 `E:\LLM\SciNet++` GPU CLI，沿用 Package B/C） | `pendulum_smoke_v1` manifest | `cosine@1`、`latent_mse@1..@3`、checkpoint SHA-256 |
| `embodied` | `LocalComputeAdapter` + 1-D 阻尼机器人动力学；**梯度下降拟合线性 world model**，5 步真实 rollout | `robot_sim_v1`（train/validation/test/**ood** 四 split，SHA-256） | `state_mse@1..@5`、`action_prediction_mse`、`task_success_rate`、`safety_violations` |
| `quant` | `LocalComputeAdapter` + 因果因子信号、**带交易成本/滑点/Benchmark 的真实回测** | `market_hist_v1`（严格按时间 60/20/20 + 独立 OOD regime） | `net_return_after_cost`、`sharpe`、`max_drawdown`、`pred_score@1/5/20`、`lookahead_leakage` |

**数据诚实声明**：

- Embodied 是**确定性仿真**（reviewer 明确要求第一版不接真实硬件）；
- Quant 是**确定性 seeded 市场面板，不是交易所真实行情** —— 本环境公网行情源
  （Stooq / Yahoo / GitHub raw）均不可达（JS challenge / 429 / timeout）。
  `manifest.provenance.is_exchange_or_hardware_data = false` 明确标注，
  `datasets.replace_with_real_csv()` 是替换真实 OHLCV 的显式 operator seam，
  **不需要改任何运行时代码**。
- 但**防未来信息泄漏机制是真实且与数据无关的**：所有输入特征只用 `rows <= t`，
  `fwd_ret_*` 仅作训练标签；`lookahead_guard()` 用截断面板重算信号做实证校验，
  违反即 `status=failed / integrity_error`；selection 只能用 `validation`，
  `test` 保留给最终确认，OOD 是不同 regime；成本、滑点、Benchmark、回撤全部实算。

真实科学现象（不是硬编码）：
- Embodied 的状态相关执行器非线性使线性 world model 的多步误差累积
  （`state_mse@1≈0.0015 → @5≈0.027`）；`model.drift_correction` 是从 **train split 残差**
  学到的真实修正，能降低长时误差。
- Quant 的 ridge 闭式拟合（替代会饱和/发散的 GD）在不同 bootstrap seed 下产生真实方差
  （`net_return_after_cost` 0.10–0.32），配对统计才有意义。

### E5 — 两个跨领域通用性缺陷（真实定位，不是绕过）

统一 runtime 第一次跑三域 benchmark 时，SciNet 报 `KeyError: 'campaign_id'`。
没有把它当成「SciNet 特例」跳过，而是逐层追到真实原因：

| 缺陷 | 根因 | 通用修复 |
|---|---|---|
| `run_research_task` 在 `get_domain("scinet")` 时，全新进程的 registry 为空（只有 `build_default_research_service` 会惰性注册，而它在读 policy **之后**才被调用） | 诊断脚本捕获到 `_guard` 返回的 `{"ok": false, "error_code": "config_key_not_allowed", "error": "model.predictor_hidden=0.1 violates its injected value rule: expected a non-empty list of layer widths"}` | ① 新增 `ensure_domain_registered()`，在读取任何 domain policy **之前**显式注册；② `run["campaign_id"]` 前先检查 `ok`，把真实 `error_code` 抛出来，不再用裸 `KeyError` 掩盖 |
| Harness planner 对所有领域的所有旋钮硬编码标量 `0.1` | SciNet 的第一个白名单键 `model.predictor_hidden` 是**列表型** `[32, 32]`；本地两域第一个键恰好是标量 `train.lr`，所以此前一直「碰巧能跑」。这是典型的只对一个靶场成立的特化假设 | `make_harness_generate` 改为从 **baseline 配置值的类型**派生干预值（int / float / 数字列表各自按 1.1× 扰动），再逐条用领域注入的 `value_rules` 校验；非法则跳过该键，绝不产出会在 config gate 被拒的干预。`diagnosis_targeted` 的领域注入干预也走同一 `_sanitise`。**没有任何 `if domain == ...` 分支** |

修复后 SciNet 通过统一 runtime 真跑（`campaign_id` 有效、配对 trial 全成功、
`dataset_version=pendulum-a0e03386be89`），两个本地域的 Harness 排序仍成立
（CAND > BASE > NEG，见下节实测）。

## 四、Package F：真正的 Harness 自我改进

`adaptive_runtime/harness_evolution.py` 的闭环：

1. **Collect** — 用 BASE Harness 跑 5 个独立真实研究任务（不同配对 seed → 不同 bootstrap 样本）。
2. **Diagnose** — `research_failure_analyzer` 从真实 `TaskResult` 结构化字段识别
   假设质量 / 实验设计 / Tool / 规划 / Workflow / 评价 / Memory / 成本 8 类失败，
   每类带实测 evidence 与建议旋钮。
3. **Generate** — 单旋钮、受 `gate_a` 预算约束、ledger 去重的 Harness 候选。
4. **Sandbox** — `probe.create_sandbox` 建隔离副本，写入**真正被执行**的 `harness.json`
   （不是标记文件），记录 base/candidate SHA-256；每个候选还有独立的
   `data_dir`（sqlite / evidence / campaign store 全部隔离）。
5. **Experiment** — Base 与 Candidate 在**相同任务 + 相同配对 seed** 上分别跑完整真实
   研究闭环（`ResearchService` → campaign → comparator → ResearchReport）。
6. **Evaluate** — **固定**的领域 `ResearchQualityEvaluator` 打分；独立 held-out
   **跨域任务**（Embodied 选择 → Quant 确认）作 OOD 验证；**负对照** Harness
   （重复已知无效干预）必须不被接受。
7. **Gate** — 真实 `gate_a` / `gate_b`（含 `min_tasks_for_t`、契约地板、配对 t 检验、
   缺 run 直接 `INCOMPLETE_EVIDENCE`），再叠加 holdout / 负对照 / substrate 未变三条件。
8. **Record** — `Ledger`（名字空间指纹）+ `Experience.Store` + `BaseVersion` 归属；
   连续无收益停止、最大轮数 / 时间预算。
9. **Promote / Rollback** — 循环**从不调用** `version.promote`；
   `promote_harness(operator=<human>)` 是独立人工门（拒绝 `system/auto`），
   `version.restore` 可回滚。

## 五、统一 CLI

```bash
python -m research_runtime.cli run --domain scinet|embodied|quant \
    [--objective ...] [--budget-experiments N] [--seeds ...] [--harness base|improved|negative]
python -m research_runtime.cli benchmark --all-domains
python -m research_runtime.cli evolve --target research_harness --rounds 2
python -m research_runtime.cli report --input <artifact.json>
python -m research_runtime.cli verify [--ci-log <pytest.log>]
```

`verify` 直接读**保存的真实产物**（benchmark summary + evolution artifact + 可选 CI
日志），逐条判定 13 条验收标准并打印 PASS/FAIL；缺少产物会直接报错，不允许凭
记忆自声明。

## 六、验收矩阵（13 条）

见 `experiments/arr_1_0*/` 下真实产物；本文末尾「回归」记录命令与结果。

| # | 验收要求 | 实现位置 |
|---|---|---|
| 1 | SciNet++ 自主研究闭环 | `run_research_task` + `build_scinet_domain_bundle`（Package B/C 已验证） |
| 2 | Embodied 仿真研究闭环 | `domains/embodied.py` + 冻结 `robot_sim_v1` |
| 3 | Quant 真实回测闭环（数据诚实声明见上） | `domains/quant.py` + 冻结 `market_hist_v1` |
| 4 | 三域共用同一 Runtime | `build_default_research_service(domain_name)` / `DomainRegistry` |
| 5 | 不提供人工假设也能自主产生 | `make_harness_generate` 注入 `LLMProposer`，确定性 Critic 仍是唯一审批门 |
| 6 | 从真实轨迹发现流程缺陷 | `research_failure_analyzer.analyze_many` |
| 7 | 自动产生 Harness 候选 | `generate_harness_candidates` |
| 8 | 候选在 Sandbox 中实际运行 | `apply_harness_patch` + 每臂独立 `data_dir` |
| 9 | Base/Candidate 公平比较 | 同任务 + 配对 seed + 固定 evaluator + substrate hash |
| 10 | 独立任务上真实改善 | held-out 跨域任务 + `holdout_confirmed` 条件 |
| 11 | 能拒绝无效/退化候选 | `degenerate_control_candidate()` 作为**完整配对候选**（同任务/同配对 seed/同 gate）进入循环并产生真实 `REJECT`；另有 `NEGATIVE_HARNESS` 单点负对照 `not_accepted` |
| 12 | 决策可追溯、可回滚 | 全链路 SHA-256 artifact + Ledger + Experience + `version.restore` |
| 13 | CI 通过，真实/Fake 分明 | workflow 真跑 5 个新测试文件；Fake 只存在于 R4 门控单测 |

## 七、明确未做 / 剩余问题（如实保留）

1. **Quant 未接入真实交易所历史数据**（网络受限）。当前是可替换的确定性模拟面板；
   接入真实 OHLCV 是 operator 动作，需要补齐来源、检索时间与许可信息。
2. **Embodied 未接真实机器人硬件 / LeRobot / Isaac**（reviewer 明确第一版用仿真）。
3. **Harness 修改范围有限**：第一版仅 6 个可控旋钮（Planner / Designer / Reviewer /
   Workflow / Memory 的配置），没有让 RRSI 自动改 Python 源码（reviewer 允许）。
4. **LLMProposer 当前走确定性 Planner**：`make_harness_generate` 是规则式
   （读真实指标 + 结构化诊断），不调用外网 LLM；这使 Base/Candidate 可复现。
   接 LLM 只需替换 `llm_generate`，Critic 仍是确定性门。
5. **5 个历史 deselect** 仍在（3 个有状态 selftest + 2 个 legacy benchmark/evolution
   真实缺陷），已逐项登记根因，未无差别重构。
6. **pre-commit 失败**仍是仓库缺 `.pre-commit-config.yaml` 的历史问题，与本轮无关。

## 八、回归

回归命令（与 GitHub Actions 完全一致）：

```bash
cd E:/agent_dev/multi-agent
RESEARCH_SKIP_E2E=1 python -m pytest tests/rt_tests/ adaptive_runtime/tests/ -q \
  -p no:cacheprovider \
  --deselect adaptive_runtime/tests/test_selftest_e2e.py \
  --deselect "adaptive_runtime/tests/test_benchmark.py::TestFakeDoesNotPollute::test_real_mode_does_write" \
  --deselect "adaptive_runtime/tests/test_evolution_version.py::TestRC3StallForcedL3::test_dominant_l1_exhausted_forces_l3"
```

结果见本文末尾的提交说明 / 最终答复中的实测数字。
