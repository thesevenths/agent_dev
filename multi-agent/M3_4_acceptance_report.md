# M3.4 验收报告 — Unified Research Orchestration（三域统一调度 + 运行隔离 + 恢复 + 证据）

> 状态：**真实引擎 E2E 已完成**（run_id=`arr_m34_001`，elapsed≈4694s / 78min，三域全 completed，无引擎调用失败）。结果表与结论已填充（见第 4/5 节）。发现并修复一处调度器聚合层 bug（决策字符串大小写），已离线从原始 `records` 重新派生正确计数，未重跑实验、未伪造任何指标。

## 0. 诚实声明（与用户方案一致，先讲清取舍）

- **真实引擎，无 FakeRunner**：SciNet++ 走真实 torch 引擎（SciNetAdapter → research_api.cli 子进程），Embodied 走真实机器人仿真，Quant 走真实回测引擎。M3.4 的调度/隔离/恢复/证据逻辑由 engine-free 单测覆盖（`test_m3_4_scheduler.py`，mock `run_dual_layer`），真实三域运行由 `M3_4_runs/run_unified.py` 端到端验证。
- **Quant 标记 synthetic**：Quant 使用合成市场数据，在 evidence_index 中 `synthetic_data=true`，**绝不描述为真实市场 Alpha**（用户方案明确要求）。
- **运行隔离的取舍（关键）**：每次运行的**可写产物**（每候选 config / metrics / checkpoint / 日志 / candidate_registry.jsonl）全部按唯一 `campaign_id = run_id_domain` 命名空间隔离；每域独立 `run_root`（registry）。**只读的 frozen dataset 与 baseline config 按设计跨运行共享**（content-addressed + per-split SHA-256 校验 + `_verify_frozen_dataset` 重校验）——共享一份不可变输入不是冲突。引擎内部存储按 run_id 命名，本就不冲突。
- **未做 B-2（引擎 CWD 隔离）并说明根因**：曾尝试把 SciNet 引擎 CWD 从 `DEFAULT_SCINET_DIR` 改为隔离 `project_dir`，但核查发现 **OOD frozen dataset 只存在于引擎仓库**（`E:\LLM\SciNet++\datasets\pendulum_smoke_ood_v1`），`M2_7_runs/scinet` 下没有；改 CWD 会让 OOD 评估失败、破坏已验证的真实引擎路径。故撤销该改动，隔离完全通过唯一 campaign_id + 每域 run_root 实现——这正修复了 M3.3 披露的真实 bug（ARR 侧共享 `dual_layer` campaign + 共享 registry）。
- **未重建任何已有模块**：调度器复用 `evolution_controller.run_dual_layer` + `CandidateConfigBuilder` + 现有统计门；未新建第二套 Runtime / ResearchService / Campaign 系统（用户方案明确要求）。
- **Harness 层用 stub**：Harness 演化与 domain 统计门正交，沿用 M3.0/M3.3 立场（stub），不跑 5 个独立 harness 任务（昂贵且超出 M3.4 范围）。

## 1. Phase 0 代码现状审计（结论）

完整审计见 `M3_4_runs/phase0_audit.md`（实际读码，不靠历史报告）。要点：

- **两套重叠 Runtime**：`cli run/benchmark` → `ResearchService`（ARR 0.x 遗产）；`cli loop` + M3.x 脚本 → `run_dual_layer`（evolution 层，真实验收路径）。M3.x 验收全部绕过 `cli loop`，直接调 `run_dual_layer`。
- **运行隔离被打破（M3.3 已披露）**：域产物绑定 `project_dir/outputs/research_campaigns/dual_layer`，M2.7/M3.0/M3.2/M3.3 全部写同一共享目录；SciNet 引擎 CWD 写死 `DEFAULT_SCINET_DIR`。
- **无跨三域统一调度器**：只有 `cli benchmark`（逐域收集，不做"决定下一轮"）与各自独立脚本。
- **无全局/跨域 checkpoint 恢复**：仅单域单 loop 的 round 级 `loop_state.json`。
- **无每运行独立证据包**：`candidate_registry.jsonl` 是候选账本，丰富指标在共享 campaign 目录。
- **无确定性唯一晋升**：`improvement_cycle.py` 多 ACCEPTED 同 target 时 last-wins。
- **CI 为空**：`.github/workflows/` 未被跟踪。
- **重复/幽灵模块**：`orchestrator.py`、`run_real_research_loop.py` 未被真实运行路径使用；两套恢复机制（loop/recovery vs campaign/runner.resume）与两套证据流（evidence_store vs candidate_registry）并存。
- **最小改动清单**：复用 run_dual_layer / CandidateConfigBuilder / 统计门 / evidence_verifier；新增 `research_runtime/scheduler.py` 统一调度 + 每域唯一 campaign_id + 每域 run_root + 原子全局状态 + 幂等恢复 + 统一证据索引 + 确定性晋升；重建 `.github/workflows/` CI。

## 2. 实施内容（按 Phase）

### Phase 1 — 运行隔离 ✅
- 调度器为每个域派生唯一 `campaign_id = run_id_domain`（经 `run_dual_layer` 的 `domain_campaign_id` 参数），所有可写产物落到 `project_dir/outputs/research_campaigns/<run_id_domain>/`，不再污染共享 `dual_layer`。
- 每域独立 `run_root = <run_root>/<domain>/`，`candidate_registry.jsonl`（证据账本）按域+运行隔离。
- SciNet 引擎 CWD 维持 `DEFAULT_SCINET_DIR`（已验证可用，且 OOD dataset 仅存于此），避免破坏 OOD 路径。

### Phase 2 — 统一 ResearchScheduler ✅
- `research_runtime/scheduler.py:UnifiedResearchScheduler`：注册三域、统一预算（seeds）、顺序执行 `run_dual_layer`、汇总、`decide next round` 留口；复用现有 Runner 与引擎，未新建第二套 Runtime。
- 新增 `M3_4_runs/run_unified.py`：一条命令完成三域调度（SciNet 真实引擎 ID+OOD / Embodied 仿真 / Quant 合成）。

### Phase 3 — 全局状态与恢复 ✅
- `scheduler_state.json`：run_id / repo_root / n_domains / completed / skipped_resumed / failed / per-domain 元数据 / elapsed。
- **幂等恢复**：已完成域（存在 `domain_summary.json`）被跳过并从磁盘重载；未完成域重跑；单域失败不破坏他域（fail-closed per domain）。状态原子写（`.tmp` → rename）。

### Phase 4 — 统一证据与决策 ✅
- `evidence_index.json`：配置 SHA 上下文、dataset_version、baseline/candidate 配对、ID/OOD 统计报告、ACCEPT/REJECT/INCONCLUSIVE、版本继承记录；Quant 标记 synthetic。
- **确定性唯一晋升（F-1 修复）**：多 ACCEPTED 候选时，按"指标方向上最大绝对改善 + 确定性 tie-break"选唯一推荐晋升，并显式记录理由；**绝不按执行顺序静默覆盖版本**。

### Phase 5 — 统一 CI 与端到端验收 ✅
- E2E 驱动 `M3_4_runs/run_unified.py` 已完成（run_id=`arr_m34_001`，真实三引擎，78min，三域全 completed）。
- CI workflow 已重建于 `.github/workflows/ci.yml`：engine-free 门禁（覆盖 M3.4 scheduler / M3.0 统计 / M3.3 ID-OOD / M3.2 embodied 真实数据契约），`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；**真实引擎 E2E 不在 CI 内冒充**，单独在有引擎的机器上运行并据实报告。
- **调度器聚合 bug 修复（诚实记录）**：E2E 首跑后，`domain_summary.json` / `evidence_index.json` 的计数全为 0——根因为 `_summarize_domain` / `_select_promotion` 用大写常量（`"ACCEPTED"` 等）匹配 `domain_layer` 记录里的小写 verdict 字符串，且证据索引以 `target`(`train.lr`) 为键导致 3 个候选坍缩为 1 条。已修复（`_norm_decision` 归一 + 以 `hyp_fingerprint` 为键），并新增 3 条回归测试（`test_m3_4_scheduler.py:TestDecisionCasingRegression`）。**原始 `records` 由真实统计门控产出、本身正确**，修复仅重派生聚合层；提供 `M3_4_runs/reaggregate.py` 离线从不变原始记录重新生成正确摘要与证据索引（不重跑实验）。

## 3. 验收清单

- [x] 每次顶层运行具备独立 run_id 与 run_root（每域独立 run_root）。
- [x] 每 Domain / Campaign / Candidate 独立命名空间（唯一 campaign_id）。
- [x] 证据/日志/配置/checkpoint 不跨运行覆盖（可写产物按 campaign_id 命名空间；只读 frozen 数据按设计共享）。
- [x] 恢复时校验配置 SHA、数据 SHA、版本继承（复用 `CandidateConfigBuilder._verify_frozen_dataset` + loop recovery）。
- [x] 不删除旧产物解决目录冲突（仅命名空间隔离）。
- [x] 统一调度：三域通过同一入口运行（scheduler + run_unified）。
- [x] 全局状态原子保存、幂等恢复（单域失败不破坏他域）。
- [x] 统一证据索引 + 确定性晋升策略（禁止执行顺序覆盖）。
- [x] Quant 合成数据明确标记 synthetic。
- [x] 真实引擎测试（engine-free 单测 + 真实三域 E2E），无 FakeRunner。
- [x] CI workflow 重建并真实运行（engine-free 门禁；真实 E2E 单独据实报告）。
- [x] E2E 结果填充（run_id=`arr_m34_001`，三域全 completed）。

## 4. 真实运行结果（ID / OOD 双轴 + 三域汇总）

**运行身份**：run_id=`arr_m34_001`，elapsed≈4694s（78min），status=`completed`，completed=[scinet, embodied, quant]，failed=[]。
**预算**：paired_seeds=5（50–54）、holdout_seeds=5（60–64）——比 M3.3 的 10+10 少，统计功效更低（见下方诚实说明）。
**协议**：`M3.0-student-t-holm`（Student-t 双侧 CI + Holm step-down + 双向判定 + holdout 方向 ≥0.6 + 硬下限 5 seed）。

### 4.1 逐域（候选 = `train.lr`，3 个值；每候选含 ID 与 OOD 合并决策）

| 域 | 引擎 | dataset_version | 候选 new | ID 决策 | OOD 决策 | mean_imp | holdout_dir | 合成 |
|---|---|---|---|---|---|---|---|---|
| scinet | 真实 torch | `pendulum-a0e03386be89` | 0.003 | inconclusive | inconclusive | +0.0112 | 1.0 | 否 |
| scinet | 真实 torch | `pendulum-a0e03386be89` | 0.005 | inconclusive | inconclusive | +0.0168 | 1.0 | 否 |
| scinet | 真实 torch | `pendulum-a0e03386be89` | 0.01  | inconclusive | inconclusive | +0.0194 | 1.0 | 否 |
| embodied | 真实仿真 | `robot_sim_v1-4979652dd897` | 0.003 | rejected | — | −0.00291 | 0.0 | 否 |
| embodied | 真实仿真 | `robot_sim_v1-4979652dd897` | 0.005 | rejected | — | −0.001998 | 0.0 | 否 |
| embodied | 真实仿真 | `robot_sim_v1-4979652dd897` | 0.01  | rejected | — | −0.000942 | 0.0 | 否 |
| quant | 真实回测 | `market_hist_v1-7640575b236b` | 0.003 | inconclusive | — | +0.002212 | 1.0 | **是** |
| quant | 真实回测 | `market_hist_v1-7640575b236b` | 0.005 | inconclusive | — | +0.001946 | 1.0 | **是** |
| quant | 真实回测 | `market_hist_v1-7640575b236b` | 0.01  | inconclusive | — | +0.001288 | 1.0 | **是** |

### 4.2 汇总

| 域 | n_accepted | n_rejected | n_inconclusive | 晋升推荐 |
|---|---|---|---|---|
| scinet | 0 | 0 | 3 | 无（无 ACCEPTED） |
| embodied | 0 | 3 | 0 | 无（无 ACCEPTED） |
| quant | 0 | 0 | 3 | 无（无 ACCEPTED） |

### 4.3 诚实解读（重要）

- **三个域本轮 0 ACCEPTED 是门控的正确行为，不是系统失败**。门控在证据不足（5+5 seed 统计功效低）或方向为负时如实给出 REJECTED / INCONCLUSIVE，没有为"看起来在跑"而伪造 PASS。
- **scinet**：改善方向为正、holdout 同向（1.0），但 5+5 seed 下 CI 未达显著阈值（M3.3 用 10+10 seed 才 ACCEPTED）——这正是统计功效效应，如实报告为 inconclusive。
- **embodied**：3 个 `train.lr` 候选全部使指标**变差**（mean_imp 为负、holdout 方向 0.0），门控正确 REJECTED。说明在该仿真里调大学习率有害。
- **quant**：改善为正、holdout 同向，但同样未达显著——INCONCLUSIVE；且在 evidence_index 中 `synthetic_data=true`，**未描述为真实市场 Alpha**（遵守用户方案）。
- **晋升**：无 ACCEPTED 候选，确定性晋升策略返回"无推荐"，**绝不按执行顺序静默覆盖 baseline 版本**。

### 4.4 隔离 / 恢复 / 证据（M3.4 核心验收项）

- **运行隔离生效**：新 campaign 目录 `arr_m34_001_scinet/embodied/quant`（唯一命名空间），旧共享 `dual_layer*` 未触碰；每域独立 `candidate_registry.jsonl`、独立 `run_root`。
- **跨域恢复已验证**：以已完成 `arr_m34_001` 为 run_root 重跑调度器（mock 引擎"被调用即失败"），结果 `skipped_resumed=[scinet,embodied,quant]`、`completed=[]`、引擎调用 0——确认每域 `domain_summary.json` 即为全局恢复检查点，单域失败不破坏他域。
- **统一证据索引**：`evidence_index.json` 含配置/数据版本上下文、逐候选（按 `hyp_fingerprint`，不坍缩）ID/OOD 决策、synthetic 标记、晋升记录；`scheduler_state.json` 原子写、幂等。

## 5. 结论

M3.4 目标——**三域统一调度、运行隔离、全局恢复、统一证据**——已通过真实引擎 E2E + engine-free 单测达成：

1. **统一调度**：`UnifiedResearchScheduler` 复用现有 `run_dual_layer`，一条命令（run_unified.py）顺序跑通 SciNet++（真实 torch / ID+OOD）、Embodied（真实仿真）、Quant（合成回测，标记 synthetic），三域各自完成 baseline/candidate 比较。
2. **运行隔离**：每运行唯一 `run_id` + 每域唯一 `campaign_id` + 每域独立 `run_root`，可写产物零跨运行覆盖（修复 M3.3 披露的共享 `dual_layer` bug）；只读 frozen 数据按设计共享（content-addressed + SHA 校验）。
3. **全局恢复**：原子 `scheduler_state.json` + 幂等"已完成域跳过/重载"，真实验证 0 引擎重跑；单域失败隔离（fail-closed per domain）。
4. **统一证据 + 确定性晋升**：`evidence_index.json` 完整记录；多 ACCEPTED 时按"最大绝对改善 + 确定性 tie-break"选唯一晋升，禁执行顺序覆盖（F-1 修复）。
5. **真实、无伪造**：本轮三域 0 ACCEPTED 是门控如实拒绝 underpowered/负向证据；统计结果真实，无 FakeRunner；发现并修复的聚合层大小写 bug 仅重派生、不重跑、不改原始门控产出。
6. **工程纪律**：独立 commit（未 merge main、未开 PR）；CI 已重建（engine-free 门禁）。

**M3.4 完成标准达成**：三域通过同一调度入口运行、拥有独立运行空间、统一状态管理、可恢复执行、统一证据、可审计结果。下一阶段进入 M4（最小可用自主选题 + SQLite/JSON 长期记忆），再进入 M5 完整闭环与持续改进验收。

## 6. 交付物

1. `research_runtime/scheduler.py` — 统一调度器（Phase 2/3/4）。
2. `research_runtime/tests/test_m3_4_scheduler.py` — engine-free 单测（隔离/聚合/晋升/恢复/证据 + 决策大小写回归）。
3. `M3_4_runs/run_unified.py` — 一条命令真实三域 E2E。
4. `M3_4_runs/reaggregate.py` — 离线从原始 `records` 重派生正确摘要/证据（聚合层 bug 修复工具，不重跑实验）。
5. `M3_4_runs/phase0_audit.md` — Phase 0 代码审计。
6. `M3_4_runs/arr_m34_001/` — 真实运行证据（scheduler_state.json / evidence_index.json / 每域 domain_summary.json / 独立 campaign 目录）。
7. `.github/workflows/ci.yml` — engine-free CI 门禁。
8. `M3_4_acceptance_report.md` — 本报告。
9. Commits：`c6bb9c3c`（M3.4 调度器/测试/E2E/审计）+ 后续修复 commit（大小写 bug + 证据索引键 + CI + 报告）。均未 merge main、未开 PR。

## 7. 未解决问题与 M4 建议

- **遗产模块**：`orchestrator.py` / `run_real_research_loop.py` 未被真实路径使用，建议清理引用后标记 deprecated（不破坏包导入）。
- **两套恢复/证据机制**：loop/recovery 与 campaign/runner.resume 并存；evidence_store 与 candidate_registry 双轨——M4 统一时须收敛。
- **自主选题与长期记忆（M4）**：本里程碑未做；下一阶段按计划进入 M4（最小可用自主选题 + SQLite/JSON 长期记忆），再进入 M5 完整闭环与持续改进验收。
