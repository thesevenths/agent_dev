# M3.4 Phase 0 — 代码现状审计（实际读码，不靠历史报告）

> 审计方法：直接读取 `research_runtime/` 各模块 + 顶层脚本，核对 import 关系与 `git` 历史；
> 关键入口文件（`cli.py`、`candidate_config.py`、`domains/scinet.py`、`adapters/scinet_adapter.py`）
> 由主代理逐行通读；其余模块（loop/、recovery、evidence_*、campaign/、improvement_cycle 等）
> 由 Explore 子代理实际读码核对。所有结论标注「直读」或「子代理核对」。

## 1. 已实现功能（带代码证据）

| 功能 | 证据 |
|---|---|
| 双轴 RRSI 协调（DOMAIN + HARNESS） | `evolution_controller.py:294 run_dual_layer`；`:327` harness_evolve + `:335-436` domain 评估（直读） |
| 双向 Student-t + Holm 族系校正统计门 | `evolution_controller.py:73-251 _significance_direction/_decide_domain/domain_family_significance`（直读 grep） |
| 独立 holdout 按 seed 方向确认 | `evolution_controller.py:106-141 _holdout_confirms`（子代理核对） |
| 硬门槛（5-seed、1% practical floor、alpha 只降不升） | `evolution_controller.py:50,60,208-213`（子代理核对） |
| ID+OOD 双轴组合判定（M3.3） | `evolution_controller.py:267-291 _combine_dual_layer_decisions`；`scinet.py:195-258` OOD 评估写 `ood_metrics.json`（直读） |
| 三域真实引擎 runner 统一接口 `runner(art)->float` | `cli.py:533-617 _build_real_domain_runner`、`domains/{scinet,embodied,quant}.py` 的 `*_compute`（直读） |
| Frozen dataset 校验（per-split SHA） | `candidate_config.py:203-259 _verify_frozen_dataset`；`controller.py` 恢复时重校验（直读 + 子代理核对） |
| 候选 config 隔离 + 防篡改复用 | `candidate_config.py:357-419`（同 inputs→复用目录；异 config→拒绝覆盖）（直读） |
| 多轮 loop + 原子 checkpoint + 版本继承 | `loop/controller.py`、`loop_state.py:129-136`（写 .tmp 后 replace）、`improvement_cycle.py`、`version_inheritance.py`（子代理核对） |
| 候选审计账本（append-only JSONL） | `candidate_registry.py:34-90`（直读 grep） |
| 可移植证据包 + 自洽校验 | `evidence_bundle.py`、`evidence_verifier.py:207-259`（子代理核对） |
| 离线 13 项验收 / 报告 | `cli.py:247-387 cmd_verify`（直读） |

## 2. 缺失功能（对照 M3.4 要求）

| 缺失 | 说明 / 证据 |
|---|---|
| **跨三域统一调度器** | 只有 `cli benchmark`（逐域收集，不做"决定下一轮"）与各自独立的 M3 脚本；`run_dual_layer` 只接受单个 `domain`（`evolution_controller.py:294`）。无"注册→分配预算→统一执行→决定下一轮"的顶层调度。 |
| **按 run_root 隔离的域产物根** | `candidate_config.py:357` campaign 根 = `project_dir/outputs/research_campaigns/<campaign_id>`；`project_dir` 由调用方传入。M3 脚本写死共享 `project_dir`（如 `M2_7_runs/scinet`），且 `domain_campaign_id` 默认 `"dual_layer"`，跨运行碰撞。 |
| **SciNet 引擎 CWD 未隔离** | `scinet.py:137` `SciNetAdapter()` 不带 `project_dir` → `resolve_engine` 用 `DEFAULT_SCINET_DIR`(E:\LLM\SciNet++)；`cli.py:592` 对 scinet 故意 `domain != "scinet"` 不传 `project_dir`。外部引擎产物落在共享仓库。 |
| **全局 / 跨域 checkpoint 恢复** | 仅单域单 loop 的 round 级 `loop_state.json`（`loop/recovery.py`）；无跨域、跨候选汇总的全局 checkpoint。 |
| **每运行独立证据包** | `candidate_registry.jsonl` 是候选账本，丰富指标在共享 campaign 目录；`evidence_bundle` 只覆盖 ARR1.0 benchmark/evolution，不覆盖 loop/run_dual_layer。 |
| **确定性唯一晋升策略** | `improvement_cycle.py:60-62` 多 ACCEPTED 同 target 时后者覆盖前者（last-wins），无"选最优"逻辑。 |
| **CI 门禁** | `.github/workflows/` 为空且未跟踪（`git ls-files .github` 返回空）。历史曾存在 CI 提交后被删除。 |
| **恢复幂等（引擎级）** | `scinet.py:148` 每次 `submit` 重训；`_scinet_run_id.txt` 存在性只用于 evaluate 找 checkpoint，崩溃后整轮重跑会重复提交训练（非幂等）。 |

## 3. 真实 Bug（带文件:行号 + 复现思路）

1. **B-1 共享 campaign 目录污染（M3.3 已披露）**：`candidate_config.py:357` + `evolution_controller.py:307`(默认 `"dual_layer"`) + M3 脚本写死 `project_dir`。复现：同 `project_dir` 跑两次 `run_dual_layer`，观察 `M2_7_runs/scinet/outputs/research_campaigns/dual_layer/` 被追加/混合。
2. **B-2 SciNet 引擎产物无 per-run 隔离**：`scinet.py:137` + `cli.py:592`。SciNet 引擎 CWD 固定 `E:\LLM\SciNet++`，任何 run 都写同一外部仓库。
3. **D-1 引擎级恢复非幂等**：`scinet.py:148` 无"已提交且成功则跳过"的 is_recoverable 守卫；崩溃重跑整轮会重复提交训练。
4. **F-1 晋升 last-wins 不取最优**：`improvement_cycle.py:60-62`，多 ACCEPTED 同 target 被后写覆盖，可能丢掉更优解。
5. **D-2 recovery 不校验 project_dir**：`recovery.py:37-46` `_assert_compatible` 比对列表不含 `project_dir`；不同 `--project-dir` 的 loop 若 `state_path` 相同会错误 resume（虽 domain/seed 会拦，但 project_dir 漂移不被发现）。
6. **C-1 `cli loop` 不能形式 ACCEPT harness**：`cli.py:655-656` harness_tasks 仅 3+3 seed，低于 `MIN_SEEDS_FOR_FORMAL_ACCEPT=5`（`evolution_controller.py:50`），harness 层永远 INCONCLUSIVE。

## 4. 重复 / 幽灵模块（带 import 证据）

- **`orchestrator.py`（`ResearchOrchestrator`）**：仅被 `tools.py:70,184,237` 与 `research_runtime/__init__.py:22` 及测试引用；**未被 `arr_runtime`/`cli`/`loop`/`run_dual_layer` 任一真实运行路径使用**（子代理核对）。与 `research_service.ResearchService` 功能重叠 → 遗产/冗余，标记 deprecated，删除前须清理 `tools.py`/`__init__.py` 引用。
- **`run_real_research_loop.py`**（ARR 0.2 Package B）：standalone，`import research_service`+`SciNetAdapter`，**无任何模块 import 它**（子代理核对）→ 死代码/演示。
- **两套恢复机制并存**：`loop/recovery.py`（loop_state.json, round 级）与 `campaign/runner.py:310 resume` + `campaign_store.py`（SQLite, campaign 级）互不感知（子代理核对）。
- **两套证据流并存**：`evidence_store.py`（全局 `evidence.jsonl`）与 `candidate_registry.jsonl`（`evolution_controller.py:323`）双轨，无统一入口（子代理核对）。
- **无字面 import 冲突**——类名/函数名均不同；问题是"功能重叠 + 决策引擎分裂"，不是 import 崩溃。

## 5. M3.4 最小改动清单

### 优先复用（禁止重建）
- `evolution_controller.run_dual_layer` + `_decide_domain`/`domain_family_significance`/`_combine_dual_layer_decisions`（统计门成熟，直接复用）。
- `domain_evolver.evaluate_domain_candidate` + `CandidateConfigBuilder`（物化/隔离/防篡改已健壮）。
- `loop/{controller,research_cycle,improvement_cycle,loop_state,recovery,version_inheritance}`（多轮+原子 checkpoint+frozen 重校验）。
- `evidence_verifier.verify_bundle_integrity`（自洽校验机制，复用于每运行证据包）。

### 必须新建 / 修改（按 Phase）
- **Phase 1 运行隔离**（核心，本审计报告已定位最小改动）：
  - `domains/scinet.py`：`scinet_compute` 增加 `project_dir` 参数并传给 `SciNetAdapter(project_dir=project_dir)`（`scinet.py:137`），使引擎 CWD 落到隔离 `project_dir`（修 B-2）。
  - `cli.py:_build_real_domain_runner`：去掉 `cli.py:592` 的 `domain != "scinet"` 守卫，scinet 也收到 `project_dir`（修 B-2 的 runner 侧）。
  - 新增 `research_runtime/scheduler.py`：`UnifiedResearchScheduler` 为每个域派生独立 `domain_run_root` + `domain_project_dir`（均在顶层 `run_root/<domain>/` 下），并传 `domain_campaign_id=<run_id>_<domain>` 给 `run_dual_layer`（修 B-1，避免跨运行 campaign 碰撞）。
- **Phase 2 统一调度器**：`scheduler.py` 注册三域、统一预算、顺序/并行执行 `run_dual_layer`、汇总、`decide next round`，复用 `run_dual_layer`，不新建第二套 Runtime。新增 `cli` 命令 `orchestrate`。
- **Phase 3 全局状态/恢复**：`scheduler.py` 写原子 `scheduler_state.json`（run_id/task_id/domain/current_round/accepted_versions/pending_tasks/completed_tasks/checkpoint/evidence_refs）；幂等恢复——已 EVALUATED 的域跳过、未完成的域重跑、单域失败不破坏他域。
- **Phase 4 统一证据/决策**：每次顶层运行生成 `evidence_index.json`（配置 SHA、数据 manifest SHA、运行记录、baseline/candidate 配对、统计报告、ID/OOD、ACCEPT/REJECT/INCONCLUSIVE、版本继承、日志、CI 结果）；多 ACCEPTED 候选在 `improvement_cycle.derive_next_version` 增加"同 target 按 primary metric 选最优"逻辑（修 F-1）。
- **Phase 5 统一 CI + E2E**：重建 `.github/workflows/` engine-free 门禁（调用 `cli verify` + 单域 smoke）；最终执行三域统一调度 E2E（SciNet 真实引擎 / Embodied 仿真 / Quant 合成，Quant 必须标记 synthetic）。

### 禁止重建
- 不重写 `research_service.py` / `CampaignRunner` / `ComparisonReportStore`（Service 层仍被 `run_research_task` 使用，campaign 原子性已健壮）。
- 不重写统计门（`evolution_controller` 的 Holm/Student-t/holdout）——M2.7/M3.0 已冻结。
- 不新建第二套证据校验——复用 `evidence_verifier.py`。
- 删除 `orchestrator.py`/`run_real_research_loop.py` 前先清理 `tools.py` 与 `__init__.py` 引用，避免破坏包导入。

---

### 一句话总结
系统已实现成熟且可复核的"单域双轴统计门 + 多轮版本继承 + 原子 checkpoint + 可移植证据包校验"，M3.4 缺的是**横向打通**：跨三域统一调度、按 `run_root` 的真正运行隔离、全局 checkpoint 恢复、每运行独立证据包、确定性唯一晋升、CI 门禁；并存在 `orchestrator.py`/`run_real_research_loop.py` 两个未被真实运行路径使用的遗产模块与两套并存的恢复/证据机制。

### 实测确认的关键事实（直读）
- `run_dual_layer` 已原生接受 `project_dir` / `run_root` / `domain_campaign_id`（默认值 `"dual_layer"`）——**隔离机制已部分存在**，缺的是调用方传入隔离路径 + scinet 真正收到 project_dir。
- `candidate_config.py:357` campaign 根 = `project_dir / campaign_root`，`config_relpath` 相对 `project_dir` 计算 —— 让 `project_dir` 隔离即可同时隔离 ARR 侧 config/metrics。
- `scinet_compute` 把 `backtest_metrics.json`/`ood_metrics.json`/`checkpoint.json` 写入 `job_dir`（`project_dir/outputs/research_campaigns/<campaign_id>/<cand>_seed_<n>`），故 `project_dir` 隔离后 SciNet 的 ARR 侧证据自然隔离；引擎体 CWD 隔离需 B-2 修复。
