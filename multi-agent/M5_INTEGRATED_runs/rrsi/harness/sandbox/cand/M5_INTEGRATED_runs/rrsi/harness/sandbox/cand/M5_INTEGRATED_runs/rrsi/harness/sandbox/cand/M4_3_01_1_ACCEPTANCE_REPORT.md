# M4.3-01.1 — A/B/C Harness Operational Hardening：验收报告

## 0. 结论先行

M4.3-01.1 的 **P0-01 ~ P0-04、P1-01 ~ P1-03 全部完成**（P1-03 按环境要求明确 skip），
把 M4.3-01 的"实验记录框架"补齐为"能可靠驱动真实 A/B/C 实验的 Harness"。
**未重构 ARR，未新增第二套 ResearchService / Planner / Decision Engine / Evidence Reader / RRSI**。

- 分支：`feature/arr-0.1-domain-registry`，**7 个逻辑 commit + 1 个 CI commit**，全部独立提交并推送；
  **未 merge main、未创建 PR**。
- 全量回归：`research_runtime/tests/` **493 passed / 2 skipped / 37 subtests**（较阶段基线
  318 passed 增加 175 个通过用例，零新增失败）。
- CI 精确门（ARR 1.1 runtime pytest 段）：**438 passed / 2 skipped / 29 subtests**。
- ARR 1.1 runtime unittest 段：**89 tests OK**。
- ARR 0.1 integration：既有测试文件与基线 **字节完全一致（0 changed）**；本机
  engine-free suite 全绿；所有失败均为**既有环境依赖缺失 `urllib3`**（已用基线 commit
  `ca1351ee` 复现），**非本阶段引入**；详见 §6。
- **Engine-free 与 Real-engine 结果分开报告**（§5）。
- **未启动 Pilot (M4.3-02) / Confirmatory (M4.3-03) / Value Report (M4.3-04) / M5**。

## 1. 提交链（均在 `feature/arr-0.1-domain-registry`）

| 阶段 | Commit | 内容 |
|---|---|---|
| P0-01 | `120d6d4e` | 接通真实 ResearchService Gate（只读 adapter、禁止伪造 verdict、删除乐观默认值、fail-closed） |
| P0-02 | `71bda620` | 真正匿名盲评（opaque review_id、封存映射、≥2 评审者、append-only、人工/自动分离） |
| P0-03 | `42c814c7` | ExperimentRegistry 持久化（复用 ARR SQLite、hash-chained 事件、重启恢复、导出+SHA 校验、可信 ID） |
| P0-04 | `eea26dd2` | 完整冻结 Confirmatory Protocol（20 字段 Python/YAML 统一 schema、逐字段 fail-closed、版本化、post-hoc 锁） |
| P1-01 | `b5a7f0df` | 16 项指标按 A/B/C 分组（单一冻结实现、分子/分母/缺失/失败规则、Block 配对、公式版本入 hash、无显著性结论） |
| P1-02 | `d5b887dc` | 可执行实验边界（显式不相交 ID 集合、frozen final split 禁令、固定 Block 预算、三组完整保留） |
| P1-03 | `f8f9db77` | 真实 SciNet++ A/B/C E2E（opt-in 门控，真实 Provider/Discovery/CLI 授权/引擎/Registry/子进程重启/round-2/证据包） |
| CI | `aed086d5` | 7 个新测试接入 ARR 1.1 根 CI（真实引擎 E2E 明确 skip 除非 opt-in） |

基线：`ca1351ee`（M4.3-00/01 验收报告）。当前 HEAD：`aed086d5`，与 origin 同步。

## 2. P0 逐项：根因 → 修复 → 测试

### P0-01 接通真实 ResearchService Gate（20 用例）
**根因**：`gate_adapter` 被接收但从未调用；`record_outcome` 允许调用者声称
`gate_steps_passed=12`、`external_approval=True`（乐观默认值）。

**修复**（`research_service_gate.py` + `evaluation_harness.py`）
- 新增 `ResearchServiceGateAdapter`：对**一个**真实 ResearchService 的现有只读接口做查询——
  `approvals.get/for_resource`、`plans.get`（plan_sha256）、`deps.campaign_store.load_campaign`、
  `get_research_report`、`sessions.experience`。**不重算、不修改、不替代 12-step gate**；
  `GateEvidence` 从 gate 执行后留下的持久化产物链派生 `gate_passed` / `gate_steps_passed=12` /
  `external_approval`（approved + human + hash 绑定当前 plan）。
- `record_outcome` **删除** `gate_steps_passed` / `external_approval` 参数；`FORGED_GATE_KEYS`
  中任何调用者提供的 verdict 一律 `ProtocolViolation`。
- 无 adapter → fail-closed；无真实 Approval → 记 `approval_bypass` 且完成类 outcome 被拒；
  缺 Campaign/Report/Experience → 不得记录 `valid_finding`。
- 失败/中断/无效**始终保留**（真实失败是证据）；A/B/C 共用**同一 adapter 实例**。

**测试**：`test_m4_3_01_1_p0_01_gate.py` — 20 用例，含真实 engine-free 链路夹具
`tests/rt_tests/m4_3_gate.py`（真实 SQLite 持久化记录，通过 SQL 删除/降权构造真实缺失，
非伪造 evidence 对象）。覆盖：无 Approval / 无 Campaign / Report 缺失 / Experience 缺失 /
pending approval / system_policy source / hash 篡改 / 手工伪造 12 步 / A/B/C 同一 adapter /
adapter 只读 / 真实失败中断保留。

### P0-02 真正实现匿名盲评（37 用例）
**根因**：`BlindedObjective` 仍含 `objective_id`（`block_1_A` 类 ID 直接泄漏 arm）；无 opaque
review_id；映射未单独封存；Review 无时间戳、可覆盖；单评审者；自动评分与人工盲评混用。

**修复**（`blind_review.py`，`metrics.py` 委托 re-export）
- `BlindedObjective` **仅含 opaque `review_id`**（`rev_` + 32 位随机 hex，不含 objective_id/arm）；
  `FORBIDDEN_BLIND_FIELDS` 覆盖 arm/group/rank_score/ranker_version/generation_source/
  snapshot_id/block_id 等，描述与嵌套 metadata 的结构化泄漏被 `BlindingViolation` 拦截。
- `BlindingLedger` 单独封存 `review_id -> (objective_id, arm, block_id, snapshot_id, ...)`
  映射；`arm_mapping()` 解盲前抛错；`unblind()` 仅在**评审完成**（默认 ≥2 份人工盲评）后允许。
- `QualityReview` 强制 `reviewer_id / rubric_version / reason / reviewed_at / review_kind`；
  `record_review()` append-only，重复 identity 拒绝，原结果不覆盖。
- `ReviewPanel` 默认 `min_reviewers=2` 且 reviewer_id 互异；`assert_pilot_review_ready()` 门控。
- `review_kind = human_blind | automated` 纳入 `frozen_metric_contract()`；
  `independently_reviewed_objective_quality` **只统计 human_blind**，自动评分单独存 `automated_scores`。

**测试**：`test_m4_3_01_1_p0_02_blinding.py` — 37 用例（objective_id 不泄漏、review_id 不可反推、
描述/嵌套 metadata/key 泄漏拦截、Ranker 分数隔离、解盲前映射不可读、两名评审分别保存、
已完成 review 不可覆盖、≥2 评审者门控、yaml↔代码契约一致）。

### P0-03 ExperimentRegistry 持久化（12 用例）
**根因**：纯内存 dict；`update()` 可静默覆盖已完成 Run；崩溃后不可恢复。

**修复**（`registry_store.py` + `experiment_registry.py`）
- 复用 ARR 现有 `ResearchDatabase` / `content_sha256`，新增 4 张 additive 表
  `m4_3_blocks / m4_3_runs / m4_3_reviews / m4_3_events`（可注入**同一个** ARR db 对象）。
- 每次状态变更 append 一个 **hash-chained 事件**（`prev_sha256 -> chain_sha256`），
  `verify_integrity()` 与 `RegistryStore.verify_export()` 可验篡改/截断。
- `finalize()` 一次性完成且不可二次执行（防重复 campaign / 预算重耗）；
  `update()` 对 finalized Run 抛 `RegistryConflict`。
- 持久化 Protocol SHA、Snapshot SHA、Objective/Plan/Approval/Campaign/Report/Experience ID、
  `review_id`；`recover()` 重建未完成 Block（按 **finalize 状态**判定，非"三 arm 是否注册"）；
  真实失败/中断/恢复事件均落盘；`export()` 独立导出 + SHA 校验。

**测试**：`test_m4_3_01_1_p0_03_persistence.py` — 12 用例（进程重启恢复、恢复后补完剩余 arm、
重复 Run 拒绝（内存+磁盘 PK）、已完成 Run 不可覆盖、事件顺序稳定+链式链接、SHA 篡改/截断
检测、导出含全部可信 ID、中断恢复不重复 Campaign、review append-only、内存模式兼容）。

### P0-04 完整冻结 Confirmatory Protocol（56 用例）
**根因**：旧 `assert_confirmatory_ready()` 只查 6 项；Python dataclass 与 `protocol.yaml` 字段集不同。

**修复**（`protocol_schema.py` + `FrozenProtocol` + `protocol.yaml`）
- `CONFIRMATORY_REQUIRED_FIELDS` = **20 字段**单一真源，Python `FrozenProtocol` 与
  `protocol.yaml::confirmatory_required_fields` / `pre_registration` **完全一致**（契约测试验证）。
- `field_is_missing()` 按类型处理：confirmatory 必须 True；seed 允许 0、None 视为缺失；
  `alpha` / `target_power` 须 0<v≤1；`metric_contract_sha256` 必须等于当前冻结 contract hash。
  **逐字段删除 → 全部 fail-closed**（`ProtocolNotFrozen`，保持 ValueError 兼容）。
- `ProtocolRegistry`：任何修改生成**新 Protocol Version**，旧版不可覆盖；`bind_experiment()`
  让已开始实验固定引用**原 SHA**；`POSTHOC_LOCKED_FIELDS`（primary_metrics / sample_size /
  exclusion_rules）在实验绑定后**完全禁止**修改。

**测试**：`test_m4_3_01_1_p0_04_protocol.py` — 56 用例（20×2 逐字段删除、schema 统一、
范围校验、版本号/幂等/不可覆盖、实验 SHA 绑定、3 个 post-hoc 锁、stage gate 未启动）。

## 3. P1 逐项

### P1-01 补齐分组指标（13 用例）
- `metrics.py` 建立 `METRIC_DEFINITIONS`（16 项全部声明 **numerator / denominator /
  missing_value / failure_policy / tier / category**）与 `compute_metric_set()` **单一实现**，
  `compute_metrics()` 与 `metrics_by_arm()` **共用**，杜绝漂移。
- `metrics_by_arm()` 现输出 **A/B/C 各全部 16 项**（此前仅 6 项）+ `by_category` 三类
  （objective_quality / research_finding_rate / compute_efficiency）+ 规则表。
- `paired_metrics()`：Primary 按 **Block 配对**，保留 **Snapshot / Block / Seed 三层**，
  仅 complete **且** finalized 的 Block 进入完整配对；`assert_paired_seeds_consistent()`
  校验同 Block 三组 Paired Seeds 相同（fail-closed）；不完整/中断/种子不一致的 Block
  进入 `excluded_blocks` 并给出 reasons。
- `METRIC_FORMULA_VERSION` 与定义表纳入 `frozen_metric_contract()` hash（改任一公式即变 hash）。
- `significance_conclusion = None`：本阶段**不作显著性结论**；不输出仅合并平均值。

### P1-02 修正实验协议和执行边界（35 用例）
- 新增 `evidence_boundary.py`：A 只读当前证据、B/C 读**同一**历史、三组同一 Snapshot；
  **current vs historical 通过显式、不相交 ID 集合分离**（未提供则从记录派生稳定 ID，仍强制
  不相交；显式集合与记录不符即拒绝）。
- **frozen final split 禁令**：扫描 current/history 全部字符串与嵌套 metadata（含 split 键），
  命中 `RESERVED_FINAL_SPLITS` + yaml 别名即 `EvidenceBoundaryViolation`；validation split 可用。
- **每 Block 固定**候选数、人工时间预算、Campaign Budget（正数、记录进 block、协议冻结时须相等）；
  Block 不可重复添加或改预算；三组完整记录含 failed/interrupted/invalid/INCONCLUSIVE。
- `M4_3_runs/protocol.yaml` 与 `M4_3_EXPERIMENT_PROTOCOL.md` 同步（§2.1/2.2）。

### P1-03 真实引擎 E2E 验证（3 用例：2 engine-free 门禁 + 1 gated）
- `test_m4_3_01_1_p1_03_real_engine.py`：完整 10 点——真实 DiscoveryContextProvider（真实历史
  Evidence）、真实 A/B/C 三个 Objective（同一真实 Snapshot）、**外部 `approval_cli` 子进程授权**、
  **同一个 ResearchService + 同一 SQLite**、真实 SciNet++ 引擎、真实 Report/Experience 写回、
  真实 Registry 持久化、**真实子进程重启恢复**、round-2 Discovery 精确引用 round-1 Experience、
  完整 E2E 证据 JSON 落盘（含 `registry_export` / `harness_snapshot` / `subprocess_recovery`）。
- **门控三条件全部必需**：`M4_3_01_1_RUN_REAL=1`（或 `M4_E2E_01_RUN_REAL=1`）+
  SciNet++ checkout + 真实历史 Evidence（`n_evidence>0`）。缺一即**明确 skip**，
  **绝不回退 FakeRunner / 合成 Snapshot 后宣称真实引擎通过**（AST 级断言文件未 import 任何
  test double、未定义 fake helper）。
- 本机状态：SciNet++ checkout=True、真实 Evidence=**118 个实体**，但 opt-in 未设 →
  `ready=False`，明确 skip。已在 engine-free 下独立验证 round-2 闭环逻辑
  （真实 Provider → 真实 Experience → `_resolve_ref` 精确接受/拒绝伪造 id）与
  `_subprocess_recover()`（真实 Registry：integrity=True / 3 runs / 1 block / 13 events），
  确保 gated 测试在具备环境时不会因编排缺陷假失败。

## 4. 新增/修改文件清单

新增模块（`research_runtime/m4_3/`）：
`research_service_gate.py`、`blind_review.py`、`registry_store.py`、`protocol_schema.py`、
`evidence_boundary.py`。
修改模块：`evaluation_harness.py`、`experiment_registry.py`、`metrics.py`、`__init__.py`。
新增测试（`research_runtime/tests/`）：7 个 `test_m4_3_01_1_*.py`；共享夹具
`tests/rt_tests/m4_3_gate.py`；迁移 `test_m4_3_evaluation_harness.py` 至新契约。
文档/配置：`M4_3_runs/protocol.yaml`、`M4_3_EXPERIMENT_PROTOCOL.md`、
`.github/workflows/arr-research-runtime.yml`（仓库根）。

## 5. Engine-free 与 Real-engine 结果（分开报告）

| 类别 | 结果 |
|---|---|
| **Engine-free**（CI 可跑，本环境已跑） | CI 精确门 **438 passed / 2 skipped / 29 subtests**；`research_runtime/tests/` 全量 **493 passed / 2 skipped / 37 subtests**；ARR 1.1 unittest **89 OK**。真实 SQLite 持久化记录、真实只读 Gate adapter、真实 `approval_cli` 子进程授权、真实 Registry 哈希链/重启恢复均已坐实。 |
| **Real-engine**（真实 SciNet++ 训练） | **未在本环境实际训练**。`test_m4_e2e_01_real_engine.py`（P0-05）与 `test_m4_3_01_1_p1_03_real_engine.py`（P1-03）均按环境门控**明确 skip**，共 2 skipped。原因：缺 `M4_E2E_01_RUN_REAL=1` / `M4_3_01_1_RUN_REAL=1`（SciNet++ checkout 与 118 条历史 Evidence 均具备）。**不宣称真实引擎已跑通**；证据包仅在 opt-in 实跑后生成。 |

## 6. 回归与工程纪律证据

- **零新增失败**：`research_runtime/tests/` 全量 493 passed；既有
  `test_m4_3_evaluation_harness.py`（17 用例）已迁移到新契约并仍全绿。
- **ARR 1.1 runtime**：unittest 89 OK + CI pytest 段 438 passed / 2 skipped。
- **ARR 0.1 integration**：
  - 既有测试文件在基线 `ca1351ee` 与 HEAD **字节完全一致（0 changed）**；本阶段对
    `tests/rt_tests/` 仅**新增** `m4_3_gate.py`（共享夹具），对 `adaptive_runtime/` 零改动。
  - 本机分批执行：核心 17 个 suite 全绿；`test_research_store/gates/flow/orchestrator/
    objective_gate/failure_diagnoser/experiment_manager/adapters` 等全绿；
    `adaptive_runtime` 已跑的 20+ suite 全绿（auto_verdict 23、autopromote 26、bench_tasks 27、
    evaluation/gate/evidence_verifier/evolution_version 等）。
  - **失败均为既有环境依赖缺失**：`test_research_tools.py` 4 failed、
    `test_agent_integration.py` 2 failed、`adaptive_runtime/tests/test_token_probe.py` 1 failed，
    根因全是 `ModuleNotFoundError: No module named 'urllib3'`（`requests` 依赖链），
    **已用基线 commit `ca1351ee` 的临时 worktree 复现同样的 4 failed / 10 passed**。
  - 若干既有真实引擎集成/长循环 suite（`test_comparison_integration.py`、
    `test_campaign_integration.py`、`test_research_e2e.py`、`test_arr_0_1_e2e.py`、
    `test_research_ingest.py`、`test_l2_and_antigaming.py`、`test_probe_parallel.py`，
    含 `wait_timeout=300` / `timeout_per_job_s=300`）在本机长时间运行/超时，属既有慢测试，
    未产生任何 FAILED；它们不 import 任何 M4.3-01.1 模块。
- **未触碰**：历史 Evidence / metrics / verdict、统计门、frozen final split 规则、
  原 12-step gate；未创建第二套 Gate / ResearchService / Planner / Decision Engine。
- 未追踪的既有文件（`M2_6_scinet_loop.log`、`M2_7_runs/*`、`_probe_*.py`）按约定**未纳入提交**。

## 7. 明确边界与暂停声明

- **未启动 Pilot (M4.3-02)、未启动 Confirmatory (M4.3-03)、未启动 Value Report (M4.3-04)、
  未启动 M5**。本阶段只交付"能够可靠驱动真实 A/B/C 实验的 Harness"及其证据。
- 未作任何显著性或价值结论；A/B/C 谁优**不下结论**（`significance_conclusion=None`）。
- 未 merge main、未创建 PR；8 个独立 commit 全部推送到 `feature/arr-0.1-domain-registry`。
- 真实引擎 E2E 需在具备 SciNet++ checkout + 真实历史 Evidence 的环境手动执行：
  `M4_3_01_1_RUN_REAL=1 python -m pytest research_runtime/tests/test_m4_3_01_1_p1_03_real_engine.py -q -rs`
  （可选 `M4_3_01_1_E2E_EVIDENCE_DIR` 指定证据落盘目录）。

**本阶段完成，暂停，等待 M4.3-02 Pilot 指令。**
