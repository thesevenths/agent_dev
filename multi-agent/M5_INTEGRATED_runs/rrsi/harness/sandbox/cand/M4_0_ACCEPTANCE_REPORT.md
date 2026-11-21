# M4.0 ACCEPTANCE REPORT — Research Knowledge Foundation

> **最终远程结论（非自述）**：HEAD `ae5928a4` 的
> [ARR Research Runtime run 43606307331](https://github.com/thesevenths/agent_dev/actions/runs/43606307331)
> `conclusion=success`，**6/6 matrix 全绿**（ubuntu/windows × py3.11/3.12/3.13）；
> [Pre-commit checks run 43606307287](https://github.com/thesevenths/agent_dev/actions/runs/43606307287)
> 亦 `success`。中途出现过真实红 run 并已修复，全过程见 §7，未隐瞒。
> `ARR 1.0 Integration` run 43606307285 在查询时仍 `in_progress`，其结论未在此声称。
>
> **状态说明**：依据工程纪律第 12 条，本文件**不宣布验收通过**；是否通过以 §7
> 记录的远程 GitHub Actions `conclusion` 与 §8 的验收标准逐项核对为准。
> 所有结论均附源码路径、函数、测试与实际运行证据。

- 分支：`feature/arr-0.1-domain-registry`（**未 merge main、未开 PR**）
- 起始基线：`2641e1c76eb159692235c275e5d9bd1876c6101f`
- 仓库：https://github.com/thesevenths/agent_dev （public）
- 工作目录：`multi-agent/`；仓库根：`E:/agent_dev`
- 本阶段**只实现 M4.0**；未开发 M4.1（objective discovery / ranker）、M4.2、M5。

---

## 1. Phase 0 只读审计（先审计，再修改）

交付 `M4_0_runs/phase0_audit.md`。核心事实（均来自实际读码 / SQLite 查询 / 复现脚本）：

| 事实 | 证据 |
|---|---|
| `research_store/db.py` 已有 `ResearchDatabase`（WAL + `BEGIN IMMEDIATE` + RLock + `schema_sql.split(";")` 逐条执行） | `db.py:49-94` |
| `ALL_SCHEMA_SQL` 是 4 个 store 的 schema 拼接，随 `ResearchStores()` 幂等迁移 | `research_store/__init__.py:56-75` |
| 已有 `ResearchObjective` 契约 | `schemas.py:70-77` |
| 真实 `research.sqlite` 存在（122 KB），但位于 `.gitignore` 的 `research_runtime/_data/`，**不入库** | `multi-agent/.gitignore:33` |
| DB 中唯一完整历史链路是 P1-06 会话 `obj_pendulum_multistep`（hypothesis / plan / approval / campaign / session / experience / report 全有明确 ID） | SQLite 实查 |
| **`arr_m34_001` 在这些表中命中 0 行** —— M3.4 走 `scheduler → run_dual_layer`，只写 JSON 文件 | 7 张表全文检索 |
| `evidence_bundle.build_bundle` 仅被自身 `main()` 调用；`cli.py` 只消费产物 | `evidence_bundle.py:307`、`cli.py:218,260,290-301` |
| 复现出 P0-2 axis 误分类反例（文件名 `ood_metrics.json` 内容实为 ID，却被判 OOD） | 复现脚本输出 |

---

## 2. 逐项交付（6 个逻辑包，独立 commit）

| Commit | 包 | 生产代码 / 文件 |
|---|---|---|
| `cbdf72af` | M4.0-01 | `research_store/objective_store.py`、`research_store/schemas.py:StoredObjective`、`research_store/__init__.py` |
| `b034707e` | M4.0-02 | `research_store/entity_link_store.py` |
| `69ba03bd` | M4.0-03 | `M4_0_runs/backfill.py` |
| `7a7a7559` | M4.0-04 | `research_runtime/evidence_read_model.py`；`research_service.py` / `scheduler.py` 各加只读 `query_run_evidence` |
| `23ac08c6` | M4.0-05 | `M4_0_runs/validate_integrity.py`；`scheduler.py` axis 分类修复 |
| `2425c986`,`17bc3ae3`,`92359d6e` | M4.0-06 | CI 自洽 seed、4 个测试模块、根 workflow |

### M4.0-01 Objective Persistence

- `objective_store.py:SCHEMA_SQL` 在**现有 `research.sqlite`** 追加 `objectives` 表，含任务书要求全部 10 列：`objective_id / domain / description / constraints_json / success_criteria_json / parent_objective_id / status / origin_experience_id / content_sha256 / created_at`；额外 `provenance_json` 用于标注「M4.0 重建」而非历史原生。
- 幂等迁移：`CREATE TABLE IF NOT EXISTS` + 索引，schema 加入 `ALL_SCHEMA_SQL`，由 `ResearchDatabase.session()` 单事务执行。`ObjectiveStore.migrate()` 亦可显式重复调用。
- `from_research_objective()` 复用现有 `ResearchObjective`；`content_sha256` 由 `db.content_sha256` 计算且排除 `created_at`；同 ID 不同内容抛 `ResearchStoreConflict`（fail-closed，不覆盖历史）。
- **未新建独立 SQLite、未引入 ORM**。

### M4.0-02 Entity Link Store

- `entity_link_store.py:SCHEMA_SQL` 追加 `entity_links`；`UNIQUE(run_id, objective_id, src_type, src_id, dst_type, dst_id, relation)`：防重复 link，同时允许一个实体有多个关系。
- 5 个 API：`add_link` / `find_by_entity`（`out|in|both` = 正向+反向）/ `find_by_objective` / `list_objective_graph` / `validate_links`；另有 `stats` 与 `ensure_link_schema`。
- 所有写操作走 `ResearchDatabase.session()`（`BEGIN IMMEDIATE`）。
- 关联 8 类实体：`objective / hypothesis / candidate / experiment_plan / campaign / research_session / evidence / experience`，**保留原实体 ID 原样**。
- `unresolved` 必须用显式哨兵 `dst_id` + 强制 reason；`add_link` 直接拒绝猜测出的实体身份。
- `add_link` 在插入前校验 `source_path` 的真实 SHA，不一致抛 `ResearchStoreConflict`；`validate_links` 复核已存 SHA 并列出 violations（只读）。
- `source_path` 一律 `Path.as_posix()`，Windows/Linux 兼容。

### M4.0-03 Historical Evidence Backfill

- `M4_0_runs/backfill.py`：只读 89 个历史文件；只用明确 ID / run_id / campaign_id / SHA 建关系：
  - `run_id` ← `evidence_index.json`
  - `campaign_id` ← 每域 `evidence_index.domains.<d>.campaign_id`
  - `candidate` ← `evidence_index.decisions` 键与 `candidate_registry.jsonl` 的 `register.fingerprint` **双来源交集**
  - `evidence` ← `sha256:<真实文件 SHA-256>`
  - P1-06 原生链 ← DB 中 7 张表的明确 ID（独立 run namespace）
- M3.4 历史运行**没有**原生 objective/hypothesis/plan/session/experience：创建 3 个明确标注 `status=reconstructed` 的 objective 作为图根（provenance 记录 basis + 源 SHA），并生成 **36 条 unresolved**（3 域 × 3 候选 × 4 缺口），每条带明确原因。**绝不按文件名或相似文本猜测**。
- 修复一个真实跨域碰撞：三域 `hyp_fingerprint` 相同，旧 UNIQUE 会合并不同 objective 的关系 → `objective_id` 纳入去重作用域；`ensure_link_schema()` 对**空**旧表原子重建、对**非空**旧表拒绝（fail-closed）。
- **真实运行结果**：PASS1 建 156 条 link（arr_m34_001 146 + P1-06 10），PASS2 **0 新增**、156 条去重命中；89 个历史文件 SHA 前后完全一致；`validate_links` 110 个 SHA 校验全部通过。
- 交付证据：`M4_0_runs/backfill_report.json`、`evidence/historical_files_sha256.json`、`evidence/backfill_idempotency.json`、`evidence/db_schema_before.json` / `db_schema_after.json` / `migration_idempotency.json`。

### M4.0-04 Unified Evidence Read Model

- `evidence_read_model.py`：`UnifiedEvidenceReader` / `EvidenceView`，**只读**枚举 6 种既有载体：`evidence_index(v2)` / `domain_summary` / `candidate_registry` / engine `*metrics*.json` / append-only `evidence.jsonl`（流式 + 结构化 run_id 校验）/ `evidence_bundle` releases。
- evidence 身份 = 真实字节 SHA-256；同一 OOD artifact 经 `evidence_index` provenance 与 `engine_metrics` 两个载体发现 → **合并为 1 个实体、2 个 carriers**（实测 40 个）。
- 保留：原始来源路径（POSIX）、**原始 verdict 字符串**（`inconclusive`/`rejected`/`proposed`/… 未重算）、dataset version、manifest SHA、checkpoint SHA、evaluation run ID。
- **不写任何文件、不重算统计 verdict、不覆盖原始证据、未改 12 步 fail-closed 门链与 `default_evidence_provider`**。
- `ResearchService.query_run_evidence` 与 `UnifiedResearchScheduler.query_run_evidence` 委托同一 `query_run_evidence()`，实测两者视图**完全相同**（`evidence/evidence_consistency.json`）。
- `evidence_bundle` 决策记录于 `M4_0_runs/evidence_bundle_decision.md`：审计确认为**手动离线发布工具**（docstring 声称接线与事实不符），**不自动接线**（会引入写副作用并改变历史 release SHA 语义），历史工具**原样保留**，其发布产物作为可选只读载体。
- 实测 `arr_m34_001`：`n_evidence=118`、6 载体、40 个合并实体、81 个实体带 checkpoint SHA / run ID。证据：`evidence/unified_evidence_view.json`。

### M4.0-05 跨系统完整性验证 + provenance 反例

- `M4_0_runs/validate_integrity.py`：逐候选验证 Objective→Hypothesis→Candidate→Campaign→Evidence→Experience 链路及反向查询。
- **真实结果**（`M4_0_runs/integrity_validation_report.json`）：
  `candidates=9`、`confirmed_by_registry=9`、`reverse_queries_ok=9`、
  `evidence_sha_verified=276`、`unresolved=36`、
  `cross_run_isolated=True`、`historical_files_unchanged=True`、
  `all_confirmable_chains_ok=True`、`problems=0`。
- P0-2 反例修复：`scheduler.py:collect_domain_provenance` 改为**内容信号优先且可否决文件名**；内容自相矛盾 → `axis=unknown` + `axis_warnings`，不污染 OOD 统计；纯文件名 fallback 标 `filename_unverified`。新增 4 个反例测试，含对真实 40 个 OOD artifacts 的回归守卫。
- 修复跨域实体 ID 碰撞：`find_by_entity` 增加 `objective_id` 作用域，**保留原始 `hyp_fingerprint` 不改名**，同时让每域图谱独立。

### M4.0-06 测试与 CI

- 新增 4 个测试模块 + 1 个 CI seed helper，覆盖任务书 §9 全部 12 项（见 §6 映射表）。
- 因 `research.sqlite` 与 38 MB `evidence.jsonl` 被 gitignore，**CI 无此二物**：提交 `evidence/p1_06_db_seed.json`（真实 P1-06 行的**只读导出**，非伪造数据，11 表 19 KB），`m4_0_seed_helper.build_equivalent_db()` 在 CI 重建等价 DB；第 6 载体由合成 workspace 测试覆盖。
- 4 个 M4.0 测试模块加入**仓库根目录实际生效**的 `.github/workflows/arr-research-runtime.yml`（共 9 个 pytest 模块）。

---

## 3. 修改前后行为对照

| 项 | 修改前 | 修改后 |
|---|---|---|
| 研究目标 | 仅 `ResearchObjective` 内存对象，`research.sqlite` 无 objectives 表 | 10 列持久化表，幂等迁移、SHA 去重、冲突 fail-closed |
| 跨实体关系 | 7 套 ID 割裂、无映射表 | `entity_links` 正/反向查询、run + objective 双层隔离 |
| Evidence 可见性 | `ResearchService` 只认 `evidence.jsonl`，看不到 M3.x 证据 | 6 载体统一只读视图，Service 与 Scheduler 一致 |
| 历史 M3.4 可追溯性 | candidate 无 objective/campaign/evidence/experience 关联 | 9/9 candidate 链路可查；36 个不可确认关系诚实标 unresolved |
| provenance axis | 文件名 `ood` 可覆盖内容信号 → 误分类 | 内容优先、可否决文件名、冲突标 unknown + 告警 |
| 恢复/迁移 | 无 | 迁移两次结构一致、历史表与行数不变 |

---

## 4. 未改动项（纪律核查）

- 未 merge main、未创建 PR；分支独立于 `main`。
- 未删除任何历史研究证据；89 个历史文件 SHA 前后一致（`historical_files_sha256.json`）。
- **未修改任何历史 verdict / metrics / records**：backfill 只新增 `objectives` / `entity_links` 行；`db_schema_before/after.json` 显示原 12 张表行数不变。
- 未降低 Student-t / Holm / holdout 门槛；统一读模型不参与任何裁决。
- 未使用 FakeRunner；未新建第二套 Memory / Evidence / Decision 系统；未引入向量库 / ORM；未新建独立 SQLite。
- `evidence_bundle.py` 一行未改；`default_evidence_provider` 一行未改。

---

## 5. 测试结果（本地 + 干净 checkout 模拟）

```bash
# 本地完整 runtime suite（真实 DB 存在）
cd E:/agent_dev/multi-agent
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest research_runtime/tests/ -q
# -> 193 passed, 37 subtests passed in 73.27s

# 干净 CI checkout 模拟（git archive 稀疏导出，无 _data / 无 research.sqlite）
SCINET_DIR="Z:/nonexistent" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  research_runtime/tests/test_m{3_0_statistics,3_2_embodied_real_data,
  3_3_scinet_ood,3_4_scheduler,3_4_embodied_horizon,
  4_0_objective_store,4_0_entity_link_store,4_0_backfill,
  4_0_evidence_read_model}.py -q
# -> 134 passed, 3 skipped, 29 subtests passed

cd E:/agent_dev && python -m pre_commit run --all-files
# -> 3/3 Passed
```

3 个 skip 为 `test_m3_3_scinet_ood.py` 中依赖引擎仓库 OOD 数据集的自跳过测试
（CI 无 `E:\LLM\SciNet++` checkout），属设计内行为，未静默改成 PASS。

---

## 6. §9 十二项测试要求 → 具体测试映射

| # | 要求 | 测试 |
|---|---|---|
| 1 | SQLite migration 幂等 | `test_m4_0_objective_store.py::test_migration_is_idempotent_and_structure_identical`；`test_m4_0_entity_link_store.py::test_migration_adds_entity_links_idempotently` |
| 2 | 重复 backfill 幂等 | `test_m4_0_backfill.py::test_backfill_is_idempotent_no_new_links` |
| 3 | Entity Link 正向查询 | `test_m4_0_entity_link_store.py::test_forward_and_reverse_queries`（direction=out） |
| 4 | Entity Link 反向查询 | 同上（direction=in）；`test_m4_0_backfill.py::test_cross_run_isolation_in_backfilled_data`；integrity report 9/9 反向 |
| 5 | 跨 run 隔离 | `test_m4_0_entity_link_store.py::test_cross_run_isolation`、`test_same_entity_id_is_isolated_by_objective`；backfill / integrity 两报告 |
| 6 | SHA 不一致 fail-closed | `test_m4_0_entity_link_store.py::test_sha_mismatch_fails_closed_and_match_succeeds`（add_link + validate_links 双路径） |
| 7 | unresolved 历史实体 | `test_m4_0_entity_link_store.py::test_unresolved_requires_sentinel_and_reason`；`test_m4_0_backfill.py::test_unresolved_relations_are_explicit_and_complete` |
| 8 | Evidence 多载体读取 | `test_m4_0_evidence_read_model.py::test_all_available_carriers_are_read`、`test_sixth_carrier_evidence_jsonl_with_synthetic_workspace` |
| 9 | Evidence 重复识别 | `test_m4_0_evidence_read_model.py::test_duplicate_evidence_is_recognised_across_carriers`、`test_synthetic_multi_carrier_dedup` |
| 10 | 原始历史文件不变 | `test_m4_0_backfill.py::test_historical_files_are_never_modified`；`test_m4_0_evidence_read_model.py::test_queries_are_read_only`；`historical_files_sha256.json` |
| 11 | 数据库事务回滚 | `test_m4_0_objective_store.py::test_transaction_rollback_leaves_no_partial_objective`；`test_m4_0_entity_link_store.py::test_transaction_rollback_keeps_no_partial_link` |
| 12 | Windows/Linux 路径兼容 | `test_m4_0_entity_link_store.py::test_source_path_is_posix_on_windows`；`test_m4_0_backfill.py::test_source_paths_are_posix_paths`；`test_m4_0_evidence_read_model.py::test_all_available_carriers_are_read` |

现有回归：M2.7（`test_m2_7_reliability.py`）/ M3.0 / M3.2 / M3.3 / M3.4 全部纳入本地
`192 passed` 全量 suite；CI 门禁覆盖 9 个 pytest 模块 + 既有 unittest 4 模块。

---

## 7. 远程 GitHub Actions 实测

所有链接均为 GitHub REST API 查到的真实 run（`conclusion`），**不凭自生成报告宣布**。

| Workflow | Run | Commit | 结论 |
|---|---|---|---|
| ARR Research Runtime | https://github.com/thesevenths/agent_dev/actions/runs/43605610197 | `92359d6e` | ❌ **failure** — ubuntu py3.11/3.12/3.13 三红（首轮接入 M4.0 测试） |
| ARR Research Runtime | https://github.com/thesevenths/agent_dev/actions/runs/43606011274 | `ce57538f` | ✅ **success，6/6 matrix**（含此前失败的 ubuntu py3.11/3.12/3.13） |
| ARR Research Runtime（最终） | https://github.com/thesevenths/agent_dev/actions/runs/43606307331 | `ae5928a4` | ✅ **success，6/6 matrix 全绿**（最终 HEAD） |
| Pre-commit checks | https://github.com/thesevenths/agent_dev/actions/runs/43605610113 | `92359d6e` | ✅ success |
| Pre-commit checks（最终） | https://github.com/thesevenths/agent_dev/actions/runs/43606307328 | `ae5928a4` | ✅ success |
| ARR 1.0 Integration | https://github.com/thesevenths/agent_dev/actions/runs/43605610117 | `92359d6e` | ✅ success（既有跨平台门禁） |

### 中途真实失败的根因与修复（如实记录，未掩盖）

首轮 `92359d6e` 的 ubuntu 三个 matrix 全红。通过 check-run annotations API 读到
确切错误（不是猜测）：

```
assert 0 == 40                                    # 跨载体合并实体数
AssertionError: 'evidence\\sub\\e.json' == 'evidence/sub/e.json'
NameError: name 'pytest' is not defined
```

三个根因与修复（commit `ce57538f`）：

1. **跨平台 artifact 路径 / SHA**：P0-2 在 `evidence_index.json` 里记录的是
   Windows 绝对路径（`E:\agent_dev\...`），Linux checkout 上不可解析；且记录的
   SHA 来自 CRLF 工作区，而 git 存的是 LF → 同一真实文件在 `evidence_index`
   与 `engine_metrics` 两个载体得到**两个不同 identity**，40 个合并塌成 0。
   新增 `evidence_read_model._resolve_artifact_path()`：用通用路径锚点
   （`M2_7_runs` / `M3_4_runs` / `research_runtime` / …）在当前 workspace
   重新解析并**哈希真实字节**。这是可移植性修复，不是伪造哈希。
2. **`Path.as_posix()` 在 Linux 不转换字面反斜杠**（Linux 上 `\` 不是分隔符）：
   `EntityLinkStore.add_link` 改为显式 `replace("\\", "/")`。
3. `test_m3_4_scheduler.py` 用了 `pytest.skip` 却未 `import pytest`（本地因其他
   模块先导入而侥幸通过）→ 补上 import。

`ce57538f` 远程 6/6 matrix **全绿**，证明修复有效。`ae5928a4` 在此之上只新增
一个跨平台路径解析回归测试。

---

## 8. 验收标准逐项核对

| 标准 | 结果 | 证据 |
|---|---|---|
| A. 迁移连续两次成功且结构一致 | ✅ | `evidence/migration_idempotency.json`：`identical=true`；真实 DB 12→14 表，两次 `ResearchStores()` 结构一致；原 12 表行数不变 |
| B. arr_m34_001 所有可确认 candidate 可追溯到 Objective / Campaign / Evidence / Experience | ✅（Experience 为明确 unresolved） | `integrity_validation_report.json`：9/9 candidate 由双来源确认，Objective/Campaign/Evidence 链路完整；每个候选 4 类缺口诚实标 unresolved（见 C） |
| C. 无法确认的关系明确列入 unresolved，不伪造 | ✅ | 36 条 unresolved，全部带 `no explicit ... in ...` 原因；哨兵 dst_id；`test_no_guessed_entity_ids` |
| D. ResearchService 与 Scheduler 同一 run Evidence 集合一致 | ✅ | `evidence/evidence_consistency.json`：`service_equals_scheduler=true`；`test_research_service_and_scheduler_agree_exactly` |
| E. 原始历史 records / verdict / metrics 不变 | ✅ | `historical_files_sha256.json` 89 文件前后一致；`db_schema_before/after.json` 历史表行数不变；`raw_verdict_census` 保留原字符串 |
| F. 所有测试通过 | ✅ | 本地 `193 passed, 37 subtests`；干净 checkout `134 passed, 3 skipped, 29 subtests`；pre-commit 3/3 |
| G. 根目录 GitHub Actions 实测成功 | ✅ | 最终 HEAD `ae5928a4` run `43606307331`：**6/6 matrix success**；修复验证轮 `ce57538f` run `43606011274` 亦 6/6 |
| H. 历史文件 SHA 快照证明未修改 | ✅ | `evidence/historical_files_sha256.json`（89 文件）+ `backfill_idempotency.json` |

> **B 的诚实说明**：M3.4 历史运行从未写入 `research_experience`（Phase 0 审计确认），
> 因此每个 candidate 的 Experience 关系是 **unresolved**，而不是伪造一条历史经验。
> M4.0 建立了可回答「最终形成了什么经验」的结构，但**没有把不存在的数据说成存在**。

---

## 9. 未解决问题

1. **M3.4 的 Objective / Hypothesis / Plan / Session / Experience 均为 M4.0 重建或 unresolved**：
   历史运行走 `run_dual_layer`，未经过 `ResearchService`，因此不能声称存在历史原生实体。
   M4.1 若要做自主选题，必须先让新运行把 objective 写进 `objectives` 表。
2. **3 个 CI skip**：M3.3 的 OOD 数据集完整性测试依赖引擎 checkout（`E:\LLM\SciNet++`），
   CI 无该目录；真实引擎 E2E 仍为 operator-driven，未冒充进 CI。
3. **`research.sqlite` 不入库**：运行时状态按设计 git-ignored；验收依赖提交的
   `p1_06_db_seed.json` 与本地真实 DB 快照。若换机器重跑 backfill，需要该 seed。
4. **`evidence_bundle` 未自动接线**：决策见 `evidence_bundle_decision.md`；其 docstring
   "wired into cli" 与事实不符，已记录但未改历史代码。
5. **三域 `hyp_fingerprint` 相同**：M4.0 用 `objective_id` 作用域隔离，未重命名原始 ID；
   未来若需全局唯一候选身份，应在 M4.1 设计 campaign-scoped ID，而不改历史。
6. **`UnifiedResearchScheduler` 仍未接入 `cli.py`**（M3.4 遗留），M4.0 未扩大范围。

---

## 10. 交付物清单与 Commit

本阶段共 **11 个独立 commit**（`2641e1c7..ae5928a4`），全部已 push 到
`origin/feature/arr-0.1-domain-registry`：

```
cbdf72af  M4.0-01 objectives 表 + 幂等迁移
b034707e  M4.0-02 entity_links 表 + 5 个 API
69ba03bd  M4.0-03 只读历史回填 + 报告 / SHA 快照
7a7a7559  M4.0-04 统一只读 Evidence 查询层（6 载体）
23ac08c6  M4.0-05 provenance 反例修复 + 完整性验证
2425c986  CI 自洽（seed helper + 测试调整）
17bc3ae3  CI seed 路径修正
92359d6e  根 workflow 纳入 4 个 M4.0 测试模块
ce57538f  修复远程 Linux CI 的 3 个真实失败
ae5928a4  跨平台路径解析回归测试
```

| 类别 | 文件 |
|---|---|
| 生产代码 | `research_runtime/research_store/objective_store.py`、`entity_link_store.py`、`schemas.py`、`__init__.py`、`research_runtime/evidence_read_model.py`、`research_service.py`、`scheduler.py` |
| SQLite migration | `objective_store.py:SCHEMA_SQL`、`entity_link_store.py:SCHEMA_SQL`（并入 `ALL_SCHEMA_SQL`） |
| Backfill 脚本 | `M4_0_runs/backfill.py` |
| Backfill report | `M4_0_runs/backfill_report.json` |
| Integrity report | `M4_0_runs/integrity_validation_report.json`、`M4_0_runs/validate_integrity.py` |
| 新增测试 | `research_runtime/tests/test_m4_0_{objective_store,entity_link_store,backfill,evidence_read_model}.py`、`m4_0_seed_helper.py`；`test_m3_4_scheduler.py` 追加 4 个反例 |
| 证据 | `M4_0_runs/evidence/*.json`（schema before/after、迁移幂等、历史 SHA、backfill 幂等、统一视图、一致性、P1-06 seed） |
| 文档 | `M4_0_runs/phase0_audit.md`、`M4_0_runs/evidence_bundle_decision.md`、本报告 |
| CI | `.github/workflows/arr-research-runtime.yml`（9 个 pytest 模块）+ §7 远程链接 |

**完成后暂停，不开发 M4.1，等待下一轮审查。**
