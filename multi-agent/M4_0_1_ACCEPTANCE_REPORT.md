# M4.0.1 验收报告 — Research Knowledge Foundation 收尾

> 结论先行：**M4.0.1 五项 P1 任务（P1-01~P1-05）全部完成、独立 commit 并推送远端，
> 全量 engine-free 回归 151 passed / 29 subtests passed / 0 failed**，未触碰 12 步
> fail-closed 统计门、未降门槛、未实现 M4.1 / Director、未引入第二套证据系统。
> 历史 Evidence / metrics / verdict / records 零改动。

---

## 1. 交付物总览

| 任务 | 主题 | Commit | 关键文件 |
|------|------|--------|----------|
| P1-01 | 修复 EvidenceReader campaign 前缀误匹配 | `fb519756` | `evidence_read_model.py` + `test_m4_0_evidence_read_model.py` |
| P1-02 | 修复 reconstructed Objective 关系语义 | `efdf5bd5` | `backfill.py` + `validate_integrity.py` + `test_m4_0_backfill.py` |
| P1-03 | 修正 SHA 验证统计口径（三向拆分） | `6bb3c20b` | `validate_integrity.py` + `test_m4_0_validate_integrity.py` + 重生成 `integrity_validation_report.json` |
| P1-04 | 统一 Reader 契约 + Discovery Adapter | `14be2cf0` | `discovery_evidence_adapter.py` + `test_m4_0_discovery_adapter.py` + `P1_04_reader_contract.md` |
| P1-05 | CI 验收 + 回归 + 验收报告 | **本提交** | `.github/workflows/arr-research-runtime.yml` + `M4_0_1_runs/` + 本报告 |

> 分支：`feature/arr-0.1-domain-registry`（未 merge main、未建 PR）。

---

## 2. 验收条件逐项核对（A–I）

| 条件 | 内容 | 状态 | 落点 |
|------|------|------|------|
| A | 当前 run 不读前缀相似其他 run | ✅ | P1-01：基于 `evidence_index.json` 精确 `campaign_id` 集合 + `_campaign_id_from_path`（`exact`/`collision`/`None`），碰撞文件进入 `unresolved` 不吸收。`test_campaign_prefix_collision_is_excluded_from_run` 实测。 |
| B | 重建关系不冒充原生 | ✅ | P1-02：objective→candidate 边**无条件** `reconstructed`；双来源确认另立 `identity_confirmed_by` 边。`test_reconstructed_objective_candidate_edges_are_never_confirmed` / `test_double_source_confirmed_candidate_still_reconstructed` 实测。 |
| C | SHA 统计可复算 | ✅ | P1-03：三向拆分字段 + `test_sha_stat_three_way_split_recomputable` 独立复算一致；同 campaign evidence 被多 candidate 引用不再报成多个文件。 |
| D | Discovery 边界明确 | ✅ | P1-04：`P1_04_reader_contract.md` 明确两套 Reader 边界；`DiscoveryEvidenceAdapter` 只委托统一读模型，静态校验不出现 `default_evidence_provider`。 |
| E | 历史 Evidence/metrics/verdict/records 不变 | ✅ | 所有脚本 READ-ONLY；`backfill.py` 失败即抛（SHA 快照前后一致）；`default_evidence_provider`、统计门、12 步链未改。 |
| F | 全量回归无退化 | ✅ | 本提交前 151 passed + 29 subtests passed + 0 failed（CI 模块全集本地重跑）。 |
| G | GitHub Actions 实测符合预期 | ⏳ | workflow 已纳入 2 个新 M4.0.1 测试模块，等待远端 Actions 跑 6/6 matrix（见 §4）。 |
| H | 每逻辑修改独立 commit 全推送 | ✅ | P1-01~P1-04 各自独立 commit 并推送；本提交为 P1-05。 |
| I | 不开发 M4.1 / 不建 Director / 不改变统计门 | ✅ | P1-04 仅交付契约 + 只读适配器 + 测试；未实现 M4.1；未建 Director；未改统计门槛。 |

---

## 3. 关键证据（P1-03 SHA 三向拆分，真实历史 `arr_m34_001`）

历史单一计数器把同一物理文件在每候选、跨域重复计数。拆分后：

| 指标 | 值 | 含义 |
|------|----|------|
| `n_unique_artifact_sha_verified` | **88** | 去重后的真实文件数（按 SHA-256） |
| `n_link_sha_verified` | **92** | 去重的证据 link 行数（run-root 级 `evidence_index.json`/`scheduler_state.json` 被 3 域共享 → 计 3 行但 1 文件） |
| `n_candidate_evidence_associations_verified` | **276** | 每 (candidate, evidence) 关联数（同文件被多候选引用 → 合法大于文件数） |
| `n_evidence_sha_verified`（历史，保留） | **276** | == 关联数，标注历史含义，**不再**把共享文件报成多个文件 |

不变量 `88 ≤ 92 ≤ 276` 成立（`test_sha_stat_three_way_split_present_and_ordered`）。
`integrity_validation_report.json` 已重生成含三新字段，历史值 276 不变。

其他 M4.0 不变式（本次重验）：`historical_files_unchanged=True`、
`cross_run_isolated=True`、`all_confirmable_chains_ok=True`、`problems=0`、
backfill `links_created={arr_m34_001: 155, p1_06: 10}`、`n_unresolved=36`（4 类缺失实体 × 3 域 × 3 候选）。

---

## 4. CI 状态与边界声明

- **矩阵**：`ubuntu-latest` × `windows-latest` × `py3.11/3.12/3.13` = **6/6**，已在
  `.github/workflows/arr-research-runtime.yml` 的 pytest gate 新增
  `test_m4_0_validate_integrity.py` 与 `test_m4_0_discovery_adapter.py`。
- **Engine-free**：CI 不签出 SciNet++ / 真实机器人 / 行情数据；research.sqlite 与 38 MB
  evidence.jsonl 被 git-ignore，测试从**已提交只读**的 `p1_06_db_seed.json` 重建等价 DB。
  因此 **CI 是 engine-free 回归，不等同真实引擎端到端**：真实引擎 E2E 仍为 operator 驱动、
  fail-closed 本地验证（与 M4.0 设计一致）。
- **3 个引擎 skip**：SciNet / Embodied / Quant 真实引擎运行不在本 engine-free gate 内执行
  （其真实数据测试 `test_m3_2/3_3/3_4_embodied_horizon` 读取已提交的历史产物，仍 engine-free）。
- **Pre-commit**：本仓库当前无 pre-commit 步骤；CI 通过 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
  规避可选 langsmith 插件崩溃，确保收集稳定。

---

## 5. 未做 / 边界（明确排除）

- ❌ **未实现 M4.1**（Research Discovery）：P1-04 仅提供 M4.1 必须依赖的只读契约与适配器。
- ❌ **未建 Autonomous Research Director**：M4.0 定位为 Research Knowledge Foundation，非 Director。
- ❌ **未改 12 步 fail-closed 门链** / **未降任何统计门槛**：`default_evidence_provider` 未改。
- ❌ **未引入第二套证据系统**：`DiscoveryEvidenceAdapter` 是统一读模型的薄门面，非新系统。
- ❌ **未 merge main / 未建 PR**（工程纪律）。

---

## 6. 交付物清单

- `M4_0_1_runs/integrity_validation_report.json` —— P1-03 三向拆分 + 不变式。
- `M4_0_1_runs/evidence/backfill_report.json` —— 回填报告（只读，155+10 links）。
- `M4_0_1_runs/evidence/seed_source.json` —— 种子来源（`real_db_copy` / `committed_p1_06_seed`）。
- `M4_0_runs/integrity_validation_report.json` —— 重生成（含三新字段，历史 276 保留）。
- `M4_0_runs/P1_04_reader_contract.md` —— Reader 契约文档。
- `.github/workflows/arr-research-runtime.yml` —— CI pytest gate 纳入 2 个新模块。
- 测试：`test_m4_0_validate_integrity.py`（3）、`test_m4_0_discovery_adapter.py`（5），全绿。
