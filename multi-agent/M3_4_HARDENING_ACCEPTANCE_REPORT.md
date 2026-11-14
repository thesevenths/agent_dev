# M3.4 HARDENING ACCEPTANCE REPORT

> **最终远程结论（非自述）**：`093fa0d3` 触发的
> [ARR Research Runtime run 43597630721](https://github.com/thesevenths/agent_dev/actions/runs/43597630721)
> 与 [Pre-commit checks run 43597630634](https://github.com/thesevenths/agent_dev/actions/runs/43597630634)
> 均为 GitHub Actions `conclusion=success`；前者的 6/6 matrix 全部 success。
> 中间曾出现真实失败并已修复，全过程见 §7（未隐瞒）。

- **状态说明**：本文按任务书要求列出修复内容、测试结果与 GitHub Actions 运行链接。
  依据工程纪律第 10 条，**本文件本身不宣布「通过」**；是否通过以本文 §6 记录的
  远程 GitHub Actions `conclusion` 为准，且任何未达标项在 §8 如实列出。

- 分支：`feature/arr-0.1-domain-registry`（**未 merge main、未开 PR**）
- 起始基线：`ef3302dfee2a4df696dc178b753d428891993af0`
- 本报告时 HEAD：`093fa0d3`（已推送至 `origin` 同名分支）
- 仓库：https://github.com/thesevenths/agent_dev （public）
- 工作目录：`multi-agent/`；仓库根：`E:/agent_dev`

---

## 1. Phase 0 源码审计发现（先复现，再改代码）

不依赖历史报告，逐文件读取真实代码 + 查询 GitHub REST API 得到的缺陷：

| # | 缺陷 | 证据 |
|---|---|---|
| A | `multi-agent/.github/workflows/ci.yml` 位于子目录，GitHub Actions **永不发现**；根目录实际生效的 `arr-research-runtime.yml` 只跑 4 个 M2.7/M3.0 unittest 模块 | `find .github -type f`；GitHub `/actions/workflows` 列表无 "ARR Engine-free CI" |
| B | `.pre-commit-config.yaml` **不存在**，而根 workflow 无条件执行 `pre-commit run --all-files` → 远程 `Pre-commit checks` 连续 failure | 本地复现 `InvalidConfigError: .pre-commit-config.yaml is not a file`；远程 run `43586499958` step 5 failure |
| C | `embodied.py` 把同一个 `rollout_horizon` 既当窗口长度又当固定 `HORIZONS` 键 | `VALUE_RULES` 允许 2..8，`HORIZONS=(1..5)` |
| D | `arr_m34_001/evidence_index.json` 中 `ood_dataset_version=null`，且无 manifest/checkpoint/run_id/OOD metrics | 实际文件；但磁盘上真实存在 40 组 `ood_metrics.json` |
| E | `scheduler_state.json` 的 `completed=[]`（被旧版恢复验证覆写），且无 4 个要求的计数字段 | 实际文件 + 复现脚本 |
| F | `longterm.py` 不在 `research_runtime/`，仅属 LangGraph Agent Runtime；6 个 M4 模块全仓零命中 | Grep / `research_runtime/` 树 |

**复现证据（修改前实测）**：

```
embodied_compute(h=2,3,4) -> IndexError: list index out of range
embodied_compute(h=5)     -> succeeded (默认，掩盖了 bug)
embodied_compute(h=6,7,8) -> KeyError: 6 / 7 / 8

scheduler.run() 第 1 次: completed=['scinet','embodied','quant']
scheduler.run() 第 2 次: completed=[]   <-- 历史被覆写
```

---

## 2. P0-1 Embodied horizon — 修复的具体文件和函数

**文件**：`multi-agent/research_runtime/domains/embodied.py`
**函数**：模块常量区、`VALUE_RULES`、`embodied_compute()` 的 `stage=evaluate` 分支。

| 项 | 修改前 | 修改后 |
|---|---|---|
| 语义 | 一个 `HORIZONS=(1..5)` 同时充当评估指标集与规划窗口 | 明确区分 `EVALUATION_HORIZONS=(1..5)`（固定指标契约，绑定 `REQUIRED_METRICS`/统计门）与 `model.rollout_horizon`（规划 horizon，2..8，`MIN/MAX_PLANNING_HORIZON` 常量为唯一真源） |
| 窗口 | `seq = group[start:start+horizon+1]`，h<5 时过短 → IndexError | `window_steps = max(max(EVALUATION_HORIZONS), planning_horizon)`，窗口恒 ≥6 |
| 误差表 | `{k: [] for k in HORIZONS}` → h>5 时 `errors[horizon]` KeyError | 覆盖两者并集；**只上报固定评估集**，规划 horizon 仅用于 `task_success_rate` |
| guardrail | `safety_violations` 随配置变化 | 只在 `EVALUATION_HORIZONS` 上累计，语义不随候选变 |
| 越界配置 | 可能崩溃 | `invalid_planning_horizon` **fail-closed** 返回 failed |
| 可审计性 | 无 | `rollout_metrics.json` 写出 `evaluation_horizons` + `planning_horizon` |
| 默认行为 | — | **逐位不变**：h=5 的 9 个指标与 180 个窗口与修改前 golden 完全一致 |

**新增测试**：`research_runtime/tests/test_m3_4_embodied_horizon.py`（2/3/4/5/6/7/8 全覆盖）
- 7 个 horizon 全部成功、无 IndexError/KeyError、9 个 `REQUIRED_METRICS` 有限、指标集恒为 1..5；
- 短 episode 不越界（fail-closed）、非法 horizon 不崩溃；
- golden 默认行为回归 + 新增「容差仍能检测真实变化」的防空洞测试；
- h=2..5 评估指标与规划 horizon 解耦。

---

## 3. P0-2 Evidence Index — OOD 全链路可追溯

**文件**：`multi-agent/research_runtime/scheduler.py`（新增只读收集器）、`multi-agent/M3_4_runs/rederive_evidence_index.py`（离线重派生）、`arr_m34_001/evidence_index.json` 等证据文件。

**函数**：`_sha256_file` / `_resolve_frozen_dir` / `_manifest_provenance` / `_collect_metric_artifacts` / `collect_domain_provenance`；`_run_domain`、`_summarize_domain`、`_build_evidence_index` 接线。

设计要点：
- **只读磁盘、绝不伪造**：通过通用 `data.frozen_dir` 定位 manifest，通过 `outputs/research_campaigns/<campaign_id>` + `*metrics*.json` 定位引擎产物；**按内容**（`config_path` / `dataset_version`）而非文件名区分 ID/OOD，跨引擎通用。
- 缺失字段显式列为 `missing` + `status=incomplete`；无 OOD 轴的域标 `not_applicable`。
- 磁盘证据优先于 record 里声明的版本，冲突显式保留 `version_conflict`。
- evidence index 升级 `m3.4-evidence-index-v2`，逐 decision 暴露门控**原始** OOD 统计量。

**真实运行 `arr_m34_001` 重派生结果**（不重跑实验、不改动任何 raw record / decision / 统计门槛）：

| 追溯项 | 值 |
|---|---|
| `provenance_status` | `complete`，`missing=[]` |
| OOD dataset_version | `pendulum-a884d327551d` |
| OOD manifest SHA-256 | `893ffd0e3ccc093b080852e7493dd4d75e4e7a668c38809099c03a796a1a0b26` |
| OOD config SHA-256 | `bc4fbdc0…96ab`（与 manifest 内声明值一致） |
| split SHA | 4 个 split 全部记录 |
| checkpoint SHA | 40 个 |
| evaluation run ID | 40 个 |
| OOD metrics 产物 | 40 组（含原始 `metrics`） |

原始 v1 索引与各域 summary **备份为 `*_v1_original.json`**；三域 `records` 与全部
`inconclusive` / `rejected` verdict 经逐字节比对**完全一致**（脚本内校验 `True`）。

---

## 4. P0-3 Scheduler Recovery — 单调状态与身份校验

**文件**：`research_runtime/scheduler.py`（`run()`、`_domain_identity`、`_validate_resume_identity`、构造函数 `strict_resume`）、`M3_4_runs/verify_recovery.py` + 恢复证据目录。

| 字段 | 语义 |
|---|---|
| `completed_this_invocation` | 本次**实际执行**成功的域 |
| `completed_total` | **累计**已完成域；由磁盘上真实完成的 `domain_summary.json` ∪ 历史 state ∪ 本次 推导，**单调不减**，可被覆写的 state 字段不再能清零它 |
| `resumed_domains` / `failed_domains` | 本次复用 / 失败的域 |
| `resume_validation` | 每域 `run_id` / `campaign_id` / `config_sha256` / `dataset_sha256` 四项校验结果 |

- 新增 **append-only** `run_history.jsonl`，每次 invocation 追加一条，不覆盖历史。
- 恢复验证**严格只读**：绝不重写 `domain_summary.json` / `candidate_registry.jsonl` / 引擎产物（有 SHA 快照测试证明）。
- `strict_resume=True`：身份不匹配时 **fail-closed 且不重跑**（重跑会覆盖历史证据）。
- 历史 v1 summary 无 identity 快照 → 标 `legacy_partial`（校验 `campaign_id` + `dataset_version`，SHA 明确标为不可验证），**绝不冒充 verified**。

**真实恢复证据**（在 `arr_m34_001` 的**副本**上运行，原始证据零触碰）：

```
engine_calls: 0
completed_this_invocation: []
completed_total: ['embodied', 'quant', 'scinet']
resumed_domains: ['scinet', 'embodied', 'quant']
failed_domains: []
historical_files_overwritten: []
original_run_unchanged: True
resume_validation: 3 x legacy_partial (campaign_id=True, dataset_version=True,
                                       config_sha256=None, dataset_sha256=None)
```

---

## 5. P0-4 CI — 根目录实际生效的 workflow + pre-commit

| 文件 | 变更 |
|---|---|
| `.github/workflows/arr-research-runtime.yml`（**仓库根，实际生效**） | 新增 step「Run M3.0 / M3.2 / M3.3 / M3.4 pytest gates (P0-4)」，在既有 `ubuntu/windows × py3.11/3.12/3.13` 6 个 matrix 上跑 5 个测试模块；安装 `pytest>=8`；失败时把 pytest 失败行发布为 `::error::` annotation（公开可诊断） |
| `.pre-commit-config.yaml`（**新建于仓库根**） | 3 个只读 `repo: local` hooks：Python AST、M3 证据 JSON 有效性、冲突标记；`files:` 精确限定 ARR 子树（实测匹配 92 / 27 / 614 个文件），不触碰 4000+ 无关历史文件；无 formatter、无网络依赖 |
| `multi-agent/tools/precommit_checks.py` | 自包含只读检查脚本（纯标准库）；已实测能真正拒绝冲突标记与非法 JSON |
| `multi-agent/.github/workflows/ci.yml` | 加 DEPRECATED 说明（**保留文件、不删除历史**），明确指向根 workflow |

**pre-commit 远程结果**：修复前 `Pre-commit checks` 连续 failure；修复后
run `43596455189` → **success**。

---

## 6. 测试结果与 GitHub Actions 运行链接

### 6.1 本地测试（可复现命令）

```bash
cd E:/agent_dev/multi-agent
# 模拟 CI：无 SciNet++ 引擎
SCINET_DIR="Z:/nonexistent_scinet" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest research_runtime/tests/{test_m3_0_statistics,
  test_m3_2_embodied_real_data,test_m3_3_scinet_ood,
  test_m3_4_scheduler,test_m3_4_embodied_horizon}.py -q
# -> 96 passed, 3 skipped, 29 subtests passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest research_runtime/tests/ -q
# -> 154 passed, 37 subtests passed

cd E:/agent_dev && python -m pre_commit run --all-files
# -> 3/3 Passed
```

3 个 skip 均为 `test_m3_3_scinet_ood.py` 中依赖引擎仓库 OOD 数据集的
self-skip（CI 无该 checkout），属设计内行为，**未被静默改成 PASS**。

### 6.2 远程 GitHub Actions（真实运行，非本地推断）

| Workflow | Run 链接 | Commit | 结论 |
|---|---|---|---|
| Pre-commit checks | https://github.com/thesevenths/agent_dev/actions/runs/43596455189 | `ed56e723` | **success**（修复前为 failure） |
| ARR Research Runtime (M2.7 + M3.0) | https://github.com/thesevenths/agent_dev/actions/runs/43596455203 | `ed56e723` | **failure**：6 个 matrix 中 4 绿（py3.12/3.13），2 红（ubuntu+windows 的 py3.11），根因见 §7 |
| ARR 1.0 Integration | https://github.com/thesevenths/agent_dev/actions/runs/43596455125 | `ed56e723` | 既有跨平台门禁（与 P0 无关） |
| ARR Research Runtime（诊断轮） | https://github.com/thesevenths/agent_dev/actions/runs/43597094422 | `c55bf13c` | py3.11 仍红，但 annotation 成功暴露确切失败行（见 §7） |
| **ARR Research Runtime（最终）** | https://github.com/thesevenths/agent_dev/actions/runs/43597630721 | `093fa0d3` | ✅ **success**，6/6 matrix 全绿（含 ubuntu+windows py3.11） |
| **Pre-commit checks（最终）** | https://github.com/thesevenths/agent_dev/actions/runs/43597630634 | `093fa0d3` | ✅ **success** |
| ARR 1.0 Integration（最终） | https://github.com/thesevenths/agent_dev/actions/runs/43597630645 | `093fa0d3` | 查询时仍 in_progress（既有跨平台门禁，与本轮 P0 无依赖） |

最终 run `43597630721` 的 matrix 明细（GitHub REST `/jobs` 实测）：

```
ubuntu-latest  / py3.11  success      windows-latest / py3.11  success
ubuntu-latest  / py3.12  success      windows-latest / py3.12  success
ubuntu-latest  / py3.13  success      windows-latest / py3.13  success
```

---

## 7. 远程 CI 失败的根因与修复（如实记录，未掩盖）

**第一次真实运行 `ed56e723` 并非全绿。** 通过 check-run annotations API 读到确切错误：

```
FAILED research_runtime/tests/test_m3_4_embodied_horizon.py::
  test_default_config_behaviour_is_bit_for_bit_unchanged
AssertionError: default behaviour changed for action_prediction_mse:
  0.2594051985526708 != 0.25940519855267075
```

根因：我在 golden 测试里要求浮点**位级相等**（`pytest.approx(abs=0, rel=0)`），
而 CPython 3.11 与 3.12+ 的浮点累加顺序产生 **1 ULP** 差异（约 5e-17）。
仅 py3.11 两个 OS 失败、3.12/3.13 全绿，与该解释完全吻合。

修复（commit `093fa0d3`）：
- 保留全部 9 个修改前 golden 数值作为回归基线；
- 比较改为 `rel=1e-12 / abs=1e-15`（只容忍平台级 1 ULP）；
- 新增 `test_golden_tolerance_still_detects_real_behaviour_change`：
  证明真实行为变化（h=5→h=6，窗口 180→168，`state_mse@1` 变化约 4e-5，
  比容差高约 8 个数量级）**仍会被捕获**，容差不会让 golden 测试变成空断言；
- **未改动任何统计门槛**：M3.0 Student-t / Holm / holdout 与全部 verdict 原样。

> 该 run 的链接与 conclusion 已回填至 §6.2；最终结论为全绿，见文件头「最终远程结论」。

---

## 8. 未解决问题 / 已知限制

1. **py3.11 已修复并验证**：`093fa0d3` 的远程 run `43597630721` 中 ubuntu+windows 的 py3.11 均 success。
2. **ARR 1.0 Integration 最终轮未纳入判定**：run `43597630645` 在报告成文时仍 `in_progress`。它是既有跨平台门禁，与本轮 P0 改动无依赖；其最终结论未在此声称。
2. **GitHub job 日志需 repo admin**：`actions/jobs/{id}/logs` 返回 403，故采用 annotation 转发失败行；这是可诊断性绕行，不是日志权限修复。
3. **历史运行 `arr_m34_001` 的 `scheduler_state.json` 仍是旧 v1**（`completed=[]`）：新代码在恢复时会从磁盘证据**修复** `completed_total`，但原始历史文件按纪律「不删除历史实验和证据」**未原地改写**；恢复验证在副本上完成。
4. **M3.3 的 3 个测试在 CI 上 self-skip**：它们校验引擎仓库内 OOD 数据集的完整性，CI 无该 checkout。真实引擎 E2E 仍是 operator-driven，未冒充进 CI。
5. **`UnifiedResearchScheduler` 未接入 `cli.py`**：目前仅 `M3_4_runs/run_unified.py` 驱动；见 M4 审计 §9。
6. **`evidence_bundle.build_bundle` 是幽灵生成器**：`cli.py` 只消费产物路径，不调用生成器。
7. **M4 仅完成架构审计**（`M4_ARCHITECTURE_AUDIT.md`），**未创建 `director/`、未写任何实现代码**，按任务书要求暂停等待新任务书。

---

## 9. 交付清单（本轮 7 个独立 commit，均已 push）

| Commit | 内容 |
|---|---|
| `084f15fe` | P0-1 embodied horizon 修复 + 7 个 horizon 测试 |
| `faabca12` | P0-2 OOD 全链路 provenance + 离线重派生 + 真实证据 |
| `900e990e` | P0-3 单调恢复状态 + 身份校验 + 恢复证据 |
| `ed56e723` | P0-4 根目录 CI workflow + pre-commit 配置 |
| `c55bf13c` | M4 架构审计（仅文档）+ CI 失败 annotation 诊断 |
| `093fa0d3` | 修复远程 py3.11 1-ULP golden 断言 |

验证：`git log ef3302df..HEAD`；远程 `origin/feature/arr-0.1-domain-registry` HEAD 与本地一致；
`main` 未合并、无 PR。

---

## 10. M4 建议（详见 `M4_ARCHITECTURE_AUDIT.md`）

- **不要新建 6 个模块**：`research_planner` / `research_memory` / `decision_engine` 已有生产等价物，重建会造成第 5 套 verdict 与重复 Memory。
- **真正缺口**：`objective` 主体 + 跨系统 ID 映射（7 套 ID 割裂、5 处 evidence 载体但 `ResearchService` 只读 1 处）。
- **建议形态**：复用 `research_store/db.py`（WAL + `BEGIN IMMEDIATE` + RLock + 幂等迁移），只**追加** `objectives` 与 `entity_links` 两张表，再加一个薄选题层；不引入向量库、不引入 ORM、不做大规模重构。
- 每阶段验收标准见审计文档 §8；核心要求：确定性、统计门槛不降、真实引擎非 FakeRunner、证据可从磁盘复算。
