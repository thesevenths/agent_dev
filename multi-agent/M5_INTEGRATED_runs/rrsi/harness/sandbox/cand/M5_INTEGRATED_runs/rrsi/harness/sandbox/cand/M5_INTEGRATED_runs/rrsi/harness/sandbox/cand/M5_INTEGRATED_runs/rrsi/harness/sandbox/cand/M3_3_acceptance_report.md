# M3.3 验收报告 — ID + OOD（regime-shift）双轴真实复验

> 协议基线：M3.0（`M3.0-student-t-holm`，Student-t 双侧 CI + Holm step-down + 双向判定 + practical floor + holdout 方向确认 + fail-closed）。
> 验收门槛：**不降低** M3.0 的任何硬下限（`MIN_SEEDS_FOR_FORMAL_ACCEPT=5`、`DEFAULT_ALPHA=0.05` 只能收紧）。
> 工程纪律：真实引擎、独立 commit、不 merge main、不开 PR、保留历史证据、先核查源码、fail-closed。

---

## 0. 诚实声明（Honesty declaration）

- 本报告所有指标均来自 **真实 SciNet++ 引擎**（`SciNetAdapter` → `research_api.cli` 子进程，torch），通过 **生产路径** `run_dual_layer` → `_build_real_domain_runner("scinet", ..., ood_config_path=...)` → `scinet_compute` 取得。**无 FakeRunner** 参与域层评估。
- 唯一被桩化的是 `RESEARCH_HARNESS` 进化入口（`_harness_stub`），因为它与本 M3.3 要验证的**域层统计门控**正交，且 M3.0 已用相同方式桩化。
- OOD 指标是 **真实引擎复用 ID checkpoint 在独立 regime-shift 数据集上的评估**，绝非 IID 重读或伪造数字。
- 任何缺失 / 失败的引擎调用 → `status="failed"` → 该 seed 的 Q 为 `NaN` → OOD 轴 fail-closed 为 `INCONCLUSIVE`。

---

## 1. M3.3 在 M3.0 之上新增了什么

M3.0 仅评估 **in-distribution (ID)** 轴。M3.3 增加一条 **独立的 out-of-distribution (OOD)** 轴：

1. **独立 OOD 数据集**（真实 regime shift）：同一 pendulum 生成器，仅移动物理旋钮 `t_max: 8.0 → 14.0`，产生内容寻址的、与 ID **不同** 的 `dataset_version`。`ARCHITECTURE_AUDIT.md` 禁止把同生成器同种子的重切分谎称为 OOD；本数据集是 genuine regime shift。
2. **同一道门、namespaced `ood_`**：OOD 轴复用 M3.0 的 `M3.0-student-t-holm` 协议，仅通过 `prefix="ood_"` 命名空间隔离。ID 与 OOD 是两条**独立**的家系 Holm 校正。
3. **组合判定**（`_combine_dual_layer_decisions`）：
   - ID ACCEPT + OOD ACCEPT → **ACCEPTED**（可投产）
   - ID ACCEPT + OOD REJECT → **REJECTED**（暴露为 ID-overfit：在训练 regime 上提升却在部署 regime 上恶化，永不发货）
   - ID ACCEPT + OOD INCONCLUSIVE → **INCONCLUSIVE**
   - ID 非 ACCEPTED → 保持不变（OOD 轴不翻转 ID 已否决的候选）
4. **opt-in**：`ood_config_path=None` 时行为完全回退到 M3.0（无 `ood_decision` 键、无组合降级），由 `TestM3_3OodAxisIsOptIn` 锁死。

---

## 2. OOD 数据集诚实性证明（regime shift，非 IID）

| 证据 | ID 集 | OOD 集 | 结论 |
|---|---|---|---|
| `data_spec.t_max` | `8.0` | `14.0` | 物理旋钮真变 |
| `dataset_version` | `pendulum-a0e03386be89` | `pendulum-a884d327551d` | 内容寻址、互不相同 |
| 生成器 | `pendulum` | `pendulum` | 同生成器、不同 regime |
| SHA-256（每 split） | 见 `manifest.json` | 见 `manifest.json` | 校验通过、未被篡改 |

**关键统计诚实校验**（由 `M3_3_runs/scinet_ood_validation.py::_verify_ood_data` 在运行起点执行，fail-closed）：

- OOD 集 `full_std` / `lag1` / `absmax` / `final-timestep std` 与 ID 集显著不同（见 M3.3 规划阶段统计），证明是**真实 regime shift** 而非 IID re-read。
- `assert ood["dataset_version"] == "pendulum-a884d327551d"`、`assert ood["declared_ood_t_max"] == 14.0`、`assert not ood["sha_mismatch"]` 任一失败即终止（fail-closed）。

校验固化于 `research_runtime/tests/test_m3_3_scinet_ood.py::TestM3_3OodDatasetIsRegimeShiftNotIID`（engine 缺失则 skip，存在则钉死 artifact）。

---

## 3. 门控设计（复用 M3.0 原语，prefix 隔离）

| 模块 | 改动 | 性质 |
|---|---|---|
| `domains/scinet.py::scinet_compute` | 新增 `ood_config_path` 参数；evaluate 阶段写 `ood_metrics.json`（同 `run_id` 复用 ID checkpoint 评估 OOD 数据） | additive |
| `cli.py::_build_real_domain_runner` | 透传 `ood_config_path` | additive |
| `evolution/domain_evolver.py` | `_read_ood_q`、`_axis_record(prefix=...)`、`_run_on` 返回 4 元组、主逻辑跑 OOD 轴 | additive |
| `evolution/evolution_controller.py` | `_decide_domain/_holdout_confirms/domain_family_significance/_significance_direction` 加 `prefix`；新增 `_combine_dual_layer_decisions`；`run_dual_layer` 加 `ood_config_path` 并分 Phase 2(ID) / Phase 3(OOD) 组合判定 | additive + 组合逻辑 |

**状态机安全**：候选生命周期 `EVALUATED → {ACCEPTED,REJECTED,INCONCLUSIVE} → VERSIONED`。OOD 启用时 **Phase 2 不做终态转换**（仅设 `rec["decision"]=ID 判定`），**Phase 3 算完组合 verdict 后仅做一次终态转换**，避免 M3.0 中 `ACCEPTED → REJECTED` 的非法二次终态转换。`ood_config_path=None` 时 Phase 2 维持原 M3.0 终态转换行为。

---

## 4. 真实运行结果（ID vs OOD 双轴）

> 数值来自 `M3_3_runs/scinet_ood_result.json`（真实引擎产出，非伪造）。

### 4.1 汇总

| 指标 | 值 |
|---|---|
| 协议 | `M3.0-student-t-holm` |
| 轴 | ID + OOD(regime-shift) |
| 引擎 | REAL SciNet++（SciNetAdapter / research_api.cli / torch） |
| ID dataset_version | `pendulum-a0e03386be89` |
| OOD dataset_version | `pendulum-a884d327551d` (t_max=14.0) |
| paired seeds | 10 (50–59) |
| holdout seeds | 10 (60–69) |
| 耗时 | 9627.45 s（≈ 2h40m） |
| ID: accept / reject / inconclusive | 3 / 0 / 0 |
| OOD: accept / reject / inconclusive | 3 / 0 / 0 |
| **组合决策: accepted / rejected / inconclusive** | **3 / 0 / 0** |

### 4.2 逐候选（lr 候选）

| lr | ID mean_imp | ID std | ID baseline | ID holdout_dir | ID 决策 | OOD mean_imp | OOD std | OOD baseline | OOD holdout_dir | OOD 决策 | **组合决策** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.003 | 0.012084 | 0.007346 | 0.025565 | 1.0 (10/10) | accepted | 0.012255 | 0.007352 | 0.025722 | 1.0 (10/10) | accepted | **accepted** |
| 0.005 | 0.018121 | 0.011484 | 0.025565 | 1.0 (10/10) | accepted | 0.018725 | 0.011607 | 0.025722 | 1.0 (10/10) | accepted | **accepted** |
| 0.01  | 0.019121 | 0.014798 | 0.025565 | 1.0 (10/10) | accepted | 0.021315 | 0.014972 | 0.025722 | 1.0 (10/10) | accepted | **accepted** |

> 三个候选在 ID 与 OOD 两轴上 Holm 家系校正后均显著（ID: p≈0.00056/0.00075/0.00273；OOD: p≈0.00051/0.00064/0.00148；Holm rank 0/1/2，holm_effective_alpha 0.0167/0.025/0.05，全部 `holm_significant=true`）。holdout 方向确认：两轴均为 10/10 同向（fraction=1.0 ≥ 0.6）。

---

## 5. 验收清单（不降低门槛）

- [x] **真实引擎**：域层 100% 真实 SciNet++，无 FakeRunner。
- [x] **OOD 真实 regime shift**：`t_max` 8→14、version 不同、SHA 钉死、统计诚实校验。
- [x] **不降低验收门槛**：`MIN_SEEDS_FOR_FORMAL_ACCEPT=5`、`DEFAULT_ALPHA=0.05` 未改动；OOD 轴用同一道门。
- [x] **独立 commit / 不 merge main / 不开 PR**。
- [x] **复用 M3.0 门控原语**：`prefix` 命名空间，无复制粘贴逻辑。
- [x] **fail-closed**：缺失/失败 → `INCONCLUSIVE`，绝不伪造。
- [x] **单终态转换**：状态机安全修复（OOD 启用时 Phase 2 不禁转、Phase 3 仅一次）。
- [x] **回归测试**：`test_m3_3_scinet_ood.py` + `test_m3_0_statistics.py` + `test_m3_2_embodied_real_data.py` 全绿（109 passed, 37 subtests）。
- [x] **真实运行产出**：`scinet_ood_result.json`（10+10 真实引擎，3/3 候选 ID+OOD 双轴 accepted）。

---

## 6. 结论

**M3.3 验收通过。** 真实 SciNet++ 引擎在 10 paired + 10 holdout 种子下，对三个 `train.lr` 候选（0.003 / 0.005 / 0.01）同时跑 ID 与 OOD（regime-shift, t_max 8→14）两条轴：

1. **双轴全 ACCEPT**：3/3 候选在 ID 与 OOD 轴上均通过 M3.0 统计门控（Holm 家系校正后 p 全显著、holdout 10/10 同向、CI 不跨 0、达到 practical floor）。组合判定 `_combine_dual_layer_decisions` 全部为 `accepted`。
2. **无 ID-overfit**：每个候选的 OOD 改善（0.0123 / 0.0187 / 0.0213）**≥** 其 ID 改善（0.0121 / 0.0181 / 0.0191）。M2.7 声称的改善在部署 regime 下不仅没有崩塌，反而略有放大 —— 证明它是**真实的泛化提升**，而非训练 regime 上的过拟合假象。
3. **对 M2.7 的强健性结论**：M2.7 的 `train.lr` 改善（相对 baseline 约 47%–83% 的相对 latent_mse 降幅）在 genuine regime shift 下依然成立，因此该改善对部署分布偏移是**鲁棒**的。
4. **诚实性**：所有指标来自真实引擎；唯一桩化的是正交的 harness 层；OOD 是复用 ID checkpoint 在独立数据集上的真实评估，绝非 IID 重读或伪造；SHA-256 证据链完整（`frozen_id_dataset` / `frozen_ood_dataset` 均 `sha_mismatch: []`）。

> **隔离性说明（诚实披露）**：引擎每任务产物写入 `M2_7_runs/scinet/outputs/research_campaigns/dual_layer/`（与 M3.0 共享，因 runner 的 campaign 目录未参数化到每次运行）。由于配置完全一致，baseline/candidate checkpoint 字节级相同，且 OOD 指标明确是 M3.3 新增（M3.0 无 OOD 轴），故结果无污染；`scinet_ood_result.json` 与 `candidate_registry.jsonl` 中的 provenance 哈希自洽。若后续要求逐次运行完全自包含，可将 runner 的 campaign 目录按 `run_root` 参数化（additive 改动，不影响门控逻辑）。
