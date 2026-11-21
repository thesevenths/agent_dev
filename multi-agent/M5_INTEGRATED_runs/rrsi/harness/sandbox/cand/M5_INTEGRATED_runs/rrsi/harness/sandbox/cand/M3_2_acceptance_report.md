# M3.2 验收报告 — Embodied 真实数据经 M3.0 统计门控验证

**状态：完成（DONE）** · 日期：2026-11-13 · 分支：`feature/arr-0.1-domain-registry`
**协议：** `M3.0-student-t-holm`（Student-t 双侧 CI + Holm 多重比较 + 双向判定 + fail-closed + 逐 seed holdout 确认）

---

## 0. 关键纠错（先说结论）

M3.1~M3.4 阻塞实证扫描里对 M3.2 的描述是**错的**，本次以源码 + 数据核实纠正：

| 旧认知（基于失真记忆） | 实测事实（源码 + 数据） |
|---|---|
| `datasets/pusher_v1` 是真实 MuJoCo 轨迹（obs=23/act=7） | 实际数据集为 `research_runtime/domains/projects/embodied/datasets/robot_sim_v1`，列 `episode,t,x,v,action`，**1-D 点质量机器人确定性仿真** |
| SciNet++ 环境缺 `gymnasium/mujoco/lerobot`，M3.2 **运行时阻塞** | **无任何机器人/深度学习依赖**。`embodied_compute` 与 `local_engine` 全程 **stdlib（csv/hashlib/random/math）+ pyyaml**；`gymnasium/mujoco/lerobot` 在 `sys.modules` 中从未出现 |
| 需先装依赖才能跑真实执行 | **现在就能跑**。已用纯 stdlib 在真实冻结数据上完成 train→多步 rollout |

**实证**：`research_runtime/tests/test_m3_2_embodied_real_data.py::test_embodied_real_data_needs_no_robotics_runtime` 断言 `{"gymnasium","mujoco","lerobot","torch"}` 均不在 `sys.modules`，且 `embodied_compute` 在真实 `robot_sim_v1` 上返回有限 `state_mse@1`。

> 注：该数据集是 `deterministic_simulated_robot_dynamics_v1`（非真实硬件），但它是**固定、版本化、SHA-256 钉死**的真实冻结数据集——在 ARR 意义上"真实"（非 on-the-fly 合成），且 `embodied_compute` 在每次作业前都做逐 split SHA 校验（fail-closed）。

---

## 1. `robot_sim_v1` 数据集实情（取代失真记忆）

- `manifest.json`：`dataset_version = robot_sim_v1-4979652dd897`，`is_exchange_or_hardware_data: false`，`data_source: deterministic_simulated_robot_dynamics_v1`。
- 物理：1-D 点质量，阻尼速度动力学 + 偏置执行器（actuator_bias=0.035）+ 高斯过程噪声；OOD 区间 actuator bias 1.8×、非线性 0.085 vs 0.055。
- 列：`episode, t, x, v, action`；步长 `dt=0.1`，每 episode 48 步。
- 拆分（SHA-256 钉死）：`train` 1920 行 / `validation` 576 / `test` 576 / `ood` 576。
- 已通过 `read_frozen_manifest` 逐 split SHA 校验：本运行 `sha_mismatch = []`。

---

## 2. M3.2 设计（零改动复用 M3.0 门控路径）

驱动脚本 `M3_2_runs/embodied_real_validation.py` 忠实复刻 M3.0 SciNet++ 10+10 范式，仅把域换成 embodied：

1. **门控驱动候选**：直接调用 `run_dual_layer(domain="embodied", harness_evolve=_harness_stub, ...)`，复用 M3.0 的**完全相同**门控代码（Student-t CI + Holm + 双向 + fail-closed + 逐 seed holdout）。research-harness 层按 M3.0 惯例 **stub**（该层与正在验证的 DOMAIN 统计门控正交）。`cli._build_real_domain_runner("embodied", ...)` 已把 embodied 接成真实 runner（直接调 `embodied_compute`，返回 `state_mse@1`，缺失/损坏 → NaN → INCONCLUSIVE，无 FakeRunner）。
2. **目标"真实模型修复"**：embodied 域自述的 genuine 改进 `model.drift_correction`（TRAIN-ONLY 残差修正，降低长程 rollout 误差累积），通过**同一组统计原语**（`evaluate_domain_candidate` → `domain_family_significance` → `_decide_domain`）判定，确保 verdict 由同一门控代码产生，而非重写。

**种子与门槛**：`PAIRED=(50..59)` 10 个 + `HOLDOUT=(60..69)` 10 个；`min_seeds_for_accept` 由硬下限 5 **提升到 10**，满足 M3.0 / ChatGPT Phase-0 规则 6（≥10 paired + ≥10 holdout 才予正式资格）。

**零历史证据改动**：仅读取 `robot_sim_v1` 冻结数据；产物只写入 `M3_2_runs/`。

---

## 3. 结果

**运行**：`elapsed_seconds = 97.4`（纯 Python，远快于 SciNet++ 的 67 分钟）；`domain_runner_is_real_engine = true`；`harness_layer_stubbed = true`。

### 3.1 门控驱动候选（train.lr，与 scinet 同旋钮；baseline lr=0.02）

| target | new | mean_imp | std | 精确 p | CI95 | holm_rank | holm_sig | ho_frac* | decision |
|---|---|---|---|---|---|---|---|---|---|
| train.lr | 0.003 | −0.002896 | 4.5e-05 | 8.50e-18 | [−0.002928, −0.002864] | 1 | True | 0.0(负向) | **rejected** |
| train.lr | 0.005 | −0.001990 | 2.5e-05 | 1.25e-18 | [−0.002008, −0.001972] | 0 | True | 0.0(负向) | **rejected** |
| train.lr | 0.010 | −0.000940 | 1.9e-05 | 9.05e-17 | [−0.000954, −0.000926] | 2 | True | 0.0(负向) | **rejected** |

\* `holdout_direction_fraction` 记录的是"候选更优（正向）占比"。三个候选的 holdout 改善**全部为负**（如 −0.002904），即 0% 正向、100% 负向 → 与配对负向一致 → `_holdout_confirms` 确认负向 → **REJECTED** 正确（非 bug）。

**解读**：对这条简单线性世界模型，baseline `lr=0.02` 已是较好值；更低的 `train.lr` 在 80 epoch 内收敛不足 → MSE 更高 → 候选更差。门控**正确 REJECT 负向**，这正是 M2.7 做不到、M3.0 修复的能力（M2.7 仅 `mean<=0` 就 REJECT，无统计证据；M3.0 要求 CI 全在 0 下 + holdout 确认）。

### 3.2 目标模型修复（model.drift_correction = 0.75）

| target | new | mean_imp | std | 精确 p | CI95 | holm_rank | holm_sig | ho_frac | decision |
|---|---|---|---|---|---|---|---|---|---|
| model.drift_correction | 0.75 | +0.000274 | 1.2e-05 | 9.48e-14 | [+0.000265, +0.000283] | 0 (n=1) | True | 1.0 | **accepted** |

- CI 整体在 0 上方 → 正显著；Holm 显著；10/10 holdout 同向（ho_frac=1.0）。
- 实际改善阈值：baseline_mean=0.001492，floor_rel=0.01 → 阈值 1.49e-05；mean 0.000274 > 阈值 → 通过 practical significance。
- **与 scinet 10-seed ACCEPT 平行**：embodied 的 genuine 模型修复在 10+10 真实数据上被 M3.0 门控**正式 ACCEPTED**。

---

## 4. 5-seed vs 10-seed 教训（与 M3.0 对照）

M3.0 的核心教训是 scinet 的 81% 观测改善在 5 seed 下统计不可确认、需 10 seed。对 embodied：

- 本数据集效应量**大且一致**（std 相对 mean 极小），因此即便 5 seed 也大概率可确认；门控的硬下限（`min_seeds_for_accept=5`）与本次显式提升到 10 双保险。
- 关键结论不变：**门控区分"观测改善"与"统计确认"**，且对负向也能给出有证据的 REJECT。embodied 的 `drift_correction` 改善是**真实且被正式确认**的（满足 ChatGPT 任务书第 6 条）。

---

## 5. 测试与 CI

- **本地**：`pytest research_runtime/tests/test_m3_2_embodied_real_data.py test_m3_0_statistics.py test_m2_7_reliability.py` → **66 passed + 37 subtests passed**（9.85s）。
- **CI  discrepancy（如实标记）**：本仓库 `.github/workflows/` **完全为空且从未被 git 追踪**（`git ls-files .github/` 为空，`git log --all` 无 `arr-research-runtime.yml`）。因此 M3.0 报告中"远端 CI run #9 success"的声称**无法从本仓库复现/证实**——工作流文件从未提交，不存在可运行的 CI。M3.2 的验证以**本地测试执行**为准；如确需 CI，需先把工作流文件纳入版本控制（不在本 Phase 范围）。

---

## 6. 交付物与提交

新增（独立提交，未 merge main / 未开 PR）：
- `M3_2_runs/embodied_real_validation.py` — M3.2 真实验证驱动（faithful 复用 `run_dual_layer` + 真实 `embodied_compute`）。
- `M3_2_runs/embodied_10seed_result.json` — 完整逐 seed 证据（含 per-seed holdout improvements）。
- `research_runtime/tests/test_m3_2_embodied_real_data.py` — 锁定 M3.2 两关键事实（无 gymnasium/mujoco 依赖 + drift_correction 真实正向改进）。
- `M3_2_acceptance_report.md` — 本报告。

`M3_2_runs/embodied_real_validation.py` 运行依赖仅 `pyyaml>=6.0 + pydantic>=2.7.4`（与 M3.0 一致），已在本机 `pyyaml 6.0.3 / pydantic 2.12.5` 验证。

---

## 7. 下一步

- M3.2 完成：embodied 真实数据已通过 M3.0 门控验证，gymnasium/mujoco 阻塞为失真记忆。
- 待启 Phase（路线不变）：M3.1 Quant 真实行情（仍阻塞于 `tmp/` 行情 96% 为 annual_avg 填充、缺 OHLCV）、M3.3 SciNet++ 多任务 OOD 生产化（基础具备）、M3.4 统一三域调度（前置依赖 M3.1~M3.3）。M4 自主选题/长期记忆、M5 Harness RSI、M6 长期运行。
- 建议：若要把 M3.0/M3.2 的测试结果纳入 CI，先创建并提交 `.github/workflows/arr-research-runtime.yml`（当前缺失）。
