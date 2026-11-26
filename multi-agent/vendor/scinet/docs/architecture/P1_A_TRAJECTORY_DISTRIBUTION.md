# P1-A — 轨迹分布分析与数据划分审计 (Trajectory Distribution & Split Audit)

**结论先行：pilot 数据被极端轨迹主导，不能直接作为第一阶段训练分布。**
但把强旋转轨迹**删除**是错误做法；正确做法是**预注册 regime 标签、按 regime 分层、把 stress-test 独立可识别地隔离为 eval-only**。

本报告覆盖任务书 **Phase 2（轨迹分布）** 与 **Phase 4（数据划分与泄漏审计）**。
所有数字来自 `scripts/analyze_p1a_pilot.py` 的实测输出：

- `datasets/controlled_pendulum_pilot/analysis/trajectory_distribution.json`
- `datasets/controlled_pendulum_pilot/analysis/split_proposal.json`
- `datasets/controlled_pendulum_pilot/analysis/figures/*.png`（6 张）

---

## 1. 全局分布（20 episodes × 201 timesteps / 200 transitions）

| 量 | min | p25 | p50 | p95 | p99 | max | std |
|----|-----|-----|-----|-----|-----|-----|-----|
| θ (rad, unwrapped) | -34.185 | -1.104 | **0.173** | 8.683 | 47.9 | **82.115** | **12.685** |
| ω (rad/s) | -27.663 | -1.955 | -0.816 | 7.010 | 20.7 | **51.264** | 7.193 |
| τ (N·m) | -4.986 | -2.500 | 0.000 | 3.472 | 4.90 | 4.9999 | 2.714 |
| Δθ / step (rad) | -0.552 | -0.041 | -0.017 | 0.141 | 0.40 | **1.0255** | 0.144 |
| \|Δθ\| / step (rad) | 0.000 | 0.010 | 0.042 | 0.238 | 0.53 | 1.0255 | 0.124 |

⚠️ **分布极度长尾**：θ 的中位数仅 0.173 rad，而最大值 82.115 rad，标准差 12.685 rad。
p50 与 p95 相差 50 倍，说明少数轨迹在数值尺度上压倒性主导。任何按全局均值/方差做的归一化都会被
这几条轨迹绑架。

---

## 2. 每场景统计

| 场景 | max\|θ\| (rad) | max\|ω\| (rad/s) | max\|Δθ\|/step | I (kg·m²) | τ_max/I (rad/s²) | E_barrier (J) | bounded / transitional / rotation |
|------|----------------|------------------|----------------|-----------|------------------|---------------|----------------------------------|
| nominal | 2.96 | 5.15 | 0.102 | 1.000 | 5.00 | 19.62 | 4 / 0 / 0 |
| **short_rod** | **82.12** | **51.26** | **1.025** | 0.090 | **55.56** | **5.89** | **2 / 1 / 1** |
| heavy_bob | 2.00 | 5.13 | 0.102 | 3.000 | 1.67 | 58.86 | 4 / 0 / 0 |
| high_damping | 2.00 | 4.30 | 0.086 | 1.000 | 5.00 | 19.62 | 4 / 0 / 0 |
| low_gravity | 9.14 | 5.16 | 0.103 | 1.000 | 5.00 | **3.24** | 2 / 2 / 0 |

**short_rod 与 low_gravity 贡献了全部 3 条越垒轨迹**：前者因为 `τ_max/I` 是 nominal 的 11.1 倍，
后者因为势垒只有 nominal 的 16.5%。

---

## 3. Bounded oscillation vs Continuous rotation（任务书要求分别统计）

### 3.1 可复现的分类判据（预注册，非事后调参）

以 `E_barrier = 2mgL` 为越顶能量阈值，`E = 0.5Iω² + mgL(1-cosθ)`：

| 判据 | 量 |
|------|----|
| `winding_number` | `(θ_T - θ_0)/(2π)` |
| `omega_sign_changes` | ω 符号翻转次数（往复摆动多，连续旋转少） |
| `fraction_timesteps_above_barrier` | `E > E_barrier` 的时间占比 |
| `continuous_rotation` | `\|winding\| ≥ 1` **且** `sign_changes ≤ 2` |
| `bounded_oscillation` | `θ_range < 2π` **且** `sign_changes ≥ 2` |
| `transitional` | 其余（越过势垒但未形成稳定单向旋转） |

### 3.2 分类结果（20 条逐条）

| 分类 | 条数 | 占比 |
|------|------|------|
| bounded_oscillation | **16** | 80% |
| transitional | **3** | 15% |
| continuous_rotation | **1** | 5% |

| episode | regime | winding | θ_final (rad) | 圈数 | ω 换向 | E>势垒占比 |
|---------|--------|---------|---------------|------|--------|------------|
| S0_random | bounded | -0.49 | -1.10 | -0.18 | 3 | 0.00 |
| S0_step | bounded | -0.79 | -2.96 | -0.47 | 2 | 0.02 |
| S0_sinusoid | bounded | -0.63 | -1.98 | -0.32 | 2 | 0.00 |
| S0_alternating | bounded | -0.53 | -1.30 | -0.21 | 3 | 0.00 |
| S1_random | bounded | +0.04 | +2.25 | +0.36 | 6 | 0.00 |
| **S1_step** | **rotation** | **+12.27** | **+79.11** | **+12.59** | **1** | **0.72** |
| **S1_sinusoid** | **transitional** | **-5.76** | **-34.19** | **-5.44** | 4 | **0.46** |
| S1_alternating | bounded | -0.10 | +1.39 | +0.22 | 5 | 0.00 |
| S2_random | bounded | -0.48 | -1.04 | -0.17 | 3 | 0.00 |
| S2_step | bounded | -0.49 | -1.07 | -0.17 | 3 | 0.00 |
| S2_sinusoid | bounded | -0.50 | -1.17 | -0.19 | 3 | 0.00 |
| S2_alternating | bounded | -0.52 | -1.30 | -0.21 | 3 | 0.00 |
| S3_random | bounded | -0.28 | +0.22 | +0.03 | 3 | 0.00 |
| S3_step | bounded | -0.34 | -0.13 | -0.02 | 3 | 0.00 |
| S3_sinusoid | bounded | -0.47 | -0.93 | -0.15 | 3 | 0.00 |
| S3_alternating | bounded | -0.31 | +0.07 | +0.01 | 3 | 0.00 |
| S4_random | transitional | -0.88 | -3.52 | -0.56 | 1 | 0.39 |
| **S4_step** | **transitional** | **+0.96** | **+8.05** | **+1.28** | 1 | **0.78** |
| S4_sinusoid | bounded | -0.57 | -1.59 | -0.25 | 2 | 0.20 |
| S4_alternating | bounded | -0.49 | -1.08 | -0.17 | 4 | 0.00 |

**注意**：ChatGPT 关注了 `S1_step` 的 79 rad；实测**还有 2 条**超出 ±2π（`S1_sinusoid` -34.19 rad、
`S4_step` +8.05 rad），共 **3/20 = 15%**。若只处理 79 rad 那一条，仍留下 2 条同类问题。

### 3.3 极端轨迹是否主导数据？——是

| 指标 | 值 |
|------|----|
| 单条最大 episode 占全部 \|θ\| 之和 | **53.5%**（`S1_step`） |
| 单条最大 episode 占全部 \|ω\| 之和 | **28.0%** |
| \|θ_final\| > 2π 的 episode | 3 条：`S1_step`, `S1_sinusoid`, `S4_step` |

**结论：极端轨迹确实主导数据集。** 1 条轨迹（5% 的样本）贡献了过半的 |θ| 总量。

---

## 4. 图表（6 张，均已生成）

| 文件 | 内容 | 关键观察 |
|------|------|----------|
| `figures/theta_trajectories.png` | 5 场景分面 θ(t) | 所有曲线 t=0 汇聚于 2.0（单一初态）；short_rod step 达 80 rad、sinusoid 达 -40 rad |
| `figures/omega_distribution.png` | 各场景 ω 直方图 | short_rod 有明显长尾至 ±50 rad/s，其余集中在 ±5 |
| `figures/torque_distribution.png` | 4 profile 的 τ 直方图 | step / alternating 仅 2 个离散值；random 连续 |
| `figures/delta_theta_distribution.png` | \|Δθ\|/step（对数 y）+ π 参考线 | 全部 < π (max 1.0255)，无虚假卷绕 |
| `figures/energy_vs_time.png` | 20 条 E(t) | short_rod / low_gravity 能量被力矩持续注入并越过势垒 |
| `figures/regime_classification.png` | winding vs ω 换向次数散点 | 清晰分离出 1 条 continuous_rotation 与 3 条 transitional |

---

## 5. Phase 4：数据划分与泄漏审计

### 5.1 原则

**必须按 episode 划分，绝不随机打散来自同一 episode 的重叠窗口。**
同一 episode 的窗口高度相关（相邻窗口只平移一个 stride），若随机分到 train/test 会造成时间窗口泄漏，
使 test 指标虚高。

### 5.2 ⚠️ 审计中发现并修复的一个真实缺陷

最初尝试"按场景分别做 70/15/15"：每个场景只有 4 条 episode，
`round(0.70*4)=3`、`round(0.15*4)=1`、余数 **0** → **test 集为空**。
已改为**全局 episode 级**划分（`analysis/split_proposal.json`）：

| split | episodes | 比例 | 目标 |
|-------|----------|------|------|
| train | **14** | 70% | 70% |
| validation | **3** | 15% | 15% |
| test | **3** | 15% | 15% |

- seed：`RandomState(master_seed % 2^31) = RandomState(20261111)`，确定性、可复现。
- dataset versioning：`dataset_version=cpilot-41bb7d668870`、`config_hash`、`npz_sha256` 均记录于 manifest；
  episode seed 规则 `SHA256("ms=<master>|s=<s_idx>|p=<p_idx>")[:4]`。

### 5.3 泄漏审计结果

| 检查 | 结果 |
|------|------|
| train ∩ validation | **空** ✅ |
| train ∩ test | **空** ✅ |
| validation ∩ test | **空** ✅ |
| 20 条是否各分配恰好一次 | ✅ |
| 是否存在空 split | ✅ 无 |

### 5.4 窗口隔离演示（`dynamics_batch_from_windows`, window_len=11, stride=5）

| split | episodes | windows | 每个 episode 的窗口数 | 全部窗口源属于本 split | 全部窗口有效 |
|-------|----------|---------|----------------------|------------------------|--------------|
| train | 14 | **546** | 39 | ✅ | ✅ |
| validation | 3 | **117** | 39 | ✅ | ✅ |
| test | 3 | **117** | 39 | ✅ | ✅ |
| 全量 | 20 | 780 | 39 | — | ✅ |

`dynamics_batch_from_windows` 已保证 `obs[s:e]` 与 `cond[s:e-1]` 严格落在单 episode 内
（`dynamics/contracts.py:297,301`），实测 780 个窗口的 `valid_mask` 全为 True、无跨 episode 拼接。

### 5.5 ⚠️ 但该划分**不能用于真实训练**

- 这是 **20 条 pilot 上的协议演示**，pilot 本身是验收产物而非可训练数据集。
- 3 条 test episode 只覆盖 **2 个场景**（`heavy_bob`, `low_gravity`），
  另外 3 个场景完全没有 test 覆盖 → 无法评估参数 OOD 泛化。
- 除 `S1_step` 外的 stress-test 轨迹未独立成组，评估会被 regime 混杂污染。
- **正式方案**：在全量数据集上按 `(scenario × profile × regime × initial-condition bin)` **分层**划分，
  为强旋转 regime 单独设 `stress_test` / `ood_rotation` 评估集，并使用独立 `dataset_version` 与 manifest。

---

## 6. 有科学依据的训练分布建议（仅提议，未执行）

### 6.1 用无量纲控制强度替代绝对力矩

当前所有场景共用一个 `τ_max = 5.0 N·m`，但 `I=mL²` 从 0.09 到 3.0，导致 `τ_max/I` 从
**1.67 到 55.56 rad/s²（33 倍）**。这不是"5 个可比场景"，而是 5 个动力学强度差异极大的系统。

建议以 **`τ_max/I`（角加速度尺度）** 和 **`E_barrier = 2mgL`（越垒能量）** 两个量共同标定：

| 场景 | I | E_barrier | E0 | 保持 `τ_max/I = 5` 时的 τ_max (N·m) | 与现值 5.0 的差异 |
|------|---|-----------|----|------------------------------------|-------------------|
| nominal | 1.000 | 19.62 | 13.89 | 5.00 | 不变 |
| **short_rod** | 0.090 | **5.89** | 4.17 | **0.45** | **需降至 1/11** |
| heavy_bob | 3.000 | 58.86 | 41.68 | 15.00 | 可提高 3× |
| high_damping | 1.000 | 19.62 | 13.89 | 5.00 | 不变 |
| **low_gravity** | 1.000 | **3.24** | 2.29 | 5.00 | 势垒仍过低，需单独处理 |

低重力场景即使角加速度与 nominal 相同，因 `E_barrier` 只有 3.24 J（E0 已占 71%），仍容易越垒；
应作为独立的"低势垒"实验组，而不是混入常规摆动组。

### 6.2 建议的训练 / 评估分布

| 集合 | 内容 | 判据 |
|------|------|------|
| **train（第一阶段）** | bounded oscillation 主导 | `\|winding\| < 1` 且 `sign_changes ≥ 2`，`E<E_barrier` 时间占比 > 90% |
| **validation** | 同 regime，不同 seed / 不同初态 | 用于超参与 early-stop，**不**含 stress-test |
| **frozen_test** | 同 regime，独立 seed，冻结 + SHA-256 | 主指标 |
| **stress_test / ood_rotation**（eval-only，独立可识别） | `S1_step`, `S1_sinusoid`, `S4_step` 及全量数据中同 regime 的轨迹 | `\|winding\| ≥ 1` 或 `E>E_barrier` 占比 > 30% |
| **parameter_ood** | 参考既有 `P2_03_ACTION_JEPA.md` 的 4 组 × 75 episodes 设计（mass / length / damping / joint） | 单参数越界，其余留 train 范围 |

### 6.3 保持 stress-test 独立可识别的机制（建议，未实现）

在 manifest 每个 episode 记录预注册的 regime 元数据，使下游无需重算即可分层：

```yaml
episode_regime: bounded_oscillation | transitional | continuous_rotation
winding_number: <float>
fraction_timesteps_above_barrier: <float>
is_stress_test: <bool>
split: train | validation | frozen_test | stress_test | parameter_ood
```

并在 config 中显式声明 `training.include_stress_test: false`（默认），
把 stress-test 作为**独立 Eval-Only 组**，绝不参与归一化统计拟合（train-only normalization）。

### 6.4 补齐初态分布（与 `P1_A_SCIENTIFIC_REVIEW.md` S1 呼应）

`θ0 ~ U(-π, π)`、`ω0 ~ U(-ω_max, ω_max)`，逐 episode 记录初态；
否则即使有 20,000 条 episode，模型也只见过一个起点。

### 6.5 数值精度

对 `short_rod` 采用更小 `dt`（如 0.005，实测误差从 7.6e-3 降至 1.1e-5）或降低 `τ_max`，
再按统一输出网格重采样，避免把积分器误差当作物理标签噪声。

---

## 7. 验收结论

| 项 | 结论 |
|----|------|
| 数据完整性 / 可复现 / 契约 | ✅ 通过 |
| episode 级划分与窗口隔离 | ✅ 协议正确、零泄漏；但 pilot 规模不足以支撑正式划分 |
| 极端轨迹主导 | ❌ **是**（单条 53.5% \|θ\|），必须隔离 |
| regime 分层清晰可辨 | ✅ 16 / 3 / 1，判据已预注册且可复现 |
| **批准全量数据生成** | ⛔ **建议暂缓**，待 §6 的训练分布方案经人工确认 |

---

## 8. 限制声明

未修改任何模拟器参数、生产代码、pilot 数据、manifest 或受限文件；
未生成全量数据、未训练 JEPA、未实现 MPC、未合并 main。
本报告的全部内容是**分析与建议**，不是实现。

## 9. 复现

```bash
python scripts/analyze_p1a_pilot.py --config configs/dynamics/controlled_pendulum_jepa.yaml
# -> analysis/trajectory_distribution.json, split_proposal.json, figures/*.png
```
