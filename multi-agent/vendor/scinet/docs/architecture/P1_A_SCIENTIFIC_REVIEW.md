# P1-A — 独立科学复核报告 (Scientific Review)

**结论先行：P0-fix.1 通过；P1-A 模拟器与数据管线的物理/数值实现正确，但 pilot 数据【尚不适合】作为 JEPA 第一阶段训练集。建议「有条件通过」并暂缓全量生成。**

本报告**不依赖**既有 `P1_A_PILOT_REPORT.md` 的 8/8 PASS 结论。所有数字来自新建的独立分析脚本
`scripts/analyze_p1a_pilot.py` 的实际运行输出，机器可读证据在
`datasets/controlled_pendulum_pilot/analysis/*.json`，图表在同目录 `figures/`。

**未修改任何模拟器参数、生产代码、受限文件、pilot 数据或 manifest**（见文末限制清单）。

---

## 1. 总体判定

| 维度 | 判定 | 依据 |
|------|------|------|
| 物理方程正确性 | ✅ 通过 | 独立 `scipy.integrate.solve_ivp(DOP853, rtol=1e-12)` 交叉验证；能量平衡恒等式残差 ≤ 1.16e-3 |
| 数值积分实现 | ✅ 通过（有边界） | 步减半误差按 ~20–27× 递减，确认 4 阶 RK4；但 `dt=0.02` 在 short_rod 下与高精度解差 **7.63e-3 rad** |
| 数据完整性与可复现 | ✅ 通过 | SHA-256 独立重算一致；20/20 episode 逐位重放一致；`DynamicsBatch(float64)` 校验通过 |
| **物理参数合理性** | ⚠️ **有条件** | `b` 是归一化阻尼率而非物理粘性系数；5 场景 `τ_max/I` 相差 **33 倍**（1.67 → 55.56 rad/s²） |
| **数据分布适合训练** | ❌ **未通过** | 单一初始条件；单条极端轨迹占 \|θ\| 总量 **53.5%**；3 条轨迹超出 ±2π |
| **可学习性 / 泛化性** | ⏸ **未证明** | 8/8 PASS 只验证数据生成管线自洽，不含任何模型训练证据 |

---

## 2. 独立验证方法与既有 audit 的差异（关键）

既有 8/8 audit 与本报告用的是**同一套仿真器代码**（`ControlledPendulum.rk4_step` / `simulate_substep`），
因此其"RK4 收敛 15.86"只证明**实现符合自身的 4 阶公式**，不能排除方程写错。本报告补上外部基准：

| 项目 | 既有 8/8 audit | 本报告（独立） |
|------|----------------|----------------|
| ODE 参考解 | 自研 `simulate_substep`（同一 `derivatives`） | **`scipy.solve_ivp(DOP853, rtol=1e-12)`**，从方程重新实现 `f(t,y)`，不复用仿真器任何函数 |
| 零力矩检验 | 小角解析解 + `b=0, τ=0` 能量守恒 | 追加 **能量平衡恒等式** `dE/dt = τω - bIω²`（含外力矩与阻尼，更强） |
| SHA-256 | 用 collect 脚本哈希函数 | 直接用 `hashlib` 重算 npz 字节 |
| 重放 | 复用 `generate_episode` | 从 manifest 独立重建 `ScenarioSpec/ProfileSpec` + 重新派生 seed |
| 分布分析 | 无 | 新增 6 张图 + 4 份 JSON，覆盖 θ/ω/τ/Δθ/能量/场景/旋转分类 |

---

## 3. 物理方程与单位核对

源码 `data/controlled_pendulum.py:152-162`（`derivatives`）、`:149`（`inertia`）：

```
theta_dot = omega
omega_dot = -(g/L)*sin(theta) - b*omega + torque/(m*L^2)
I = m*L^2
```

| 量 | 单位 | 核对 |
|----|------|------|
| θ | rad（unwrapped） | ✅ 从不 `mod 2π` |
| ω | rad/s | ✅ |
| τ (torque) | N·m | ✅ 被 `I` 归一化为 rad/s² |
| g | m/s² | ✅ 9.81 / 1.62 |
| L | m | ✅ 1.0 / 0.3 |
| m | kg | ✅ 1.0 / 3.0 |
| dt | s | ✅ 0.02 |
| I = mL² | kg·m² | ✅ |
| **b** | **1/s（归一化阻尼率）** | ⚠️ 见下 |

### 3.1 ⚠️ 发现 S4：`b` 的语义必须写清（不是错误，是文档缺口）

`- b*omega` 直接乘 ω，而 τ 项除以了 `I`。因此 `b` 是 **c/(mL²)**（归一化阻尼率，单位 1/s），
**不是**物理粘性阻尼系数 `c`（单位 N·m·s）。后果：5 场景 `I` 从 0.09 到 3.0，若把 `b` 误读为 `c`，
则同一 `b=0.1` 在各场景对应的物理阻尼系数相差 33 倍。当前实现在"归一化阻尼率一致"的意义下是
**自洽**的，但 `controlled_pendulum_jepa.yaml` 与模块 docstring 均未声明该语义。

> **建议（未执行）**：在配置与文档中把 `b` 明确标注为 `normalized_damping_rate_1_per_s`，
> 并注明 `c = b * m * L^2`。不改动数值。

### 3.2 五场景物理参数（独立重算）

| 场景 | g | L | m | b | I=mL² | τ_max/I (rad/s²) | E_barrier=2mgL (J) | E0 (J) |
|------|---|---|---|---|-------|------------------|--------------------|--------|
| nominal | 9.81 | 1.0 | 1.0 | 0.1 | 1.000 | **5.00** | 19.62 | 13.89 |
| short_rod | 9.81 | **0.3** | 1.0 | 0.1 | **0.090** | **55.56** | **5.89** | **4.17** |
| heavy_bob | 9.81 | 1.0 | 3.0 | 0.1 | 3.000 | **1.67** | 58.86 | 41.68 |
| high_damping | 9.81 | 1.0 | 1.0 | **1.0** | 1.000 | 5.00 | 19.62 | 13.89 |
| low_gravity | **1.62** | 1.0 | 1.0 | 0.1 | 1.000 | 5.00 | **3.24** | 2.29 |

**这是 79 rad 的定量根因**：`short_rod` 的控制角加速度是 nominal 的 **11.1×**，而越顶势垒只有 nominal 的
**0.30×**，且初态能量 `E0=4.17 J` 已达势垒 `5.89 J` 的 **71%**。三者叠加使"轻微驱动即可连续越顶旋转"。
`low_gravity` 同理（势垒仅 3.24 J）。

---

## 4. RK4 数值精度（独立交叉验证）

### 4.1 与 scipy DOP853 的逐 episode 状态误差

`independent_verification.json → independent_ode_crosscheck_scipy_dop853`

| episode | max \|state error\| vs scipy |
|---------|------------------------------|
| **S1_step** | **7.628e-03** |
| S1_sinusoid | 3.149e-04 |
| S1_random | 2.373e-04 |
| S1_alternating | 8.344e-05 |
| S0_step | 5.184e-06 |
| 其余 15 条 | ≤ 2.9e-06 |

**全局最大 7.628e-03 rad，全部集中在 short_rod**。这比既有 audit 报告的"小角解析解误差 1.35e-4"大
约 56 倍 —— 不是矛盾，而是**检查条件不同**：既有 audit 用小角度、零力矩、b=0 的温和情形；
本报告覆盖全部 20 条真实 episode（含强驱动）。

### 4.2 步减半收敛（确认是截断误差，不是方程错）

`step_convergence.json`，对最坏的 `S1_step`，用 `np.repeat` 保持**完全相同**的物理力矩波形：

| dt | max \|err\| vs scipy | 缩减比 |
|----|----------------------|--------|
| 0.02000 | 7.628e-03 | — |
| 0.01000 | 2.804e-04 | 27.20× |
| 0.00500 | 1.095e-05 | 25.62× |
| 0.00250 | 4.744e-07 | 23.08× |
| 0.00125 | 2.287e-08 | 20.74× |

误差按 4 阶规律收敛（理论 16×，实测 20–27×），**证明方程正确、误差来自 RK4 截断**。

> ⚠️ **发现 S3（数值精度边界）**：`dt=0.02` 对 short_rod 偏粗，单步目标含 ~7.6e-3 rad 的积分器误差。
> 相对该轨迹 79 rad 的总量仅 ~1e-4，但相对"单步 Δθ≈1.0 rad"是 ~0.8% 的标签噪声。
> 若 JEPA 要学精确的单步动力学，该场景应使用更小 `dt` 或降低 `τ_max`。**本报告不改参数，仅提出。**

---

## 5. 能量行为（比守恒更强的检验）

解析恒等式（由方程直接推导）：

```
E = 0.5*I*ω² + m*g*L*(1 - cos θ)
dE/dt = τ·ω - b·I·ω²          (τ 做功 - 阻尼耗散)
```

用梯形近似逐步比对 `ΔE_observed` 与 `∫power dt`：

| episode | 相对残差 |
|---------|----------|
| S1_step | 1.163e-03 |
| S1_alternating | 1.095e-03 |
| S1_sinusoid | 1.040e-03 |
| S1_random | 9.162e-04 |
| S2_alternating | 4.469e-04 |
| 其余 | 更小 |

全局最大 **1.163e-03**。残差主要来自梯形法自身的 O(dt²) 近似，且同样集中在高 ω 的 short_rod。
这**独立确认了 τ/(mL²) 与 -bω 两项符号和系数都正确**（若符号错，残差会是 O(1) 而非 1e-3）。

---

## 6. ⚠️ 发现 S1：初始条件退化为单点（最高优先级）

```
initial_condition_distribution:
  distinct_theta0 = [2.0]
  distinct_omega0 = [0.0]
  n_distinct_initial_conditions = 1
```

20 条 episode **全部**从 `(θ0=2.0, ω0=0.0)` 出发。图 `figures/theta_trajectories.png` 中 5 个子图
所有曲线在 t=0 都汇聚于 2.0，直观可见。

**后果**：pilot 只覆盖"单一初态 + 5 组物理参数 + 4 种动作序列"。它无法支持任何关于
"模型对不同初始状态泛化"的结论，也无法验证 JEPA 学到的是状态-动作联合动力学。
若直接按此配置生成全量数据（仅增加 seed 数量），这个缺陷会被放大到整个数据集。

> **建议（未执行）**：全量数据必须显式采样初态分布，例如
> `θ0 ~ U(-π, π)`、`ω0 ~ U(-ω_max, ω_max)`，并在 manifest 中记录每个 episode 的初态与其分布参数。
> 这是比 79 rad 更根本的科学限制。

---

## 7. ⚠️ 发现 S5：动作分布与执行器边界

`independent_verification.json → action_distribution`：

| 指标 | 实测 |
|------|------|
| 全局 τ 范围 | [-4.9863, +4.9999] N·m，边界 ±5.0 |
| 越界样本数 | **0** |
| 99% 饱和比例 | 0.00225（几乎不饱和，执行器范围合理 ✅） |
| 全局 mean / std | 0.0187 / 2.714 |

按 profile：

| profile | 取值 | std | **unique values** | zero_fraction |
|---------|------|-----|-------------------|---------------|
| random | [-4.986, 4.999] | 2.843 | **1000** | 0.00 |
| step | ±3.0 | 3.000 | **2** | 0.00 |
| sinusoid | ±3.5 | 2.475 | **51** | 0.02 |
| alternating | ±2.5 | 2.500 | **2** | 0.00 |

⚠️ 观察：
- `step` / `alternating` 各只有 **2 个**动作取值 → 动作多样性低，模型可能靠"识别 profile 类型"而非真正积分力矩来预测。
- **数据集中没有纯 τ=0 的 episode**（`zero_fraction` 最大仅 0.02），而 E0（Persistence）与 E4（零动作）消融
  需要无控基线；目前 τ=0 只存在于 audit 的独立探针中。
- 所有 episode 的初态相同，导致"动作序列 → 轨迹"近乎一一映射，缺乏"同状态不同动作 / 同动作不同状态"
  的配对覆盖（见 §9 的受控对照，那是在分析脚本里额外生成的，不在 pilot 数据里）。

---

## 8. 角度展开 / 完整性 / 契约（复核通过项）

| 检查 | 结果 |
|------|------|
| Unwrapped θ | ✅ 从不 `mod 2π`；`θ0=2π+0.1` 与 `θ0=0.1` 两条轨迹全程保持精确 2π 偏移（单测 `test_unwrapped_theta_not_reduced_mod_2pi`） |
| 单步 \|Δθ\| 最大值 | **1.0255 rad** < π（无虚假卷绕跳变） |
| SHA-256（独立 hashlib 重算） | ✅ `8d769ff5…` 与 manifest 一致 |
| 确定性重放 | ✅ 20/20 obs 与 cond **逐位一致**，seed 派生 20/20 一致 |
| `DynamicsBatch` float64 | ✅ `B=20, T=200, D=2, U=1, conditioned=True`, 全部 4000 个 transition 有效 |
| 有限值 | ✅ 无 NaN/Inf |

---

## 9. short_rod 79 rad 专题（ChatGPT 关注点）

### 9.1 事实核对

| 项 | 值 |
|----|----|
| 轨迹 | `S1_step`（scenario=short_rod, profile=step） |
| θ_final | **79.112 rad** = **12.59 圈** |
| winding number | **+12.27** |
| max \|ω\| | 51.26 rad/s（全局最大） |
| ω 换向次数 | **1**（几乎不反向 → 持续旋转） |
| E > E_barrier 的时间占比 | **0.72** |
| 且不止一条：`S1_sinusoid` | θ_f = **-34.19 rad**（-5.44 圈），winding -5.76 |
| `S4_step`（low_gravity） | θ_f = **+8.05 rad**（+1.28 圈） |

**超出 ±2π 的共 3 条**（ChatGPT 提到的只是最极端的一条）。分布被极端轨迹主导：
`share_of_abs_theta_from_top1_episode = 0.535` —— **单条 `S1_step` 占全部 \|θ\| 的 53.5%**；
全局 θ 的 `std = 12.68 rad`、`p50 = 0.17 rad`、`max = 82.12 rad`，是极端长尾分布。

### 9.2 判断

- **数学上正确**：unwrapped θ 保留累计转角是刻意设计，避免人为 ±2π 跳变；scipy 交叉验证与能量平衡
  都确认该轨迹是方程的真解（误差 7.6e-3 属数值精度，不是物理错误）。
- **实验设计上不宜混入第一阶段训练集**：它属于"强驱动连续旋转"regime，与其余 16 条 bounded
  oscillation 是**不同物理机制**。混在一起会让归一化统计、latent 尺度、loss 被单一 regime 主导，
  并使"JEPA 是否学到动作条件动力学"的结论不可解释。
- **不要凭空规定最大力矩**：应由 `I=mL²`、`E_barrier=2mgL` 与期望的 `τ/I` 共同决定（见 §3.2 表）。

### 9.3 建议（均未执行，等待人工决策）

1. **不删除** `short_rod` 强驱动轨迹，保留为独立、可识别的 **stress-test / OOD 评估集**。
2. 用 `winding_number`、`E > E_barrier` 占比、ω 换向次数三个量做**预注册的 regime 标签**，写入 manifest，
   让训练/评估可按 regime 分层。
3. 第一阶段训练分布以 bounded oscillation 为主；若需覆盖旋转 regime，单独设实验组并单独报告。
4. 若要让各场景动力学强度可比，按 `τ_max ∝ I`（保持 `τ_max/I` 一致）配置每场景力矩，
   而不是全局一个 5.0 N·m。当前 5 场景 `τ_max/I` 从 1.67 到 55.56，相差 **33 倍**。

---

## 10. Phase 3：动作信息分析（受控对照）

`analysis/action_information.json`。设计：**同一 nominal 场景、同一初态 (2.0, 0.0)**，仅改变力矩序列；
`shuffled` / `reversed` 复用 `random` 的**完全相同动作多重集**（只做排列）。

| 臂 | 终态 θ (rad) | 终态 ω (rad/s) | 末能量 (J) | 净功 (J) |
|----|--------------|----------------|------------|----------|
| zero (τ=0) | -1.294 | 2.366 | 9.925 | 0.000 |
| positive_constant (+2.5) | -0.273 | 3.197 | 5.473 | -5.761 |
| negative_constant (-2.5) | **-23.016** | -9.340 | 58.526 | +62.306 |
| random | -1.741 | 0.669 | 11.698 | +2.059 |
| shuffled_random | -1.067 | 2.372 | 7.892 | -2.539 |
| reversed_random | -1.589 | 1.664 | 11.375 | +1.177 |

| 与 positive_constant 的 \|Δθ_final\| | 值 |
|---|---|
| vs zero | 1.021 |
| vs negative_constant | **22.743** |
| vs random | 1.468 |
| vs **shuffled_random** | **0.794** |
| vs reversed_random | 1.316 |

**同一动作、不同初态**（4 个初态）→ 4 个互不相同的终态 θ（-0.273 / -0.549 / 0.751 / 0.418）。

### 这些结果建立了什么、没建立什么

| 建立了 ✅ | 没建立 ❌ |
|-----------|-----------|
| 相同初态 + 不同动作 → 不同未来（动作敏感性） | 任何模型**已学会**利用动作 |
| 相同动作 + 不同初态 → 不同未来（状态依赖性） | E2 优于 E1 的收益来自动作信息 |
| 动作**顺序**重要：shuffled 与 random 终态差 0.674 rad | 打乱动作对照在训练时一定失效 |
| 力矩做功与能量变化方向一致（负力矩注入 +62.3 J → E 升至 58.5 J），物理自洽 | 多步预测不会发散 / latent 不 collapse |

> **关键区别（呼应 ChatGPT 判断）**：以上全部是**数据生成过程**的属性。真正证明"模型利用了动作"，
> 必须在 P2-03A 的 E2 vs E3(shuffle)/E4(zero) 消融中，用 paired bootstrap CI 验证
> `Δ_action = L_without_action - L_with_action > 0` 且 shuffle 对照回退。详见 `P2_03A_READINESS_PLAN.md`。

---

## 11. "8/8 PASS" 的证明边界（必须明确）

| 8/8 PASS 证明了 | 8/8 PASS **没有**证明 |
|-----------------|------------------------|
| 模拟器正确执行了自己定义的物理与数据契约 | 物理参数选择合理（`τ_max/I` 33× 差异、单一初态） |
| 数据完整、可位级重放、SHA-256 未被篡改 | 数据分布适合训练（极端轨迹占 53.5% \|θ\|） |
| RK4 是 4 阶、能量/解析解自洽 | 该 dt 在全部场景下数值精度足够（short_rod 差 7.6e-3） |
| 动作能影响状态 | 模型能学到状态-动作联合动力学 |
| `DynamicsBatch` 契约兼容 | 多步预测稳定、latent 不 collapse、可泛化到新初态 |

**因此：8/8 PASS 是「工程验收通过」的充分证据，但不是「科学验收通过 / 可批准全量数据」的充分证据。**

---

## 12. Phase 1 验收建议

| 项 | 建议 |
|----|------|
| P0-fix.1 | ✅ 批准 |
| P1-A 工程验收 | ✅ 有条件通过 |
| P1-A 科学验收 | ⚠️ **暂缓**，需先解决 S1（初态分布）、S2（极端主导）、S3（dt 精度） |
| 全量数据生成 | ⛔ **暂缓** |
| JEPA 训练 E0–E5 | ⛔ **暂缓** |

需人工决策的 3 个问题（本报告不擅自决定）：
1. 第一阶段训练目标是**仅 bounded oscillation**，还是同时包含 continuous rotation？
2. `short_rod` / `low_gravity` 强驱动是否统一划为 stress-test（eval-only）？
3. 是否按 `τ_max ∝ I` 重新标定各场景力矩上限，以及是否对 short_rod 采用更小 `dt`？

---

## 13. 限制与未执行项（合规声明）

- 未修改 `data/controlled_pendulum.py`、`configs/dynamics/controlled_pendulum_jepa.yaml`、
  `scripts/collect_controlled_pendulum.py`、`scripts/audit_controlled_pendulum.py`、
  pilot `npz`/`manifest`、`audit_report.json`。
- 未修改任何受限 Core / Adapter 文件、未动 `datasets/pusher_v1/`、未合并 main、未开 PR。
- 未生成全量数据集、未实现 `trajectory_windows.py` / `ActionConditionedWorldModel`、
  未训练 JEPA、未实现 MPC。
- 新增内容仅：`scripts/analyze_p1a_pilot.py`（只读分析）、`analysis/` 下的 JSON + PNG、本报告。

## 14. 复现

```bash
python scripts/analyze_p1a_pilot.py --config configs/dynamics/controlled_pendulum_jepa.yaml
# 输出: datasets/controlled_pendulum_pilot/analysis/{independent_verification,
#       step_convergence, trajectory_distribution, action_information,
#       split_proposal}.json  + figures/*.png (6)
```

依赖实测版本：numpy 2.2.6 / scipy 1.16.3 / matplotlib 3.10.8 / torch 2.14.1 / PyYAML 6.0.3 /
pytest 9.1.1（Python `D:/python313/python.exe`）。

### 数字可追溯性核验（沿用 P2-02 方法学）

```bash
python scripts/verify_p1a_reports.py
# -> number traceability: PASS (28/28)
# -> datasets/controlled_pendulum_pilot/analysis/number_traceability.json
```

对四份报告中的每个高精度数值，验证「字面量出现在报告中」**且**「与 `analysis/*.json`
的实测值在报告所用舍入精度内数值一致」。实测 **28/28 全部可追溯，0 虚构**。
全量回归 `pytest tests/ --ignore=tests/test_pysr.py` → **255 passed + 101 subtests**；
既有 `audit_controlled_pendulum.py` 复核仍 **8/8 PASS**（本轮未改动任何数据或管线）。
