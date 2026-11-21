# ARR 1.1 — M3.0 验收报告：统计门控修复（M2.7 Statistical Gate Correction）

> 分支：`feature/arr-0.1-domain-registry`（基线 `9326a9cb` = M2.7 最终 HEAD）
> 提交清单以 `git log --oneline 9326a9cb..HEAD` 为准，不在报告内硬编码自身 HEAD。
> 本轮未 merge main、未开 PR、未重建 Runtime、未提前开发 Package H。
> M2 / M2.5 / M2.6 / M2.7 全部历史实现与验收产物保留。

---

## 0. 结论先行

| 项目 | 结论 |
|---|---|
| ChatGPT 复查指出的 6 条统计缺陷 | **逐条源码核实属实**，另发现 2 条同源问题，共 8 项已修复 |
| 修复性质 | 纯统计修复包，**不重建 Runtime**、不加新模块层级、不新增第二套研究流程 |
| 判定协议 | 已冻结：双侧 Student-t（df=n-1）+ Holm 家族校正 + 双向显著 + 缺失统计量 fail-closed + 逐 seed holdout 方向确认；`alpha=0.05` 只能收紧不能放宽 |
| **M2.7 的 SciNet++ `lr=0.01` 正式 ACCEPT** | **资格不成立**：t 分布 CI 下界 -0.005306，Holm 家族不显著。历史候选标注 `provisional`，历史 artifacts 一字未改 |
| 约 81% 的观测指标改善 | **仍是真实的观测指标变化**，本轮不否认；但它不是经过统计验证的正式改进，两者不混为一谈 |
| 测试 | **89 / 89** 通过（本地 + 干净 `PyYAML`+`pydantic` venv 双验证），新增 34 项 |
| 远端 CI | 见 §7（以 GitHub REST API 核实，未用本地 PASS 代替） |
| M3.1 ~ M3.4 | 本轮完成**真实依赖核查**，阻塞项见 §9；未用假数据 / FakeRunner 顶替 |
| 真实 10+10 seed 重验证 | 见 §6（真实 SciNet++ 引擎、真实冻结数据集） |

**一句话**：M2.7 的工程闭环仍成立，但它的「正式统计 ACCEPT」结论在正确的小样本统计下不成立；M3.0 把判定协议修正并冻结，之后不再为了多拿 ACCEPT 而动门槛。

---

## 1. 源码核查：缺陷逐条坐实（不是照抄复查结论）

核查对象均为本地真实源码，行号对应修复前的 `9326a9cb` 版本。

| # | 位置（修复前） | 实际代码表现 | 应有行为 |
|---|---|---|---|
| 1 | `evolution/evolution_controller.py:110` | `ci_lower = mean - 1.96 * se` —— n=5 用正态临界值 | df=4 双侧 95% 应用 **t=2.776445** |
| 2 | `:141-142` | `if mean <= 0: return REJECTED` —— 负向无任何统计证据要求 | CI **上界 < 0** 才正式 REJECT；不显著负向应 INCONCLUSIVE |
| 3 | `:108` | `std is not None and math.isfinite(std) and std > 0` 为假时**整段 CI 检查被跳过**，直接落到 practical floor → 可能 ACCEPT | 缺失 / NaN / Inf std 一律 INCONCLUSIVE |
| 4 | `:116-118` | `baseline_mean is not None ...` 为假时**跳过实际改善阈值** | 缺失 / 0 / 非有限 baseline 一律 INCONCLUSIVE |
| 5 | `evolution/domain_evolver.py:269-271` + `controller:122-139` | holdout 只落 `holdout_improvement`（均值）与 `holdout_n_valid`；判定只看均值 > 0 | 保存**每个 holdout seed 的原始结果**，按同向确认规则验证 |
| 6 | `controller:212-217` | `run_dual_layer` 循环内逐候选独立 `_decide_domain`，一轮 3 个 LR 各用 95% CI，挑选最好者 | 增加多重比较控制（Holm step-down） |
| 7（额外） | `controller:108` | `std == 0`（所有配对改善完全相同）被归入「跳过 CI」分支 | 这是**确定性改善**（t=∞, p=0），应显式判定，而非当作缺失 |
| 8（额外） | `domain_evolver.py:266-274` | holdout 部分 seed 返回 NaN 时 `ho_valid` 变小但不记录失败数，`holdout_n_seeds` 不存在 | 任一 holdout seed 失败即 fail-closed，不得静默减少样本 |

**数值复核**（用 M2.7 历史真实 mean/std/n=5 重算，与复查表格逐位吻合）：

| 候选 LR | Mean | Std | 旧正态 CI 下界 | **Student-t CI 下界** | 双侧 p |
|---|---|---|---|---|---|
| 0.003 | 0.011228 | 0.008678 | +0.003621 | **+0.000453** | 0.04442 |
| 0.005 | 0.016844 | 0.014152 | +0.004439 | **-0.000728** | 0.05631 |
| 0.010 | 0.019355 | 0.019861 | +0.001946 | **-0.005306** | 0.09484 |

再加 Holm（m=3，阈值 0.0167 / 0.025 / 0.05）：最小 p=0.04442 > 0.0167 → **三个候选全部不显著**。

---

## 2. 修复内容与数据流变化

### 2.1 修改文件清单（7 个逻辑 commit）

| 文件 | 变化 |
|---|---|
| `research_runtime/evolution/statistics.py` | **新增**。纯标准库统计原语：正则化不完全贝塔函数（Lentz 连分数）→ `P(|T_df|>t)=I_{df/(df+t²)}(df/2,1/2)` → 二分求临界值；`paired_t_statistic` / `paired_p_value` / `paired_confidence_interval` / `holm_step_down` |
| `research_runtime/evolution/evolution_controller.py` | `_decide_domain` 重写为双向 t 判定 + fail-closed；新增 `_finite_number` / `_nonneg_int` / `_significance_direction` / `_holdout_confirms` / `domain_family_significance`；`run_dual_layer` 改两阶段并落 Holm 审计字段 |
| `research_runtime/evolution/domain_evolver.py` | holdout 逐 seed 原始结果落库（7 个新字段），原字段保持兼容 |
| `research_runtime/tests/test_m3_0_statistics.py` | **新增 34 项**统计回归测试 |
| `research_runtime/tests/test_m2_7_reliability.py` | `_rec()` 补齐逐 seed 证据（**提高测试证据，不是降门槛**）；docstring 指向新协议 |
| `research_runtime/cli.py` | 仅同步 `--help` 措辞，**未新增任何可调门槛的参数** |
| `.github/workflows/arr-research-runtime.yml` | 纳入新测试 + 远端断言 `t_df4 == 2.776445` |
| `M3_0_runs/*` | 只读历史重估脚本与结果、真实 10+10 重验证 driver（证据产物） |

### 2.2 数据流（修复前 → 修复后）

```
修复前：  evaluate(cand_i) → _decide_domain(rec_i, 1.96 正态) → 立刻出判定
          （3 个候选各自独立 95% CI，无家族校正；holdout 只有均值）

修复后：  Phase 1  evaluate(cand_1..m)  → 全部 rec 落库（含逐 seed holdout 原始结果）
          Phase 2  domain_family_significance(recs)  → 每候选双侧 t 检验 p 值
                   → Holm step-down → holm_effective_alpha / rank / significant
          Phase 3  _decide_domain(rec_i, holm_alpha=α_i)
                   → t 分布 CI（α_i）双向判定
                   → practical floor / 逐 seed holdout 方向确认
                   → ACCEPT / REJECT / INCONCLUSIVE
          registry / payload / rec["statistics"] 全量持久化，可用 SHA 链重算判定
```

### 2.3 冻结的判定协议（不得放宽）

| 规则 | 实现 |
|---|---|
| 小样本用 t 不用 1.96 | `student_t_crit_two_sided(alpha, n-1)`；df=4 → 2.776445 |
| 多重比较 | Holm step-down，family = 一轮内全部候选；`alpha` 只能收紧 |
| 双向显著 | CI 全在 0 上 → 才可能 ACCEPT；全在 0 下 → REJECT；跨 0 → INCONCLUSIVE |
| 缺失即失败 | `n_valid` / `n_invalid` / `mean` / `std` / `baseline_mean` 任一缺失或非有限 → INCONCLUSIVE |
| `std == 0` | 显式确定性改善（不是跳过检查） |
| 独立 holdout | 全部 seed 有效（`n_seeds == n_valid`、`n_invalid == 0`）+ 均值同向 + ≥60% seed 同向 + 逐 seed 原始结果落盘；**不参与候选选择** |
| 硬下限 | `MIN_SEEDS_FOR_FORMAL_ACCEPT=5`、`PRACTICAL_FLOOR_REL=0.01` 只能被提高 |
| dev/provisional | 关闭 holdout 确认时正向候选只能 INCONCLUSIVE |

**没有新增任何可以放宽门槛的 CLI 参数**——这是刻意设计，避免以后「为了多拿 ACCEPT」调 alpha。

---

## 3. 测试

| 场景 | 结果 |
|---|---|
| t 临界值对标准表（df=1/2/4/9/29/119/1e7） | 误差 < 1e-6 |
| `p(t_crit) == alpha` 反算 | 通过 |
| Holm：单候选不校正 / 阈值收紧 / step-down 停止 / 非有限 p 不通过 / alpha 只能收紧 | 通过 |
| 显著正向→ACCEPT、正向不显著→INCONCLUSIVE、显著负向→REJECT、负向不显著→INCONCLUSIVE | 通过 |
| 缺失 std / baseline_mean / 样本计数 → 全部 INCONCLUSIVE | 通过 |
| holdout 逐 seed 缺失 / 部分失败 / 不同向 / 负向矛盾 → INCONCLUSIVE | 通过 |
| 3 个 p=0.033 候选「各自显著但家族无胜出」→ 全部 INCONCLUSIVE | 通过 |
| `run_dual_layer` 端到端真正写入 Holm 审计字段 | 通过 |
| M2.7 历史三组真实数据重算 + Holm 结论锁定 | 通过 |
| **合计** | **89 / 89**（本地 + 干净 venv 仅 `PyYAML>=6.0` + `pydantic>=2.7.4`） |

统计模块**不引入 numpy / scipy / pandas**（完全标准库），因此远端 CI 的最小依赖集没变——这一点在干净 venv 里实测过。

### 3.1 门控收紧后的 FP / FN 复核（「不按下葫芦起了瓢」）

| 检查 | 结果 |
|---|---|
| 极端 Holm alpha（m=1/3/10/50/100 → 0.05…0.0005） | `t_df4` 分别 2.7764 / 3.9608 / 5.5976 / 8.6103 / 10.3063，反算 `p(t)=alpha` 误差 < 1e-7，无数值失稳 |
| 100 个 p=0.04 的边际候选组成的家族 | Holm 后显著数 **0**（未因多重比较产生新假阳性） |
| n=10 的真实强改善（mean=0.010, std=0.003） | 单候选 **ACCEPTED**；置于 3 候选 Holm 家族（alpha=0.0167）下**仍 ACCEPTED**（收紧未制造新漏报） |
| 该改善的镜像负向 | **REJECTED**（双向对称） |

即：新协议只淘汰「证据不足」的判定，不牺牲对真实显著改进的识别能力。

---

## 4. 历史重估：M2.7 的正式 ACCEPT 资格

脚本 `M3_0_runs/reevaluate_m2_7.py` **只读** `M2_7_runs/*/round_*/candidate_registry.jsonl`，输出到 `M3_0_runs/m2_7_reevaluation.json`。历史文件 SHA-256 前后一致：

```
0ad2e6dc6c3ff24a0c7e683791c1ed0e384d72a7d0e9b2a9ada5fd552de9a3ab  scinet/round_1/candidate_registry.jsonl
c75944d06f7cc66d67874c8cba5e9d672d5fcbdd5b037d7136aa853d5b251d74  scinet/baseline_v2.yaml
```

### 4.1 SciNet++ round 1（M2.7 曾 3 × ACCEPTED）

| LR | 旧判定 | 旧正态 CI 下界 | t CI 下界 | p | Holm 显著 | M3.0 判定 | 最有利反事实 |
|---|---|---|---|---|---|---|---|
| 0.003 | accepted | +0.003621 | +0.000453 | 0.04442 | 否 | inconclusive | inconclusive |
| 0.005 | accepted | +0.004439 | -0.000728 | 0.05631 | 否 | inconclusive | inconclusive |
| **0.010** | **accepted** | +0.001946 | **-0.005306** | 0.09484 | **否** | **inconclusive** | **inconclusive** |

**关键**：「最有利反事实」假设 M2.7 当时保存了逐 seed holdout 且全部同向（对结论最有利），结果**仍是 INCONCLUSIVE**。所以 `lr=0.01` 的问题出在**统计本身**，不是「历史没存新字段」——这个结论不依赖任何缺失证据。

### 4.2 三域全部 27 个历史候选

| 域 | 旧判定分布 | M3.0 重算 |
|---|---|---|
| SciNet++ | 3 accepted / 6 rejected | 全部 inconclusive（provisional） |
| Embodied | 9 rejected | 全部 inconclusive（provisional） |
| Quant | 9 inconclusive | 全部 inconclusive |

**如何正确理解这张表**（避免误读成「推翻 M2.7」）：
- 历史 artifacts 的**真实执行记录、观测指标、SHA-256 证据链、版本继承结果全部有效且保留**，本轮一字未改。
- 变化的是**判定协议**（更严格的小样本统计 + 多重比较校正）和**新增的逐 seed holdout 证据要求**（M2.7 未保存 → 按新协议 fail-closed）。
- 因此所有历史终端判定标注 `provisional`，**不伪造重新验收、不改写历史、不删除 `baseline_v2.yaml`**。
- 约 81% 的 `baseline_mean` 下降（0.023939 → 0.004584）仍是**真实观测**，但它不构成「经统计验证的正式改进」。

---

## 5. 修复的真实 Bug 清单（含归属）

| Bug | 归属 | 根因 |
|---|---|---|
| n=5 用 1.96 正态临界值 | **M2.7 本轮引入**（P0） | 小样本未用 Student-t |
| `mean<=0` 即 REJECT | **M2.7 本轮引入**（P0） | 单向门控，负向无统计证据 |
| 缺失 std 跳过 CI | **M2.7 本轮引入**（P0） | 条件短路而非 fail-closed |
| 缺失 baseline_mean 跳过实际阈值 | **M2.7 本轮引入**（P0） | 同上 |
| holdout 只有均值、无逐 seed | **M2.7 本轮引入**（P0） | 证据粒度不足 |
| 多候选无 Holm | **M2.7 本轮引入**（P0） | 逐候选独立判定 |
| `std==0` 被当作缺失跳过 | 本轮额外发现 | 语义混淆 |
| holdout 部分 seed 失败静默吞掉 | 本轮额外发现 | 无 `n_seeds` / `n_invalid` 计数 |

另有一处**非 Bug 但需说明的语义变化**：`run_dual_layer` 从「每候选 evaluate→decide 紧邻」改为两阶段，registry 中 EVALUATED 事件先集中、decision 事件其后；证据链 verifier 按 `event`/`to` 过滤，不依赖紧邻顺序。

---

## 6. 真实运行证据：10 paired + 10 holdout seeds 重验证

对应 ChatGPT Phase 0 第 6 条：用 **≥10 paired seeds + 独立 holdout seeds** 重新验证重要改进。

### 6.1 运行方式（真实引擎，非 FakeRunner）

```bash
cd /e/agent_dev/multi-agent
python -u M3_0_runs/scinet_10seed_validation.py > M3_0_runs/scinet_10seed.log 2>&1
```

| 参数 | 值 |
|---|---|
| 引擎 | **真实 SciNet++**（`SciNetAdapter` → `research_api.cli` 子进程 → torch）；domain runner = 生产 `_build_real_domain_runner` |
| 数据集 | `M2_7_runs/scinet/datasets/pendulum_smoke_v1`，`dataset_version=pendulum-a0e03386be89`，4 个 split 逐 SHA-256 复核通过 |
| paired seeds | `50..59`（10 个） |
| independent holdout seeds | `60..69`（10 个，与 paired 完全不相交） |
| 候选族 | 一轮内 `generate_domain_candidates` 产出的 3 个 `train.lr` 候选（Holm 的 family 大小 m=3） |
| 统计协议 | M3.0：`run_dual_layer` → 两阶段 → 双侧 Student-t（df=9）→ Holm → 双向判定 + 逐 seed holdout 确认 |
| 唯一 stub | **仅 RESEARCH_HARNESS 层入口**（`harness_evolve`）被 stub，因为 M3.0 验证的是 DOMAIN 统计门控，harness 层与之正交且会额外 spawn 大量引擎任务。**domain 层 100% 真实引擎，无 FakeRunner、无预置结果** |

> 明确声明：这不是 M3.4 的「统一三域调度」，也不含 Quant / Embodied 真实数据；它是 M3.0 统计协议在真实外部引擎上的一次独立复核。

### 6.2 真实运行结果

真实 SciNet++ 引擎跑了约 **67 分钟**（4024 s）真实训练+评估，10 paired + 10 holdout seeds，**domain 层 100% 真实引擎、无 FakeRunner**。三个 `train.lr` 候选**全部通过 M3.0 门控成为正式 ACCEPT**：

| 候选 | n_valid | mean | std | t(df=9) | p(双侧) | Holm α | Holm 显著 | holdout 同向占比 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| lr=0.003 | 10 | +0.012084 | 0.007346 | 5.20 | 0.000563 | 0.0167 | ✓ | 10/10 (1.0) | **accepted** |
| lr=0.005 | 10 | +0.018121 | 0.011484 | 4.99 | 0.000749 | 0.0250 | ✓ | 10/10 (1.0) | **accepted** |
| lr=0.01  | 10 | +0.019121 | 0.014798 | 4.09 | 0.002733 | 0.0500 | ✓ | 10/10 (1.0) | **accepted** |

**关键对照（这正是 M3.0 门控正确性的最强证据）**：同一个 SciNet++ 问题，在 M2.7 的 **5 seed** 数据上，用新协议重算结果是 **三个全部 INCONCLUSIVE**（t 分布下界 +0.000453 / −0.000728 / −0.005306，Holm 均不显著）；而本轮 **10 seed** 真实重验则**三个全部 ACCEPTED**。

| 候选 | 5-seed（M2.7 历史，M3.0 重算） | 10-seed（M3.0 真实重验） |
|---|---|---|
| lr=0.003 | INCONCLUSIVE（p=0.04442，Holm 不显著） | **ACCEPTED**（p=0.000563，Holm 显著） |
| lr=0.005 | INCONCLUSIVE（p=0.05631，Holm 不显著） | **ACCEPTED**（p=0.000749，Holm 显著） |
| lr=0.01  | INCONCLUSIVE（p=0.09484，Holm 不显著） | **ACCEPTED**（p=0.002733，Holm 显著） |

> **结论**：M2.7 报告的「~81% 指标改善」是**真实观测**（baseline_mean 0.023939 → 0.004584），但 5 个配对 seed 在 Student-t（df=4）下**无法建立正式统计 ACCEPT 资格**；M3.0 用 **10 paired + 10 holdout** seeds 重新验证后，三个候选才真正取得正式 ACCEPT 资格。这同时满足 ChatGPT 任务书第 6 条「用至少 10 个配对 seed 重新验证重要正式改进」与「加固后的门控必须能接受真正改善（不只会拒绝）」两个要求。重验结果与 M2.7 观测方向完全一致（lr=0.01 改善最大），且与历史 artifacts 不矛盾——后者只是证据量不足。

---

## 7. 远端 CI（不以本地 PASS 代替）

**远端实证，非本地 PASS 替代。** 本机 `gh` 未认证，改用 GitHub REST API 直接核实：
`api.github.com/repos/thesevenths/agent_dev/actions/workflows/arr-research-runtime.yml/runs`
与 `.../actions/runs/43392991350/jobs`。

| 项 | 远端实测 |
|---|---|
| Workflow 名 | `ARR Research Runtime (M2.7 + M3.0)`（仓库根 `.github/workflows/`，GitHub 自动发现） |
| 本轮最终 run | **#9 / id 43392991350**，head_sha=`d751d59b`（M3.0 最终 commit），[链接](https://github.com/thesevenths/agent_dev/actions/runs/43392991350) |
| status / conclusion | `completed` / **`success`** |
| 矩阵 jobs | **6 / 6 全部 success**：ubuntu-latest × py3.11/3.12/3.13、windows-latest × py3.11/3.12/3.13 |
| 关键步骤 | 每个 job 的 `Run unified Research Runtime engine-free suite` 与 `Verify CLI safety defaults + frozen statistical protocol` 均 success |
| 远端跑过的断言 | 89 项测试（含新增 34 项）+ `student_t_crit_two_sided(0.05,4) == 2.776445` 且 `> 2.77`，**从远端证明旧 1.96 已被替换** |
| 工作流累计 | **9 次 run 全部 success**，0 failure（#1..#9） |

顺带更正复查中的一处状态：ChatGPT 复查时记录「最后 docs-only HEAD 对应运行仍为 `in_progress`」，该 run（`9326a9cb`）现已核实为 **Success**。保留的 [ARR 0.1 Integration](https://github.com/thesevenths/agent_dev/actions/workflows/arr-0.1.yml) 旧 workflow 未被修改。

---

## 8. Commit 清单

每个逻辑改动独立 commit，已推送 `feature/arr-0.1-domain-registry`，未 merge main、未开 PR。以 `git log --oneline 9326a9cb..HEAD` 为准：

| 主题 | 内容 |
|---|---|
| `statistics` | 新增纯标准库 Student-t / Holm 原语 |
| `evolution_controller` | 双向 t 判定 + Holm + fail-closed + 两阶段编排 |
| `domain_evolver` | 逐 seed holdout 原始结果落库 |
| `tests` | 34 项新测试 + `_rec()` 提高证据 |
| `cli` | 帮助文本同步（无新门槛参数） |
| `ci` | 纳入新套件 + 远端断言 t_df4 |
| `docs/evidence` | 只读历史重估 + 10+10 重验证 driver |

---

## 9. 未解决问题与 M3.1 ~ M3.4 阻塞项（实证核查，非推测）

按 ChatGPT「遇到无法完成的外部依赖，明确报告阻塞，不用假实现替代」的要求，本轮**实际扫描了本机数据**：

| Phase | 需要的外部依赖 | 本机实证 | 结论 |
|---|---|---|---|
| **M3.1 Quant 真实行情** | 合法取得的真实历史 OHLCV CSV | `tmp/btc_daily.csv` 1011 行中 **960 行 `annual_avg` 填充**、仅 51 行 `real`；`tmp/qqq_daily.csv` 723 行中 705 填充、18 real；列仅 `date, close, source`，**缺 open/high/low/volume** | **阻塞**。不能把年度均值填充序列冒充真实行情；需取得完整 OHLCV 且可追溯来源的数据集后再接 `quant_compute` |
| **M3.2 Embodied 真实数据** | LeRobot 兼容数据集 + gymnasium/mujoco 运行时 | 两个 Python 环境（默认 + `scinet_gpu`）均**未安装** `lerobot` / `gymnasium` / `mujoco`。但发现 SciNet++ 已有 `datasets/pusher_v1`：`Pusher-v5`、obs_dim=23 / act_dim=7、`train=240` / `validation=30` / `frozen_test=15` / **`ood_test=15` episodes**、逐文件 SHA-256 | **部分基础已具备、运行时依赖阻塞**。该数据集是 MuJoCo 轨迹（**非 LeRobot 格式**），且已有独立 OOD split，是比 1-D toy 仿真更好的起点；但需先补依赖并明确「非 LeRobot」的适配工作 |
| **M3.3 SciNet++ 生产化** | 多独立物理任务 + OOD | 已有 `pendulum.yaml` / `double_pendulum.yaml` / `newton.yaml` / `pendulum_smoke.yaml`、`configs/robotics`、`datasets/controlled_pendulum_v2_pilot`、`datasets/pusher_v1` | **基础已具备**，无需重接线；可直接做任务级验证 / 缓存 / 预算 / OOD 泛化 |
| **M3.4 统一三域调度** | 前三 Phase 各自通过真实引擎验收 | 未开始 | 依赖 M3.1 ~ M3.3 |

**诚实标注**：本轮 Runtime 内的 Quant 仍是**合成市场数据**、Embodied 仍是**1-D 数值仿真**，不得声称已接入真实行情或完成真实机器人验证。SciNet++ 是真实外部引擎。

### 其他未解决
1. **Harness 层有机 ACCEPT 仍未证明**（M5 的科研能力分水岭），本轮未触碰。
2. 小样本 CI 现用精确 t 分布；若未来样本量很大，t 会自然收敛到 1.96，无需再改代码。
3. Holm 控制的是 family-wise error rate，比 Bonferroni 稍宽松但仍是保守校正；若未来引入「预注册唯一主候选」模式需单独设计，本轮未做（避免开后门）。
4. SciNet++ 单轮真实实验耗时长（见 §6），M6 的算力/预算调度仍是后续问题。
5. **非关键一致性细节（本轮刻意未改）**：`holdout_direction_fraction` / `holdout_n_positive` 这两个审计展示字段由未 round 的原始改善值计算，而判定权威 `holdout_improvements` 落盘时 round 到 6 位；仅在改善量 < 1e-6 的极端边界，两者可能相差 1 个计数。判定始终以落盘的逐 seed 列表为准，不影响任何结论（本轮真实数据改善量约 1e-2）。按「不纠缠非关键细节、控制工程建设比例」原则留待后续统一，未在运行中的真实实验期间改动源码以避免版本歧义。

---

## 10. 对 M3 后续路线的建议（沿用 ChatGPT 的资源分配）

- **70% 真实科研能力**：先解决 §9 的两个数据阻塞（真实 OHLCV、机器人运行时依赖），这是 M3.1 / M3.2 的实际门槛，不是再写框架代码。
- **20% Harness 自我改进**：M5 再证明，不要在 M3 阶段用「让 harness 多产 ACCEPT」来提高数字。
- **10% 工程维护**：M3.0 已把统计协议冻结；后续除非发现新的统计错误，不应再动门控。
- 顺序保持 **M3.1 → M3.2 → M3.3 → M3.4**：先让每个领域独立完成真实研究任务，最后才做统一多域调度；三域的 Dataset / CandidateRegistry / VersionRecord / State 必须相互隔离。
