# M7.6 Multi-Round Capability Report

## 0. 原则

L1（工程）、L2（科学能力）、L3（递归研究）分别判定，绝不把工程成功当科学能力成功。
允许并保留 INCONCLUSIVE / REFUTED / REJECTED 作为合法结论。

## 1. 三轮真实研究流程（生产 E2E）

| 轮次 | 输入 | 系统行为 | 结论 |
|---|---|---|---|
| R1 初始发现 | broad objective | 真实 OpenAlex 19 篇→真实 LLM 假设→价值排序(EVIDENCE_BASED_PRIORITY)→预注册→带外审批→正式 Runtime 执行→证据/SHA 校验→holdout→replication | INCONCLUSIVE（effect < MME，power_gate=NOT_ASSESSED） |
| R2 Memory 引导 | 同任务 + verified memory | M7.3 四臂对照（OFF/ON/SHUFFLED/EQUAL_TOKEN），GT-based repeat_failure 测量 | Memory **未改善**：ON accuracy 0.10 < OFF 0.70；SHUFFLED harmful 0.30 |
| R3 递归改进 | 前两轮真实失败 | 从 base 轨迹诊断瓶颈→无明确瓶颈→不生成 Candidate→无 adoption/rollback | INCONCLUSIVE（NO_CLEAR_BOTTLENECK） |

R3 的"无瓶颈就不改进"是合法的科学判断：base 在 6 个 development case 上全对（0 失败），
因此不存在可从证据诊断的方法论瓶颈。系统拒绝凭空制造改进——这正是用户要求的 Scientific Judgment。

## 2. Scientific Judgment（M7.2，修复后多主结局）

| 指标 | 值 |
|---|---|
| Novelty accuracy | 1.000（标签分布窄，不能外推） |
| Novelty precision | NOT_ASSESSED（无系统预测为 LIKELY_NOVEL） |
| Falsifiability accuracy | 0.857 |
| Evidence sufficiency accuracy | 0.643 |
| Abstention F1 | 0.750（CI 0.454–0.883，过度弃答 FP=4） |
| Verdict | **INCONCLUSIVE**（n=14<30 且专家盲评 NOT_PERFORMED） |

关键改进：novelty_accuracy=1.0 **不再**单独产生 PASS；evidence_sufficiency 为主要结局，
低于阈值会 FAIL。这是对"Novelty 单票通过"缺陷的根因修复。

## 3. Research Memory（M7.3，GT-based 修复后）

| Arm | repeat_failure_accuracy（GT） | predicted rate | memory usage | harmful |
|---|---:|---:|---:|---:|
| OFF | 0.70 | 0.30 | – | – |
| ON | **0.10** | 0.90 | 1.00 | 0.00 |
| SHUFFLED | 0.70 | 0.30 | 0.00 | 0.30 |
| EQUAL_TOKEN | 0.50 | 0.50 | – | – |

关键发现：修复前用模型"预测重复失败率"时 ON 看起来更差；修复后用 GT 真值，
ON 的 grounded accuracy 反而显著更差（0.10）。无论哪种口径，**都没有证据表明 Memory 改善研究**。
且 ON 用了全部相关 memory 却产生系统性误判，SHUFFLED 出现 30% harmful usage。
Verdict：**INCONCLUSIVE**（专家评审未做），但方向性证据为负面。

## 4. Recursive Improvement（M7.4，证据驱动修复后）

- 瓶颈诊断：从真实 base 轨迹统计 evidence/novelty/falsifiability/parse 失败分布。
- 本轮结果：6/6 development cases 全对，counts 全 0 → `NO_CLEAR_BOTTLENECK` → 不生成 Candidate。
- 无 Base/Candidate 配对结果、无 holdout delta、无 adoption、无 rollback（NOT_APPLICABLE_NO_ADOPTION）。
- Verdict：**INCONCLUSIVE**。

这是"改进机制存在但改进未发生"的诚实记录。M6 的 generate_candidate 管道已具备瓶颈驱动能力，
本轮没有满足触发条件，不伪造瓶颈。

## 5. Integrated（M7.5，family-level 修复后）

| Group | evidence acc | Δ vs base | CI 下界 | met MME |
|---|---:|---:|---:|---|
| A_BASE | 0.500 | – | – | – |
| B_MEMORY | 0.500 | 0.000 | 0.000 | no |
| C_RSI | 0.625 | +0.125 | 0.000 | no |
| D_FULL | 0.625 | +0.125 | 0.000 | no |

- n_unique_families = 7（< 预注册 min_families=8），verdict **INCONCLUSIVE**。
- family-level paired delta + cluster bootstrap CI：C/D 的 CI 下界为 0，未达 MME。
- Memory 无增量（B=A）；无证据表明 Memory × RSI 协同。

## 6. 跨轮结论

1. **能否自主发现有价值的 RQ？** 部分：能从真实文献+LLM 生成并证据排序，但价值依据来自 LLM critic 非人类专家。
2. **能否区分真 gap 与表面新颖？** 部分：Novelty 已 fail-closed（SCREENING_ONLY），不伪装确认。
3. **能否产出可证伪竞争假设？** 能（H1/H2/H3/H0 + Critic 独立审查）。
4. **能否设计控制实验？** 能（selection/replication/holdout/capacity control）。
5. **能否走真实生产 Runtime？** **能（本轮最强证据，L1 PASS 级）。**
6. **是否真正等待人类审批？** 是（两阶段 + 带外 CLI + SHA-bound，driver 零自批）。
7. **能否保存并校验证据？** 能（28 journal 事件、12 result SHA、SHA chain）。
8. **Candidate/Verified 分层？** 是（INCONCLUSIVE 保持 candidate tier）。
9. **Memory 是否阻止历史失败？** **否**（本轮反向证据）。
10. **能否诊断自身瓶颈？** 机制存在，本轮无瓶颈可诊断（诚实 INCONCLUSIVE）。
11. **能否提出基于证据的改进？** 机制存在，本轮未触发。
12. **Candidate 是否在独立任务上胜过 Base？** 本轮无 Candidate。
13. **能否拒绝失败改进？** 能（无瓶颈时不生成；此前 M7.4 holdout 失败会 REJECT）。
14. **跨轮能力是否提升？** **未被证明**（样本与专家评审不足）。

## 7. 统计与诚信

- 所有 verdict 由预注册标准产生，未降低 MME、未放宽门槛、未挑选有利任务。
- 负面结果（Memory 无效、RSI 未触发、样本不足）全部保留。
- 无把 API 调用成功当科学成功、无把工程 Gate PASS 当能力 PASS。
