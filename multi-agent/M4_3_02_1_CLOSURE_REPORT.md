# M4.3-02.1 Pilot Closure Report

**结论先行**：Pilot Closure 的最小必要修复已全部完成并推送（7 个 commit，远端 HEAD
`8960b1e7`）。Pilot v1 历史证据 **SHA 完全未变**，修复后的分析作为**新版本**
`M4_3_runs/pilot/closure_v1_1/` 生成，原报告保留。真实结论仍是
`evidence_insufficient`：**11 条盲评目标 0 条已由真人提交**，样本量两指标均无法
估计，Confirmatory 按设计拒绝启动。这不是缺陷，是规格要求的行为。

分支：`feature/arr-0.1-domain-registry`。未 merge main，未创建 PR。

---

## 1. Part A 交付总览（每项独立 commit + 推送）

| 项 | Commit | 内容 |
|---|---|---|
| P0-01 | `93005d34` | 配对资格单一真源 `paired_eligibility.py`：三组全部 finalized + 成功科学结论 + 同 block/snapshot/protocol；Analysis 与 Value Report 共享 paired n；`excluded_blocks` 列原因 |
| P0-02 | `66fc4168` | Outcome 语义：`inconclusive`/`not_evaluated` 保留 inconclusive；`rejected` 仅在真实成功 Campaign + 有效科学结论时才映射 negative_finding；FAILED/INTERRUPTED/STOPPED/INVALID 分别记录；未知状态 fail-closed 保留原始状态 |
| P1-01 | `2cac8e60` | 双 Primary Metric 样本量：两个指标各用自己的配对方差与预注册 SEOI；零方差/无人工评分/缺配对数据各有明确状态；最保守汇总；`seoi_valid_findings` 真正生效 |
| P1-02 | `9513d259` | 固定预注册检验家族（缺失对比 p=1.0，家族不缩小）；Primary/Secondary 层级显式声明；`significance_claimed` 仅由 C-B on primary metric 驱动；结论绑定冻结决策对比/指标 |
| P1-03 | `cf3eea38` | 组别描述准确（A/B delegated、C autonomous，非人工选题）；审批边界区分 CLI 子进程授权 / 真人审批 / 身份认证；无人工盲评时 objective quality 显示缺失 + n_reviewed=0 |
| P1-04 | `05ad911d` | 盲评闭环桥（非侵入 overlay，≥2 不同真人 + 解盲后才可解析，自动评分不填补）；`review resolve-quality` CLI；版本化分析（原报告不动）；真实队列 0 提交 → 不伪造 |
| P1-05 | `8960b1e7` | 6 个新测试模块接入根 CI workflow |

---

## 2. 每项要求的证据链

### P0-01 配对资格统一
- 单一函数 `select_paired_blocks()` 被 `pilot_analysis.py` 与 `value_report.py`
  同时调用；测试 `test_analysis_and_value_report_agree_on_paired_n` 证明相同数据
  产出相同 paired n（本 Pilot = 3）。
- 真实 block 4 因 `A/B not finalized` + `outcome interrupted/invalid` 被排除并
  列出原因；三组完整 finalized 的 block 1-3 才可配对。
- 失败/中断/无效记录仍留在 `outcome_counts` / 所有总体分母（见 closure_v1_1 报告：
  interrupted=2、invalid=1、negative_finding=9 全部保留）。

### P0-02 Outcome 语义
- 纯函数 `classify_director_outcome()` 参数化测试覆盖所有 report outcome 与
  director terminal state（25 例）。
- 关键行为：`inconclusive`/`not_evaluated` 永不为 negative_finding；`rejected`
  缺真实成功 campaign 或有效结论时降为 inconclusive 并记录原因；
  FAILED/INTERRUPTED/STOPPED/INVALID 映射为 4 个不同 outcome。

### P1-01 双 Primary Metric 样本量
- `estimate_dual_primary_sample_size()` 为 quality 与 findings 各算配对差值、
  方差、样本量，各用 `seoi_quality=0.5` / `seoi_valid_findings=0.01`；测试证明
  改 findings 的 SEOI 只改变 findings 的 n。
- 零方差 → `variance_status=zero_variance`（**不是 n=1**）；无人工评分 →
  `no_human_review`；缺配对 → `insufficient_pairs`；总推荐取所有必需指标最大值，
  任一不可估计则 `evidence_insufficient`，不编造数字。
- 本 Pilot：两指标均不可估计 → `status=evidence_insufficient`、
  `n_blocks_recommended=null`。

### P1-02 多重比较与结论一致性
- `analyse_primary_contrasts(family=...)` 用冻结的 3 对比家族；缺失对比保留在家族
  中（p=1.0、significant=False），校正分母始终为 3；测试证明 C-A/B-A 缺失前后
  C-B 的 effective_alpha 相同。
- `significance_claimed` 与结论仅由 C-B（primary contrast）在 PRIMARY_METRICS 上
  驱动。`test_secondary_only_significance_cannot_claim_a_value`：C-A/B-A 显著而
  C-B 不显著时，`significance_claimed=False`、结论为 "no clear incremental value"。

### P1-03 组别与审批边界
- A/B 描述为 "delegated baseline"，明确 NOT independent human；C 为 Autonomous
  Objective Discovery；报告中不再出现 "A manual / B memory-assisted"。
- 审批边界显式区分：CLI 子进程授权（`approval_cli grant --yes`，本 Pilot 的
  `approved_by=operator`）、真人审批、身份认证；后两者 `confirmed=False`；
  证据链每条 run 记录 `approval_source` / `approval_approved_by`。
- 无人工盲评时 value report 显示 `value=missing, n_reviewed=0`，不呈现为 0 分。

### P1-04 人工盲评闭环
- `review_bridge.py` 从 append-only ReviewQueue 解析：≥2 不同真人、`human_blind`
  类型、解盲后才可解析；测试证明自动评分不计数、同一 reviewer 不可覆盖、
  未解盲不产出、全解盲后解析出 3 个 run 的 quality。
- **真实 Pilot 队列**：11 条 pending、0 条 human review 提交。因此不产出任何
  quality overlay，不伪造真人评分。
- 版本化分析：`python -m research_runtime.m4_3.cli analyze-pilot
  --analysis-version closure_v1_1`。原 `pilot_analysis.json` /
  `pilot_value_report.json` / `M4_3_PILOT_VALUE_REPORT.md` SHA 未变（见 §4）。

### P1-05 CI 与回归
- 6 个 `test_m4_3_02_1_*.py` 接入根 workflow `arr-research-runtime.yml`，YAML 校验通过。
- **Engine-free**：`pytest research_runtime/tests/ -q` →
  **634 passed, 3 skipped, 37 subtests**；3 skip 为 opt-in real-engine，未当作通过。
- **Pre-commit**：`precommit_checks.py ast` 对全部变更 Python 通过（无输出=干净）；
  本机未安装 `pre-commit` 可执行文件，workflow 用的是同一 ast 脚本入口。
- 历史 SHA / integrity / campaign 唯一性见 §4。

---

## 3. 真实 Pilot 修复后分析（closure_v1_1，来自同一 SQLite 重算）

| 项 | 值 |
|---|---|
| blocks | 4 注册，**3 可配对**（block 4 排除：A/B not finalized + interrupted/invalid） |
| runs | 12（negative_finding 9 / interrupted 2 / invalid 1 / 其余 0） |
| 配对 n（C-B，两指标） | 3 |
| Review closure | `pending_human_reviews`（`closure_applied=false`，0 run 由真人评审解析） |
| Objective quality | 三组均 `value=missing, n_reviewed=0`（未伪造） |
| 双指标 SEOI | quality=0.5 / findings=0.01（各自方差） |
| Sample size | `evidence_insufficient`，两指标均不可估计，`n=null` |
| significance_claimed | false |
| 结论 | `evidence insufficient` |
| 审批边界 | CLI subprocess grant under identity `operator`；independent human / identity auth 均未确认 |

---

## 4. 历史文件 SHA 不变证明（P1-05）

| 文件 | 基线 SHA256 | 当前 SHA256 | 一致 |
|---|---|---|---|
| `research_runtime/_data/evidence.jsonl` | `379c2206…` | `379c2206…` | ✅ |
| `M4_3_runs/pilot/pilot_analysis.json` | `e155c3e6…` | `e155c3e6…` | ✅ |
| `M4_3_runs/pilot/pilot_value_report.json` | `523b1d18…` | `523b1d18…` | ✅ |
| `M4_3_runs/pilot/M4_3_PILOT_VALUE_REPORT.md` | `77ec9e92…` | `77ec9e92…` | ✅ |

Registry integrity = `true`；12 runs / 9 distinct campaigns / duplicate_campaign_ids = `[]`。
冻结 metric contract SHA `3138acff…` 全程未变。历史 Campaign/Report/Experience/
Verdict 均未修改。

---

## 5. Part B 启动条件（当前状态）

Pilot Closure 已完成最小修复，但进入 M4.3-03/04 的前置条件**尚未满足**：

1. ❌ **真人盲评未完成**：11 条待评目标 0 条由两名不同真人提交。这是必须由真人在
   CLI（`review submit --reviewer <真人> --scores … --reason …`）完成的行为，Agent
   不得代填。
2. ❌ **两 Primary Metric 样本量不可估计**：无人工评分 → quality 无方差；findings
   三组有效发现率全 0 → 零方差。两者均为 `evidence_insufficient`。
3. ❌ **无 Confirmatory Protocol**：`prepare/freeze-confirmatory` 按设计拒绝
   （样本量不足）——正确行为。

**下一步（真人操作）**：完成 ≥11×2 条真人盲评 → `review unblind` →
`review resolve-quality` → `analyze-pilot --analysis-version closure_v2` →
若两指标方差可估计则 `prepare/freeze-confirmatory` → 真人审批绑定 SHA →
`run --stage confirmatory`。若方差仍不足，创建 **Pilot v2**（新协议版本 + 新
Block），不得修改 v1。

---

## 6. 未完成项 / 限制

- 人工盲评由真人提交，非 Agent 可代劳；本报告如实标注 pending。
- 本轮 A/B 选题仍是预注册的 delegated-operator 规则，非独立人类选题。
- Approval 是 CLI 子进程 `--yes` 授权，未证明独立真人身份与身份认证。
- Confirmatory / M4.3-04 Final Value Report 未启动（前置条件不满足）。
- M5-01 跨域调度、M5-02 双层 RRSI 演化不在本阶段。
