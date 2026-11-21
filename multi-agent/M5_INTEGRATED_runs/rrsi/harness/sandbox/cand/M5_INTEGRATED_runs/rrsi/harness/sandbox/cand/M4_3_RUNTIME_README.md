# M4.3 A/B/C Value Validation Runtime — 使用手册

本运行时在 **不改动** `ResearchService` / Planner / Evidence Reader / Decision
Engine / 12-step fail-closed Gate / RRSI 的前提下，一次性提供：

| 组件 | 模块 |
|---|---|
| Pilot Runner（真实 SciNet++ A/B/C） | `research_runtime/m4_3/pilot_runner.py` |
| Confirmatory Runner（人工授权后执行） | `research_runtime/m4_3/confirmatory_runner.py` |
| Pilot 分析 / 样本量估计 | `pilot_analysis.py` / `sample_size.py` |
| 协议冻结 | `confirmatory_protocol_builder.py` |
| 统计分析（配对 + Holm） | `statistical_analysis.py` |
| M4.3-04 价值报告 | `value_report.py` |
| 统一控制器 / 状态 / 恢复 | `experiment_controller.py` / `experiment_state.py` / `experiment_recovery.py` |
| 人工盲评提交入口 | `review_queue.py`（`cli review ...`） |
| 统一 CLI | `cli.py` |

---

## 1. 真实引擎环境变量

运行时**永远不会**用 FakeRunner 顶替真实引擎；缺少任意一项就 fail-closed。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SCINET_DIR` | `E:\LLM\SciNet++` | SciNet++ checkout（需含 `research_api/cli.py`） |
| `SCINET_PYTHON` | `D:\conda_envs\scinet_gpu\python.exe` | 带 torch 的引擎解释器 |
| `SCINET_CONFIG` | `configs/pendulum_smoke.yaml` | 默认配置 |
| `RESEARCH_DATA_DIR` | `research_runtime/_data` | evidence / 历史证据所在目录 |

Agent 侧解释器（跑本 CLI 的那个）只需要 `pydantic` + `PyYAML`，**不需要** torch。

## 2. 数据位置

| 内容 | 位置 |
|---|---|
| 真实历史 Evidence | `research_runtime/_data/evidence.jsonl`（只读） |
| 每个阶段的 SQLite（registry / controller / review） | `M4_3_runs/<stage>/research.sqlite` |
| Director 状态与 journal | `M4_3_runs/<stage>/director/<block>/<arm>/` |
| A/B/C 选题记录（append-only） | `M4_3_runs/<stage>/objective_selections.jsonl` |
| Pilot 运行记录 | `M4_3_runs/pilot/pilot_run.json` |
| Pilot 分析产物 | `M4_3_runs/pilot/{pilot_analysis.json,pilot_report.md,sample_size_recommendation.json}` |
| Pilot 价值报告 | `M4_3_runs/pilot/{pilot_value_report.json,M4_3_PILOT_VALUE_REPORT.md}` |
| 冻结协议 + 人工批准 | `M4_3_runs/confirmatory/protocol/{confirmatory_protocol.json,confirmatory_approval.json}` |
| 最终价值报告 | `M4_3_runs/final/{value_report.json,M4_3_VALUE_REPORT.md}` |

> `M4_3_runs/` 是运行产物（默认被 `.gitignore` 之外的路径管理；需要留存证据时
> 显式 `git add -f`）。

## 3. 命令

```bash
# 0) 环境自检（不写任何东西；不 ready 时返回码 2）
python -m research_runtime.m4_3.cli preflight

# 1) 真实 Pilot（4 个 Block × A/B/C）
python -m research_runtime.m4_3.cli run --stage pilot \
    --blocks 4 --seeds 42 43 --candidate-budget 3 \
    --approver operator --max-campaigns 12

# 2) Pilot 分析 + 样本量建议 + Pilot 价值报告
python -m research_runtime.m4_3.cli analyze-pilot

# 3) 人工盲评（两名独立 reviewer，匿名 ID）
python -m research_runtime.m4_3.cli review list
python -m research_runtime.m4_3.cli review show <review_id>
python -m research_runtime.m4_3.cli review submit <review_id> \
    --reviewer alice \
    --scores '{"evidence_grounding":3,"novelty":2,"falsifiability":3,
               "intervention_feasibility":2,"metric_measurability":3,
               "expected_information_gain":2}' \
    --reason "evidence is specific; intervention plausible"
python -m research_runtime.m4_3.cli review status
python -m research_runtime.m4_3.cli review unblind      # 只在全部完成后允许

# 4) 正式实验准备（Pilot 证据不足时会被拒绝，并写出 blocked 草稿）
python -m research_runtime.m4_3.cli prepare-confirmatory --blocks 8 --seoi 0.5
python -m research_runtime.m4_3.cli freeze-confirmatory  --blocks 8 --seoi 0.5
python -m research_runtime.m4_3.cli approve-confirmatory --approved-by alice

# 5) 正式实验（默认禁止自动启动；必须已冻结 + 已批准）
python -m research_runtime.m4_3.cli run --stage confirmatory --blocks 8

# 6) 报告 / 状态 / 恢复 / 导出 / 校验
python -m research_runtime.m4_3.cli report --stage pilot
python -m research_runtime.m4_3.cli report --stage final
python -m research_runtime.m4_3.cli status
python -m research_runtime.m4_3.cli recover --dry-run
python -m research_runtime.m4_3.cli recover
python -m research_runtime.m4_3.cli export --out M4_3_runs/export.json
python -m research_runtime.m4_3.cli verify-integrity
```

返回码：`0` 成功；`2` preflight 未就绪；`3` 被门拒绝（缺真实引擎 / 缺评审 /
缺冻结协议 / 缺人工批准）；`4` registry 完整性校验失败。

## 4. 失败处理与恢复

* **进程被杀 / 断电**：直接重跑同一命令。`ExperimentController` 从持久化状态恢复，
  `experiment_recovery` 会报告已完成 / 未完成的 Block，**不会**重复创建 Campaign。
* **Director 已跑一半**：`research_runtime.m4_3.cli recover` 依据 persisted
  Campaign/Report/Experience 对账；只有真实完成的 chain 才会被 finalize。
* **预算不足**：`--max-campaigns/--max-seconds` 用尽时以 `BudgetExhausted` 停止，
  Controller 进入 `STOPPED` 并记录 `stopped_reason`，绝不静默截断样本。
* **环境缺失**：`preflight` 返回 `ready=false` 与具体原因（checkout / 引擎解释器 /
  真实证据），`run` 直接拒绝。
* **Gate 证据不完整**：该 arm 记为 `invalid`（附 `invalid_reason`），不会被升级为
  `valid_finding`。

## 5. 语义边界（必须遵守）

* `valid_finding` 需要 **成功** 的 Campaign（`completed` / `completed_with_failures`
  且至少一个 trial 成功）＋ 报告有真实科学结论 ＋ Plan/Approval/Campaign/Session/
  Report/Experience 精确互相关联 ＋ `experience_refs` 能解析到真实行。
* `negative_finding` 允许“真实完成但没有预期正向发现”。
* `failed` / `cancelled` / `interrupted` 永远不算 Gate 成功，且全部保留。
* 一个 Campaign 的成功证据不能被第二个 run / block / arm 复用。
* Pilot **不宣布**胜出；四种价值结论只属于冻结后的正式分析。
* 没有真实人工盲评时，`independently_reviewed_objective_quality` 记为缺失，
  不做插补、不用自动评分冒充。

## 6. 测试

```bash
# 引擎无关（CI）
python -m pytest research_runtime/tests/test_m4_3_02_*.py -q

# 真实引擎（opt-in，训练模型；CI 中显式 skip）
M4_3_RUN_REAL=1 python -m pytest research_runtime/tests/test_m4_3_02_real_engine.py -q -rs
```
