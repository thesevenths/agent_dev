# RRSI 达成判据

- 结论：**未达成**（硬条件 8/9）
- 基线版本：`base/v2`；轮数 6；promote 1 / demote 0
- 生成时间：2026-10-25T18:18:18

| 判据 | 结果 | 证据 |
| --- | --- | --- |
| `noise_calibrated`（硬）噪声已**实测**标定（δ 不是假设值） | ✅ | `{"exists": true, "passed": true, "sd_diff_null": 14.9922, "k": 3, "validity": [], "target": 15.0, "tasks": 8, "n_pairs": 45}` |
| `loop_self_contained`（硬）每轮诊断只吃本轮自己跑出的轨迹 | ✅ | `{"rounds": 6, "offenders": []}` |
| `holdout_quarantined`（硬）holdout 从未参与诊断/提案（泛化层干净） | ✅ | `{"rounds_checked": 6, "leaked": [], "holdout_ids": [], "excluded_run_counts": []}` |
| `auto_promote_happened`（硬）闭环自己至少换过一次基线 | ✅ | `{"auto_promotes_in_window": 1, "auto_promotes_total": 1, "tags": ["base/v2"], "targets": ["AGENT_MAX_ITERATIONS_CODE"]}` |
| `no_human_promote_in_run`（硬）本次运行窗口内没有任何人工签字 | ✅ | `{"human_promotes": []}` |
| `behavior_actually_changed`（硬）改动真正落到生产（内容指纹改变） | ✅ | `{"content_stamp_before": "abc0793462d6c409", "content_stamp_after": "23dd8b2d79c3d661", "content_stamp_now": "23dd8b2d79c3d661", "promote_records": 1, "note": "内容指纹变了才说明生产 harness 真的换了一版；只升版本号不改内容 = 自欺"}` |
| `holdout_paired_gain`（硬）holdout 配对增益 > 0（换没参与训练的集上也更好） | ❌ | `{"rounds_with_pairs": 1, "best": -2.855, "series": [{"round": 2, "holdout_delta": -2.855, "pairs": 10}]}` |
| `post_promote_stable`（硬）换基线后重测 holdout 未退步 | ✅ | `{"checks": [{"tag": "base/v2", "drift": -0.493, "tol": -5.0, "n": 10}], "bad": 0}` |
| `no_unresolved_demote`（硬）没有「回滚后仍带着坏基线继续跑」 | ✅ | `{"promotes": 1, "demotes": 0, "last_event": "re-snapshot", "note": "回滚发生过又没重新升上去 = 当前基线就是那个被否掉的版本，必须让人看见"}` |
| `environment_healthy`（观察）端点可达、没有成片的 infrastructure 废跑（观察项） | ✅ | `{"infra_runs": 0, "runs_with_usage": 1020, "samples": [], "note": "这些 run 测的是端点可达性，已被排除在样本外；不是 harness 的质量信号"}` |
| `improvement_capacity`（观察）仍有可检出的改进空间（观察项） | ✅ | `{"accepts": 1, "no_candidate_rounds": [], "rounds": 6}` |

## 未达成时缺什么

- **holdout_paired_gain** holdout 配对增益 > 0（换没参与训练的集上也更好） → {"rounds_with_pairs": 1, "best": -2.855, "series": [{"round": 2, "holdout_delta": -2.855, "pairs": 10}]}

## 这份判据为什么不是自证

1. 判据读的是磁盘上的产物（diagnosis.json / policy.json / base_history.jsonl / learning_curve.json），不是运行时内存里的印象；任何人可以删掉报告重跑 `auto --verify-only`。
2. `holdout_paired_gain` 用的是**配对差**，不是两组均值 —— 后者会把「两次不同的噪声」当成改进。
3. `behavior_actually_changed` 要求生产内容指纹真的变了：只升版本号而不改行为的闭环，在这张表上过不了。
4. `post_promote_stable` 是换基线**之后**重测的：它防的是「探针里赢了、上线后不行」。
5. 任一条不过 ⇒ 直接写未达成，并列出缺的那条。不做加权平均，不折算成『进度 80%』。
