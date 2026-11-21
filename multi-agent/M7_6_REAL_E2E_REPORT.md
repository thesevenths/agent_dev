# M7.6 Real E2E Report

> 真实生产执行，非 FakeRunner/FakeProvider。证据目录：`M7_6_EVIDENCE/run2/`

## 1. 结论

**生产链路：完整接通并成功执行。科学结论：INCONCLUSIVE（诚实）。**

```
status=COMPLETED, runtime_integration=FORMAL_PRODUCTION_RUNTIME,
scientific_verdict=INCONCLUSIVE, result_sha=572fab83...
```

## 2. 真实依赖清单

| 依赖 | 实证 |
|---|---|
| 学术 API | OpenAlex 真实请求，19 unique papers，success_rate 0.667，6 queries |
| LLM | vLLM OpenAI-compatible `/models/Qwen3.8-27B-FP8`，2 次真实调用（novelty 3828 tokens、hypotheses 1719 tokens），全部 ok |
| Harness | 真实 M6 已批准 harness `6c197910...`，经真实 HarnessLoader.resolve_for_campaign 解析 |
| 审批 | 带外 approval_cli 独立进程，`source=human`，`approved_by=user_explicit_2026-11-17`，SHA 绑定 `65066556...` |
| 引擎 | 真实 DomainPack embodied_world_model，selection + replication |
| Runtime | 真实 M5 RuntimeState：28 journal 事件、2 reservation 全 consumed/succeeded |

## 3. 身份分离（关键）

| 身份 | 值 |
|---|---|
| git_commit_sha（运行源码） | `a51284f20df152aeaf67ff1b267e65ce7a7787f2` |
| harness_sha（M6 已批准方法版本） | `6c1979100dda02f035a0ee7160580cdc6d4f29500f71ec064e20cdd1df6b000d` |
| preregistration_sha | `650665564951b7aa3673cb23910ff1ee570f6b0b06977d2d7421274353bb3146` |

三者独立，harness SHA **不是** git HEAD（M7 基线缺陷已修复）。

## 4. 两阶段审批流（关键）

1. `prepare`：真实文献→假设→实验设计→冻结 preregistration→**停在 HUMAN_APPROVAL**，创建 pending approval。
2. 真人通过 AskUserQuestion 显式批准；operator 字符串 `user_explicit_2026-11-17`（类比 M6 的 `user_explicit_2026-11-16`）。
3. 带外 `approval_cli grant approval_1e09bf25b3fa --approved-by user_explicit_2026-11-17 --yes`。
4. `resume`：重读审批、复核 SHA-bound、经正式 Runtime 执行。driver 全程未写审批。

信任边界：`approved_by` 是未认证字符串（M4.3），无签名/证书；审批是带外、SHA-bound、fail-closed 的，不等于企业身份认证。

## 5. 正式链路证据（journal 摘录）

```
budget_reserved → campaign_created(harness_sha=6c197910...) → campaign_running
  → campaign_completed(outcome=succeeded)        [EXPERIMENT_EXECUTION]
budget_reserved → campaign_created(harness_sha=6c197910...) → campaign_running
  → campaign_completed(outcome=succeeded)        [REPLICATION]
```

- campaign_id：`camp_edbfbf8b743cc8e...`、`camp_4ad8280ca57c640...`（真实派生，非合成 `m7sel_`）
- 2 reservations：state=consumed, outcome=succeeded
- journal：28 events, head_sha=f2ff1fe2..., ok=true
- result_integrity：12/12 result SHA 校验通过

## 6. 价值评审证据（不伪造）

`M7_6_EVIDENCE/value_evidence.json`：由真实 LLM critic 对 10 篇真实文献生成独立评审记录 + 显式 EIG observation model。
明确标注：reviewer = LLM critic，**非人类独立专家**；仅用于生产排序门，不用于 L2 专家盲评。

## 7. 科学结论为何是 INCONCLUSIVE

- Novelty：2 个 claim 均 `NOVELTY_UNCERTAIN` / `SCREENING_ONLY`（Jaccard 不再冒充证明）。
- Selection verdict：INCONCLUSIVE（effect below MME）。
- Replication verdict：INCONCLUSIVE（power_gate=NOT_ASSESSED，无真实 pilot families）。
- Memory tier：`candidate`（INCONCLUSIVE 结论未提升为 verified）。

这些不是工程失败：系统正确拒绝在证据不足时下结论。
