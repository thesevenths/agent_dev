# M6 Benchmark Specification

## 1. 定位

这是**研究方法评测基准**，不是模型能力基准。任务要求研究者（Harness）针对一个真实研究机会
选择问题、提出假设、设计实验、诊断失败；每个任务都必须可复现、有家族归属、有冻结答案。

它是初始工程基准，**不构成统计充分性声明**。确认性扩展需依据 pilot power analysis。

## 2. 规模与切分

| split | 任务数 | 用途 |
|---|---|---|
| dev | 36 | 失败分析、候选生成的唯一可见数据 |
| selection | 12 | Base/Candidate 配对选择（预注册主指标） |
| holdout | 12 | 独立确认；**task id 与 task family 都与 selection 不相交** |

三域均衡：SciNet++（科学机器学习）、Embodied AI（世界模型）、Quant（受控合成研究数据，
仅用于系统工程流程验证，**不构成 alpha 证据**）。

## 3. 每个任务的必填字段

`TaskSpec`：`task_id / domain / family_id / split / source / allowed_tools / budget /
seed / primary_metric / grading_spec / gold_ref / rubric_ref`。

- `source`：真实可复现入口。`scinet` 指向外部 SciNet++ 仓库的 dataset + config +
  `research_api.cli`；`embodied`/`quant` 指向 `research_runtime/domains/projects/*` 的真实
  config.yaml 与 local engine。
- `allowed_tools`：允许的工具白名单。
- `budget`：`max_campaign_rounds / max_calls / max_wall_s`，防止无限消耗。
- `seed`：固定配对 seed。
- `primary_metric` / `grading_spec`：冻结指标与评分定义。

## 4. 家族分组与泄漏防护

```
SciNet++     selection families: control_quality, ood_robustness   holdout: tta_efficiency
Embodied     selection families: dynamics_accuracy, safety_compliance   holdout: transfer_robustness
Quant        selection families: signal_validity, cost_efficiency   holdout: regime_robustness
```

- `validate_benchmark()` 拒绝：重复 task id、selection/holdout task 重叠、**family 重叠**、
  缺 source / tools / budget / seed / metric、sealed 文件 SHA 不符。
- `EvidenceBoundary.assert_allowed()` 拒绝：任何 selection/holdout task id 或
  `gold_ref/rubric_ref/expected_direction/minimum_meaningful_effect` 出现在候选生成器载荷中。
- `ContaminationLedger` 记录每个候选可见的 task id 集合；`assert_clean()` 可事后证明 holdout
  从未暴露给生成器。

## 5. sealed gold / rubric

gold 与 rubric 写入**独立文件** `benchmark/sealed/gold_rubric.json`，其文件 SHA 写进
`benchmark_manifest.json`。manifest 只保留 `gold_ref`/`rubric_ref` 占位符，不含任何答案。
`generator_view()` 只返回 dev 任务且剥离所有 sealed 字段（对抗测试验证）。

## 6. benchmark 版本与不可变性

`BENCHMARK_VERSION = m6_bench_v1`；manifest SHA 进入晋级证据的 `dataset_manifest_sha`，
并由 `verify_holdout_evidence(manifest_path=...)` 重新读盘复核。改动任何任务定义都会改变
manifest SHA，导致旧晋级证据失效（fail-closed）。

## 7. power analysis

`pilot_power_analysis()` 复用 `m4_3.sample_size.estimate_sample_size`，输入 pilot 配对差值
与**预注册的 SEOI**（`SEOI_BY_METRIC`），输出每对比与敏感度分析。它只给建议，不自动扩大
确认性基准；不足时明确标注 underpowered，不宣称统计充分。
