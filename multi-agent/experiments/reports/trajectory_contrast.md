# 成功 / 失败轨迹对比（弱化版）

> 弱化版：跨任务分布对比，不是同一任务的逐步对照 —— 只能说明「失败 run 在这些指标上更差」，不能归因到单条轨迹。

- 成功 run：16 / 失败 run：24

## 分布对比

| 指标 | 成功 run 均值 | 失败 run 均值 | 差 |
|---|---|---|---|
| 工具往返 / 步 | 2.66 | 1.949 | -0.711 |
| 重试次数 / 步 | 0.0 | 0.0 | +0.000 |
| 契约达成率 | 1.0 | 0.175 | -0.825 |
| 通过步骤数 | 2.125 | 0.625 | -1.500 |

## 按 agent 的每步重试数

| agent | 成功 | 失败 | 差 |
|---|---|---|---|
| ChatAgent | 0.0 | 0.0 | +0.000 |
| CrawlerAgent | 0.0 | 0.0 | +0.000 |
| CodeAgent | 0.0 | 0.0 | +0.000 |

## 失败聚类（kind × agent × step）

| kind | agent | step | 次数 |
|---|---|---|---|
| recursion | CodeAgent | 2 | 4 |
| recursion | CodeAgent | 3 | 3 |
| exception | ChatAgent | 3 | 1 |
| recursion | CodeAgent | 1 | 1 |
| recursion | CodeAgent | 4 | 1 |
| truncated | CodeAgent | 2 | 1 |
| truncated | CodeAgent | 3 | 1 |
| truncated | CodeAgent | 4 | 1 |
| unknown | chat_agent | 4 | 1 |
| unknown | code_agent | 2 | 1 |
| unknown | code_agent | 3 | 1 |
| unknown | code_agent | 5 | 1 |
