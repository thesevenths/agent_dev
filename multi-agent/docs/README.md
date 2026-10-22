# 文档索引

本目录记录 `Adaptive Runtime` 从方案提出到落地验收的完整过程。文档按「提案 → 评审 → 定稿 → 实施 → 验证」排列，每一步都保留原始结论与后续修正记录，便于回溯某项决策的形成依据。

---

## 阅读路径

| 顺序 | 文档 | 读者关心的问题 |
|---|---|---|
| 1 | [`adaptive_agent_runtime_rebuild_plan.md`](adaptive_agent_runtime_rebuild_plan.md) | 外部方案（V1）：要解决什么问题、模块如何划分、研究议程是什么 |
| 2 | [`reviews/review-01-runtime-rebuild.md`](reviews/review-01-runtime-rebuild.md) | 对 V1 的评审：哪些设计可保留、哪六个问题会阻塞实施 |
| 3 | [`adaptive_agent_runtime_v2_implementation_spec.md`](adaptive_agent_runtime_v2_implementation_spec.md) | 修订方案（V2）：可开工的规格，含 Gate 骨架与目录结构 |
| 4 | [`reviews/review-02-implementation-spec.md`](reviews/review-02-implementation-spec.md) | 对 V2 的评审：剩余六个结构性问题、四个落地缺口，以及可直接替换的五段规格 |
| 5 | [`adaptive_agent_runtime_final_plan.md`](adaptive_agent_runtime_final_plan.md) | 定稿（V3）：合并上述结论后的唯一施工依据 |
| 6 | [`adaptive_runtime_implementation_status.md`](adaptive_runtime_implementation_status.md) | 实施结果：方案 §2–§18 逐功能点 → 代码位置 → 验证方式；含暂未启用项的取舍说明 |
| 7 | [`adaptive_runtime_verification.md`](adaptive_runtime_verification.md) | 验证结论：判定器可信度、统计功效、真实成本与证据边界 |

若只读两份：先读 **定稿（第 5 项）**，再读 **验证报告（第 7 项）**。

---

## 文档定位

| 文档 | 类型 | 状态 |
|---|---|---|
| `adaptive_agent_runtime_rebuild_plan.md` | 外部提案（V1） | 已归档，保留作为评审基线，不作为施工依据 |
| `adaptive_agent_runtime_v2_implementation_spec.md` | 外部提案（V2） | 已归档，其骨架被定稿吸收 |
| `adaptive_agent_runtime_final_plan.md` | 设计定稿（V3） | 现行施工依据 |
| `adaptive_runtime_implementation_status.md` | 实施对照与验收记录 | 随代码同步更新 |
| `adaptive_runtime_verification.md` | 验证报告 | 随实测数据更新 |
| `reviews/review-01-runtime-rebuild.md` | 设计评审 | 已完成 |
| `reviews/review-02-implementation-spec.md` | 设计评审 | 已完成 |

---

## 关于评审文档

`reviews/` 下的两份文档是对外部提案的正式评审记录，保留其原始判断与严重度分级（阻塞 / 严重）。评审结论并非全部被采纳——若后续实测与评审判断冲突，以实测结果为准，并在实施对照表中记录该分歧。

文档中的严重度定义：

| 级别 | 含义 |
|---|---|
| 阻塞（P0） | 不修复则无法产生可信结论，或实施第一天即会走错方向 |
| 严重 | 可在实施中规避，但会导致返工或结论失真 |
| 建议 | 改进项，不影响正确性 |

---

## 相关代码入口

| 位置 | 内容 |
|---|---|
| `adaptive_runtime/` | 自优化实验层（27 个模块） |
| `adaptive_runtime/run_adaptive.py` | 唯一 CLI 入口，15 个顶层子命令 |
| `adaptive_runtime/tests/` | 165 个单元 / 集成测试 |
| `adaptive_runtime/settings.py` | 全部阈值与阶段开关（含取值依据注释） |
