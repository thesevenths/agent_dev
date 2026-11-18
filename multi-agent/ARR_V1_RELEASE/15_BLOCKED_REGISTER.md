# ARR v1.0 — 15 Blocked Register

> 外部阻塞如实记录；不伪造、不自动标记完成。随项目推进更新。

| ID | 阻塞项 | 影响的 gate | 状态 | 说明 |
|---|---|---|---|---|
| B-01 | 独立人类科学专家盲评 | L2 Scientific Workflow / L4 Generalization / L3 部分 | BLOCKED | 无外部专家资源；50-case benchmark ground truth 由 curator 制定，非独立盲评；LLM-as-judge 仅为诊断，不能替代 |
| B-02 | ≥8 独立 Research Family + ≥3 paired seeds | L4 Generalization | BLOCKED | 现有 embodied DomainPack family 有限；M7.5 pilot 实测 7 family < 8；不降低 min_families |
| B-03 | 外部 benchmark 访问/许可（ScienceAgentBench / FIRE-Bench / DISCERN / ResearchClawBench / SCI-AGENT-EXAM / SciAgentArena） | L4 external_validation | BLOCKED | 未验证访问与许可；不假设可用、不伪造结果；提供 adapter 与 coverage matrix，实测状态记录为 NOT_RUN/BLOCKED |
| B-04 | 独立 authentication（operator identity / principal） | WP-01 approval 强度 | PARTIAL | M4.3 approved_by 是未认证字符串；带外 CLI + SHA-bound 已实现，但无签名/证书/ACL；如实声明，不声称企业身份认证 |
| B-05 | 更多算力用于真实 SciNet++ 多轮训练 | L3/L4 统计功效 | PARTIAL | 以 CPU/小 GPU 仿真为主；如功效不足标记 INCONCLUSIVE，不虚构样本 |

## 处理策略
- 上述阻塞不阻断所有独立工程任务（WP-01/02/05/07/10 等继续）。
- 每次外部资源到位后重新评估对应 gate；不在报告中把 BLOCKED 转为 PASS。
- B-01/B-03 若最终无法获得，L2/L4 保持 INCONCLUSIVE/BLOCKED 并在 verdict 体现。
