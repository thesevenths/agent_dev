# ARR v1.0 — 11 Expert Review Report

## 状态：NOT_PERFORMED（BLOCKED）

独立人类科学专家盲评**未进行**。当前环境无可用外部领域专家资源。

## 事实记录（不伪造）

- M7.2 50-case benchmark 的 Ground Truth 由 curator 制定，`reviewer_annotations` 标注
  `independent_expert_blind_review: NOT_PERFORMED`。
- 价值评审证据由真实 LLM critic 生成，已明确标注 `reviewer_identity: LLM critic (NOT
  an independent human expert)`，仅用于生产排序门，不用于科学能力认证。
- 无任何"专家签名/评审表/审查意见"被生成或伪造。

## 为什么这很重要

任务书明确：LLM-as-judge 只能作诊断工具，不能替代独立专家验证；盲评需隐藏 method/round/
arm identity，测量 inter-rater agreement，并文档化争议解决。这些都**没有**发生。

## 影响

| Gate | 影响 |
|---|---|
| L2 Scientific Workflow | 无法 PASS（要求独立专家评估通过） |
| L3 Recursive Improvement | 部分受影响（holdout 通过不完全依赖专家，但认证链需要） |
| L4 Generalization | 无法 PASS（要求专家评估完成） |

## 恢复路径

当有 2+ 名相关领域独立专家时：盲评 50 cases（隐藏 arm/round/method）→ 记录评分与理由 →
计算 inter-rater agreement（如 Cohen's κ）→ 争议由第三人裁决 → 更新本报告与 verdict。
在此之前 L2/L4 保持 INCONCLUSIVE。
