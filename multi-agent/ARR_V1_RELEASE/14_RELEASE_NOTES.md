# ARR v1.0 — 14 Release Notes

## 交付定位
从"科研模块存在"升级为"模块通过真实 Runtime 协作 + 可跨轮编排 + 证据可复现"的**研究平台**。

## 亮点
- L1 生产执行链路真实跑通（M7.6 E2E，带外审批 + 正式 M5/M6 + 真实引擎）。
- WP-01 修复 3 个 P0：重复 dispatch 二次执行 backend、INCONCLUSIVE 伪造成 REFUTES、无原子 campaign claim。
- WP-02 薄 program 层：3 轮真实 LLM program，跨轮 memory 递增，planning 由前轮 artifacts 驱动。
- WP-05/06 六类记忆 + 七状态归因 + 负迁移测量。
- WP-07/08 method registry + 外部 benchmark adapter（诚实 BLOCKED）。
- WP-10 运维 CLI + 可复现包导出。
- 869 passed / 12 skipped，净增 36 测试，零回归。

## 限制（不隐藏）
- L2/L3/L4 均 INCONCLUSIVE：缺独立专家盲评、≥8 family、真实瓶颈触发的 RSI 全链。
- 外部 benchmark 未获数据/许可；不声称通过。
- 审批身份仍为未认证字符串。

## 结论
可发布为 **Engineering Release（研究平台）**；不可声称 Full ARR v1.0 Scientific Certification。
