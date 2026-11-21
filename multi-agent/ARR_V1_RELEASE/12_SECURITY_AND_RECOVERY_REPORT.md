# ARR v1.0 — 12 Security and Recovery Report

## 1. 恢复（WP-01 实测）

- 原子 campaign claim（BEGIN IMMEDIATE）保证 at-most-once dispatch；并发/不同 executor 被拒。
- 状态机 CREATED→CLAIMED→RUNNING→COMPLETED/FAILED；非法转换抛错。
- crash 后 CLAIMED/RUNNING → RECOVERY_REQUIRED，不盲重后端（exactly-once 诚实）。
- reservation 在结果已知后结算；失败执行结算为 failed，不伪装成功。
- journal hash-chain 完整（28 事件，verify ok）；program manifest SHA 重启可校验。

## 2. 安全边界

- 审批：M4.3 带外 CLI、SHA-bound、fail-closed；approved_by 是未认证字符串（如实声明）。
- 无任意 shell：arr_cli 无 os.system / shell=True / subprocess（测试断言）。
- 最小权限：生产执行只在带外审批后经正式 Runtime；holdout 访问有 ledger。
- 冻结数据/阈值/历史结果不可被自动流程修改（架构约束 + 测试覆盖）。

## 3. 未实现
企业级 operator 身份认证（B-04，PARTIAL）；如需要需外部 PKI/ACL，不在 v1.0 自造。
