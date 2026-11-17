# M7.6 Implementation Report — Production Runtime Closure

## 1. Scope

从"科研模块存在"升级到"科研模块通过真实生产 Runtime 协作，并可跨多轮评估"。
无架构重写、无新建 Runtime/Campaign/审批系统、无 production FakeRunner、无自动审批。

## 2. 基线

- 基线分支：`feature/arr-m7-capability-validation` @ `6888ac95`
- M7.6 分支：`feature/arr-m7-6-production-runtime-closure`
- 两个 feature 分支祖先关系已核对，discovery 是 validation 的祖先，无需 reconcile。

## 3. 修改文件与根因/修复

| 文件 | 根因 | 修复（复用优先） |
|---|---|---|
| `discovery/approval_gate.py`（新） | M7 自审 gate 只做字符串黑名单 | 新建只读验证器，复用 M4.3 ApprovalStore，校验 status/source/scope/SHA-bound，fail-closed |
| `discovery/discovery_coordinator.py` | `_check_approval` 信任 callable/dict；harness SHA 来自参数；direct executor 旁路；异常不结算 reservation；value fallback 静默选首 | 生产模式只走外部 store；formal seam 强制检查；统一 `_run_experiment` funnel（生产走 adapter，禁 direct）；异常结算；生产缺 value evidence BLOCKED；resume 可恢复 HUMAN_APPROVAL |
| `discovery/formal_runtime.py`（新） | M7 执行绕过 M5/M6 | 薄适配层复用 M5 RuntimeState（journal/reservation）+ M6 HarnessLoader（SHA 绑定）+ 真实 DomainPack 引擎；campaign 生命周期写入 M5 journal；生产缺契约 BLOCKED |
| `m7_production_assembly.py`（新） | 无可用的真实 Loader 装配 | 从真实 M6 已批准 artifact（SHA 6c197910、user_explicit_2026-11-16 审批）装配 HarnessLoader + adapter；不伪造身份 |
| `run_m7_e2e.py` | 自动审批回调、git HEAD 冒充 harness SHA、单阶段 | 两阶段 prepare/resume；禁自动审批；harness SHA 来自 Loader；git SHA 独立记录；`--evidence-map` 要求真实价值证据 |
| `research_service.py` | 恢复路径 `campaign_id` 未赋值先引用（NameError） | 用 `existing.campaign_id` |
| `capability_bench/scientific_judgment.py` | verdict 仅由 novelty_accuracy 决定 | 预注册 primary/co-primary/secondary 多结局门控；novelty 只作 secondary |
| `capability_bench/memory_bench.py` | repeat_failure 用模型预测率 | 用 GT `expected_repeat_failure` 计算混淆矩阵/准确率；verdict 比较 grounded accuracy + shuffled 对照 |
| `capability_bench/integrated_bench.py` | 采样返回不足 n；用行数冒充 families；无 cluster CI | 精确 n 采样；family 一级字段；复用 `_clustered_bootstrap`；family-level paired delta + CI 门控 |
| `capability_bench/rsi_bench.py` / `run_m7_4_pilot.py` | Candidate 写死、OOD 未执行 | 从真实 base failure 诊断瓶颈再生成 Candidate；执行 OOD split；无瓶颈则不生成改进 |
| `discovery/novelty/assessor.py` | Jaccard 直接决定 NOT_NOVEL | Jaccard 降为 screening；强制 POSSIBLE_PRIOR_ART + SCREENING_ONLY + confirmed_novelty=False |
| `discovery/questions.py` | 逐 abstract 生成问题；RQ 结构缺字段；EIG 默认常数 | 多论文主题综合（可审计 token 聚合）；扩展 RQ 结构；EIG 仍需显式 observation model |

## 4. 关键架构约束（诚实记录）

M5 `ResearchService.run_research_campaign` 执行的是 ML `ExperimentPlan`（hypothesis/candidate/paired-seed），
M7 DomainPack 执行的是科学实验（arms/seeds/splits）。两者抽象不同，完整合并属于架构重写（M7.6 禁止）。
因此采用**正式 Runtime 适配层**：复用 M5 RuntimeState（journal/预算/claim）+ M6 HarnessLoader（SHA 绑定）
+ 真实引擎，而不是新建第二套 Runtime。真实生产 E2E 证据证明该 seam 满足生产链路要求
（campaign_created/running/completed ×2，真实 harness SHA，reservation 全结算）。

## 5. 信任模型（不得夸大）

- `approved_by` 是未认证的操作员字符串，无签名/证书/ACL（M4.3 `APPROVAL_BOUNDARY_M4_3.md`）。
- 审批是带外 `approval_cli` 独立进程写入的 SHA-bound 决策；driver 绝不写审批。
- 价值评审由真实 LLM critic 生成，明确标注 **非人类独立专家**；仅用于生产排序门，不用于 L2 专家盲评。
- Git commit SHA 与 Harness SHA 是两个独立身份，分别记录。

## 6. 兼容性

- 所有新增字段（RQ/novelty）均为可选默认值，M7.1–M7.5 存量 artifact 可继续读取。
- engine-free 测试夹具必须显式 `fixture_approval=True`，且生产模式拒绝；无夹具标记则 fail-closed。
- 完整测试：833 passed / 12 skipped / 37 subtests（见 M7_6_TEST_REPORT.md）。
