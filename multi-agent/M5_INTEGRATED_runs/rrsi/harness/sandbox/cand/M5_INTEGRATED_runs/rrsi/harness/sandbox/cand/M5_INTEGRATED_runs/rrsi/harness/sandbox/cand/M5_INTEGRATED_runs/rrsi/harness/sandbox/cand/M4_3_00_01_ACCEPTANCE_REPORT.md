# M4.3-00 + M4.3-01 验收报告（ARR Director Value Validation / Phase 00 + 01）

**分支**：`feature/arr-0.1-domain-registry`　**HEAD**：`fc0fd3a0`　**日期**：2026-11-15
**范围**：仅 M4.3-00 Closure（P0-01..P0-06）与 M4.3-01 A/B/C Evaluation Harness。
**未开始**：M4.3-02 Pilot、M4.3-03 Confirmatory、M4.3-04 Value Report、M5。
**工程纪律**：未 merge main、未建 PR、每个逻辑改动独立 commit + push、未改历史 Evidence/metrics/verdict、未降低统计门、未绕过 frozen split、未创建第二套 ResearchService/Planner/Decision Engine、未用 FakeRunner 冒充真实引擎。

---

## 1. 结论先行

M4.3-00 六处遗留缺陷**全部修复并独立验证**，M4.3-01 评价 Harness **完整落地但未运行任何实验**。全量回归 **318 passed / 1 skipped / 37 subtests**；CI 精确门 **263 passed / 1 skipped / 29 subtests**。Engine-free 与真实引擎 E2E 分开报告：真实 SciNet++ E2E 因 opt-in/环境门**明确 skip**，不宣称通过。本阶段**暂停**，未启动 Pilot/Confirmatory，未下任何价值结论。

---

## 2. 提交链（9 个独立 commit，均已 push）

| Commit | 内容 |
|---|---|
| `f0cd30c5` | P0-01 Failure Evidence Reference（typed scope + 精确 event_type 核对真实 signals） |
| `48d0ba42` | P0-02 OOD 证据语义（真实可比 ID/OOD 指标，4 维核对 + frozen split 排除 + evidence_insufficient） |
| `d0750141` | P0-03 Campaign 幂等/恢复（round cap 仅新建 binding、状态分类恢复、非 exactly-once 语义） |
| `f1040078` | P0-04 Director Recovery（Report+Experience 三层证明、RECOVER_COMPLETE 合法路径、全转换经 StateMachine） |
| `04347c25` | P0-05 真实 SciNet++ E2E 证明范围（单一 ResearchStores、真实 Provider/Discovery、外部 CLI、round-2 闭环） |
| `7fc354ab` | P0-06 审批边界文档（准确信任模型）+ 可执行事实核查测试 |
| `da1eabe0` | M4.3-01 A/B/C Harness（冻结指标、独立盲评、区块配对、全量保留、预注册 shell） |
| `fc0fd3a0` | CI 接入 M4.3-00/01 测试 |

（前一阶段 M4-E2E-01.1 基线 `6776776a` 之上。）

---

## 3. M4.3-00 六项 P0 逐条验收

### P0-01 Failure Evidence Reference
- **根因**：`_resolve_ref` 把 session_id/objective_id 混入同一集合、不区分前缀类型，且**完全不校验 `:event:<type>`**，伪造事件即可"可解析"。
- **修复**：共享引用契约 `make_failure_ref`/`parse_failure_ref`（generator）+ 严格解析器（Critic），只按 scope 对应的 id 字段核对 `failures.session_failure_signals` 的 `(id, event_type)` 组合。
- **测试**（`test_m4_3_p0_01_failure_refs.py`，10）：真实 Session/Objective Failure 通过；不存在 ID、伪造事件、scope/type 混淆、malformed 均拒绝；Failure-derived Candidate 通过 Critic。
- **跨项目通用**：契约不含任何特定框架/项目名（session/objective/event 通用）。

### P0-02 OOD Candidate 证据语义
- **根因**：`_from_ood_gaps` 仅凭 `ood_dataset_version` 元数据就生成"存在 gap"结论；现有 Evidence View 无 ID/OOD 指标数值载体。
- **修复**：定义框架无关的"可比较 ID/OOD 指标记录"契约（`id_ood_comparable_metrics`），校验 domain / dataset version / metric identity / evaluation split 四维度一致、两侧数值有限、溯源到真实 evidence_id、排除 frozen final split（复用 `RESERVED_FINAL_SPLITS`，不修改规则）。仅合规记录生成 gap claim；仅有 version 时生成 `id_ood_exploration` 候选并标记 `evidence_insufficient`（经 `success_criteria` 落地持久化）。Critic 增加 gap-claim 守卫与 frozen split 显式拒绝。
- **测试**（`test_m4_3_p0_02_ood_evidence.py`，12）：有真实可比指标→允许 gap candidate；仅 version→无 gap claim；版本不匹配/相同、metric 不一致、split 不一致、frozen split、不可溯源、非有限值、domain 不匹配全部拒绝；View 自定义保留 split 生效。不重算 verdict、未改统计门/frozen split 规则。

### P0-03 Campaign 幂等与恢复
- **根因**：`_run_campaign` 在 `campaign_for()` 之前无条件检查 `campaigns_for_objective()`（计的是 **binding 数**），`max_campaign_rounds=1` 时"崩溃后恢复"被误判 `round_limit_reached`；已有 CampaignRecord 一律 `already_submitted`，掩盖未完成执行。
- **修复**：round cap **仅在新建 binding 时**执行；新增通用 `classify_campaign_state`（not_started/preflighting/running/paused/terminal）；仅 terminal 为幂等 no-op；in-flight 状态以 `resume=True` 从持久化状态恢复，`CampaignRunner.run()` 对 in-flight record 不再 `duplicate_run_ignored` 而是 `resume()`；明确真实执行语义 `EXECUTION_SEMANTICS`——**at-most-once submission + resume-in-place，不宣称严格 exactly-once**（依赖确定性 job_id、PERSIST BEFORE EXECUTE、job probe、引擎 PK 去重）。
- **测试**（`test_m4_3_p0_03_campaign_recovery.py`，11）：cap=1 崩溃→恢复成功；cap=1 成功→重复提交无新 Campaign；CampaignRecord 创建后/配置生成后/running 后/paused 崩溃均恢复；job ID 提交前持久化且恢复不重复提交已提交 job；恢复不重复消耗预算；外部 binding 无 record 仍重启一次。runner 既有 suite 无回归（17+10+27+1+2+9+14 passed）。

### P0-04 Director Recovery 状态与审计语义
- **根因**：`reconcile_director_state` 把"Report 存在"等同于"Experience 存在"直接返回 `S_COMPLETED`；而 `S_COMPLETED` 不在 `ACTIVE_STATES`，`recover()` 落入 halt 分支用 `E_FAIL` 写 COMPLETED；step 方法直接改 `context.state` 绕过 StateMachine。
- **修复**：`report_has_real_experience` 三层证明（真实 report_id + 非空 experience_refs + Experience Store 真实行 id 匹配）；`finalize()`/`record_experience()`/`recover()` 全部强制三层验证；状态机新增唯一合法 `RECOVERING --RECOVER_COMPLETE--> COMPLETED`；`_record` 统一经 `StateMachine.require`；halt 选择正确事件（`E_FAIL` 只能写 FAILED）。
- **测试**（`test_m4_3_p0_04_director_recovery.py`，8）：Report 存在但 Experience 缺失→不 COMPLETED（回 `RECORDING_EXPERIENCE`）；三层齐全→COMPLETED；finalize 无真实行 fail-closed；恢复用 `RECOVER_COMPLETE` 从不用 `E_FAIL`；状态机层面禁止 `E_FAIL→COMPLETED`；非法写被拒；Journal Event 与最终状态一致性；端到端故障不谎报 COMPLETED。

### P0-05 真实 SciNet++ E2E 证明范围
- **修复**：`test_m4_e2e_01_real_engine.py` 重写为 10 点规格：① 单一 `service.stores` 实例（同一 SQLite）贯穿 Objective/Session/Hypothesis/Plan/Approval/Experience；② 移除手工 `sha256:neg1`；③④ 真实 `DiscoveryContextProvider` 读真实历史 run `arr_m34_001` + 真实 `ObjectiveDiscovery` 生成 Objective；⑤ Director 驱动真实引擎；⑥ 外部 `approval_cli` 子进程授权；⑦ 真实 Evaluation + Experience 写回；⑧⑨ 同一 Provider 构建 round-2 Snapshot 并精确引用 round-1 Experience；⑩ 缺真实 Evidence/SciNet++/opt-in 时**明确 skip，不回退合成 Snapshot**。
- **验证**：真实 Provider 实测 `arr_m34_001` → **118 个真实 evidence 实体、6 类 carrier、3 domains**，真实 Discovery 产出 **14 候选/14 approved**；round-2 闭环用真实 Provider + 真实 Experience 单独验证通过（`resolve exact=True / fabricated=False / cites exact=True`）。本环境 E2E 因 `M4_E2E_01_RUN_REAL` 未开而明确 skip（`SciNet++ present=True, real evidence=118`），**不宣称真实引擎已跑通**。

### P0-06 审批边界文档准确性
- **文件**：`research_runtime/APPROVAL_BOUNDARY_M4_3.md` + 事实核查测试。
- **确认**：Director 不自授权（代码强制，有测试）；Agent-facing tool surface 11+4 个工具**无任何 grant/approve 工具**（代码枚举验证）；`approval_cli` 为独立进程入口；**`ApprovalStore.grant()` 仍是可被任何 Python 调用方调用的公开 Store API**（约定而非沙箱）。
- **信任模型**：`approved_by` 仅校验非空，**不是强身份认证**；`source="human"` 是调用方字符串；本阶段不实现企业级认证；`approval_bypass_count` 作为 Safety 指标。测试用伪造身份串证明"仅非空校验"，防止文档夸大。

---

## 4. M4.3-01 A/B/C Evaluation Harness

**交付物**：`M4_3_EXPERIMENT_PROTOCOL.md`、`M4_3_runs/protocol.yaml`、`research_runtime/m4_3/{__init__,metrics,experiment_registry,evaluation_harness}.py`、`research_runtime/tests/test_m4_3_evaluation_harness.py`。**未改 ResearchService**。

- **三组**：A 人工（不读历史 Experience）、B 记忆辅助人工、C 自动 Discovery；B/C 读**相同**历史上下文，harness 主动为 A 屏蔽历史且 `build_arm_context` 直接层 fail-closed 双保险。
- **对照规则**：同 Snapshot/同 Seed 区块配对、固定 model/prompt/Ranker/DomainPolicy、固定候选与时间预算、同一 12-step gate、外部 Approval、同 Campaign Budget、同 Paired Seeds、**全部结果保留**（valid/negative/INCONCLUSIVE/invalid/failed/interrupted）、主指标冻结（`assert_no_posthoc_metric_change`）。
- **指标**：Primary 2 / Secondary 9 / Safety 5，冻结哈希 `frozen_metric_contract()`；`compute_metrics()` 断言指标集合不变。
- **独立盲评**：`blind()` 拒绝 arm/rank_score/ranker_version 等字段；`IndependentReviewer` 用固定 6 维 rubric `[0,4]`、强制 reason；**Ranker 无路径给自己评分**；arm C 返回的 ranked 信号在评审前剥离。
- **预注册**：`FrozenProtocol.assert_confirmatory_ready()` 在 confirmatory 未冻结字段时拒绝；`protocol.yaml` 的 `pre_registration` 全部留空、`confirmatory=false`、`M4_3_02_pilot`/`M4_3_03_confirmatory` = NOT_STARTED；允许 4 种最终结论，禁止强行 C 胜出。
- **测试**（17）：三组定义/上下文、A 屏蔽、B=C 历史一致、盲评剥离、Ranker 不自评、reason 强制、指标冻结、主指标不可事后改、全量保留、Safety 计数、12 步门、冻结协议、层级（Snapshot/Seed/Paired Seeds）、YAML 与代码契约零漂移、Pilot 未启动。

---

## 5. 回归与 CI 证据

| 检查 | 结果 |
|---|---|
| 全量 `research_runtime/tests/` | **318 passed, 1 skipped, 37 subtests**（0 failed） |
| CI 精确文件门（24 个文件） | **263 passed, 1 skipped, 29 subtests** |
| runner/resume/gates 既有 suite | 17 / 10 / 27 / 1 / 2 / 9 / 14 passed |
| `arr-research-runtime.yml` | `yaml.safe_load` 合法；已纳入全部 M4.3 新测试 |
| 既有 `test_research_tools.py` 4 failed | **基线对比确认既有失败**（缺 `urllib3`，HEAD 同样 4 failed），与本次改动无关 |
| 分支/远端 | 同一分支、与 origin 同步、9 个独立 commit 全 push |

**新增测试合计 64 个**（P0-01 10、P0-02 12、P0-03 11、P0-04 8、P0-06 6、M4.3-01 17）。P0-05 为 gated 真实引擎 E2E（1 个，明确 skip）。

---

## 6. 明确未做 / 风险与边界

- **真实引擎 E2E 未实际执行**（需 `M4_E2E_01_RUN_REAL=1` + SciNet++ checkout 训练，分钟~小时级）；仅证明 skip 判定与 round-2 闭环逻辑正确，**不宣称真实训练通过**。
- **Pilot / Confirmatory / Value Report 未启动**；未做任何显著性结论，未回答 M4.3-04 的 7 个价值问题。
- **审批边界是工具面/部署约定，不是沙箱**：`ApprovalStore.grant()` 可被任何同进程 Python 调用；`approved_by` 非强身份认证（详见 `APPROVAL_BOUNDARY_M4_3.md`）。
- **Campaign 执行语义**：at-most-once submission + resume-in-place，非严格 exactly-once 物理执行。
- 未跟踪文件（`M2_6_scinet_loop.log`、`M2_7_runs/*`、`_probe_*.py`）按前阶段约定**刻意排除**，未纳入提交。

---

## 7. 暂停声明

M4.3-00 与 M4.3-01 已全部完成、验证、提交、推送。**按规格暂停**：不开始 M4.3-02 Pilot、不开始 M4.3-03 Confirmatory、不开始 M4.3-04 Value Report、不开始 M5、不 merge main、不建 PR。等待下一步指令。
