# M5-01 Cross-domain Research Scheduler Architecture

**结论先行**：M5-01 以薄模块完成跨域调度骨架：通用 Scheduler 只按 capability 与独立预算路由，不写死领域 metric/dataset/config key；三个真实域（SciNet++ / embodied 仿真 / quant 回测）均已注册并完成真实 Smoke Campaign。跨域 Experience 保留 `source_domain`，可被下一轮 Discovery 读取，但不被当作其它域的科学事实。

## 1. 新增薄模块

| 文件 | 职责 |
|---|---|
| `research_runtime/domain_adapter_registry.py` | 统一 Adapter Contract：domain_id / capabilities / preflight / build_snapshot / run_campaign / collect_result / resource_estimate / health_check |
| `research_runtime/domain_budget_manager.py` | 每域独立持久化预算：limit / reserved / consumed / remaining；reservation 先于执行；同 campaign 不重扣 |
| `research_runtime/domain_run_record.py` | 全局 run_id + objective_id + domain_id + campaign_id + approval/plan/session/report 不可变关联记录 |
| `research_runtime/cross_domain_scheduler.py` | capability 路由、capability 不满足即拒绝、失败隔离、外部审批后运行真实 Director、统一 Experience 读取与跨域 Snapshot |

复用未改：`ResearchService`、`ResearchDirector`、`ObjectiveDiscovery`、Decision Engine、Evidence Reader、12-step Gate、DomainBundle、CampaignStore、ResearchStores。

## 2. 三个已注册域

| domain_id | 真实引擎 | capabilities（节选） | 数据/来源 |
|---|---|---|---|
| `scinet` | 外部 SciNet++ CLI | `real_engine`, `paired_seeds`, `validation_split` | 冻结 pendulum 数据集，真实训练/验证 |
| `embodied` | 本地 robot world-model simulator | `simulation`, `paired_seeds`, `validation_split` | `robot_sim_v1` frozen manifest + SHA |
| `quant` | 本地 causal-factor backtest | `backtest`, `paired_seeds`, `synthetic_data` | `market_hist_v1` frozen manifest；**明确 synthetic，非真实市场数据** |

Domain-specific metric/config 只存在于各域模块与 policy 中；Scheduler 只比较通用 capability 字符串。

## 3. 调度与隔离语义

1. Objective 从任一域 Store 定位；`required_capabilities` 来自 Objective constraints。
2. `route()` 检查目标域 capabilities；不满足则 `DomainCapabilityError`，不执行。
3. 每域独立 budget（本 Smoke 各 1 campaign）；一域耗尽不影响其它域。
4. `ResearchDirector` 负责 Hypothesis → Plan → Approval request；Scheduler 只调用外部 approver，绝不自行 grant。
5. Plan SHA 绑定 approval；确定性 campaign_id 在真实 Campaign 前 reservation。
6. `run_objectives()` 逐域 try/except；失败记录为 `ok=False` 并继续其它域。
7. `read_experience()` 从每个真实域 Store 读取 Experience，并附 `source_domain` / `source_run_id`。
8. `build_cross_domain_snapshot()` 合并真实 Snapshot，显式附 `source_domain`；不会把某域结论直接标为另一域事实。

## 4. 真实 Smoke 结果

| 域 | Objective | Campaign | Report | Director | Budget |
|---|---|---|---|---|---|
| scinet | `obj_m5s2_scinet` | `campaign_07cf3e22efff5f17` | `report_40d85e60e496` | COMPLETED | consumed=1 |
| embodied | `obj_m5s2_embodied` | `campaign_fcb6cc156d6859d2` | `report_6c8178069472` | COMPLETED | consumed=1 |
| quant | `obj_m5s2_quant` | `campaign_8b247041db585098` | `report_c05cd8405281` | COMPLETED | consumed=1 |

本地域的真实 grounding experiment `m5s2_src_embodied` / `m5s2_src_quant` 由各自真实 engine 的 train+evaluate 生成，只作为本域结构化诊断证据；没有跨领域借用科学结论。

下一轮 Discovery 读取跨域 Snapshot（3 objectives / 3 sessions / experience_available=true / unresolved_gaps=[]），产出 3 个新 Experience-based Proposed Objectives。

## 5. 边界

- 当前为 supervised multi-domain：Scheduler 自动路由，真实实验仍要求原有 Human Approval。
- 不做大规模训练；Smoke 预算每域 1 campaign、2 paired seeds。
- quant 的 synthetic data 在 preflight/record 中显式标记，不能当作真实市场结论。
