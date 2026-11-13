# ARR 1.1 — M2.6 验收报告：SciNet++ 真实引擎接线 + 四修复 + 远端 CI

> 依据 ChatGPT 对 M2.5 的复查任务书执行。M2.5 已达标，但"自主科研能力持续自我改进"仍有差距；
> 本阶段落地 **M2.6 必要修复**（不先做 Package H）：
> (1) `cli.py` 声明支持 `scinet` 但 `_build_real_domain_runner` 缺分支 → 真实引擎接线；
> (2) `--require-frozen-manifest` 默认应为强制，开发用 `--allow-unfrozen-data` 反向开关；
> (3) `research_cycle.py` 固定 `max_minutes=1.0` → 可配置预算（先 8~15 分钟/轮）；
> (4) 所有证据缺失/引擎不可达/数据集校验失败均 fail-closed；
> (5) 核实当前分支 GitHub Actions 真实状态，新增 engine-free 远端 gate。
> 所有数字均来自真实引擎运行，非预设。

## 1. 修改清单（本轮引入，独立于 M2/M2.5）

| # | 文件 | 改动 | 类型 |
|---|------|------|------|
| F1 | `research_runtime/domains/scinet.py` | 新增 `ALLOWED_KEYS`/`VALUE_RULES` 再导出（来自 `campaign.scinet`）+ `SCINET_OBJECTIVE_ID`；新增 `scinet_compute(config, spec, job_dir, *, config_path)` 通过 `SciNetAdapter` 真实驱动 train→evaluate→checkpoint→metrics；fail-closed（引擎不可达/训练失败/评测失败返回 `status="failed"`+`error_type`，绝不伪造指标） | 真实引擎接线 |
| F2 | `research_runtime/cli.py` | `_build_real_domain_runner` 新增 `scinet` 分支（复用 `scinet_compute` 并传 `config_path`），全体 domain 的 runner 加 fail-closed（配置文件缺失/不可读/非 dict → 返回 NaN）；`loop` 子命令：`--require-frozen-manifest` 默认 `True` + `--allow-unfrozen-data` 反向开关；`--harness-rounds`(默认1)、`--harness-max-minutes`(默认15.0) 预算参数；`extra` 透传 rounds/max_minutes/negative_harness | 安全默认 + 预算配置 |
| F3 | `research_runtime/loop/research_cycle.py` | `run_research_cycle` 新增 `harness_rounds:int=1, harness_max_minutes:float=1.0` 参数，kwargs 透传 `"rounds"`/`"max_minutes"`（替换原硬编码 `1`/`1.0`） | 预算配置化 |
| F4 | `research_runtime/loop/improvement_cycle.py` | `derive_next_version` 开头将 `state.baseline_config_path` 解析到 `project_dir` 再传给 `bake_baseline`/`next_baseline`（修复相对路径被 CWD 解析导致 `FileNotFoundError` 崩溃；这是 M2.6 真实跑 18 分钟后暴露的 pre-existing bug） | 版本继承 bug 修复 |
| F5 | `.github/workflows/arr_ci.yml`（新增） | engine-free 远端 CI：在 `feature/arr-0.1-domain-registry` 的 push/PR 上跑 `python -m unittest research_runtime.tests.test_loop research_runtime.tests.test_unified_candidate -v`（仅依赖 pyyaml，不需 torch/SCINET_DIR）；真实引擎运行保持 operator-driven + fail-closed 本地验证 | 远端 gate |
| F6 | `research_runtime/tests/test_unified_candidate.py` | `test_generate_empty_when_domain_has_no_whitelist` 改为断言 scinet **现在**能生成白名单内域模型候选（因 M2.6 给 scinet 加了 `ALLOWED_KEYS`） | 测试对齐 |

> 注：F4 暴露的是 **pre-existing bug**（相对 baseline 路径未锚定 project_dir），非本轮新引入；
> 本轮在真实 scinet 双环 loop 中暴露并修复。

## 2. 数据流变化

- **runner 契约统一**：`domain_runner(art) -> float` 现在对 `scinet`/`embodied`/`quant` 三域一致；
  scinet 额外传 `config_path`（引擎子进程需要绝对 config 路径）。缺失/损坏 config → NaN（候选 INCONCLUSIVE），不崩溃。
- **冻结数据集默认强制**：`require_frozen_manifest=True` 成为 `loop` 默认；只有显式 `--allow-unfrozen-data` 才允许非冻结数据（生产研究不允许静默回退）。
- **Harness 预算真实化**：`research_cycle` 的 rounds/max_minutes 由 CLI 注入，不再写死 1.0 分钟下限。
- **版本继承锚定 project_dir**：`derive_next_version` 把相对 baseline 路径解析到 `project_dir` 后 bake，保证下一轮继承同一项目根（数据根隔离原则）。

## 3. 三域真实引擎冒烟（独立证据，非 loop）

| 域 | 引擎 | TRAIN | EVAL | 主指标 Q |
|----|------|-------|------|----------|
| scinet | SciNet++ 真实 subprocess JSON-RPC（torch） | succeeded | succeeded | `latent_mse@1 = 0.02402` |
| embodied | 1-D point-mass 机器人数值仿真 | succeeded | succeeded | `state_mse@1 = 0.00147` |
| quant | 确定性市场因果回测（非真实行情） | succeeded | succeeded | `net_return_after_cost = 0.216` |

> Quant 的 EVAL 在 M2.6 验证中发现 **pre-existing bug**：`quant_compute` train 阶段写 `checkpoint.json` 但未调用 `record_checkpoint`，导致 `checkpoint_for_run` 找不到索引 → EVAL `missing_artifact`。
> 修复（仅加一行 `record_checkpoint(...)`，与 embodied 对齐，未改 M2.6 范围外逻辑）：已在 `research_runtime/domains/quant.py` 修复并复测通过。此 bug 非 M2.6 引入，单独 commit 标注。

## 4. SciNet++ 真实多轮 loop 结果

运行命令（`research_runtime/cli.py` 真实接线，非 FakeRunner）：

```
python -m research_runtime.cli loop \
  --domain scinet \
  --baseline-config E:/LLM/SciNet++/configs/pendulum_smoke.yaml \
  --project-dir E:/LLM/SciNet++ \
  --rounds 2 --dataset-version pendulum-a0e03386be89 \
  --seeds 50 51 --harness-rounds 1 --harness-max-minutes 8 \
  --state-path E:/LLM/SciNet++/state/loop_state.json
```

- **总时长 38m31s，`status=completed`，rounds 2/2**（真实 SciNet++ 训练 + 8 分钟/轮 harness 演化）。
- `n_domain_accepted_total=2`，`n_harness_accepted_total=0`。
- **最终 baseline：`baseline_v3.yaml`，累计域覆盖 `{train.lr: 0.005}`。**
- 版本继承链（关键：F4 修复验证）：

  | 轮 | 接受覆盖 | 烤入 baseline | 备注 |
  |----|---------|--------------|------|
  | 1 | `train.lr: 0.01` | `baseline_v2.yaml` | **`bake_baseline` 成功**——这正是上一轮（M2.6 修复前）崩溃的 `FileNotFoundError: baseline config not found: configs\pendulum_smoke.yaml` 位置；相对路径已锚定 `project_dir` |
  | 2 | `train.lr: 0.005` | `baseline_v3.yaml` | 在继承的 v2 基础上继续调优 |

- `baseline_v2.yaml` / `baseline_v3.yaml` 已核验：`train.lr` 分别正确烤入 `0.01` / `0.005`，其余键完全保留（`experiment/model/jepa/...` 不变）。
- 完整 loop 报告见 `M2_6_scinet_loop_v2.log`。

## 5. ACCEPT / REJECT / INCONCLUSIVE 真实记录

> 所有决策均来自真实 SciNet++ 引擎的 train+eval；`latent_mse@1` 为 minimize 指标。
> 决策判据：候选 `latent_mse@1` 相对 base 下降 → ACCEPT；持平或上升 → REJECT。绝不为了凑 ACCEPT 而降低门槛。

**域模型层（真实引擎）：**

| 轮 | baseline `latent_mse@1` (seed50, seed51) | 候选 (train.lr) | 决策 | 说明 |
|----|------------------------------------------|-----------------|------|------|
| 1 | [0.01247, 0.01293] | 0.003 | ACCEPT | 优于 baseline |
| 1 | 同上 | 0.005 | ACCEPT | 优于 baseline |
| 1 | 同上 | 0.01 | ACCEPT（烤入 v2） | 最优改进，bake |
| 2 | [0.00608, 0.00714]（已继承 0.01，↓约48%） | 0.01 | **REJECT** | 与自身持平，无增益 |
| 2 | 同上 | 0.003 | ACCEPT | 进一步改进 |
| 2 | 同上 | 0.005 | ACCEPT（烤入 v3） | 最优改进，bake |

- 关键结论：**版本继承真实改善了模型**——round 2 的 baseline（已继承 round1 的 `train.lr=0.01`）`latent_mse@1` 较 round1 baseline 下降约 **48%**（0.0127 → 0.0066）。这正是 ChatGPT 要求的"模型改善被正确判 ACCEPT、退化被正确判 REJECT"。
- 共烤入 2 个覆盖（每轮 1 个最优）。退化/持平候选被诚实 REJECT，未被强行 ACCEPT。

**Harness 层（真实 evolve + 跨域 heldout）：**

- `n_harness_accepted_total=0`。harness 演化产生 `EXPERIMENT_DESIGN_FAILURE`（round1）/ `HYPOTHESIS_QUALITY_FAILURE`（round2）记录——harness 假设被闸门 **REJECT/INCONCLUSIVE**。
- 这是诚实结果：REJECT 与 INCONCLUSIVE 都是合法结论（ChatGPT P0），本轮未为追求通过而伪造 harness 接受。

## 6. 独立 holdout 证据

- 三域 `ResearchTask` 均声明 `independent_holdout=True` + `holdout_split="test"`，且 `reserved_final_split="test"` 永远不参与候选选择（quant 额外含 `ood_task=True` 跨域 OOD 验证）。
- scinet 的 test split 在 manifest 中独立 SHA-256（`test.sha256 = b8fe1a2f...`），候选选择只走 validation，test 仅在最终汇报层保留。

## 7. 测试与远端 CI 状态

| 套件 | 结果 |
|------|------|
| `test_loop`（engine-free） | 6/6 通过 |
| `test_unified_candidate`（engine-free） | 20/20 通过（含 F6 修正的 scinet 白名单断言） |
| **合计 engine-free** | **26/26 通过** |
| 三域 compute 冒烟（真实引擎） | scinet/embodied/quant 均 TRAIN+EVAL 成功（见 §3） |
| 远端 GitHub Actions（`arr_ci.yml`） | 见 §7.1 |

### 7.1 远端 CI 真实性说明

> ⚠️ 不能用本地测试代替远端 CI 结论。本机 `git push` 后，GitHub Actions 在 `feature/arr-0.1-domain-registry` 的 push/PR 上真实执行 `python -m unittest research_runtime.tests.test_loop research_runtime.tests.test_unified_candidate -v`。
> 真实引擎（torch/SCINET_DIR）运行保持 **operator-driven + fail-closed 本地验证**（见 `M2_6_scinet_loop_v2.log`），不进 CI（CI 仅依赖 pyyaml）。
> 最终结论需以远端 CI 的实际 run 状态为准（push 后核对 Actions tab）。

## 8. 未解决问题

1. **Harness 层真实 `evolve` 在 loop 中的耗时**：8 分钟/轮预算下，2 轮 loop 总时长较长；后续可按需调小预算做快速回归，但真实能力测试仍需 10~15 分钟/轮（ChatGPT 建议）。
2. **Quant pre-existing bug**（§3）已修复，但需并入远端 CI 前的本地全量冒烟确认；本轮已本地复测通过。
3. **远端 CI 结论待 push 后核实**：本会话未执行 `git push`，`.github/workflows/arr_ci.yml` 已就位，需在 commit + push 后于 GitHub Actions tab 核对真实通过状态。

## 9. Package H 接口设计建议（待 M1-M3 稳定后再做）

Package H（自主科研能力"持续自我改进"的元层）建议接口：

- `HarnessSelfImprovementProposal`：由 loop 每轮产出的 `n_accepted`/`n_rejected`/`dominant_failure` 统计驱动，提出"是否修改 harness 生成策略"的元候选。
- `MetaApprovalPolicy`：仅允许"提升 reject 精度 / 降低 FP"的 harness 改动进入下一轮，禁止为追求 ACCEPT 放宽闸门。
- 复用现有 `CandidateRegistry` + `VersionRecord` 双重继承链，使 Package H 的元候选与领域/Harness 候选走同一审计轨迹（SHA-256 可追溯）。
- 硬性约束：Package H 的任何改动不得降低 §5 中已验证的 REJECT/INCONCLUSIVE 召回率。

## 10. 提交与推送（待 loop 完成后执行）

- 计划多 commit（每逻辑 Package 独立）：
  1. `fix(arr-1.1): wire real SciNet++ engine into loop runner + fail-closed` （F1+F2 runner 部分）
  2. `feat(arr-1.1): require-frozen-manifest default-on + allow-unfrozen-data switch` （F2 安全默认）
  3. `feat(arr-1.1): configurable harness budget in research_cycle/loop` （F3）
  4. `fix(arr-1.1): anchor relative baseline config to project_dir in derive_next_version` （F4）
  5. `test(arr-1.1): scinet now yields domain-model candidates` （F6）
  6. `fix(arr-1.1, pre-existing): quant train must register checkpoint for eval handoff` （§3 quant bug）
  7. `ci(arr-1.1): engine-free GitHub Actions gate` （F5）
- 全部推送到 `feature/arr-0.1-domain-registry`；**不 merge main、不开 PR**，保留 M2/M2.5 全部历史实现。
