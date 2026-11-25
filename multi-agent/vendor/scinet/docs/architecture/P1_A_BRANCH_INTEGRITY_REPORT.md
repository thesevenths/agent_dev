# P1-A — Git 分支完整性报告 (Branch Integrity Report)

> ## ⚠️ 历史审计快照声明（P1-A' review fix P5）
>
> **本文件 §1–§6 是 `a5b8396` 时的历史审计快照，其内容与数字（HEAD `a5b8396`、
> 相对 main 9 个提交、18 个文件变更、4853 insertions）均按当时实测记录，
> 现予以保留、不修改、不覆盖**，作为该时间点的不可变证据。
>
> 该快照之后分支又新增了两个审查提交（`eb1dc51` analysis、`3143772` docs）。
> **当前实际状态见本报告末尾的 §7「当前 HEAD 审计」**：
> HEAD `3143772...`、`0 11`（**11** ahead / **0** behind）、**36** files
> （34 A + 2 M + 0 D）。请勿再用 §1–§6 的旧数字描述当前分支。

**结论先行（历史快照）：✅ 在 `a5b8396` 时，`feature/p2-03a-action-jepa` 完整包含 P2-02 的全部代码与证据；与 `origin/main` 无冲突、无重复实现、无旧代码重新引入。**

- 核验时间：2026-11-11（**快照基准 HEAD `a5b8396`**）
- 快照时分支 / HEAD：`feature/p2-03a-action-jepa` @ `a5b8396`（=`origin/feature/p2-03a-action-jepa`）
- 参照：`origin/main` @ `e4eccc0`、此前报告 P2-02 HEAD `e4eccc0`
- 仓库根：`E:/LLM`（远程 `https://github.com/thesevenths/LLM.git`），`SciNet++/` 为子目录
- **未执行** `merge` / `rebase` / `reset` / `force-push` 中的任何一项（仅 `git fetch` + 只读查询）

---

## 1. 对 ChatGPT 四个问题的逐条回答

| # | 问题 | 结论 | 证据 |
|---|------|------|------|
| 1 | 当前 feature branch 是否包含 P2-02？ | **是，完整包含** | `git merge-base origin/main HEAD` = `e4eccc0`；`git merge-base --is-ancestor origin/main HEAD` 返回 0（成功），即 P2-02 的 HEAD 本身就是当前分支的**直接祖先** |
| 2 | 若不包含，P2-02 代码与结果在哪里？ | **不适用（已包含）**。P2-02 同时存在于 `origin/main`、`origin/feature/p2-02-raw-state-dynamics`、`origin/HEAD` | `git branch -a --contains e4eccc0` 列出 6 个 ref；`git ls-tree -r HEAD` 中 `SciNet++/outputs/dynamics/pusher_v1/` 下 **78 个证据文件**完好 |
| 3 | 六个提交会与 main 上 A0/A1/A2/P2-02 冲突吗？ | **不会。结构上不可能冲突** | `git rev-list --left-right --count origin/main...HEAD` = `0  9` —— main 侧 0 个新提交，是纯线性快进链；相对 main 的 diff 中 **0 个删除(D)**，仅 2 个修改(M) + 16 个新增(A) |
| 4 | 有重复实现或旧版本代码被重新引入吗？ | **没有** | ① 2 个 M 均为纯增量的校验加固（见 §4），未删除任何既有生产逻辑；② 新模块未重定义 `DynamicsBatch` / `PendulumGenerator` / `RobotTransition` / `Trajectory` / `dynamics_batch_*`（grep 结果 NONE）；③ 6 个 P2-02 关键文件在 main 与 HEAD 的 **blob hash 完全相同**（见 §4 表） |

> **对"不能仅凭两个分支分别报告完成就认为包含"这一谨慎态度的回应**：该怀疑是合理的，但已用命令证伪 —— `origin/main` 自 P2-02 后**没有前进任何提交**，因此本分支不是"平行开发的另一条线"，而是从 P2-02 线性长出的快进分支。

---

## 2. 分支关系实际输出（逐条，未加工）

```
$ git branch -avv
  feature/a1-dynamics-contract          7da4bbe [origin/...] fix(dynamics): make conditions=None ...
  feature/a2-conditional-predictor     5f4b278 [origin/...] feat(models): add independent conditional predictor
  feature/p2-02-raw-state-dynamics     e4eccc0 [origin/...] docs(dynamics): add P2-02 final report
* feature/p2-03a-action-jepa           a5b8396 [origin/...] data(p1-a): add 20-episode controlled-pendulum pilot + report
  main                                 e4eccc0 [origin/main] docs(dynamics): add P2-02 final report
  remotes/origin/HEAD                  -> origin/main
  remotes/origin/feature/p2-03a-action-jepa  a5b8396 ...
  remotes/origin/main                  e4eccc0 ...

$ git merge-base origin/main HEAD
e4eccc0b1da2c8a365706f97ded95f3723e0f5e9      # == e4eccc0 (P2-02 final report)

$ git rev-list --left-right --count origin/main...HEAD
0   9                                          # main 0 ahead, HEAD 9 ahead

$ git log --oneline --graph --decorate --all -40   (节选，单链无分叉)
* a5b8396 (HEAD -> feature/p2-03a-action-jepa, origin/feature/p2-03a-action-jepa) data(p1-a) ...
* f9ca24f test(p1-a): add controlled-pendulum simulator unit tests
* 5a63cdc feat(p1-a): add pilot collection + audit scripts
* 68611e0 feat(p1-a): add torque-controlled pendulum simulator + config
* 0e305d6 docs(p2-03a): clarify ablation capacity wording
* d28e007 fix(core): reject boolean normalization arrays
* 0c06b78 fix(core): harden dynamics validation
* d3382bb docs(p2-03a): clarify experimental protocol
* 144a300 docs(architecture): P2-03A Phase 0 architecture audit (read-only)
* e4eccc0 (origin/main, origin/feature/p2-02-raw-state-dynamics, origin/HEAD, main) docs(dynamics): add P2-02 final report
* fe55f80 experiment(dynamics): P2-02 five-seed baseline results and evidence
* bafb828 feat(dynamics): add P2-02 training and evaluation pipelines
* 08bb09a feat(dynamics): add train-only normalisation, metrics and open-loop evaluation
* 9590011 feat(models): add raw-state residual dynamics baselines
* 5716e21 feat(dynamics): add P2-02 Phase 0 Pusher-v1 dataset audit (STOP GATE)
* 5f4b278 (origin/feature/a2-conditional-predictor) feat(models): add independent conditional predictor
* 7da4bbe (origin/feature/a1-dynamics-contract) fix(dynamics): make conditions=None ...
* ac60b57 feat(dynamics): introduce domain-independent dynamics contract
...

$ git branch -a --contains e4eccc0
  feature/p2-02-raw-state-dynamics
* feature/p2-03a-action-jepa
  main
  remotes/origin/HEAD -> origin/main
  remotes/origin/feature/p2-02-raw-state-dynamics
  remotes/origin/feature/p2-03a-action-jepa
  remotes/origin/main
```

`git log --oneline --graph --all -40` 显示 `a5b8396 → e4eccc0 → ... → 5f4b278 → 7da4bbe → ac60b57` 为**单一线性链**，无 merge 分叉、无旁支重新汇入。

---

## 3. 提交数量口径澄清（重要，避免"六个 vs 九个"的误解）

ChatGPT 的任务书写道"本次六个提交"。实测存在**两个不同口径**，两者都正确，但不可混用：

| 口径 | 命令 | 提交数 | 内容 |
|------|------|--------|------|
| 相对 `origin/main`（P2-02 验收基准） | `git log --oneline origin/main..HEAD` | **9** | 3 个已批准的 Phase 0 / P0-fix 提交 + 6 个本轮提交 |
| 相对上一轮验收点 `0c06b78`（ChatGPT 上轮 STOP 时的 HEAD） | `git log --oneline 0c06b78..HEAD` | **6** | 本轮新增：2 个 P0-fix.1 + 4 个 P1-A |

**9 = 3 + 6**，明细如下：

| 分组 | 提交 | 说明 |
|------|------|------|
| 此前已批准（3） | `144a300` Phase 0 架构审计、`d3382bb` 实验协议澄清、`0c06b78` 校验加固 | 上一轮 STOP 时已存在 |
| 本轮 P0-fix.1（2） | `d28e007` 拒绝布尔归一化数组、`0e305d6` 澄清消融容量措辞 | 本次条件批准项 |
| 本轮 P1-A（4） | `68611e0` 仿真器+配置、`5a63cdc` 采集+审计脚本、`f9ca24f` 单测、`a5b8396` pilot 数据+报告 | 本次新增交付 |

`git diff --stat origin/main...HEAD` = **18 files changed, 4853 insertions(+), 20 deletions(-)**；其中 `20 deletions` 全部来自 2 个 M 文件里被替换的旧校验代码（见 §4），**没有任何文件被删除**（`name-status` 中 D 计数 = 0）。

### 六个提交的文件级变更（`git log --stat 0c06b78..HEAD`）

| 提交 | 变更 |
|------|------|
| `a5b8396` | 4 文件 / +1402：pilot `npz`(104966 B) + `manifest.json` + `audit_report.json` + `P1_A_PILOT_REPORT.md` |
| `f9ca24f` | 1 文件 / +146：`tests/dynamics/test_controlled_pendulum.py` |
| `5a63cdc` | 2 文件 / +647：`collect_controlled_pendulum.py`(289) + `audit_controlled_pendulum.py`(358) |
| `68611e0` | 2 文件 / +474：`controlled_pendulum.py`(376) + `controlled_pendulum_jepa.yaml`(98) |
| `0e305d6` | 1 文件 / +7 -3：`P2_03_ACTION_JEPA.md` §5.2 措辞 |
| `d28e007` | 2 文件 / +59：`normalization.py`(+17) + `test_p0_fix_validation.py`(+42) |

---

## 4. "旧代码重新引入 / 重复实现" 的正面验证

### 4.1 相对 main 只有 2 个文件被修改，且均为纯增量校验加固

```
$ git diff --name-status origin/main...HEAD
A  (16 files: config / simulator / scripts / tests / docs / pilot dataset)
M  SciNet++/dynamics/normalization.py
M  SciNet++/models/state_dynamics.py
D  (无)
```

两个 M 的 diff 内容（`git diff origin/main...HEAD -- <2 files>`）已逐行核对：

- **`dynamics/normalization.py`**：新增 `import math/numbers/numpy`、`Mapping`、`_REQUIRED_FIELDS`、`_ALLOWED_SOURCE_SPLITS`、`_validate_vector()`；`from_dict` 由"无校验直接 `torch.tensor(...)`"改为"严格 schema 校验后构造"。**`to_dict` / `save` / `load` / `fit_normalization`、既有字段与语义未变**；已落盘 P2-02 `normalization.json` 经回归测试仍可 `load()`。
- **`models/state_dynamics.py`**：`_check_hidden_dims` 由 `tuple(int(h) for h in ...)` 改为 `numbers.Integral` + 显式排除 `bool` 的逐元素校验。仅增加 `import numbers`（stdlib），其余模型前向逻辑零改动。

两处均属上一轮 **P0-fix 已批准**范围 + 本轮 **P0-fix.1** 布尔数组拦截；不存在把旧版本 `from_dict` / 宽松 `int(h)` 重新引入的情况。

### 4.2 P2-02 关键文件在 main 与 HEAD 字节一致（blob hash 对比）

| 文件（`SciNet++/` 下） | main blob | HEAD blob | 一致 |
|---|---|---|---|
| `docs/architecture/P2_02_FINAL_REPORT.md` | `2e3828806c21` | `2e3828806c21` | ✅ SAME |
| `dynamics/contracts.py` | `28078ed0e599` | `28078ed0e599` | ✅ SAME |
| `models/conditional_predictor.py` | `1b3a15f2fc91` | `1b3a15f2fc91` | ✅ SAME |
| `scripts/train_dynamics.py` | `e611d6bae8be` | `e611d6bae8be` | ✅ SAME |
| `outputs/dynamics/pusher_v1/normalization.json` | `d524135a0997` | `d524135a0997` | ✅ SAME |
| `dynamics/normalization.py` | `d2f714ca7ec0` | `68d47fa38159` | 预期 DIFF（P0-fix） |
| `models/state_dynamics.py` | `ee8882e0358a` | `730f79ddffdd` | 预期 DIFF（P0-fix） |

`git ls-tree -r HEAD --name-only | grep -c "outputs/dynamics/pusher_v1/"` = **78**，与上一轮日志记录的 P2-02 证据文件数一致，零缺失。

### 4.3 新增模块没有重复实现既有抽象

```
$ grep -nE "class (DynamicsBatch|PendulumGenerator|RobotTransition|Trajectory)|def dynamics_batch" \
       data/controlled_pendulum.py scripts/collect_controlled_pendulum.py scripts/audit_controlled_pendulum.py
NONE (no duplicated contract/generator classes)
```

- `data/controlled_pendulum.py` 的 import 仅为 `hashlib` / `math` / `dataclasses` / `typing` / `numpy`；**不 import `robotics`，也不 import `dynamics.contracts`**（仿真器保持领域无关）。
- `DynamicsBatch` 契约只在 `audit_controlled_pendulum.py` 中**被消费/校验**（`DynamicsBatch(torch.as_tensor(..., dtype=float64), ...)`），未另造一套窗口/批处理逻辑；数据以 `observations/conditions/valid_mask/episode_ids` 四元组存储，与 `dynamics/contracts.py` 的契约字段名一致。
- `data/pendulum.py`（阻尼振荡器）与 `robotics/data/*` 全部为 `SAME`/未触碰。

---

## 5. 边界与诚实声明

1. **本报告只证明 Git 结构完整性**，不证明 P1-A 数据适合训练 World Model（那属于 `P1_A_SCIENTIFIC_REVIEW.md` 与 `P1_A_TRAJECTORY_DISTRIBUTION.md`）。
2. `origin/main` 当前**仍停在 `e4eccc0`**；这份"快进无冲突"的结论只在 main 未前进的前提下成立。若后续有他人向 main 推送新提交，合并前必须重新执行 `git fetch` + `git rev-list --left-right --count origin/main...HEAD`。
3. 未验证远端 PR 状态（本任务不涉及）；未创建 PR，未合并 main。
4. 全部数字来自本次实际命令输出；未使用记忆值或推断值。

---

## 6. 复现命令

```bash
cd E:/LLM/SciNet++
git fetch origin
git branch -avv
git log --oneline --graph --decorate --all -40
git merge-base origin/main HEAD
git merge-base --is-ancestor origin/main HEAD; echo $?
git branch -a --contains e4eccc0
git log --oneline origin/main..HEAD
git log --oneline 0c06b78..HEAD
git rev-list --left-right --count origin/main...HEAD
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
git log --stat 0c06b78..HEAD
git ls-tree -r HEAD --name-only | grep -c "outputs/dynamics/pusher_v1/"
```

---

## 7. 当前 HEAD 审计（P1-A' review fix P5，追加章节，不覆盖 §1–§6）

以下为**追加审计**，基准为 2026-11-11 16:01 的当前 HEAD。全部数字来自
`scripts/capture_git_integrity.py` 的实际 git 命令输出，机器可读快照保存在
`datasets/controlled_pendulum_pilot/analysis/git_integrity.json`。

### 7.1 当前状态实测

| 项 | 实测值 |
|----|--------|
| 分支 | `feature/p2-03a-action-jepa` |
| HEAD full SHA | `3143772cf8ede786254a01ee6d0013a4d4c243d1` |
| `origin/main` full SHA | `e4eccc0b1da2c8a365706f97ded95f3723e0f5e9` |
| merge base | `e4eccc0b1da2c8a365706f97ded95f3723e0f5e9`（= `origin/main`，即 main 是 HEAD 的祖先） |
| `git rev-list --left-right --count origin/main...HEAD` | **0** behind / **11** ahead |
| 相对 main 文件变更 | **36** files（34 A + 2 M + 0 D） |
| P2-02 证据文件数（`git ls-tree` 实读） | **78** |
| 工作树 | clean（采集时有本轮未提交的新增文件，故快照记 `false`；提交后复核） |

### 7.2 `a5b8396` 之后新增的两个审查提交

```
$ git log --oneline a5b8396..HEAD
3143772 docs(p1-a): add four acceptance-review reports (STOP, no JEPA training)
eb1dc51 analysis(p1-a): add independent review scripts + machine-readable evidence
```

这两个提交**只新增**分析脚本、机器可读证据和四份审查报告，
未修改任何生产代码、simulator、配置、pilot 数据或已有历史证据。

### 7.3 与 ChatGPT 独立 GitHub 核验的一致性

| 检查项 | ChatGPT GitHub 实测 | 本次本地实测 | 一致 |
|--------|---------------------|--------------|------|
| Base | `main` | `origin/main` | ✅ |
| Head | `feature/p2-03a-action-jepa` | 同 | ✅ |
| Merge base | `e4eccc0` | `e4eccc0b1da...` | ✅ |
| Ahead | 11 commits | **11** | ✅ |
| Behind | 0 | **0** | ✅ |
| 文件变更 | 34 added + 2 modified + 0 deleted | **36** = 34 + 2 + 0 | ✅ |
| 删除文件 | 0 | 0 | ✅ |

### 7.4 结论（当前 HEAD）

- ✅ 当前 feature 分支仍**完整包含 P2-02 基线**（`origin/main` 是直接祖先），可快进、无分叉。
- ✅ 78 个 P2-02 证据文件与 6 个关键 blob hash 未被覆盖或改动。
- ✅ 两个新审查提交为净新增，不含重复实现或旧代码回退。
- ✅ **未合并 main、未开 PR、未 rebase/reset/force-push**。
- ⚠️ 数字口径：§1–§6 中的「9 个提交 / 18 文件」是 `a5b8396` 快照值；
  当前值应为「**11** 个提交 / **36** 文件」。后续引用须注明基准。

### 7.5 复现

```bash
python scripts/capture_git_integrity.py
# -> datasets/controlled_pendulum_pilot/analysis/git_integrity.json
```
