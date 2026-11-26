# Multi-Agent Research Runtime

[中文](#中文说明) | [English](#english)

---

<a name="中文说明"></a>

## 中文说明

### 这是什么

一个**让 LLM 自主做实验**的研究运行时。核心闭环：

> 人给研究想法 → LLM 自己规划并设定实验参数 → SciNet++ 引擎真实训练/评估 →
> LLM 读取实测数据复盘 → 自查扫描发现关键变量 → 改进方案 → 再实验 →
> LLM 自己判断达标完成或继续 → 迭代直到达标或预算耗尽

两层交付物：
1. **通用运行时**（`research_runtime/`）：多智能体 LangGraph 编排（规划—执行—再规划），面向长周期研究任务。
2. **ARR-07 自主科研循环**：LLM 直接操控真实 SciNet++ 世界模型引擎（train / evaluate / analyze / symbolic），带冻结判据、防伪造护栏、可审计证据链。

### 实测能力（有真实产物，不是演示）

- LLM 自主扫描发现 `train.lr=0.01` 是关键变量（MSE −51.9%、cosine 0.544→0.928），并据此达标后自行判定 finish。
- 配对 3 seed 验证：95% CI [0.524, 1.045]，改善显著。
- LLM 读取 physics probe 的概念发现指标（γ/ω 的 R²）做机制层反思。
- 20 个单元测试通过。

诚实边界：目前验证在 **64 样本 / 1 epoch 的 pendulum smoke 数据集**上；PySR 公式发现依赖 Julia 环境（未配置时会明确标 BLOCKED，绝不伪造公式）；概念 γ/ω 是数据集预标注的真值，agent 尚未"自主发现新概念"。

### 安装

环境要求：
- **Python ≥ 3.10**（主 agent 运行时）
- **一个装有 torch/scipy/PyYAML 的 Python 环境**（SciNet++ 引擎用；CPU 即可，非必须 GPU）

```bash
# 1. 进入项目目录
cd multi-agent

# 2. 创建主环境并安装依赖
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 3. 安装 SciNet++ 引擎依赖（引擎源码已随仓库附带，位于 vendor/scinet/）
pip install -r vendor/scinet/requirements.txt   # 装到引擎用的那个 Python 环境
```

SciNet++ 引擎源码和冻结 smoke 数据集已 vendored 在 `vendor/scinet/`（约 2MB），无需单独下载 SciNet++ 项目。运行时的 Python 解释器无法随 git 分发，需自行准备环境。

### 配置

在项目根目录创建 `.env`（**不要提交**）：

```ini
# LLM 网关（ARR-07 实测用本地 vLLM；换成在线模型则改 base_url/key/model）
LLM_BASE_URL=http://192.168.50.46:8888/v1
LLM_API_KEY=<your-key>
LLM_MODEL=qwen3.7-flash
LLM_VERIFY_SSL=false

# SciNet++ 引擎 Python（装了 torch 的那个解释器，改成你的实际路径）
SCINET_PYTHON=/path/to/python-with-torch
# SCINET_DIR 默认指向仓库内 vendor/scinet，无需设置
```

### 使用：跑一次自主科研循环

```bash
PYTHONPATH=. python -u -c "
from pathlib import Path
from research_runtime.arr07.autonomous_scinet_loop import run_autonomous
r = run_autonomous(
    task_id='my_study',
    root=Path('experiments/campaigns/MY-RUN'),
    seed=42, max_iterations=3, target=0.10)
print(r.to_dict())
"
```

**可选能力开关**（在 campaign 目录放标记文件）：
- `enable_agent_screening`：让 LLM 先自主规划单变量扫描、自己发现关键参数（无需人工喂答案）
- `enable_concept_discovery`：运行 `analyze.py`，让 LLM 读取物理概念 probe 的 R²
- `enable_formula_discovery`：运行 `symbolic.py`（PySR 公式发现）；Julia 未配置时自动标记 BLOCKED，不影响循环

**每个 campaign 的外部配置**（写在 campaign 目录，代码不写死）：
- `contract.json`：冻结的多目标阈值（例如 MSE 至少降 10%、cosine 不回退）
- `dataset_scale.json`：训练样本数（用于结构合理性检查）
- `prior_evidence.json`：可选，外部已验证的结论（如不做自查而直接喂证据）

### 验证

```bash
PYTHONPATH=. python -m pytest research_runtime/generic/tests -q
```

预期：28 passed。

运行时产物在 `experiments/campaigns/<RUN>/`：
- `raw_data/autonomy/`：LLM 每次提案/判断原文、search_memory.json、result.json
- `raw_data/scinet/`：真实 train/evaluate 记录（config hash / run_id / 指标）
- `raw_data/analyze/`、`raw_data/symbolic/`：概念发现与公式发现产物
- `final/`：campaign 报告

### 关键模块

| 路径 | 作用 |
|---|---|
| `research_runtime/arr07/autonomous_scinet_loop.py` | 自主循环主入口 |
| `research_runtime/arr07/scinet_backend.py` | SciNet++ 真实引擎调用（train/evaluate/analyze/symbolic） |
| `research_runtime/generic/llm_autonomy.py` | LLM 决策权：提案、自查规划、判定 finish |
| `research_runtime/generic/research_contract.py` | 冻结判据、结构检查、搜索记忆、配对 seed 统计 |
| `research_runtime/generic/identifiability.py` | 实验可分辨性审计 |
| `research_runtime/generic/experiment_design.py` | 区分性实验设计 |

---

<a name="english"></a>

## English

### What is this

A research runtime that lets an LLM **run real experiments autonomously**. Core loop:

> User gives a research idea → LLM plans and sets experiment parameters →
> SciNet++ engine trains/evaluates for real → LLM reviews measured data →
> self-planned single-variable screening finds the key variable → agent improves
> the plan → experiments again → LLM decides finish or continue → repeat until
> the frozen goal is met or the budget is exhausted.

Two deliverables:
1. **Generic runtime** (`research_runtime/`): LangGraph multi-agent orchestration (plan–execute–replan) for long-horizon research tasks.
2. **ARR-07 autonomous loop**: LLM directly drives a real SciNet++ world-model engine (train / evaluate / analyze / symbolic), with frozen criteria, anti-fabrication guardrails, and an auditable evidence trail.

### Verified capabilities (with real artifacts, not a demo)

- The LLM self-discovered `train.lr=0.01` as the key variable via its own screening sweep (MSE −51.9%, cosine 0.544→0.928) and judged finish itself after reaching the goal.
- Paired 3-seed validation: 95% CI [0.524, 1.045], significant improvement.
- The LLM reflects on physics-probe concept-discovery metrics (R² of γ/ω) at the mechanism level.
- 20 unit tests pass.

Honest boundaries: validated on a **64-sample / 1-epoch pendulum smoke dataset**; PySR formula discovery needs a Julia environment (marked BLOCKED, never fabricated, when absent); concepts γ/ω are dataset-prelabeled ground truth — the agent has NOT discovered new concepts on its own.

### Installation

Requirements:
- **Python ≥ 3.10** for the main agent runtime
- **One Python environment with torch/scipy/PyYAML** for the SciNet++ engine (CPU is fine)

```bash
cd multi-agent
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -r vendor/scinet/requirements.txt   # into the engine Python environment
```

SciNet++ source and the frozen smoke dataset are vendored at `vendor/scinet/` (~2MB); no separate download needed. The Python interpreter itself cannot be shipped via git — you must prepare the environment.

### Configuration

Create `.env` in the project root (**never commit it**):

```ini
LLM_BASE_URL=http://192.168.50.46:8888/v1
LLM_API_KEY=<your-key>
LLM_MODEL=qwen3.7-flash
LLM_VERIFY_SSL=false
SCINET_PYTHON=/path/to/python-with-torch
```

### Usage: run one autonomous research loop

```bash
PYTHONPATH=. python -u -c "
from pathlib import Path
from research_runtime.arr07.autonomous_scinet_loop import run_autonomous
r = run_autonomous(task_id='my_study',
    root=Path('experiments/campaigns/MY-RUN'),
    seed=42, max_iterations=3, target=0.10)
print(r.to_dict())
"
```

Optional feature flags (drop a marker file in the campaign dir):
- `enable_agent_screening` — LLM plans its own single-variable screening first
- `enable_concept_discovery` — run `analyze.py`; LLM reasons on concept probe R²
- `enable_formula_discovery` — run `symbolic.py` (PySR); marks BLOCKED if Julia is absent

Per-campaign external config (in the campaign dir, not hardcoded):
- `contract.json` — frozen multi-objective thresholds
- `dataset_scale.json` — training sample count for structure sanity checks
- `prior_evidence.json` — optional externally verified findings

### Verify

```bash
PYTHONPATH=. python -m pytest research_runtime/generic/tests -q
```

Expected: 20 passed.

Artifacts: `experiments/campaigns/<RUN>/` holds `raw_data/autonomy/` (LLM decisions), `raw_data/scinet/` (real engine runs), `raw_data/analyze/` + `raw_data/symbolic/` (concept/formula discovery), and `final/` (campaign reports).

### Key modules

| Path | Role |
|---|---|
| `research_runtime/arr07/autonomous_scinet_loop.py` | Main autonomous loop |
| `research_runtime/arr07/scinet_backend.py` | Real SciNet++ engine calls (train/evaluate/analyze/symbolic) |
| `research_runtime/generic/llm_autonomy.py` | LLM decision rights: propose, plan screening, judge finish |
| `research_runtime/generic/research_contract.py` | Frozen criteria, sanity checks, search memory, paired-seed stats |
| `research_runtime/generic/identifiability.py` | Experiment identifiability audit |
| `research_runtime/generic/experiment_design.py` | Discriminative experiment design |
