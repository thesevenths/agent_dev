"""Benchmark 任务集：三层划分与外部分布外集接入。

为什么必须分层
--------------
RRSI 论文所针对的失效模式，正是"演化循环反复观看同一批任务"：
每一轮提案都依赖此前在同一 evolve 集上的测量，于是搜索过程逐渐把
**任务本身**编码进 harness，而不是发现可复用的机制。表现为
evolve 集大涨、held-out 集持平甚至倒退。

因此任务集至少要有三层：

| 层 | 是否参与演化 | 用来回答 |
|---|---|---|
| Evolve | 是 | 这个候选在当前任务上是否更优 |
| ID held-out | 否 | 同分布任务上是否泛化 |
| OOD | 否 | 跨环境 / 跨任务族是否泛化 |

只有 **OOD 同步上升** 才构成"普遍变强"的充分证据；
evolve 集上升只是必要条件。

关于本仓库现有任务集
--------------------
`experiments/tasks/replay.json` 由历史日志反推而来（`tasks init`），
而历史日志同时是诊断层的输入。若不分层，等同于"用作业当考卷"，
测得的提升无法与记忆效应区分。这是当前验证链路最需要修正的一环。

外部基准的可获取性
------------------
RRSI 论文在 8 个基准上评测：Terminal-Bench 2.1、SWE-bench Verified、
Harvey LAB、JobBench、GDPval、APEX-Agents、EngDesign、Frontier-Eng。
其中公开可下载且与本仓库"专业交付物生成"定位最匹配的是
APEX-Agents（HuggingFace `mercor/apex-agents`，CC-BY 4.0，480 任务 /
33 worlds，覆盖投行、管理咨询、法律）。

但接入外部基准需要两件独立工作，本模块只定义接口而不代为实现：
1. **adapter**：把基准任务映射为本仓库的 `query` + `family`；
2. **grader**：把本仓库产物映射回基准的 rubric 判定。

在这两项完成前，外部 OOD 结论不可声称。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

#: 任务必需字段。缺任一字段的任务会在加载期被拒，避免静默产出不可比分数。
REQUIRED_FIELDS = ("task_id", "query")


@dataclass
class TaskSet:
    """三层任务集。"""

    evolve: list[dict] = field(default_factory=list)
    id_heldout: list[dict] = field(default_factory=list)
    ood: list[dict] = field(default_factory=list)

    @property
    def all_ids(self) -> list[str]:
        return [t["task_id"] for t in (*self.evolve, *self.id_heldout, *self.ood)]

    def summary(self) -> dict[str, Any]:
        return {
            "evolve": len(self.evolve),
            "id_heldout": len(self.id_heldout),
            "ood": len(self.ood),
            "overlap": _overlap(self.evolve, self.id_heldout, self.ood),
        }


def normalize(raw: Iterable[dict], source: str = "") -> list[dict]:
    """把任意来源的任务记录规整为统一 schema。

    不同来源的字段名不同（``query`` / ``prompt`` / ``instruction``），
    统一在此收敛，避免下游到处做兼容判断。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for r in raw or ():
        if not isinstance(r, dict):
            continue
        q = (r.get("query") or r.get("prompt") or r.get("instruction") or "").strip()
        if len(q) < 8:
            continue
        tid = str(r.get("task_id") or r.get("id") or "").strip()
        if not tid:
            tid = "T-" + hashlib.sha1(q.encode("utf-8")).hexdigest()[:8]
        if tid in seen:
            continue
        seen.add(tid)
        out.append({
            "task_id": tid,
            "query": q,
            "family": r.get("family") or "general",
            "source": r.get("source") or source or "unknown",
        })
    return out


def split_three_way(tasks: list[dict], evolve_ratio: float = 0.5,
                    id_ratio: float = 0.2, seed: int = 0) -> TaskSet:
    """按 task_id 把任务集切成 evolve / ID held-out / OOD 三层。

    划分依据为 **task_id 的稳定哈希**而非列表下标：
    - 同一任务在任何一次运行中都落到同一层，保证跨轮可比；
    - 新增任务不会打乱已有分层，历史结论仍可复现。

    ``evolve_ratio`` 与 ``id_ratio`` 之和须小于 1，余数归 OOD。
    """
    ts = normalize(tasks)
    if evolve_ratio + id_ratio >= 1.0:
        raise ValueError("evolve_ratio + id_ratio 必须小于 1，余量归 OOD")

    def _bucket(tid: str) -> float:
        h = hashlib.sha256(f"{seed}:{tid}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    evolve, idh, ood = [], [], []
    for t in sorted(ts, key=lambda x: x["task_id"]):
        b = _bucket(t["task_id"])
        if b < evolve_ratio:
            evolve.append(t)
        elif b < evolve_ratio + id_ratio:
            idh.append(t)
        else:
            ood.append(t)
    return TaskSet(evolve=evolve, id_heldout=idh, ood=ood)


def _overlap(*groups: list[dict]) -> list[str]:
    seen: set[str] = set()
    dup: set[str] = set()
    for g in groups:
        for t in g:
            tid = t.get("task_id", "")
            if tid in seen:
                dup.add(tid)
            seen.add(tid)
    return sorted(dup)


def required_n(sd: float, effect: float, power: float = 0.80,
               alpha: float = 0.05) -> int:
    """配对 t 检验达到目标功效所需的**配对数**。

    近似式：``n ≈ (z_{1-α/2} + z_power)² · sd² / effect²``。
    sd 应取**配对差值的实测标准差**（null 差值），不是单臂标准差。
    """
    if effect <= 0:
        raise ValueError("effect 必须为正")
    z_a, z_p = _z(1 - alpha / 2), _z(power)
    from math import ceil
    return int(ceil((z_a + z_p) ** 2 * sd * sd / (effect * effect)))


#: 标准正态分位数（仅收录有把握的常用点，不覆盖的 p 会显式报错）。
#: 宁可拒绝计算，也不要用一个记不准的近似产出看似精确的样本量。
_Z_TABLE: dict[float, float] = {
    0.80: 0.8416,
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.995: 2.5758,
}


def _z(p: float) -> float:
    """标准正态分位数查表。

    刻意不实现有理近似：近似系数需要逐位核对，凭记忆写出会静默产生
    数量级错误（实测曾把 n=188 算成 n=271570）。此处只返回表中已有值，
    未收录的 p 直接抛错，把不确定性暴露在调用点。
    """
    if p in _Z_TABLE:
        return _Z_TABLE[p]
    raise ValueError(
        f"未收录的分位数 p={p}；请扩展 _Z_TABLE（当前支持 {sorted(_Z_TABLE)}）")


# ---------------------------------------------------------------- 外部分布外集


@dataclass
class ExternalSpec:
    """外部基准接入规格。用于把接入工作显式化，避免"以为接了其实没接"。

    当前状态：仅定义接口。``loader`` 未实现时调用会抛 ``NotImplementedError``，
    以便任何声称"已在外部基准上验证"的结论都能被机器证伪。
    """

    name: str
    #: 数据集来源（URL 或 HuggingFace repo id）
    source: str
    #: 是否已实现 loader。False 表示该基准尚不可用于验证。
    available: bool = False
    #: 任务族标注，用于判断与本仓库的匹配度
    families: tuple[str, ...] = ()


#: 候选外部分布外基准。``available`` 全部为 False —— 尚未接入任何一个。
EXTERNAL_BENCHMARKS: tuple[ExternalSpec, ...] = (
    ExternalSpec(
        name="APEX-Agents",
        source="huggingface.co/datasets/mercor/apex-agents",
        families=("investment_banking", "consulting", "legal"),
    ),
    ExternalSpec(
        name="GDPval",
        source="OpenAI GDPval gold subset (220 tasks)",
        families=("professional_deliverables",),
    ),
    ExternalSpec(
        name="Terminal-Bench 2.1",
        source="terminal-bench (containerized terminal tasks)",
        families=("coding", "terminal"),
    ),
    ExternalSpec(
        name="SWE-bench Verified",
        source="SWE-bench Verified",
        families=("coding", "repository"),
    ),
)


def external_status() -> list[dict[str, Any]]:
    """返回外部基准接入状态，供 `status` / 报告输出。"""
    return [
        {"name": s.name, "source": s.source, "available": s.available,
         "families": list(s.families)}
        for s in EXTERNAL_BENCHMARKS
    ]


# ---------------------------------------------------------------- 降本任务集（轻量画像）

#: 为什么需要这条路径
#: ------------------
#: 统计所需的配对样本量 n 只取决于噪声 sd，**不**取决于单次成本；但单次成本
#: 决定了「预算内能否跑得起 n 对」。当前完整研报任务单 run ≈ 65 万 token，
#: 在实测 sd≈24.5 下检出 ΔQ=4 需要 n≈295 对 → 成本 ≈ 295×2×65万 ≈ 3.8 亿 token，
#: 远超 1500 万预算上限。于是「验证」在数学上成立、在经济上不可行。
#:
#: 降本路径把单 run 成本压到 ≈ 5 万 token（窄口径定量简报，约 1/13），
#: 使「预算内可达 n」与「统计所需 n」重新对齐：同一 sd=24.5 下 ΔQ=10
#: 仅需 n≈48 对 → 成本 ≈ 4.8M token ≤ 15M 上限，验证即落在预算内。
#: （ΔQ≈4 仍需把 sd 压到 ~15，见降噪通道；降本是其必要前提。）


# ---------------------------------------------------------------- 成本估算


@dataclass
class CostPlan:
    """一次 benchmark 的执行规模与成本估算。"""

    train_tasks: int
    reps_per_task: int
    rounds: int
    holdout_tasks: int
    tokens_per_run: int
    seconds_per_run: float

    @property
    def evolve_runs(self) -> int:
        return self.train_tasks * self.reps_per_task * 2 * self.rounds

    @property
    def holdout_runs(self) -> int:
        return self.holdout_tasks * self.rounds

    @property
    def total_runs(self) -> int:
        return self.evolve_runs + self.holdout_runs

    @property
    def total_tokens(self) -> int:
        return self.total_runs * self.tokens_per_run

    @property
    def wall_hours(self) -> float:
        return self.total_runs * self.seconds_per_run / 3600.0


#: 实测基线：一次真实研报 run ≈ 65 万 token / 1109 秒
MEASURED_TOKENS_PER_RUN = 650_000
MEASURED_SECONDS_PER_RUN = 1108.6


@dataclass(frozen=True)
class RunProfile:
    """一次 run 的成本画像。降本路径 = 切换 Profile，不动其余闭环逻辑。"""

    name: str
    tokens_per_run: int
    seconds_per_run: float
    note: str = ""


#: 实测基线画像：完整研报任务，单 run ≈ 65 万 token
MEASURED_PROFILE = RunProfile(
    name="measured", tokens_per_run=MEASURED_TOKENS_PER_RUN,
    seconds_per_run=MEASURED_SECONDS_PER_RUN,
    note="完整研报任务：多工具编排 + 长输出，单 run 约 65 万 token",
)
#: 降本目标画像：轻量微任务 ≈ 5 万 token / 120 秒（约为实测的 1/13、1/9）
LIGHTWEIGHT_PROFILE = RunProfile(
    name="lightweight", tokens_per_run=50_000, seconds_per_run=120.0,
    note="窄口径定量简报：少量工具调用 + 短输出，单 run 约 5 万 token",
)


def plan_cost(train_tasks: int, reps_per_task: int, rounds: int,
              holdout_tasks: int,
              tokens_per_run: int = MEASURED_TOKENS_PER_RUN,
              seconds_per_run: float = MEASURED_SECONDS_PER_RUN,
              *, profile: "RunProfile | None" = None) -> CostPlan:
    """成本计划。

    传入 ``profile`` 时以画像的 token/秒为准（用于切换降本路径）；
    否则沿用显式 ``tokens_per_run`` / ``seconds_per_run``（向后兼容）。
    """
    if profile is not None:
        tokens_per_run = profile.tokens_per_run
        seconds_per_run = profile.seconds_per_run
    return CostPlan(train_tasks=train_tasks, reps_per_task=reps_per_task,
                    rounds=rounds, holdout_tasks=holdout_tasks,
                    tokens_per_run=tokens_per_run,
                    seconds_per_run=seconds_per_run)


def lightweight_cost_plan(train_tasks: int, reps_per_task: int, rounds: int,
                         holdout_tasks: int) -> CostPlan:
    """降本路径成本计划：直接套用 ``LIGHTWEIGHT_PROFILE``。"""
    return plan_cost(train_tasks, reps_per_task, rounds, holdout_tasks,
                     profile=LIGHTWEIGHT_PROFILE)


def cost_vs_detectability(effects: tuple[float, ...] = (4.0, 10.0, 15.0, 20.0),
                         sds: tuple[float, ...] = (15.0, 24.5),
                         cost_cap: int | None = None) -> list[dict]:
    """「效果 × 噪声 × 画像」成本对照表 —— 「降本让验证真正成立」的量化证据。

    每行给出：在给定 sd 与 ΔQ 下所需配对数、两种画像下的总成本、是否落预算。
    """
    if cost_cap is None:
        try:
            from .settings import CFG
            cost_cap = CFG.cost_cap_round
        except Exception:
            cost_cap = 15_000_000
    rows: list[dict] = []
    for sd in sds:
        for eff in effects:
            try:
                n = required_n(sd, eff)
            except ValueError:
                continue
            for prof in (MEASURED_PROFILE, LIGHTWEIGHT_PROFILE):
                cost = n * 2 * prof.tokens_per_run
                rows.append({
                    "sd": sd, "effect": eff, "required_n": n,
                    "profile": prof.name,
                    "tokens_per_run": prof.tokens_per_run,
                    "total_tokens": cost,
                    "cost_cap": cost_cap,
                    "fits_budget": cost <= cost_cap,
                })
    return rows


# ---------------------------------------------------------------- 轻量任务生成器

_LW_FAMILIES: tuple[dict, ...] = (
    {
        "family": "market_sizing",
        "query": (
            "估计 {region} 的 {sector} 在 {year} 年的市场规模（美元）。\n"
            "必须覆盖：(a) 测算口径与关键假设；(b) 至少 2 个独立来源并标注出处；"
            "(c) 近两年年复合增长率；(d) 一个主要驱动因素与一个主要风险。"
        ),
    },
    {
        "family": "competitor_teardown",
        "query": (
            "拆解 {region} 的 {sector} 赛道中 3 家头部公司的定位差异。\n"
            "必须覆盖：(a) 每家 1 句定位；(b) 至少 1 个量化指标并标注来源；"
            "(c) 一个共同风险与一个分化点。"
        ),
    },
    {
        "family": "policy_impact",
        "query": (
            "评估 {region} 近期一项 {sector} 相关政策对从业者的实际影响。\n"
            "必须覆盖：(a) 政策要点 1 句；(b) 至少 2 个受影响方并各给 1 个量化或来源支撑；"
            "(c) 一个不确定性。"
        ),
    },
    {
        "family": "metric_benchmark",
        "query": (
            "给出 {sector} 行业的 4 个核心经营指标及其典型区间。\n"
            "必须覆盖：(a) 每个指标的定义；(b) 区间数值并标注至少 1 个来源；"
            "(c) 一个指标间的联动关系。"
        ),
    },
)

_LW_REGIONS = ("中国", "美国", "欧盟", "东南亚", "日本")
_LW_SECTORS = ("SaaS", "新能源车", "医疗器械", "跨境电商", "工业机器人",
               "半导体设备", "储能", "数字广告")


def generate_lightweight_tasks(n: int, seed: int = 0) -> list[dict]:
    """生成 n 个确定性、带改进余量的轻量微任务。

    任务刻意设计成「窄口径 + 显式子问题 + 必须引用来源」：
    - 子问题清单给 C（完整性）留余量 —— 基线 agent 常漏答某一项；
    - 「至少 N 个来源」给 E（证据率）留余量 —— 基线 agent 常断言无出处；
    因此 RRSI 闭环若在 evolve 轮吃到来「漏答 / 无来源」的惩罚，
    就能在 ID / OOD 上体现真实提升，而非靠任务记忆刷分。

    确定性：相同 (n, seed) 永远产出同一任务集，保证跨轮可比与可复现。
    """
    import random
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        tpl = _LW_FAMILIES[i % len(_LW_FAMILIES)]
        region = _LW_REGIONS[rng.randrange(len(_LW_REGIONS))]
        sector = _LW_SECTORS[rng.randrange(len(_LW_SECTORS))]
        year = rng.choice((2024, 2025, 2026))
        query = tpl["query"].format(region=region, sector=sector, year=year)
        out.append({
            "task_id": f"LW-{i:04d}",
            "query": query,
            "family": tpl["family"],
            "source": "lightweight-synthetic",
        })
    return out
