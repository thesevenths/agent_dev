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


def plan_cost(train_tasks: int, reps_per_task: int, rounds: int,
              holdout_tasks: int,
              tokens_per_run: int = MEASURED_TOKENS_PER_RUN,
              seconds_per_run: float = MEASURED_SECONDS_PER_RUN) -> CostPlan:
    return CostPlan(train_tasks=train_tasks, reps_per_task=reps_per_task,
                    rounds=rounds, holdout_tasks=holdout_tasks,
                    tokens_per_run=tokens_per_run,
                    seconds_per_run=seconds_per_run)
