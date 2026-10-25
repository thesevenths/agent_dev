"""演化层：候选生成（P0 只开放 L1 参数，L2/L3 由 max_edit_level 关闭）。

搜索空间是**真实参数**，不是 V2 里那批编造的名字（MAX_REPLAN / REVIEW_THRESHOLD ...）。
判据只有一条：**改它会不会改变判分口径**。会 → 冻结；不会 → 可搜。

两个真实结论已写进搜索空间：
- ``AGENT_CRITIC_MIN_LEN`` 是惰性机制（默认 4 字符，注释明确「不是质量门槛」）→ 不搜；
- ``AGENT_MAX_ITERATIONS_CODE`` 已是全局 15 的 4 倍仍产生 28 次 recursion 失败 →
  只作为**证伪候选**，用来把「调大预算就能修好」这个假设彻底否掉。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Iterable

from .budget import estimate_round_tokens
from .diagnosis import Taxonomy
from .io_util import new_id, now_iso
from .ledger import Ledger, fingerprint
from .settings import CFG, Settings

#: 真实搜索空间。level: 1=参数 2=提示 3=结构
SEARCH_SPACE: dict[str, dict] = {
    "AGENT_MAX_ITERATIONS_CODE": {
        "values": ["60", "80", "100"], "level": 1,
        "taxonomies": (Taxonomy.COMPLETION_FAILURE, Taxonomy.EXECUTION_FAILURE),
        "rationale": "已是全局 15 的 4 倍仍 28 次失败 → 预期无效，仅作证伪",
        "expectation": "falsify",
    },
    "AGENT_MAX_ITERATIONS": {
        "values": ["15", "25", "40"], "level": 1,
        "taxonomies": (Taxonomy.COMPLETION_FAILURE, Taxonomy.EXECUTION_FAILURE),
        "rationale": "爬虫类任务的全局 ReAct 预算；未单独配置时落在这里",
        "expectation": "improve",
    },
    "AGENT_PARSE_TRIES": {
        "values": ["2", "3", "5"], "level": 1,
        "taxonomies": (Taxonomy.EXECUTION_FAILURE, Taxonomy.COMPLETION_FAILURE),
        "rationale": "结构化解析重试次数，影响计划/路由的稳定性",
        "expectation": "improve",
    },
    "AGENT_REPLAN_ARTIFACT_MAX": {
        "values": ["2", "3", "5"], "level": 1,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "产物缺失时触发 replan 的门槛（弱效应）",
        "expectation": "weak",
    },
    "AGENT_REPLAN_NOOP_MAX": {
        "values": ["2", "3"], "level": 3,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "改它等于改控制流，P2 才允许", "expectation": "structure",
    },
    "AGENT_REPLAN_OBS_WINDOW": {
        "values": ["20", "10"], "level": 3,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "观测窗口，改它等于改控制流", "expectation": "structure",
    },
    "AGENT_PLAN_GROW_MAX": {
        "values": ["1", "2"], "level": 3,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "计划增长上限，改它等于改控制流", "expectation": "structure",
    },
}

#: 结构层的真实当前值（部分不在 .env 里，取代码默认值）
CURRENT_DEFAULTS: dict[str, str] = {
    "AGENT_MAX_ITERATIONS_CODE": "60",
    "AGENT_MAX_ITERATIONS": "15",
    "AGENT_PARSE_TRIES": "3",
    "AGENT_REPLAN_ARTIFACT_MAX": "3",
    "AGENT_REPLAN_NOOP_MAX": "2",
    "AGENT_REPLAN_OBS_WINDOW": "20",
    "AGENT_PLAN_GROW_MAX": "1",
}

#: 泄漏检测：候选文本里出现这些就不许进 probe
_LEAK_PATTERNS = (
    re.compile(r"\b(FRS|CAL|REG|OOD)-\d{2,}\b"),
    re.compile(r"task_id\s*=="),
    re.compile(r"if\s+task(_id)?\s*=="),
)


@dataclass
class Candidate:
    candidate_id: str
    target: str
    level: int
    old: Any
    new: Any
    hypothesis: str
    taxonomy: str = ""
    edits: int = 1
    est_cost: int = 0
    expected_behavior: list[str] = field(default_factory=list)
    patch: dict = field(default_factory=dict)
    base_commit: str = ""
    created_by: str = "proposer"
    hyp_fingerprint: str = ""
    created_at: str = field(default_factory=now_iso)

    # ---- 便于 gate 使用 ----
    @property
    def is_env(self) -> bool:
        return self.patch.get("kind") == "env"

    def env_overrides(self) -> dict[str, str]:
        return {self.target: str(self.new)} if self.is_env else {}

    def diff_text(self) -> str:
        if self.is_env:
            return f"{self.target}: {self.old} -> {self.new}"
        return str(self.patch.get("diff") or self.patch.get("new") or "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_env"] = self.is_env
        return d


def leakage_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pat in _LEAK_PATTERNS:
        m = pat.search(text or "")
        if m:
            hits.append(m.group(0))
    return hits


def _adjacent(values: list[str], old: str) -> list[str]:
    """优先尝试相邻值（改动最小），再试远端值。"""
    try:
        i = values.index(str(old))
    except ValueError:
        return list(values)
    order = [i + 1, i - 1, i + 2, i - 2]
    out = []
    for j in order:
        if 0 <= j < len(values):
            out.append(values[j])
    for v in values:
        if v not in out:
            out.append(v)
    return out


def propose(dominant: str, ledger: Ledger, cfg: Settings | None = None,
            avg_run_tokens: int = 200_000, n_pairs: int | None = None,
            limit: int | None = None) -> list[Candidate]:
    """按失败模式生成候选。确定性规则，不花 LLM 钱。"""
    cfg = cfg or CFG
    n_pairs = n_pairs or cfg.n_default
    est = estimate_round_tokens(avg_run_tokens, n_pairs)
    out: list[Candidate] = []

    ordered = [k for k, v in SEARCH_SPACE.items() if dominant in v["taxonomies"]]
    ordered += [k for k, v in SEARCH_SPACE.items() if dominant not in v["taxonomies"]]

    for target in ordered:
        spec = SEARCH_SPACE[target]
        if spec["level"] > cfg.max_edit_level:
            continue
        old = CURRENT_DEFAULTS.get(target, "")
        for new in _adjacent(list(spec["values"]), old):
            if new == old:
                continue
            fp = fingerprint(target, old, new, dominant)
            dup, prev = ledger.is_duplicate(fp)
            if dup:
                continue
            cand = Candidate(
                candidate_id=new_id("cand"),
                target=target, level=spec["level"], old=old, new=new,
                hypothesis=f"{target} {old}→{new} 可缓解 {dominant}（{spec['rationale']}）",
                taxonomy=dominant, edits=1, est_cost=est,
                expected_behavior=[spec.get("expectation", "improve")],
                patch={"kind": "env", "key": target, "old": old, "new": new},
                hyp_fingerprint=fp,
            )
            out.append(cand)
            break
        if len(out) >= (limit or cfg.candidates_per_round):
            break
    return out


def falsification_candidate(taxonomy: str = Taxonomy.COMPLETION_FAILURE,
                            avg_run_tokens: int = 200_000,
                            cfg: Settings | None = None) -> Candidate:
    """阴性对照 2：故意变差（把 code agent 的预算砍到全局默认值以下）。"""
    cfg = cfg or CFG
    return Candidate(
        candidate_id=new_id("neg"),
        target="AGENT_MAX_ITERATIONS_CODE", level=1, old="60", new="12",
        hypothesis="阴性对照：预算砍到 12 应使完成率下降（用于验证比较方向没反）",
        taxonomy=taxonomy, est_cost=estimate_round_tokens(avg_run_tokens, cfg.n_default),
        expected_behavior=["worse_expected"],
        patch={"kind": "env", "key": "AGENT_MAX_ITERATIONS_CODE", "old": "60", "new": "12"},
        created_by="selftest",
    )


def zero_change_candidate(cfg: Settings | None = None) -> Candidate:
    """阴性对照 1：零改动（只改一行注释，env 不变）。"""
    cfg = cfg or CFG
    return Candidate(
        candidate_id=new_id("zero"),
        target="__comment__", level=1, old="", new="",
        hypothesis="阴性对照：零改动必须被判 WITHIN_NOISE",
        taxonomy=Taxonomy.UNKNOWN, edits=1, est_cost=0,
        expected_behavior=["no_change"],
        patch={"kind": "comment", "file": "adaptive_runtime/__init__.py",
               "old": "# no-op", "new": "# no-op "},
        created_by="selftest",
    )
