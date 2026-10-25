"""经验层：经过跨任务验证的机制，可检索、可复核、可人工 promote。

为什么经验不是「普通 memory」
---------------------------
memory 记的是事实（用户偏好、市场数据）；经验记的是**在什么失败模式下、改了什么、
在多少个任务上有效/无效**。它带负面证据与统计置信度，过期或被证伪必须能降级。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .io_util import append_jsonl, new_id, now_iso, read_jsonl
from .settings import CFG, PATHS, Settings


def confidence(n_pairs: int, n_target: int, regression_rate: float, p_value: float,
               weights: tuple[float, float, float] = (0.4, 0.3, 0.3)) -> float:
    """置信度 = 0.4·样本充分度 + 0.3·稳健性 + 0.3·显著度。

    缺一不可：只跑 2 对但全胜（p 小）不该算高置信，因为样本不足；
    跑满 8 对但 30% 任务变差，也不该算高置信。
    """
    w1, w2, w3 = weights
    sample = 1.0 if n_target <= 0 else max(0.0, min(1.0, n_pairs / n_target))
    robust = max(0.0, min(1.0, 1.0 - max(0.0, regression_rate)))
    sig = max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, p_value))))
    return round(max(0.0, min(1.0, w1 * sample + w2 * robust + w3 * sig)), 4)


@dataclass
class Experience:
    experience_id: str
    problem: str                       # 失败模式指纹：TAXONOMY|agent|step
    claim: str
    component: str                     # L1/env/AGENT_...
    edit: dict = field(default_factory=dict)
    hypothesis_id: str = ""
    positive_tasks: list[str] = field(default_factory=list)
    negative_tasks: list[str] = field(default_factory=list)
    dQ: float = 0.0
    dB: float = 0.0
    dC: float = 0.0
    regression_rate: float = 0.0
    transfer_score: float | None = None
    confidence: float = 0.0
    status: str = "validated"          # validated | weakened | refuted
    judge_version: str = ""
    created_at: str = field(default_factory=now_iso)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def lesson(self, verified: bool | None = None) -> str:
        """生成一条可注入提示的经验。verified 缺省时按置信度推导，避免把低置信经验标成「已验证」。"""
        if verified is None:
            verified = self.confidence >= CFG.exp_min_confidence
        tag = "已验证" if verified else "待验证"
        return (f"[{self.experience_id} · {tag} · conf {self.confidence:.2f}] "
                f"{self.problem} → {self.claim}")


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else PATHS.experience / "experiences.jsonl"

    def rows(self) -> list[dict]:
        return read_jsonl(self.path)

    def latest(self) -> list[dict]:
        """按 experience_id 取最新一条（refute/降级是 append-only，不能只看首条）。"""
        by_id: dict[str, dict] = {}
        for r in self.rows():
            eid = r.get("experience_id") or ""
            by_id[eid] = r
        return list(by_id.values())

    def add(self, exp: Experience) -> Experience:
        for r in self.rows():
            if (r.get("problem"), r.get("component"), (r.get("edit") or {}).get("patch_hash")) == \
               (exp.problem, exp.component, exp.edit.get("patch_hash")):
                return exp  # 幂等：同一经验不重复落库
        append_jsonl(self.path, [exp.to_dict()])
        return exp

    def create(self, problem: str, claim: str, component: str, edit: dict,
               n_pairs: int, n_target: int, p_value: float, regression_rate: float,
               dQ: float, dC: float, positive: list[str], negative: list[str],
               dB: float = 0.0, judge_version: str = "", hypothesis_id: str = "") -> Experience:
        conf = confidence(n_pairs, n_target, regression_rate, p_value)
        status = "validated" if conf >= CFG.exp_min_confidence else "weakened"
        exp = Experience(experience_id=new_id("exp"), problem=problem, claim=claim,
                         component=component, edit=edit, hypothesis_id=hypothesis_id,
                         positive_tasks=list(positive), negative_tasks=list(negative),
                         dQ=dQ, dB=dB, dC=dC, regression_rate=regression_rate,
                         confidence=conf, status=status, judge_version=judge_version)
        return self.add(exp)

    # ---- 检索 ----
    def retrieve(self, problem: str, top: int | None = None) -> list[Experience]:
        """按失败模式检索：先精确命中 ``TAXONOMY|agent|step``，再退回 ``TAXONOMY`` 前缀。"""
        top = top or CFG.exp_retrieve_top
        rows = [r for r in self.latest() if r.get("status") != "refuted"]
        taxonomy = (problem or "").split("|")[0]

        def key(r: dict) -> tuple[int, float]:
            p = r.get("problem", "")
            exact = 2 if p == problem else (1 if p.split("|")[0] == taxonomy else 0)
            return (exact, float(r.get("confidence", 0.0)))

        picked = [r for r in sorted(rows, key=key, reverse=True) if key(r)[0] > 0][:top]
        return [_from_row(r) for r in picked]

    def lessons_for(self, problem: str, top: int | None = None) -> list[str]:
        out = []
        for e in self.retrieve(problem, top):
            out.append(e.lesson(verified=e.confidence >= CFG.exp_min_confidence))
        return out

    # ---- promote ----
    def promotion_candidates(self, min_families: int = 2, min_conf: float = 0.5) -> list[dict]:
        """达到「可人工合并进 Base」标准的经验：≥2 个任务族验证通过。"""
        out = []
        for r in self.latest():
            fams = {t.split("-")[0] for t in (r.get("positive_tasks") or [])}
            if len(fams) >= min_families and float(r.get("confidence", 0)) >= min_conf:
                out.append(r)
        return out

    def refute(self, experience_id: str, note: str = "") -> bool:
        for r in self.rows():
            if r.get("experience_id") == experience_id:
                row = dict(r)
                row["status"] = "refuted"
                row["notes"] = (row.get("notes", "") + " | " + note).strip(" |")
                append_jsonl(self.path, [row])
                return True
        return False


def _from_row(r: dict) -> Experience:
    fields = {k: v for k, v in r.items() if k in Experience.__dataclass_fields__}
    return Experience(**fields)
