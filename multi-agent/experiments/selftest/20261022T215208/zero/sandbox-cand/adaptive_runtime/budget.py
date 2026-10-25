"""预算治理：probe 是双份执行，没有硬上限必然失控。

默认上限（见 settings）：单轮 8M token，单候选 4M token。
预算不足时的正确做法是**降低 n 并把结论标成 inconclusive**，
而不是「少跑几次但结论照旧」。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .settings import CFG, Settings


class BudgetExceeded(RuntimeError):
    """预算越界。调用方必须终止本轮，而不是继续跑。"""

    def __init__(self, message: str, *, spent: int = 0, cap: int = 0):
        super().__init__(message)
        self.spent = spent
        self.cap = cap


@dataclass
class Budget:
    cap_round: int = field(default_factory=lambda: CFG.cost_cap_round)
    cap_cand: int = field(default_factory=lambda: CFG.cost_cap_cand)
    spent_round: int = 0
    spent_cand: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, cfg: Settings) -> "Budget":
        return cls(cap_round=cfg.cost_cap_round, cap_cand=cfg.cost_cap_cand)

    # ---- 查询 ----
    @property
    def remaining_round(self) -> int:
        return max(0, self.cap_round - self.spent_round)

    def remaining_cand(self, cand_id: str) -> int:
        return max(0, self.cap_cand - self.spent_cand.get(cand_id, 0))

    def can_spend(self, n: int, cand_id: str | None = None) -> bool:
        if n > self.remaining_round:
            return False
        if cand_id is not None and n > self.remaining_cand(cand_id):
            return False
        return True

    # ---- 记账 ----
    def spend(self, n: int, cand_id: str | None = None) -> None:
        if n < 0:
            raise ValueError("token 数不能为负")
        if not self.can_spend(n, cand_id):
            raise BudgetExceeded(
                f"预算不足：需要 {n:,}，轮剩余 {self.remaining_round:,}"
                + (f"，候选剩余 {self.remaining_cand(cand_id):,}" if cand_id else ""),
                spent=self.spent_round, cap=self.cap_round)
        self.spent_round += n
        if cand_id is not None:
            self.spent_cand[cand_id] = self.spent_cand.get(cand_id, 0) + n

    def to_dict(self) -> dict:
        return {"cap_round": self.cap_round, "cap_cand": self.cap_cand,
                "spent_round": self.spent_round, "spent_cand": dict(self.spent_cand),
                "remaining_round": self.remaining_round}


def estimate_round_tokens(avg_run_tokens: int, n_pairs: int, overhead: int = 200_000) -> int:
    """一轮 probe 的 token 估算：n 对 × 2 臂 + proposer/judge 开销。"""
    return max(0, avg_run_tokens) * max(0, n_pairs) * 2 + overhead


def fits(avg_run_tokens: int, n_pairs: int, cfg: Settings | None = None,
         overhead: int = 200_000) -> tuple[bool, int]:
    cfg = cfg or CFG
    est = estimate_round_tokens(avg_run_tokens, n_pairs, overhead)
    return est <= cfg.cost_cap_round, est


def degraded_n(avg_run_tokens: int, cfg: Settings | None = None,
               overhead: int = 200_000) -> int:
    """预算内最多能跑几对（至少 2 对，否则不应得出结论）。"""
    cfg = cfg or CFG
    if avg_run_tokens <= 0:
        return cfg.n_default
    n = (cfg.cost_cap_round - overhead) // (avg_run_tokens * 2)
    return max(0, min(cfg.n_default, int(n)))
