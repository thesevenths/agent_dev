"""演化层的两个正则化机制：Exploration（防搜索坍缩）与 Pruning（淘汰无效组件）。

为什么需要这两件事
------------------
`proposer` 是「按主导失败模式挑参数」的贪心搜索，天然会反复打磨同一层、同一类参数，
把搜索空间坍缩到局部最优。RRSI 的两条对策在本仓库的落地形态是：

- **Exploration**：按**被接受**次数（不是被提出次数）统计各层使用率；一旦某层被
  过度使用，下一轮**强制**在被冷落的 (失败模式 × 层) 组合里提候选。注意本仓库的层
  极不均衡（L1 只有 4 个可搜参数，很容易穷举完），所以冷落对象必须是组合而不是层。
- **Pruning**：某组件（= 一个 target 参数）连续若干次 `WITHIN_NOISE`，说明「改它不影响
  任何判分结果」→ 进入 `dormant`，一段时间内不再由它生成候选。它真正的产出不是省算力，
  而是**「这类机制在这个 harness 上是惰性的」这条结论本身**。

两者都只读台账，不写状态机：`dormant` 由台账时间戳现算，避免多一份需要同步的状态。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

from .gate import Reason
from .proposer import SEARCH_SPACE
from .settings import CFG, Settings

#: 冷落判据（V3 §10.4）：max(usage) > 4 × min(usage) 且 max ≥ 5
COLD_RATIO = 4
COLD_MIN = 5

#: 剪枝判据（V3 §10.6）：某 target 连续 10 次 WITHIN_NOISE → dormant 30 天
PRUNE_STREAK = 10
DORMANT_DAYS = 30

#: RC3（停滞触发 L3 强制探索）：连续多少轮 0 accept 即强制把搜索推到结构层。
#: 本仓库每轮只产 1 个候选（candidates_per_round=1），故「连续非 accept 行」≈「连续 0 accept 轮」。
FORCE_AFTER_NO_ACCEPT = 2


# ---------------------------------------------------------------- 台账行归类

def outcome_of(row: dict) -> str:
    """把一行台账归一化为 accepted / within_noise / inconclusive / rejected / other。

    为什么不能只看 ``status``：``within_noise`` 在台账里被记成 ``inconclusive``
    （「判不出来」而非「已证伪」），而 pruning 关心的恰恰是这一类。
    """
    status = row.get("status")
    reason = (row.get("evidence") or {}).get("reason")
    if status == "accepted":
        return "accepted"
    if reason == Reason.WITHIN_NOISE.value:
        return "within_noise"
    if status == "inconclusive":
        return "inconclusive"
    if status == "rejected":
        return "rejected"
    return "other"


def level_of(target: str, search_space: dict[str, dict] | None = None) -> int | None:
    spec = (search_space or SEARCH_SPACE).get(target)
    return int(spec["level"]) if spec else None


def _target_of(row: dict, search_space: dict[str, dict]) -> str | None:
    """台账行不一定存 target，但 fingerprint 的第一段一定是 target。"""
    t = row.get("target")
    if isinstance(t, str) and t in search_space:
        return t
    fp = row.get("fingerprint") or ""
    head = fp.split("|", 1)[0]
    return head if head in search_space else None


def _taxonomy_of(row: dict) -> str:
    fp = row.get("fingerprint") or ""
    parts = fp.split("|")
    return parts[2] if len(parts) >= 3 else (row.get("taxonomy") or "")


# ---------------------------------------------------------------- Exploration

def usage_by_level(rows: Iterable[dict], search_space: dict[str, dict] | None = None,
                   ) -> dict[int, int]:
    """各层**被接受**的次数。按提出次数统计会被「大量提 L1」这种噪声带偏。"""
    ss = search_space or SEARCH_SPACE
    out: dict[int, int] = {}
    for r in rows:
        if outcome_of(r) != "accepted":
            continue
        lv = level_of(_target_of(r, ss) or "", ss)
        if lv is None:
            continue
        out[lv] = out.get(lv, 0) + 1
    return out


def usage_by_pair(rows: Iterable[dict], search_space: dict[str, dict] | None = None,
                  ) -> dict[tuple[str, int], int]:
    """各 (失败模式 × 层) 被接受的次数 —— exploration 真正的作用单位。"""
    ss = search_space or SEARCH_SPACE
    out: dict[tuple[str, int], int] = {}
    for r in rows:
        if outcome_of(r) != "accepted":
            continue
        target = _target_of(r, ss)
        lv = level_of(target or "", ss)
        if target is None or lv is None:
            continue
        out[(_taxonomy_of(r), lv)] = out.get((_taxonomy_of(r), lv), 0) + 1
    return out


def should_explore(rows: Iterable[dict], cfg: Settings | None = None,
                   search_space: dict[str, dict] | None = None) -> tuple[bool, dict]:
    """是否触发一次强制探索。返回 (是否, 依据)。"""
    cfg = cfg or CFG
    ss = search_space or SEARCH_SPACE
    usage = usage_by_level(rows, ss)
    reachable = sorted({int(v["level"]) for v in ss.values() if int(v["level"]) <= cfg.max_edit_level})
    if len(reachable) < 2:
        # 只有一层可选时不存在「层间冷落」，这是 P0 的常态，不是异常
        return False, {"reason": "single_level", "reachable_levels": reachable, "usage": usage}
    counts = [usage.get(lv, 0) for lv in reachable]
    hi, lo = max(counts), min(counts)
    hit = hi >= COLD_MIN and hi > COLD_RATIO * max(lo, 0)
    return hit, {"reason": "cold_shoulder" if hit else "balanced",
                 "usage": usage, "max": hi, "min": lo,
                 "reachable_levels": reachable}


def neglected_pairs(rows: Iterable[dict], search_space: dict[str, dict] | None = None,
                    max_level: int = 3) -> list[tuple[str, int]]:
    """搜索空间里「有参数但从未被接受过」的 (失败模式 × 层) 组合，按层升序。"""
    ss = search_space or SEARCH_SPACE
    used = usage_by_pair(rows, ss)
    pairs: set[tuple[str, int]] = set()
    for spec in ss.values():
        lv = int(spec["level"])
        if lv > max_level:
            continue
        for tax in spec.get("taxonomies", ()):
            pairs.add((tax, lv))
    unused = [p for p in pairs if used.get(p, 0) == 0]
    return sorted(unused, key=lambda p: (p[1], p[0]))


def exploration_pairs(rows: Iterable[dict], cfg: Settings | None = None,
                      search_space: dict[str, dict] | None = None) -> list[tuple[str, int]]:
    """本轮若要探索，应该去覆盖哪些组合（按「用得最少」优先）。"""
    cfg = cfg or CFG
    ss = search_space or SEARCH_SPACE
    used = usage_by_pair(rows, ss)
    pairs = neglected_pairs(rows, ss, max_level=cfg.max_edit_level)
    reachable = {int(v["level"]) for v in ss.values() if int(v["level"]) <= cfg.max_edit_level}
    # 优先被冷落的层：层内被接受次数最少的那些
    by_level = {lv: sum(1 for (t, l) in used if l == lv) for lv in reachable}
    pairs.sort(key=lambda p: (by_level.get(p[1], 0), p[1], p[0]))
    return pairs


# ---------------------------------------------------------------- Pruning

def _ts(row: dict) -> datetime | None:
    raw = row.get("created_at") or row.get("ts") or ""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def within_noise_streak(rows: list[dict], target: str,
                        search_space: dict[str, dict] | None = None) -> int:
    """某 target 从最新往回数，连续 ``WITHIN_NOISE`` 的次数。"""
    ss = search_space or SEARCH_SPACE
    streak = 0
    for row in reversed(rows):
        if _target_of(row, ss) != target:
            continue
        kind = outcome_of(row)
        if kind == "within_noise":
            streak += 1
        elif kind in ("accepted", "rejected"):
            break
        # inconclusive / other：不打断也不计数（「没判出来」不等于「没用」）
    return streak


def prune_candidates(rows: list[dict], cfg: Settings | None = None,
                     search_space: dict[str, dict] | None = None) -> list[dict]:
    """达到剪枝阈值的组件清单（不含 dormant 有效期，只是「已判定惰性」）。"""
    ss = search_space or SEARCH_SPACE
    out: list[dict] = []
    for target in ss:
        streak = within_noise_streak(rows, target, ss)
        if streak >= PRUNE_STREAK:
            out.append({"target": target, "level": level_of(target, ss),
                        "within_noise_streak": streak,
                        "verdict": "该组件在此 harness 上未表现出可测收益（惰性）"})
    return out


def dormant_targets(rows: list[dict], cfg: Settings | None = None,
                    search_space: dict[str, dict] | None = None,
                    now: datetime | None = None) -> set[str]:
    """仍在 dormant 有效期内的组件：剪枝判定后 ``DORMANT_DAYS`` 天内不再生成候选。"""
    ss = search_space or SEARCH_SPACE
    now = now or datetime.now()
    dormant: set[str] = set()
    for item in prune_candidates(rows, cfg, ss):
        target = item["target"]
        last = None
        for row in reversed(rows):
            if _target_of(row, ss) == target and outcome_of(row) == "within_noise":
                last = _ts(row)
                break
        if last is None or (now - last) <= timedelta(days=DORMANT_DAYS):
            dormant.add(target)
    return dormant


# ---------------------------------------------------------------- 报告

def report(rows: list[dict], cfg: Settings | None = None) -> dict:
    cfg = cfg or CFG
    hit, why = should_explore(rows, cfg)
    return {
        "usage_by_level": usage_by_level(rows),
        "usage_by_pair": {f"{t}@L{lv}": n for (t, lv), n in usage_by_pair(rows).items()},
        "should_explore": hit,
        "explore_reason": why,
        "exploration_pairs": [f"{t}@L{lv}" for t, lv in exploration_pairs(rows, cfg)],
        "dormant": sorted(dormant_targets(rows, cfg)),
        "pruned": prune_candidates(rows, cfg),
        "thresholds": {"cold_ratio": COLD_RATIO, "cold_min": COLD_MIN,
                       "prune_streak": PRUNE_STREAK, "dormant_days": DORMANT_DAYS},
    }


# ---------------------------------------------------------------- RC3 停滞触发

def _consecutive_no_accept(rows: Iterable[dict], n: int = FORCE_AFTER_NO_ACCEPT) -> bool:
    """从最新台账行往前数，连续「非 accept」行是否已达 ``n``。

    本仓库每轮只产 1 个候选（``candidates_per_round=1``），故「连续非 accept 行」
    即「连续 0 accept 轮」。达到阈值说明闭环在原地打转，需要跳出 L1 旋钮去试结构层。
    """
    if n <= 0:
        return False
    count = 0
    for r in reversed(rows):
        if outcome_of(r) == "accepted":
            break
        count += 1
        if count >= n:
            return True
    return False


def _dominant_l1_exhausted(rows: Iterable[dict], dominant: str,
                           search_space: dict[str, dict] | None = None) -> bool:
    """dominant 失败模式下，其全部 L1 旋钮都已被试过且无一被接受 → 视为穷举。

    此时继续在 L1 旋钮上做相邻值搜索是浪费，应把搜索推到 L3 结构候选（B1–B4）。
    """
    ss = search_space or SEARCH_SPACE
    l1 = [k for k, v in ss.items()
          if int(v["level"]) == 1 and dominant in v.get("taxonomies", ())]
    if not l1:
        return False
    tried = {_target_of(r, ss) for r in rows
             if _target_of(r, ss) in l1 and outcome_of(r) != "accepted"}
    return len(tried) == len(l1)


def evolve_plan(rows: list[dict], dominant: str, cfg: Settings | None = None) -> dict:
    """给一轮演化用的决策：跳过哪些 target、是否强制探索、强制到哪个组合。"""
    cfg = cfg or CFG
    dormant = dormant_targets(rows, cfg)
    hit, why = should_explore(rows, cfg)
    forced: tuple[str, int] | None = None
    if hit:
        pairs = exploration_pairs(rows, cfg)
        if pairs:
            forced = pairs[0]
    else:
        # RC3：冷落机制（should_explore）依赖「被接受次数」，低 accept 区永远不触发，
        # 导致 L3 结构候选（B1–B4）被 L1 旋钮饿死。这里用「连续 0 accept」或
        # 「dominant 的 L1 旋钮已穷举」两个停滞信号，直接强制 (dominant, 3)，
        # 把结构候选顶到 propose 的 ordered 首位（propose 内部再把 code 级顶到最前）。
        if _consecutive_no_accept(rows) or _dominant_l1_exhausted(rows, dominant):
            forced = (dominant, 3)
            why = {"reason": "rc3_stall_forced_l3",
                   "consecutive_no_accept": _consecutive_no_accept(rows),
                   "dominant_l1_exhausted": _dominant_l1_exhausted(rows, dominant)}
    return {"exclude_targets": sorted(dormant), "explore": hit or forced is not None,
            "explore_reason": why, "forced_pair": forced, "dominant": dominant}
