"""Regression 集：只用于否决、永不参与演化的固定任务集（V3 §13.4）。

为什么需要它
------------
演化只看「本轮的主导失败模式」，于是很容易出现：某候选把 A 类失败修好了，把 B 类
任务弄坏了，而 B 类根本不在本轮任务集里 —— 一次漂亮的正 Δ 掩盖了一次回归。
Regression 集的作用就是把「体检项」固定下来：

- 任一被接受的候选都必须在这个集上复跑；
- 该集**永不**用于生成假设（只用于否决），否则它会被一起过拟合掉；
- 判据 ``regression_rate = 显著负 Δ 的任务数 / 总任务数 ≤ R_MAX``。

诚实的边界
----------
如果本轮 probe 没有真的覆盖 regression 集（P0 只有历史回放任务，常常没有），
``check`` 必须返回 ``covered=False`` 而不是给出一个假的 0% —— 没测不等于没坏。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io_util import append_jsonl, now_iso, read_json, read_jsonl, write_json
from .settings import CFG, PATHS, Settings

#: 规模要求：≥12 个任务，覆盖 6 个任务族，每族 ≥2
MIN_TASKS = 12
MIN_FAMILIES = 6
MIN_PER_FAMILY = 2


def dir_path() -> Path:
    return PATHS.regression


def tasks_path() -> Path:
    return dir_path() / "tasks.json"


def history_path() -> Path:
    return dir_path() / "history.jsonl"


def load() -> list[dict]:
    data = read_json(tasks_path(), default=None) or {}
    tasks = data.get("tasks") if isinstance(data, dict) else data
    return [t for t in (tasks or []) if t.get("query")]


def save(tasks: list[dict], source: str = "") -> None:
    dir_path().mkdir(parents=True, exist_ok=True)
    write_json(tasks_path(), {"updated_at": now_iso(), "source": source,
                              "count": len(tasks), "tasks": tasks})


def add_from_accepted(tasks: Iterable[dict], candidate_id: str = "") -> int:
    """把一个被接受候选的正面任务并入 regression 集。返回新增数量。

    只追加不删除：删掉就等于把曾经证明过没回归的证据抹掉。
    """
    cur = load()
    seen = {t.get("task_id") for t in cur}
    added: list[dict] = []
    for t in tasks:
        tid = t.get("task_id")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        added.append({"task_id": tid, "query": t.get("query", ""),
                      "family": t.get("family", "general"),
                      "added_at": now_iso(), "added_by": candidate_id})
    if added:
        save(cur + added, source=f"accepted:{candidate_id}")
        append_jsonl(history_path(), [{"ts": now_iso(), "candidate_id": candidate_id,
                                       "added": [t["task_id"] for t in added]}])
    return len(added)


def coverage(tasks: list[dict] | None = None) -> dict:
    tasks = tasks if tasks is not None else load()
    fams: dict[str, int] = {}
    for t in tasks:
        f = t.get("family") or "general"
        fams[f] = fams.get(f, 0) + 1
    ok = (len(tasks) >= MIN_TASKS and len(fams) >= MIN_FAMILIES
          and all(v >= MIN_PER_FAMILY for v in fams.values()))
    return {"count": len(tasks), "families": fams, "meets_spec": ok,
            "requires": {"min_tasks": MIN_TASKS, "min_families": MIN_FAMILIES,
                         "min_per_family": MIN_PER_FAMILY}}


def select_for_probe(tasks: list[dict], budget: int) -> list[dict]:
    """从 regression 集里挑 ``budget`` 个任务参加本轮 probe（按族轮转，避免偏科）。"""
    if budget <= 0 or not tasks:
        return []
    by_family: dict[str, list[dict]] = {}
    for t in tasks:
        by_family.setdefault(t.get("family") or "general", []).append(t)
    out: list[dict] = []
    i = 0
    while len(out) < min(budget, len(tasks)):
        progressed = False
        for fam in sorted(by_family):
            if i < len(by_family[fam]) and len(out) < min(budget, len(tasks)):
                out.append(by_family[fam][i])
                progressed = True
        if not progressed:
            break
        i += 1
    return out


def check(comparison_any: Any, regression_task_ids: Iterable[str],
          cfg: Settings | None = None) -> dict:
    """在 regression 子集上算 ``regression_rate``。

    ``comparison_any`` 是 ``comparator.Comparison``（或含 ``per_task`` 的对象）。
    """
    cfg = cfg or CFG
    ids = {str(x) for x in regression_task_ids}
    per_task = getattr(comparison_any, "per_task", None) or {}
    covered = [k for k in per_task if k in ids]
    if not covered:
        return {"covered": False, "n": 0, "regression_rate": None,
                "ok": None, "r_max": cfg.r_max,
                "note": "本轮 probe 未覆盖 regression 集 → 不判定（没测 ≠ 没坏）"}
    bad = [k for k in covered if per_task[k].get("mean_delta", 0.0) < -cfg.regression_min_points]
    rate = len(bad) / len(covered)
    return {"covered": True, "n": len(covered), "regression_rate": round(rate, 4),
            "regressed": sorted(bad), "ok": rate <= cfg.r_max, "r_max": cfg.r_max}


def summary() -> dict:
    tasks = load()
    cov = coverage(tasks)
    return {"tasks": len(tasks), "coverage": cov,
            "history_events": len(read_jsonl(history_path())),
            "path": str(tasks_path())}
