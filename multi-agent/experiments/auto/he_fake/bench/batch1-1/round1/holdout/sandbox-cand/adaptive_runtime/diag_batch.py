"""批量诊断：把「本轮自己跑出来的轨迹」变成失败模式统计。

为什么不能继续用 `_newest_log()`
--------------------------------
原实现（``benchmark.cmd_round`` 同款）是 ``log = _newest_log()`` —— 拿 ``log/`` 里
**mtime 最新**的那份日志当诊断输入。人工在 Studio 里跑一次提问，闭环的诊断输入就被换掉了；
更糟的是无人值守时它根本不闭合：闭环要改进的是「它自己刚跑出来的那批失败」，
而它读的却是「别人刚才随手跑的那一次」。这会让每一轮的 dominant 失败模式与本轮 probe
毫无因果关系，闭环看起来在动，实际上是在对着一份随机历史开药方。

本模块把诊断的输入源改成 **本轮 run 目录里的 trace.json + score.json**：

- 只读本轮产物 ⇒ 诊断与探针是同一批证据；
- 只读 train 臂 ⇒ holdout 一旦进诊断池就是泄漏，这里用 ``exclude_task_ids`` **机器排除**，
  不靠调用方自觉；
- 沙箱会被 ``cleanup()`` 删掉，但 trace/score 落在 ``experiments/`` 下，
  所以诊断必须在 cleanup 之前或之后都能重放 —— 这正是把结论写成可复核工件的原因。

为什么需要 ``load_trace`` / ``load_score``
------------------------------------------
``Trace``/``Score`` 只实现了 ``to_dict()``（``steps`` 的 int 键被 str 化、并附加一个
``derived`` 块），没有反向构造。诊断层却要求真实的 dataclass（``trace.steps`` 需要
``StepState.achieved`` 这类 property）。反序列化必须由**写入方所在模块之外**的适配层承担：
放在 ``trace_export`` 里会让它被 Frozen Core 清单覆盖（改判分链路 = 越界），
放在 ``diagnosis`` 里会引入 evaluator→diagnosis 的循环依赖。所以单独成文件。
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Iterable

from . import diagnosis
from .diagnosis import Finding
from .evaluator import Score
from .settings import CFG, Settings
from .trace_export import StepState, Trace


# ---------------------------------------------------------------- 反序列化

def _pick(dc_cls, data: dict) -> dict:
    """只取 dataclass 认识的键：多余键（如 ``derived``）必须丢掉，否则构造直接抛。"""
    known = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in (data or {}).items() if k in known}


def load_trace(data: dict) -> Trace:
    """``trace.json`` → ``Trace``。

    ``to_dict()`` 把 ``steps`` 写成 ``{"1": {...}}``（JSON 不支持 int 键），
    这里必须还原成 int 键 + ``StepState`` 实例：``diagnosis.classify`` 会访问
    ``step.failures`` / ``step.achieved``，传 dict 进去会静默拿到 0，
    于是「全部跑崩」被诊断成「无失败信号」—— 那是比崩溃更糟的失效。
    """
    d = dict(data or {})
    raw_steps = d.pop("steps", {}) or {}
    d.pop("derived", None)          # 派生块由 property 重算，不能当构造参数
    tr = Trace(**_pick(Trace, d))
    tr.steps = {}
    for k, v in raw_steps.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        tr.steps[idx] = StepState(**_pick(StepState, v))
    return tr


def load_score(data: dict) -> Score | None:
    """``score.json`` → ``Score``。缺文件/缺关键字段返回 None（调用方走硬失败通道）。"""
    if not data:
        return None
    if "Q" not in data and "A" not in data:
        return None
    return Score(**_pick(Score, data))


# ---------------------------------------------------------------- 扫描 run 目录

def iter_run_dirs(probe_dir: str | Path, arm: str = "base") -> list[Path]:
    """probe 输出目录下的单次 run 目录（含 trace.json），按名字排序保证可复现。"""
    root = Path(probe_dir) / arm
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir()
                   if p.is_dir() and (p / "trace.json").exists()),
                  key=lambda p: p.name)


def load_run(run_dir: str | Path) -> tuple[Trace, Score | None]:
    from .io_util import read_json
    p = Path(run_dir)
    tr = load_trace(read_json(p / "trace.json", default={}) or {})
    sc = load_score(read_json(p / "score.json", default={}) or {})
    return tr, sc


def is_fake_run(run_dir: str | Path) -> bool:
    """这个 run 是假跑（受控轨迹）吗？假跑产物绝不得驱动真实轮次的诊断。

    ``runner_once`` 在 ``result.json`` 里写了 ``fake`` 标志：假轨迹的失败模式是按
    剧本造出来的（dQ 恒为 +38.1），拿它当「症状」，闭环就是在给自己编的病历开药。
    读不到 result.json 时保守当作非假（旧产物里没有这个字段，不能因此把真实证据丢了）。
    """
    from .io_util import read_json
    data = read_json(Path(run_dir) / "result.json", default={}) or {}
    return bool(data.get("fake"))


def from_run_dirs(probe_dir: str | Path, *, arm: str = "base",
                  cfg: Settings | None = None,
                  exclude_task_ids: Iterable[str] = (),
                  use_score: bool = True,
                  allow_fake: bool = False) -> tuple[list[Finding], dict]:
    """把一批 run 的诊断结果聚合成一份 findings + summary。

    ``exclude_task_ids``：**泄漏防线**。holdout 任务只允许出现在「泛化证据」里，
    一旦出现进诊断池，闭环就会针对泛化集开药方，那条 ΔQ 立刻一文不值。
    这里按 ``score.task_id``（回落到 trace 里的 goal/query 无法判定，则按目录名）排除。

    ``allow_fake``：整轮都是假跑时（selftest / 管线自检）才设 True。
    默认 False 是故意的：假轨迹不能驱动真实轮次的诊断，但也不能让
    「拿假跑验管线」这条路直接变成「没有诊断输入」—— 那个情况下闭环会空转。
    """
    cfg = cfg or CFG
    banned = {str(x) for x in (exclude_task_ids or ())}
    findings: list[Finding] = []
    n_runs = n_scored = n_excluded = n_fake = 0
    task_ids: list[str] = []

    for rd in iter_run_dirs(probe_dir, arm):
        if not allow_fake and is_fake_run(rd):
            n_fake += 1
            continue
        tr, sc = load_run(rd)
        tid = str((sc.task_id if sc else "") or rd.name.rsplit("-r", 1)[0])
        if tid in banned:
            n_excluded += 1
            continue
        n_runs += 1
        task_ids.append(tid)
        score = sc if (use_score and sc is not None) else None
        if score is not None:
            n_scored += 1
        findings.extend(diagnosis.classify(tr, cfg, score=score))

    summary = diagnosis.summarize(findings, cfg)
    summary.update({
        "source": "round_traces",          # 让报告能证明「诊断吃的是本轮产物」
        "arm": arm,
        "n_runs": n_runs,
        "n_scored": n_scored,
        "n_tasks": len(set(task_ids)),
        "holdout_runs_excluded": n_excluded,
        "fake_runs_excluded": n_fake,
        "task_ids": sorted(set(task_ids)),
    })
    return findings, summary


def from_probe_outcome(probe_dir: str | Path, *, cfg: Settings | None = None,
                       exclude_task_ids: Iterable[str] = (),
                       arm: str = "base",
                       allow_fake: bool = False) -> tuple[list[Finding], dict]:
    """一轮 probe 后的诊断入口。

    默认读 **base 臂**：候选还没被接受时，cand 臂的行为属于「尚未上线的假设」，
    拿它当诊断输入会让「候选自己声称要修的问题」反过来决定下一个候选 ——
    这是自我证成。base 臂才是「当前生产 harness 的真实症状」。
    """
    return from_run_dirs(probe_dir, arm=arm, cfg=cfg, exclude_task_ids=exclude_task_ids,
                         allow_fake=allow_fake)


def write_diagnosis(out_dir: str | Path, findings: list[Finding], summary: dict,
                    cfg: Settings | None = None) -> Path:
    """落盘本轮诊断（供复核 / 供下一轮 provenance 检查）。"""
    from .io_util import write_json
    cfg = cfg or CFG
    per_task: dict[str, dict] = {}
    for f in findings:
        per_task.setdefault(f.taxonomy, {"count": 0, "agents": set(), "sample": ""})
        per_task[f.taxonomy]["count"] += 1
        if f.agent:
            per_task[f.taxonomy]["agents"].add(f.agent)
        if not per_task[f.taxonomy]["sample"]:
            per_task[f.taxonomy]["sample"] = (f.reason or "")[:160]
    readable = {k: {"count": v["count"], "agents": sorted(v["agents"]),
                    "sample": v["sample"]} for k, v in per_task.items()}
    payload = {"findings": [f.to_dict() for f in findings],
               "summary": summary, "by_taxonomy": readable,
               "unknown_max": cfg.unknown_max}
    return write_json(Path(out_dir) / "diagnosis.json", payload)


def is_usable(summary: dict) -> tuple[bool, str]:
    """诊断是否足以驱动提案：没有轨迹 / UNKNOWN 爆表 都不该继续。"""
    if not summary.get("n_runs"):
        return False, "本轮没有任何可诊断的 run（probe 未产出 trace.json）"
    if summary.get("hint"):
        return False, str(summary["hint"])
    return True, ""


def recent_trace_dirs(limit: int = 1) -> list[Path]:
    """在 ``experiments/`` 下按 mtime 找最近跑出的 probe 目录（其下有 ``base/``）。

    只给单轮命令 ``round`` 用：它需要在不跑 bootstrap 的前提下拿到「闭环自己跑过的
    症状」。找不到时调用方应退回人工日志，并把 ``source`` 标成 ``manual_log`` ——
    让报告能看出这一轮的诊断不是自洽产物。
    """
    from .settings import PATHS

    roots = [PATHS.runs, PATHS.experiments / "benchmark",
             PATHS.experiments / "auto", PATHS.experiments / "calibration"]
    found: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        if not r.exists():
            continue
        for base_arm in r.rglob("base"):
            if not base_arm.is_dir() or base_arm in seen:
                continue
            real = [p for p in base_arm.iterdir()
                    if p.is_dir() and (p / "trace.json").exists() and not is_fake_run(p)]
            if not real:          # 全是假跑 ⇒ 不能作为真实轮次的诊断输入
                continue
            seen.add(base_arm)
            found.append(base_arm.parent)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:max(1, limit)]
