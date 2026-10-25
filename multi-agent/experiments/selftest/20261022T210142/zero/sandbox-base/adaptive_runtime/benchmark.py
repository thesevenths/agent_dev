"""连续多轮 benchmark：回答「闭环跑起来之后，它真的变强了吗」。

为什么单轮 round 不够
----------------------
单轮只能回答「这一个候选在这一批任务上有没有显著收益」。用户真正要问的是
「这套 RRSI 机制能不能让 agent 自己越跑越好」。这需要两个额外证据：

1. **提升曲线**：第 1..k 轮的 ΔQ 是什么形状。若第 2 轮起就一直是 WITHIN_NOISE，
   说明搜索空间已被穷举或诊断信号不足 —— 这也是结论，而且是重要结论。
2. **holdout 平稳性**：在**没有 promote** 的轮次里，holdout 任务的 base Q 应当保持平稳。
   若它自己涨了，说明有泄漏（例如 holdout 任务参与了诊断/提案/经验写入），
   那么所有「提升」都不可信。这是最容易自欺的一环，所以必须每轮都测。

诚实边界（不要把这些说成已实现）
-------------------------------
- ``promote`` 是人工门（``version.promote`` 要求非 system 操作者）。因此**自动化闭环本身
  不会改 base 版本**；benchmark 的 base Q 只在人工 promote 之后才应发生变化。
- 本模块因此把「未 promote 期间 holdout 不动」当作正确行为来断言，
  而不是期待它自动上升。
- 多轮真实运行的 token 成本 ≈ rounds × n × 2 × 单run tokens，跑之前先估。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import diagnosis, evolution, proposer, version
from .comparator import summarize as cmp_summary
from .diagnosis import Taxonomy
from .evaluator import judge_version
from .experience import Store
from .gate import Decision, explain, gate_a, gate_b
from .io_util import now_stamp, write_json
from .ledger import Ledger
from .probe import ProbeOutcome, cleanup, probe_candidate
from .settings import CFG, PATHS, Settings
from .trace_export import export_from_log


# ---------------------------------------------------------------- 任务划分

def _rank_key(task: dict) -> str:
    """确定性排序键：用 task_id 的 sha256，不用 random，保证同一输入同样的划分。"""
    return hashlib.sha256(str(task.get("task_id", "")).encode("utf-8")).hexdigest()


def split(tasks: list[dict], holdout_ratio: float = 0.3,
          min_holdout: int = 1) -> tuple[list[dict], list[dict]]:
    """按 id 哈希切 train / holdout。

    为什么不用「前 N 个当 train、后 M 个当 holdout」：任务集来自历史日志，天然按时间有序，
    顺序切分会把「新任务」全放进 holdout，于是泛化差距里混入了时间漂移。
    """
    if holdout_ratio <= 0:
        return list(tasks), []
    ordered = sorted(tasks, key=_rank_key)
    k = max(min_holdout, round(len(ordered) * holdout_ratio))
    k = min(k, max(0, len(ordered) - 1))     # train 至少要留 1 个
    if k <= 0:
        return ordered, []
    return ordered[k:], ordered[:k]


# ---------------------------------------------------------------- 单轮结果

@dataclass
class RoundResult:
    round_no: int
    dominant: str = ""
    unknown_ratio: float = 0.0
    target: str = ""
    level: int = 0
    patch_hash: str = ""
    decision: str = "SKIP"
    reason: str = ""
    n_tasks: int = 0
    mean_delta: float = 0.0
    delta_crit: float = 0.0
    p_value: float = 1.0
    tokens: int = 0
    holdout_base_q: float = 0.0
    holdout_n: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _newest_log() -> Path | None:
    logs = sorted(PATHS.log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=True) if PATHS.log_dir.exists() else []
    return logs[0] if logs else None


# ---------------------------------------------------------------- 主流程

def run_benchmark(tasks: list[dict], *, rounds: int = 3, n: int = 4,
                  holdout_ratio: float = 0.3, level: int | None = None,
                  trace: str | Path | None = None, out_dir: str | Path | None = None,
                  task_label: str = "bench", fake_profiles: dict[str, str] | None = None,
                  keep_sandbox: bool = False,
                  cfg: Settings | None = None) -> dict:
    """跑 rounds 轮闭环，返回曲线与结论。不写 base 版本（promote 是人工门）。"""
    cfg = cfg or CFG
    train, holdout = split(tasks, holdout_ratio)
    if not train:
        return {"error": "train 集为空，无法开始"}

    out = Path(out_dir) if out_dir else (PATHS.experiments / "benchmark" /
                                         f"{task_label}-{now_stamp()}")
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger()
    store = Store()
    results: list[RoundResult] = []
    base_before = version.current()
    judge = judge_version(cfg)

    for r in range(1, rounds + 1):
        rr = RoundResult(round_no=r)

        # ---- 诊断（真实模式用最新日志；fake 模式没有新日志时复用同一份）----
        log = Path(trace) if trace else _newest_log()
        if not log or not Path(log).exists():
            rr.notes.append("没有可用于诊断的日志，跳过本轮")
            results.append(rr)
            continue
        tr = export_from_log(log)
        _findings, dsum = diagnosis.from_trace(tr)
        rr.dominant = dsum["dominant"]
        rr.unknown_ratio = float(dsum.get("unknown_ratio") or 0.0)

        # ---- 提案 ----
        plan = evolution.evolve_plan(ledger.rows, rr.dominant)
        cands = proposer.propose(rr.dominant, ledger, cfg=cfg,
                                 exclude_targets=plan["exclude_targets"],
                                 prefer_pair=plan["forced_pair"])
        if level is not None:
            cands = [c for c in cands if c.level == level]
        if not cands:
            rr.decision = "SKIP"
            rr.reason = "no_candidate"
            rr.notes.append("搜索空间已穷举或全部在台账里 —— 这本身就是结论")
            results.append(rr)
            continue
        cand = cands[0]
        cand.base_commit = base_before.base_commit if base_before else ""
        rr.target, rr.level, rr.patch_hash = cand.target, cand.level, cand.patch_hash

        # ---- Gate A ----
        from . import frozen as frozen_mod
        dec_a = gate_a(cand, frozen_mod.verify(), proposer.leakage_hits(cand.diff_text()))
        if not dec_a.accepted:
            rr.decision, rr.reason = dec_a.decision, dec_a.reason
            ledger.update(cand.hyp_fingerprint, "rejected",
                          {"reason": dec_a.reason, "stage": "gate_a"})
            results.append(rr)
            continue

        # ---- 探针（train 集）----
        try:
            po: ProbeOutcome = probe_candidate(
                cand, train, out / f"round{r}", cfg=cfg, n=n,
                keep_sandbox=keep_sandbox, fake_profiles=fake_profiles)
        except Exception as e:                      # 单轮失败不应中断整条曲线
            rr.notes.append(f"probe 失败：{type(e).__name__}: {e}")
            results.append(rr)
            continue

        rr.tokens = po.tokens
        if po.comparison is None:
            rr.notes.append("没有配对结果（两臂未齐）")
            cleanup(po)
            results.append(rr)
            continue

        ev = po.comparison.evidence(cfg)
        pr = ev.paired(cfg.alpha)
        rr.n_tasks = len(ev.task_deltas)
        rr.mean_delta, rr.delta_crit, rr.p_value = pr.mean, pr.delta_crit, pr.p_value

        dec: Decision = gate_b(ev, cfg)
        # Stage 1 止损优先（与 round 保持一致）
        if po.stopped == "stop_loss":
            rr.decision, rr.reason = "REJECT", "regression"
            rr.notes.append("Stage 1 止损")
        else:
            rr.decision, rr.reason = dec.decision, dec.reason

        if dec.accepted and po.stopped != "stop_loss":
            negative = [k for k, v in po.comparison.per_task.items()
                        if v.get("mean_delta", 0.0) < -cfg.regression_min_points]
            store.create(
                problem=f"{cand.taxonomy}|{cand.target}|0",
                claim=cand.hypothesis, component=f"L{cand.level}/{cand.target}",
                edit={"patch_hash": cand.patch_hash, "old": cand.old, "new": cand.new},
                n_pairs=pr.n, n_target=cfg.n_default, p_value=pr.p_value,
                regression_rate=ev.regression_rate, dQ=round(pr.mean, 4), dC=round(ev.dC, 4),
                positive=[t["task_id"] for t in train], negative=negative,
                judge_version=judge, hypothesis_id=cand.hyp_fingerprint)
        ledger.update(cand.hyp_fingerprint,
                      "accepted" if dec.accepted and po.stopped != "stop_loss"
                      else ("inconclusive" if rr.reason == "within_noise" else "rejected"),
                      {"n": pr.n, "dQ": round(pr.mean, 4), "p": round(pr.p_value, 6),
                       "round": r, "reason": rr.reason},
                      tasks=[t["task_id"] for t in train])

        # ---- holdout 平稳性（未 promote 时不该动）----
        if holdout:
            hq = _holdout_base_quality(holdout, out / f"round{r}", cfg,
                                       fake_profiles=fake_profiles)
            rr.holdout_base_q, rr.holdout_n = hq[0], hq[1]
        cleanup(po)
        results.append(rr)

    base_after = version.current()
    report = _summarize(results, holdout, base_before, base_after, judge)
    write_json(out / "benchmark.json",
               {"curve": [r.to_dict() for r in results], **report,
                "train_ids": [t["task_id"] for t in train],
                "holdout_ids": [t["task_id"] for t in holdout]})
    (out / "benchmark.md").write_text(_render(report, results), encoding="utf-8")
    report["out_dir"] = str(out)
    return report


def _holdout_base_quality(tasks: list[dict], out: Path, cfg: Settings,
                          fake_profiles: dict[str, str] | None) -> tuple[float, int]:
    """只跑 base 臂，用于观察「未 promote 的轮次里 holdout 有没有自己动」。"""
    from .probe import run_one
    from .probe import create_sandbox

    sb = create_sandbox(out / "holdout-base", "base", keep=False)
    qs: list[float] = []
    for i, task in enumerate(tasks):
        ro = run_one(sb, task, 0, "base", out / "holdout" / f"{task['task_id']}-r0",
                     cfg=cfg, fake_profile=(fake_profiles or {}).get("base"))
        if ro.ok:
            qs.append(ro.Q)
    return (_mean(qs), len(qs))


def _summarize(results: list[RoundResult], holdout: list[dict],
               base_before, base_after, judge: str) -> dict:
    accepted = [r for r in results if r.decision == "ACCEPT"]
    rejected = [r for r in results if r.decision == "REJECT"]
    skipped = [r for r in results if r.decision == "SKIP"]
    deltas = [r.mean_delta for r in results if r.n_tasks]
    holdout_qs = [r.holdout_base_q for r in results if r.holdout_n]

    promoted = bool(base_before and base_after and
                    base_before.version != base_after.version)
    drift = 0.0
    if len(holdout_qs) >= 2:
        drift = holdout_qs[-1] - holdout_qs[0]

    verdict = []
    verdict.append(f"跑完 {len(results)} 轮：ACCEPT {len(accepted)} / REJECT {len(rejected)} "
                   f"/ SKIP {len(skipped)}")
    if deltas:
        verdict.append(f"ΔQ 序列：{['%+.1f' % d for d in deltas]}")
    if accepted:
        verdict.append("至少有一个改动真正过了噪声门 —— 闭环能产出有效结论")
    elif results:
        verdict.append("没有任何改动过门：要么搜索空间已穷举，要么诊断信号不足（看 UNKNOWN 占比）")
    if holdout and not promoted:
        if abs(drift) <= CFG.regression_min_points:
            verdict.append(f"holdout 平稳（drift={drift:+.2f}），未 promote 期间没有泄漏或漂移")
        else:
            verdict.append(f"⚠ holdout 漂移 {drift:+.2f}，但版本未 promote —— "
                           "存在泄漏或非确定性干扰，本轮结论不可信")
    if promoted:
        verdict.append(f"base 版本 {base_before.version}→{base_after.version}（人工 promote），"
                       "此时 holdout 上升才算泛化证据")

    return {"judge_version": judge, "rounds": len(results),
            "accepted": len(accepted), "rejected": len(rejected), "skipped": len(skipped),
            "delta_series": [round(d, 4) for d in deltas],
            "holdout_series": [round(q, 2) for q in holdout_qs],
            "holdout_drift": round(drift, 4),
            "promoted": promoted,
            "base_version": [getattr(base_before, "version", None),
                             getattr(base_after, "version", None)],
            "verdict": verdict}


def _render(report: dict, results: list[RoundResult]) -> str:
    lines = ["# Benchmark 报告", "",
             f"- 判分口径：`{report['judge_version']}`",
             f"- 轮数：{report['rounds']}（ACCEPT {report['accepted']} / "
             f"REJECT {report['rejected']} / SKIP {report['skipped']}）",
             f"- ΔQ 序列：{report['delta_series']}",
             f"- holdout Q 序列：{report['holdout_series']}（drift {report['holdout_drift']:+.2f}）",
             "", "## 结论", ""]
    lines += [f"- {v}" for v in report["verdict"]]
    lines += ["", "## 逐轮", "",
              "| 轮 | dominant | 候选 | L | 决策 | n | ΔQ | δ | p | tokens |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r.round_no} | {r.dominant or '-'} | `{r.target or '-'}` | "
                     f"{r.level or '-'} | {r.decision}/{r.reason} | {r.n_tasks} | "
                     f"{r.mean_delta:+.2f} | {r.delta_crit:.2f} | {r.p_value:.3f} | "
                     f"{r.tokens:,} |")
    for r in results:
        if r.notes:
            lines.append(f"\n- 第 {r.round_no} 轮备注：{'; '.join(r.notes)}")
    lines += ["", "## 怎么读这份报告", "",
              "1. **ΔQ 序列**才是「有没有在变强」的证据。全为正且逐步过门 = 有效；"
              "全在噪声内 = 搜索空间或诊断信号已经不够了。",
              "2. **holdout drift** 在未 promote 时应≈0。若明显漂移，先查泄漏，别急着庆祝提升。",
              "3. **单轮 ACCEPT 不等于累积提升**。累积要靠 promote 换基线，再重跑本 benchmark 对比。"]
    return "\n".join(lines) + "\n"
