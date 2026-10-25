"""自动换基线：用「比人工更硬的证据」替代人工门，而不是取消门槛。

人工门原本在防什么
------------------
``version.promote`` 禁止 system 操作者，是因为**无锚漂移**：如果闭环可以拿
「上一轮候选」当新基线，那么 Q 的正向微小噪声会被逐轮累积成「看起来一直在进步」，
而任何一次负噪声也会被后续轮次当成新常态。这是 RRSI 类系统最典型的失效模式 ——
不是不改进，而是**改进的度量本身被改进动作污染**。

要满足「无人工干预」，唯一诚实的做法不是拆掉这道门，而是把它换成一组**机器可判定、
且比人工评审更难通过**的条件。本模块就是那组条件，逐条留痕：

- ``calibration``   标定必须**实测通过**：δ 来自真实 sd(Δ) 而不是假设值。
  没有它，「过门」只是「碰巧大于一个拍脑袋的数」。
- ``evolve_gate``   候选在演化集上通过配对门（Gate B accepted，且未触发止损）。
- ``holdout_evidence`` holdout 上必须**真跑了配对**（≥2 个任务），
  只看 holdout 均值不叫泛化证据 —— 那是「两次不同的噪声」。
- ``holdout_no_regression`` holdout 配对 mean(Δ) > -holdout_tol：泛化层不许退步。
- ``regression_gate`` 回归集若被覆盖则必须无回归；**未覆盖只告警不阻断**
  （首轮 probe 只覆盖 train 任务，要求覆盖会让任何机制都上不了线）。
- ``frozen_intact`` Frozen Core 完整：候选没碰判分口径。
- ``promote_budget`` 单次运行内自动 promote 次数上限（跑飞保险）。

任何一条不满足 → 不升基线，候选进台账（accepted 但不 promote 是不一致状态，
所以这里把「升基线」当作 accept 的唯一出口：policy 不过就把决策降级为 HOLD）。
"""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from . import landing, version
from .io_util import now_iso, write_json
from .settings import CFG, PATHS, Settings


@dataclass
class Check:
    name: str
    passed: bool
    blocking: bool = True
    detail: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed:
            return "ok"
        return "BLOCK" if self.blocking else "warn"

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "blocking": self.blocking,
                "status": self.status, **self.detail}


@dataclass
class Policy:
    allowed: bool
    checks: list[Check] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "blocking": self.blocking,
                "warnings": self.warnings, "checks": [c.to_dict() for c in self.checks]}


def calibration_ok(cfg: Settings | None = None) -> tuple[bool, dict]:
    """标定是否**实测通过**（存在 + passed + 带真实 sd_diff_null + 报表自认为可信）。

    ``validity`` 非空 = 样本退化（比如所有 run 都因为同一个网络故障而拿到同一个地板分），
    那种报表里 sd=0 会把 δ 拉成 0，于是“任何正数的 ΔQ 都显著”。
    """
    from .io_util import read_json
    cal = read_json(PATHS.calibration, default=None) or {}
    detail = {"exists": bool(cal), "passed": bool(cal.get("passed")),
              "sd_diff_null": cal.get("sd_diff_null"), "k": cal.get("k"),
              "validity": cal.get("validity") or []}
    ok = (bool(cal) and bool(cal.get("passed"))
          and cal.get("sd_diff_null") not in (None, 0)
          and not cal.get("validity"))
    return ok, detail


def evaluate(*, cand: Any, dec_b_accepted: bool, stopped: str = "",
             holdout_cmp: Any = None, regression: dict | None = None,
             frozen_violations: list | None = None, promotes_done: int = 0,
             cfg: Settings | None = None) -> Policy:
    """把上面那组条件算成一份可审计的判定表。"""
    cfg = cfg or CFG
    checks: list[Check] = []

    cal_ok, cal_detail = calibration_ok(cfg)
    checks.append(Check("calibration", cal_ok, blocking=True, detail=cal_detail))

    checks.append(Check("evolve_gate", bool(dec_b_accepted) and stopped != "stop_loss",
                        blocking=True,
                        detail={"gate_b": bool(dec_b_accepted), "stopped": stopped,
                                "target": getattr(cand, "target", "")}))

    # ---- holdout 配对证据 ----
    h_n = 0
    h_mean = None
    if holdout_cmp is not None:
        pairs = getattr(holdout_cmp, "pairs", [])
        h_n = len(getattr(holdout_cmp, "per_task", {}) or {})
        if pairs:
            h_mean = round(sum(p.delta for p in pairs) / len(pairs), 4)
    checks.append(Check("holdout_evidence", h_n >= cfg.min_tasks_for_t, blocking=True,
                        detail={"holdout_tasks": h_n, "requires": cfg.min_tasks_for_t}))

    tol = cfg.holdout_tol
    h_ok = (h_mean is not None) and (h_mean > -tol)
    checks.append(Check("holdout_no_regression", bool(h_ok), blocking=True,
                        detail={"holdout_mean_delta": h_mean, "tol": -tol}))

    # ---- 回归集（未覆盖 → 告警，不阻断）----
    reg = regression or {}
    if reg.get("covered"):
        checks.append(Check("regression_gate", bool(reg.get("ok")), blocking=True,
                            detail={"regression_rate": reg.get("regression_rate"),
                                    "regressed": reg.get("regressed"), "r_max": reg.get("r_max")}))
    else:
        # passed=False + blocking=False ⇒ status="warn"，于是它会出现在 policy.warnings 里。
        # 写成 passed=True 的话，「没测」在汇总层面等于「测过且干净」——那是把盲区报成安全区。
        checks.append(Check("regression_gate", False, blocking=False,
                            detail={"covered": False, "note": "本轮未覆盖回归集 → 没测不等于没坏"}))

    v = frozen_violations or []
    checks.append(Check("frozen_intact", not v, blocking=True,
                        detail={"violations": [str(x) for x in v][:8]}))

    budget_ok = promotes_done < cfg.auto_promote_max
    checks.append(Check("promote_budget", budget_ok, blocking=True,
                        detail={"promotes_done": promotes_done,
                                "max": cfg.auto_promote_max}))

    blocking = [c.name for c in checks if not c.passed and c.blocking]
    warnings = [c.name for c in checks if not c.passed and not c.blocking]
    return Policy(allowed=not blocking, checks=checks, blocking=blocking, warnings=warnings)


def promote_and_land(cand: Any, *, policy: Policy, holdout_summary: dict,
                     note: str = "", cfg: Settings | None = None,
                     root: "Path | None" = None) -> dict:
    """升基线 + 落地到生产 + 落地后自检；任何越界立刻整体回滚。

    顺序是刻意的：**先升版本再落地**，因为落地记录需要写上 ``base/vN`` 标签；
    而落地后必须立刻复查 Frozen Core —— 万一候选碰了判分口径，回滚要在这一次调用里
    完成，不能留给下一轮（那时它已经污染了一次真实运行）。
    """
    cfg = cfg or CFG
    if not policy.allowed:
        raise RuntimeError(f"policy 未通过，拒绝自动 promote：{policy.blocking}")
    prev = version.current()
    evidence = [dict(policy.to_dict()), {"holdout": holdout_summary},
                {"candidate": getattr(cand, "to_dict", lambda: {})()}]
    new = version.promote(evidence, operator="auto:adaptive",
                          note=note or f"自动 promote：{getattr(cand, 'target', '')}",
                          require_operator=False)
    landed: dict = {}
    if cfg.land_changes:
        landed = landing.land(cand, version_tag=new.tag, root=root)
        integrity = landing.check_integrity(root=root)
        if integrity:
            landing.rollback_to(prev.tag if prev else "base/v0", root=root)
            version.restore(prev, note=f"落地后 Frozen/锚点越界，自动回滚：{integrity[:5]}"
                            if prev else "落地后越界，回退至无基线")
            write_json(PATHS.experiments / "landing" / f"aborted_{new.tag.replace('/', '_')}.json",
                       {"ts": now_iso(), "integrity": integrity, "landed": landed})
            raise RuntimeError(f"落地后自检越界，已整体回滚：{integrity}")
    record = {"ts": now_iso(), "event": "auto_promote", "tag": new.tag,
              "target": getattr(cand, "target", ""), "level": getattr(cand, "level", 0),
              "patch_hash": getattr(cand, "patch_hash", ""),
              "content_stamp": landing.content_stamp(), "landed": landed,
              "holdout": holdout_summary, "policy": policy.to_dict()}
    if landed:
        # 落地完才能重采 env 快照：不重采，base/vN 记的就是改动**前**的配置
        stamp = record["content_stamp"]
        version.refresh_snapshot(new, base_commit=f"{version.git_head()}+landed:{stamp}"
                                 if version.git_head() else f"landed:{stamp}")
    write_json(PATHS.experiments / "landing" / f"promote_{new.tag.replace('/', '_')}.json", record)
    if cfg.auto_commit and cfg.land_changes:
        record["commit"] = _git_commit_landed(new.tag, getattr(cand, "target", ""))
    return record


def verify_after_promote(*, holdout_prev_mean: float, holdout_new_mean: float,
                         tol: float, prev_tag: str,
                         prev_by_task: dict[str, float] | None = None,
                         new_by_task: dict[str, float] | None = None,
                         cfg: Settings | None = None) -> dict:
    """换基线后的回归检查：新 base 在**同一批 holdout 任务**上若明显更差 → 该回滚。

    这是自动闭环里唯一能区分「真变强」与「换了把尺子」的观测：promote 前后各测一次
    holdout，用的任务、判分口径、离线模式都相同，唯一变化就是那次落地改动。

    为什么有了配对输入就不能只看两个均值的差：holdout 只有几个任务，两个独立均值的差
    自带 sd ≈ sd(Δ)/√n ≈ 5 分，而 tol 也是 5 分 —— 于是「一次正常的噪声」就会触发回滚，
    把已经过门的改进扔掉。误回滚比漏回滚更伤闭环：它同时打掉 ``auto_promote_happened``
    与 ``no_unresolved_demote`` 两条硬判据。以同一任务为单位配对后，任务难度被完全抵消，
    剩下的才是改动的效应；此时要求「幅度超 tol 且配对检验显著」两件事同时成立才回滚。
    """
    cfg = cfg or CFG
    drift = round(holdout_new_mean - holdout_prev_mean, 4)
    out = {"prev_tag": prev_tag, "prev_mean": round(holdout_prev_mean, 4),
           "new_mean": round(holdout_new_mean, 4), "drift": drift, "tol": -tol}

    common = sorted(set(prev_by_task or {}) & set(new_by_task or {}))
    if len(common) >= max(2, cfg.min_tasks_for_t):
        from .stats import paired_stats
        deltas = [float(new_by_task[t]) - float(prev_by_task[t]) for t in common]
        pr = paired_stats(deltas, alpha=cfg.alpha)
        sig_worse = pr.n >= 2 and pr.mean < -pr.delta_crit
        out.update({"mode": "paired", "paired_n": pr.n,
                    "paired_mean": round(pr.mean, 4),
                    "paired_delta_crit": round(pr.delta_crit, 4),
                    "significant_regression": bool(sig_worse),
                    "tasks": common})
        out["ok"] = not (drift < -tol and sig_worse)
        return out

    # 没配对输入（旧调用方）或共同任务不足：沿用纯幅度判据，并在记录里说清证据等级。
    out.update({"mode": "means_only", "common_tasks": len(common),
                "note": "共同 holdout 任务不足以配对 → 只看均值差，证据等级低于配对"})
    out["ok"] = drift > -tol
    return out


def rollback(base_prev: Any, *, reason: str, cfg: Settings | None = None,
             root: "Path | None" = None) -> dict:
    """整体回滚：撤销落地改动 + Base 退回上一版本。"""
    cfg = cfg or CFG
    prev_tag = base_prev.tag if base_prev is not None else "base/v0"
    dropped = landing.rollback_to(prev_tag, root=root)
    restored = version.restore(base_prev, note=f"自动回滚：{reason}") if base_prev else None
    out = {"ts": now_iso(), "event": "auto_demote", "to_version": prev_tag,
           "reason": reason, "landing": dropped,
           "restored": restored.to_dict() if restored else None}
    write_json(PATHS.experiments / "landing" / f"demote_{now_iso()[:19].replace(':', '')}.json", out)
    return out


def _git_commit_landed(tag: str, target: str) -> str:
    """可选：把落地产物提交（仅在 ADAPTIVE_AUTO_COMMIT=1 时执行）。

    只 add 明确的两个文件，不用 ``-A``：闭环无权把用户工作区里的其它改动打包提交。
    """
    try:
        subprocess.run(["git", "add", "--", ".env", "prompt.py"], cwd=str(PATHS.root),
                       capture_output=True, text=True, timeout=60, check=True)
        p = subprocess.run(["git", "commit", "-m",
                            f"adaptive runtime: land {tag} ({target})"],
                           cwd=str(PATHS.root), capture_output=True, text=True, timeout=60)
        return (p.stdout or "").strip()[:200] if p.returncode == 0 else f"commit failed: {p.stderr[:200]}"
    except (OSError, subprocess.SubprocessError) as e:
        return f"commit skipped: {type(e).__name__}: {e}"
