"""无人值守闭环：``诊断 → 提案 → Gate A → train 探针 → Gate B → holdout 配对 → 自动换基线 → 落地 → 复查``。

这个模块存在的唯一理由
----------------------
在此之前，闭环每跑一轮都要**人来按 promote**。那不是自我改进，那是「自动化的人工评审」：
证据收集是机器做的，但「改进是否成为新基线」这个决定是人做的，于是系统在两次人工之间
原地踏步 —— 而 ``benchmark`` 里那句「未 promote 期间 holdout 应当平稳」恰好说明了后果：
不 promote，改进就永远只在探针里成立，从没在真实任务上生效过。

``auto`` 把这道人工门换成 ``autopromote.evaluate`` 的一组机器判据（标定实测通过 +
evolve 配对过门 + holdout 配对不退 + 回归集不命中 + Frozen 完整 + 限次 + 可回滚）。
注意这不是「拆门」：人工门防的是**无锚漂移**，而机器判据防的是同一件事，只是判据更硬、
且每次判定都落盘可复核。

跑完必须能回答「怎么知道它实现了 RRSI」
--------------------------------------
``assess()`` 把达成判据算成一张表并写 ``RRSI_VERDICT.md``。判据不是「跑了很多轮」这种
工作量指标，而是**闭环性质**的观测：

- 自洽性：每一轮的诊断只吃本轮自己跑出来的轨迹（``diag_source == round_traces``）；
- 泛化层干净：holdout 任务 id 从未进入诊断/提案（机器排除，写进 diagnosis.json 可查）；
- 无人：本次运行窗口内的 promote 事件全部 ``created_by=auto:*``，没有一次人工签字；
- 真落地：生产内容指纹 ``landing.content_stamp()`` 因这次自动改进而改变，且自检无越界；
- 真的更强：holdout 配对 mean(Δ) > 0，且换基线后重测 holdout 的漂移 > -tol；
- 尺子可信：噪声是**实测标定**过的（否则 δ 是假设值，上面所有「显著」都不成立）；
- 可撤销：出现退步时 demote 事件存在，且最终状态不是「带着一个坏基线继续跑」。

任何一条不成立，报告直接写「未达成 + 缺哪一条」，不粉饰。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import autopromote, benchmark, diag_batch, frozen, landing, version
from .io_util import now_iso, read_json, write_json
from .settings import CFG, PATHS, Settings, validate

STATE_NAME = "state.json"
CURVE_NAME = "learning_curve.json"
VERDICT_NAME = "RRSI_VERDICT.md"

#: 达成判据里哪些是「不满足就没实现」的硬条件（其余为观察项）
BLOCKING_CRITERIA = ("noise_calibrated", "loop_self_contained", "holdout_quarantined",
                     "auto_promote_happened", "no_human_promote_in_run",
                     "behavior_actually_changed", "holdout_paired_gain",
                     "post_promote_stable", "no_unresolved_demote")


# ---------------------------------------------------------------- 运行目录与状态

def _dir(label: str, out_dir: str | Path | None = None) -> Path:
    root = Path(out_dir) if out_dir else (PATHS.experiments / "auto")
    p = root / label
    p.mkdir(parents=True, exist_ok=True)
    return p


def _task_hash(tasks: list[dict]) -> str:
    """任务集指纹：续跑必须确认「还是同一批任务」，否则曲线上的轮次不可比。"""
    blob = "|".join(sorted(str(t.get("task_id")) for t in tasks)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _state_path(out: Path) -> Path:
    return out / STATE_NAME


def load_state(out: Path) -> dict:
    return read_json(_state_path(out), default=None) or {}


def save_state(out: Path, state: dict) -> None:
    state["updated_at"] = now_iso()
    write_json(_state_path(out), state)


# ---------------------------------------------------------------- 自动换基线钩子

@dataclass
class PromoteTally:
    """本次运行内的换基线情况（``promotes_done`` 要给 policy 当限次保险用）。"""
    promotes: int = 0
    demotes: int = 0
    verifies: list[dict] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)


def make_on_accept(tally: PromoteTally, cfg: Settings, *, verify: bool = True,
                   fake_profiles: dict[str, str] | None = None,
                   keep_sandbox: bool = False):
    """生成 ``benchmark.run_benchmark(on_accept=...)`` 的回调。

    为什么 holdout 复查要**重跑**而不是复用探针里的数：落地前 holdout 的 cand 臂是在
    **沙箱**里跑的（补丁只打在副本上），落地后生产代码才等于候选。只有用新的生产
    沙箱重测一次同一批 holdout 任务，才真的回答「这次改动上线后，泛化集变好了吗」。
    """

    def handler(cand, rr, ctx: dict) -> dict:
        out = Path(ctx["out_dir"])
        violations = frozen.verify()
        policy = autopromote.evaluate(
            cand=cand, dec_b_accepted=bool(getattr(ctx["decision"], "accepted", False)),
            stopped=ctx.get("stopped", ""), holdout_cmp=ctx.get("holdout_comparison"),
            regression=ctx.get("regression"), frozen_violations=violations,
            promotes_done=tally.promotes, cfg=cfg)
        write_json(out / "policy.json", policy.to_dict())
        if not policy.allowed:
            tally.blocked.append({"round": ctx["round_no"], "blocking": policy.blocking,
                                  "target": getattr(cand, "target", "")})
            rr.notes.append("机器判据未通过，不升基线：" + ",".join(policy.blocking))
            return {"promoted": False, "blocked": policy.blocking}

        prev_base = version.current()
        holdout_summary = {
            "pairs": rr.holdout_pairs, "tasks": rr.holdout_n,
            "mean_delta": rr.holdout_mean_delta, "delta_crit": rr.holdout_delta_crit,
            "base_q": rr.holdout_base_q, "evolve_mean_delta": rr.mean_delta,
        }
        try:
            rec = autopromote.promote_and_land(cand, policy=policy,
                                               holdout_summary=holdout_summary, cfg=cfg)
        except RuntimeError as e:            # promote_and_land 内部越界已自行回滚
            tally.demotes += 1
            rr.notes.append(f"自动 promote 失败（已回滚）：{e}")
            return {"promoted": False, "error": str(e)}
        tally.promotes += 1
        result: dict[str, Any] = {"promoted": True, "tag": rec["tag"],
                                  "landed": bool(rec.get("landed")), "record": rec}

        if verify and ctx.get("holdout"):
            try:
                ho_tasks = ctx["holdout"]
                sweep = benchmark.base_sweep(ho_tasks, out / "post_promote", cfg,
                                             reps=cfg.holdout_reps,
                                             fake_profiles=fake_profiles,
                                             keep_sandbox=keep_sandbox)
                mean_new, n_new, by_task_new = _holdout_mean_from_probe(sweep)
                if n_new:
                    chk = autopromote.verify_after_promote(
                        holdout_prev_mean=float(rr.holdout_base_q or 0.0),
                        holdout_new_mean=mean_new, tol=cfg.holdout_tol,
                        prev_tag=(prev_base.tag if prev_base else "base/v0"),
                        prev_by_task=_holdout_prev_by_task(ctx.get("holdout_comparison")),
                        new_by_task=by_task_new, cfg=cfg)
                    chk.update({"round": ctx["round_no"], "tag": rec["tag"], "n": n_new})
                    tally.verifies.append(chk)
                    write_json(out / "post_promote_check.json", chk)
                    if not chk["ok"]:
                        autopromote.rollback(prev_base,
                                             reason=f"换基线后 holdout 漂移 {chk['drift']:+.2f} "
                                                    f"低于容忍 {-cfg.holdout_tol} 且配对检验显著"
                                                    f"（paired={chk.get('paired_mean')}"
                                                    f"±{chk.get('paired_delta_crit')}）", cfg=cfg)
                        tally.promotes -= 1
                        tally.demotes += 1
                        rr.notes.append(f"换基线后退步，已自动回滚（drift={chk['drift']:+.2f}）")
                        result.update({"rolled_back": True, "verify": chk, "promoted": False})
                    else:
                        result["verify"] = chk
                else:
                    rr.notes.append("换基线后 holdout 重测没有可用 run，无法验证漂移")
            except Exception as e:
                rr.notes.append(f"换基线后复查异常：{type(e).__name__}: {e}")
                result["verify_error"] = f"{type(e).__name__}: {e}"
        return result

    return handler


def _latest_benchmark(out: Path) -> dict:
    """找最后一批的 benchmark.json（续跑后旧路径不存在）。"""
    cands = sorted(out.glob("bench/**/benchmark.json"), key=lambda p: p.stat().st_mtime)
    return (read_json(cands[-1], default=None) or {}) if cands else {}


def _holdout_mean_from_probe(probe_dir: Path) -> tuple[float, int, dict[str, float]]:
    """读落地后 base 臂重测的 holdout：返回 ``(总均值, 可用 run 数, 按任务均值)``。

    第三个量是为了配对：跟 promote 前那轮 holdout 的 base 臂按同一 task_id 对齐后，
    任务难度抵消，剩下的才是落地改动本身的效应。
    """
    qs: list[float] = []
    by_task: dict[str, list[float]] = {}
    for rd in diag_batch.iter_run_dirs(probe_dir, "base"):
        _tr, sc = diag_batch.load_run(rd)
        if sc is None or getattr(sc, "Q", None) is None:
            continue
        qs.append(float(sc.Q))
        tid = str(getattr(sc, "task_id", "") or "").strip() or rd.parent.name
        by_task.setdefault(tid, []).append(float(sc.Q))
    return ((sum(qs) / len(qs)) if qs else 0.0, len(qs),
            {k: sum(v) / len(v) for k, v in by_task.items()})


def _holdout_prev_by_task(holdout_cmp: Any) -> dict[str, float]:
    """从 promote **前**的 holdout 配对探针里取 base 臂的任务级均值（无则空 dict）。"""
    acc: dict[str, list[float]] = {}
    for p in getattr(holdout_cmp, "pairs", None) or []:
        acc.setdefault(str(p.task_id), []).append(float(p.base_Q))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# ---------------------------------------------------------------- 主入口

def run_auto(tasks: list[dict], *, rounds: int = 3, n: int | None = None,
             holdout_ratio: float = 0.3, label: str = "auto",
             out_dir: str | Path | None = None, cfg: Settings | None = None,
             resume: bool = True, verify: bool = True,
             fake_profiles: dict[str, str] | None = None,
             keep_sandbox: bool = False, on_progress=None,
             force_dominant: str | None = None) -> dict:
    """跑（或续跑）无人值守闭环，返回 ``{"report", "state", "verdict"}``。

    续跑语义：``state.json`` 里记了 ``rounds_done`` 与任务集指纹。同一 label 且任务集
    未变时，本次只补跑「还差几轮」，曲线累加；任务集变了则**另起一条曲线**（旧的
    移到 ``learning_curve.json.history``），因为跨任务集的 ΔQ 不构成同一条学习曲线。
    """
    cfg = cfg or CFG
    problems = validate()
    out = _dir(label, out_dir)
    th = _task_hash(tasks)
    state = load_state(out) if resume else {}
    if state and state.get("task_hash") != th:
        state = {}                      # 任务集换了 ⇒ 旧曲线不可续
    #: 每实验隔离的假设台账：续跑载入已有记忆；全新跑（--no-resume）或任务集变更时清空，
    #: 避免跨实验污染，也保证「全新曲线」不带旧假设记忆（全局 ledger 这两点都做不到）。
    ledger_path = out / "ledger.jsonl"
    if (not resume or not state) and ledger_path.exists():
        ledger_path.unlink()
    done = int(state.get("rounds_done") or 0)
    if done >= rounds:
        _say = on_progress or (lambda m: None)
        _say(f"已完成 {done}/{rounds} 轮，无需续跑；直接出具判据")
        rep = read_json(out / "bench" / "benchmark.json", default=None) or _latest_benchmark(out)
        return {"report": rep, "state": state,
                "verdict": write_verdict(out, state)}

    to_run = max(0, rounds - done)
    #: 每一批续跑单独一个目录：benchmark 的轮次目录叫 round1..roundN，同一批里续跑会
    #: **覆盖上一轮的 diagnosis.json**，于是「哪一轮诊断吃了哪些 run」这份证据就丢了。
    bench_dir = out / "bench" / f"batch{done + 1}-{done + to_run}"
    # 续跑时必须把旧批的记账**带进来**：promotes/demotes 早就是带着的（限次保险要看累计值），
    # verifies/blocked 同样不能漏 —— 判据 8（post_promote_stable）读的是 state["verifies"]，
    # 若第 2 批没发生 promote 就把列表置空，那第 1 批「换基线后已复查通过」这份证据会被
    # 抹掉，判据 8 由绿转红。那是记账错误，不是实验结果。
    tally = PromoteTally(promotes=int(state.get("promotes", 0) or 0),
                         demotes=int(state.get("demotes", 0) or 0),
                         verifies=list(state.get("verifies") or []),
                         blocked=list(state.get("blocked") or []))
    content_before = str(state.get("content_stamp_before") or landing.content_stamp())
    started_at = state.get("started_at") or now_iso()
    if not state:
        state = {"label": label, "task_hash": th, "started_at": started_at,
                 "content_stamp_before": content_before,
                 "measure_mode": cfg.measure_mode()}

    report = benchmark.run_benchmark(
        tasks, rounds=to_run, n=n or cfg.n_default, holdout_ratio=holdout_ratio,
        out_dir=bench_dir, task_label=label, fake_profiles=fake_profiles,
        keep_sandbox=keep_sandbox, cfg=cfg,
        on_accept=make_on_accept(tally, cfg, verify=verify,
                                 fake_profiles=fake_profiles,
                                 keep_sandbox=keep_sandbox),
        ledger_path=ledger_path, force_dominant=force_dominant)
    if "error" in report:
        return {"error": report["error"], "state": state}

    # 续跑时轮次必须接着往后排：benchmark 每被调一次都从 1 重新编号，而学习曲线要的是
    # 「同一批任务上的第 1..N 轮」，两个 round 1 混在一起就没人能读出趋势。
    curve = list(state.get("curve") or []) + [
        dict(r, round_no=done + int(r.get("round_no") or 0)) for r in (report.get("curve") or [])]
    state.update({"rounds_done": done + to_run, "promotes": tally.promotes,
                  "demotes": tally.demotes, "verifies": tally.verifies,
                  "blocked": tally.blocked[-8:], "curve": curve,
                  "content_stamp_after": str(landing.content_stamp()),
                  "base_version": report.get("base_version"),
                  "holdout_ids": report.get("holdout_ids"),
                  "train_ids": report.get("train_ids"),
                  "last_report": {k: report.get(k) for k in
                                  ("delta_series", "holdout_delta_series", "holdout_drift",
                                   "diag_sources", "accepted", "rejected", "skipped")}})
    save_state(out, state)
    write_json(out / CURVE_NAME, {"label": label, "task_hash": th,
                                  "measure_mode": cfg.measure_mode(),
                                  "curve": curve,
                                  "promotes": tally.promotes, "demotes": tally.demotes})
    verdict = write_verdict(out, state)
    if on_progress:
        for line in report.get("verdict") or []:
            on_progress(line)
        on_progress(f"判据：{'达成' if verdict['achieved'] else '未达成'} "
                    f"（{verdict['passed_blocking']}/{verdict['n_blocking']} 硬条件）")
    return {"report": report, "state": state, "verdict": verdict, "problems": problems}


# ---------------------------------------------------------------- 达成判据

def _diagnosis_leakage(out: Path, holdout_ids: list[str]) -> tuple[bool, dict]:
    """逐轮核对 diagnosis.json：holdout 任务是否曾进入诊断池。"""
    banned = {str(x) for x in (holdout_ids or [])}
    seen: list[str] = []
    checked = 0
    for d in sorted(out.glob("bench/**/round*/diagnosis.json")):
        data = read_json(d, default=None) or {}
        sm = data.get("summary") or {}
        if sm.get("source") != "round_traces":
            continue
        checked += 1
        for tid in (sm.get("task_ids") or []):
            if str(tid) in banned:
                seen.append(f"{d.parent.name}:{tid}")
    return (not seen), {"rounds_checked": checked, "leaked": seen}


def assess(out: Path, state: dict, *, cfg: Settings | None = None) -> dict:
    """把「是否实现 RRSI」算成一张判据表（可从磁盘产物重放，不需要在跑闭环）。

    额外一条环境健康观察项：模型端点不可达时，那些 run 已被 runner_once 报成失败，
    但报告里必须看得见它 —— 否则“未达成”会被读成“agent 不行”，而实际问题在链路。
    """
    cfg = cfg or CFG
    curve = state.get("curve") or []
    hist = version.history()
    cal = read_json(PATHS.calibration, default=None) or {}
    holdout_ids = list(state.get("holdout_ids") or [])
    promotes = [h for h in hist if h.get("event") == "promote"]
    demotes = [h for h in hist if h.get("event") == "demote"]
    started = str(state.get("started_at") or "")

    def _in_window(h: dict) -> bool:
        return started and str(h.get("created_at") or "") >= started

    crit: list[dict] = []

    def add(name: str, passed: bool, detail: dict, blocking: bool | None = None) -> None:
        crit.append({"name": name, "passed": bool(passed), "blocking":
                     (name in BLOCKING_CRITERIA) if blocking is None else blocking,
                     "detail": detail})

    # 1) 尺子可信：噪声必须是实测标定的（判据单一真源：autopromote.calibration_ok）
    cal_ok, cal_detail = autopromote.calibration_ok(cfg)
    add("noise_calibrated", cal_ok, dict(cal_detail, target=cal.get("target"),
                                          tasks=cal.get("tasks"), n_pairs=cal.get("n_pairs")))

    # 2) 自洽性：诊断只吃本轮轨迹
    bad_src = [{"round": r.get("round_no"), "source": r.get("diag_source")}
               for r in curve
               if (r.get("reason") not in ("no_candidate",) and r.get("diag_source")
                   and r.get("diag_source") != "round_traces")]
    add("loop_self_contained", not bad_src,
        {"rounds": len(curve), "offenders": bad_src})

    # 3) 泛化层干净：holdout 从未参与诊断
    leak_ok, leak_detail = _diagnosis_leakage(out, holdout_ids)
    excluded = [int((read_json(d, default=None) or {}).get("summary", {})
                    .get("holdout_runs_excluded") or 0)
                for d in sorted(out.glob("bench/**/round*/diagnosis.json"))]
    add("holdout_quarantined", leak_ok,
        dict(leak_detail, holdout_ids=holdout_ids,
             excluded_run_counts=[e for e in excluded if e]))

    # 4) 闭环自己换过基线（只算**本次运行窗口内**：上一次 auto 的 promote 不能为本次作证）
    auto_promotes = [h for h in promotes if str(h.get("created_by", "")).startswith("auto")]
    auto_here = [h for h in auto_promotes if _in_window(h)]
    add("auto_promote_happened", bool(auto_here),
        {"auto_promotes_in_window": len(auto_here),
         "auto_promotes_total": len(auto_promotes),
         "tags": [h.get("tag") for h in auto_here][-6:],
         "targets": [((h.get("evidence") or [{}])[-1].get("candidate") or {}).get("target")
                     for h in auto_here][-6:]})

    # 5) 本次窗口内没有任何人工签字
    human = [h for h in promotes if not str(h.get("created_by", "")).startswith("auto")
             and _in_window(h)]
    add("no_human_promote_in_run", not human,
        {"human_promotes": [{"tag": h.get("tag"), "created_by": h.get("created_by"),
                             "created_at": h.get("created_at")} for h in human]})

    # 6) 改动真的落到生产（不是「版本升了行为没变」）
    before = str(state.get("content_stamp_before") or "")
    after = str(state.get("content_stamp_after") or landing.content_stamp())
    live = str(landing.content_stamp())
    add("behavior_actually_changed", bool(before) and (after != before or live != before),
        {"content_stamp_before": before, "content_stamp_after": after,
         "content_stamp_now": live, "promote_records": len(promotes),
         "note": "内容指纹变了才说明生产 harness 真的换了一版；只升版本号不改内容 = 自欺"})

    # 7) holdout 配对增益 > 0（泛化层，不是 evolve 集）
    gained = [r for r in curve if (r.get("holdout_pairs") or 0) > 0
              and float(r.get("holdout_mean_delta") or 0.0) > 0.0]
    add("holdout_paired_gain", bool(gained),
        {"rounds_with_pairs": sum(1 for r in curve if (r.get("holdout_pairs") or 0) > 0),
         "best": max((float(r.get("holdout_mean_delta") or 0.0) for r in curve
                      if r.get("holdout_pairs")), default=None),
         "series": [{"round": r.get("round_no"), "holdout_delta": r.get("holdout_mean_delta"),
                     "pairs": r.get("holdout_pairs")} for r in curve
                    if r.get("holdout_pairs")]})

    # 8) 换基线后退步已排除
    verifies = state.get("verifies") or []
    bad_drift = [v for v in verifies if not v.get("ok")]
    add("post_promote_stable", bool(verifies) and not bad_drift,
        {"checks": [{"tag": v.get("tag"), "drift": v.get("drift"), "tol": v.get("tol"),
                     "n": v.get("n")} for v in verifies], "bad": len(bad_drift)})

    # 9) 最终状态不是「带着一个已知的坏基线继续跑」
    last_event = (hist[-1].get("event") if hist else "")
    unresolved = bool(demotes) and (not promotes or
                                    str(demotes[-1].get("created_at") or "") >
                                    str(promotes[-1].get("created_at") or ""))
    add("no_unresolved_demote", not unresolved,
        {"promotes": len(promotes), "demotes": len(demotes),
         "last_event": last_event,
         "note": "回滚发生过又没重新升上去 = 当前基线就是那个被否掉的版本，必须让人看见"})

    # 观察项：基础设施健康度。端点不可达的 run 已被 runner_once 标为失败（不会进样本），
    # 但必须让人看见“为什么本轮没数据”：否则“未达成”会被误读成“agent 不行”。
    infra = _infra_summary(out)
    add("environment_healthy", not infra["infra_runs"],
        infra, blocking=False)

    # 观察项：改进是否已经穷尽（搜索空间边界，不是缺陷）
    accepted = sum(1 for r in curve if r.get("decision") == "ACCEPT")
    exhausted = [r.get("round_no") for r in curve if r.get("reason") == "no_candidate"]
    add("improvement_capacity", accepted > 0 and len(exhausted) < len(curve),
        {"accepts": accepted, "no_candidate_rounds": exhausted, "rounds": len(curve)},
        blocking=False)

    n_block = sum(1 for c in crit if c["blocking"])
    n_pass_block = sum(1 for c in crit if c["blocking"] and c["passed"])
    return {"achieved": n_pass_block == n_block, "passed_blocking": n_pass_block,
            "n_blocking": n_block, "criteria": crit,
            "base_version": (version.current().version if version.current() else 0),
            "rounds": len(curve), "promotes": len(promotes), "demotes": len(demotes),
            "generated_at": now_iso(), "out_dir": str(out)}


def _infra_summary(out: Path) -> dict:
    """扫本次运行目录里的 usage.json，数出「因端点不可达而报废」的 run。

    判据与 runner_once._infra_error 一致：一次调用都没成功（errors >= calls > 0）。
    旧的产物里没有 calls/errors 字段，那种文件一律不计（不能把历史抹成故障）。
    """
    infra_runs = 0
    total_runs = 0
    samples: list[str] = []
    for f in sorted(out.glob("bench/**/usage.json")):
        data = read_json(f, default=None) or {}
        calls = int(data.get("calls") or 0)
        if not calls:
            continue
        total_runs += 1
        errs = int(data.get("errors") or 0)
        if errs >= calls:
            infra_runs += 1
            if len(samples) < 4:
                samples.append(f"{f.parent.name}: {errs}/{calls} 调用失败")
    return {"infra_runs": infra_runs, "runs_with_usage": total_runs, "samples": samples,
            "note": "这些 run 测的是端点可达性，已被排除在样本外；不是 harness 的质量信号"}


def write_verdict(out: Path, state: dict, *, cfg: Settings | None = None) -> dict:
    v = assess(out, state, cfg=cfg)
    write_json(out / "verdict.json", v)
    (out / VERDICT_NAME).write_text(_render_verdict(v, state), encoding="utf-8")
    return v


def _render_verdict(v: dict, state: dict) -> str:
    zh = {
        "noise_calibrated": "噪声已**实测**标定（δ 不是假设值）",
        "loop_self_contained": "每轮诊断只吃本轮自己跑出的轨迹",
        "holdout_quarantined": "holdout 从未参与诊断/提案（泛化层干净）",
        "auto_promote_happened": "闭环自己至少换过一次基线",
        "no_human_promote_in_run": "本次运行窗口内没有任何人工签字",
        "behavior_actually_changed": "改动真正落到生产（内容指纹改变）",
        "holdout_paired_gain": "holdout 配对增益 > 0（换没参与训练的集上也更好）",
        "post_promote_stable": "换基线后重测 holdout 未退步",
        "no_unresolved_demote": "没有「回滚后仍带着坏基线继续跑」",
        "environment_healthy": "端点可达、没有成片的 infrastructure 废跑（观察项）",
        "improvement_capacity": "仍有可检出的改进空间（观察项）",
    }
    lines = ["# RRSI 达成判据", "",
             f"- 结论：**{'已达成' if v['achieved'] else '未达成'}**"
             f"（硬条件 {v['passed_blocking']}/{v['n_blocking']}）",
             f"- 基线版本：`base/v{v['base_version']}`；轮数 {v['rounds']}；"
             f"promote {v['promotes']} / demote {v['demotes']}",
             f"- 生成时间：{v['generated_at']}", "",
             "| 判据 | 结果 | 证据 |", "| --- | --- | --- |"]
    for c in v["criteria"]:
        mark = "✅" if c["passed"] else ("❌" if c["blocking"] else "⚠️")
        kind = "硬" if c["blocking"] else "观察"
        lines.append(f"| `{c['name']}`（{kind}）{zh.get(c['name'], '')} | {mark} | "
                     f"`{_short(c['detail'])}` |")
    lines += ["", "## 未达成时缺什么", ""]
    fails = [c for c in v["criteria"] if not c["passed"]]
    if not fails:
        lines.append("全部硬条件通过：agent 在没有人工介入的情况下，把自己换到了一个"
                     "「在训练集与泛化集上都显著更好、且可自动回滚」的新基线，"
                     "并且这条结论的证据链每一步都由它自己产出。")
    else:
        for c in fails:
            lines.append(f"- **{c['name']}** {zh.get(c['name'], '')} → {_short(c['detail'])}")
    lines += ["", "## 这份判据为什么不是自证", "",
              "1. 判据读的是磁盘上的产物（diagnosis.json / policy.json / base_history.jsonl / "
              "learning_curve.json），不是运行时内存里的印象；任何人可以删掉报告重跑 `auto --verify-only`。",
              "2. `holdout_paired_gain` 用的是**配对差**，不是两组均值 —— 后者会把「两次不同的噪声」"
              "当成改进。",
              "3. `behavior_actually_changed` 要求生产内容指纹真的变了：只升版本号而不改行为的"
              "闭环，在这张表上过不了。",
              "4. `post_promote_stable` 是换基线**之后**重测的：它防的是「探针里赢了、上线后不行」。",
              "5. 任一条不过 ⇒ 直接写未达成，并列出缺的那条。不做加权平均，不折算成『进度 80%』。"]
    return "\n".join(lines) + "\n"


def _short(d: Any, limit: int = 300) -> str:
    import json
    try:
        s = json.dumps(d, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(d)
    return s if len(s) <= limit else s[:limit] + "…"


# ---------------------------------------------------------------- 报告（供 CLI）

def render_curve(state: dict) -> str:
    curve = state.get("curve") or []
    if not curve:
        return "（还没有曲线数据）"
    rows = ["| 轮 | dominant | 候选 | 决策 | n | ΔQ | δ | holdout ΔQ/对 | 诊断源 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in curve:
        hq = (f"{r.get('holdout_mean_delta', 0):+.2f}/{r.get('holdout_pairs', 0)}"
              if r.get("holdout_pairs") else "-")
        rows.append(f"| {r.get('round_no')} | {r.get('dominant') or '-'} | "
                    f"`{r.get('target') or '-'}` | {r.get('decision')}/{r.get('reason')} | "
                    f"{r.get('n_tasks', 0)} | {r.get('mean_delta', 0):+.2f} | "
                    f"{r.get('delta_crit', 0):.2f} | {hq} | "
                    f"{r.get('diag_source') or '-'}({r.get('diag_runs', 0)}) |")
    return "\n".join(rows)
