"""统一入口：日 0 检查、标定、评分、单轮演化、自检、冻结、报告。

    python -m adaptive_runtime.run_adaptive selftest
    python -m adaptive_runtime.run_adaptive freeze
    python -m adaptive_runtime.run_adaptive verify
    python -m adaptive_runtime.run_adaptive day0
    python -m adaptive_runtime.run_adaptive tasks init --limit 24
    python -m adaptive_runtime.run_adaptive calibrate --k 6 --limit 6        # 真实，很贵
    python -m adaptive_runtime.run_adaptive calibrate --fake                 # 自检用
    python -m adaptive_runtime.run_adaptive score --log log/xxx.log
    python -m adaptive_runtime.run_adaptive round --fake --case good
    python -m adaptive_runtime.run_adaptive round --tasks-file experiments/tasks/replay.json
    python -m adaptive_runtime.run_adaptive report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import calibrate as calib
from . import diagnosis, frozen, proposer
from .budget import Budget
from .comparator import summarize as cmp_summary
from .evaluator import score_trace, write_score
from .experience import Store
from .gate import Decision, Reason, explain, gate_a, gate_b
from .io_util import now_stamp, read_json, write_json, write_text
from .ledger import Ledger
from .probe import create_sandbox, probe_candidate, run_one
from .settings import CFG, PATHS, validate
from .trace_export import export_from_log, iter_logs, write_trace
from . import selftest as selftest_module

STAGE_LABEL = {"P0": "能看见/能量/能拒绝", "P1": "泛化与经验", "P2": "正则化 RSI"}


# ---------------------------------------------------------------- 小工具

def _say(msg: str = "") -> None:
    print(msg)


def _load_tasks(path: str | Path | None, limit: int = 8) -> list[dict]:
    p = Path(path) if path else PATHS.tasks / "replay.json"
    data = read_json(p, default=None)
    if not data:
        raise SystemExit(f"任务集不存在：{p}（先跑 `tasks init`）")
    tasks = data.get("tasks") if isinstance(data, dict) else data
    tasks = [t for t in (tasks or []) if t.get("query")]
    if not tasks:
        raise SystemExit(f"任务集为空：{p}")
    return tasks[:limit]


def _newest_log() -> Path | None:
    logs = list(iter_logs(PATHS.log_dir))
    return logs[0] if logs else None


def _write_decision(out: Path, decision: Decision, extra: dict | None = None) -> None:
    payload = decision.to_dict()
    if extra:
        payload.update(extra)
    write_json(out / "decision.json", payload)


# ---------------------------------------------------------------- tasks

def cmd_tasks_init(args) -> int:
    PATHS.ensure()
    logs = list(iter_logs(PATHS.log_dir))[: args.limit]
    tasks = []
    seen: set[str] = set()
    for log in logs:
        tr = export_from_log(log)
        # 优先用 supervisor 重写后的 goal：它是自包含的；原始 query 常是「这个具体是怎么做的」这类
        # 依赖上下文的追问，单独拿去当任务跑会答非所问。
        query = (tr.goal or tr.query or "").strip()
        if len(query) < 8 or query in seen:
            continue
        seen.add(query)
        tasks.append({
            "task_id": f"REP-{len(tasks) + 1:03d}",
            "query": query,
            "family": _guess_family(query),
            "source_log": str(log),
            "plan_steps": tr.total_steps,
            "contract_rate": round(tr.contract_rate, 3),
        })
    if not tasks:
        _say("没有可用的历史任务（日志里未找到 query/goal）")
        return 1
    out = PATHS.tasks / "replay.json"
    write_json(out, {"created_at": now_stamp(), "source": "history_logs",
                     "count": len(tasks), "tasks": tasks})
    _say(f"已写入 {out}（{len(tasks)} 个任务）")
    fams: dict[str, int] = {}
    for t in tasks:
        fams[t["family"]] = fams.get(t["family"], 0) + 1
    _say(f"任务族分布：{fams}")
    return 0


def _guess_family(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ("比特币", "btc", "以太", "crypto", "加密")):
        return "crypto"
    if any(k in ql for k in ("纳斯达克", "标普", "a股", "港股", "美股", "指数", "etf")):
        return "index"
    if any(k in ql for k in ("财报", "估值", "营收", "毛利")):
        return "fundamental"
    if any(k in ql for k in ("代码", "脚本", "python", "爬", "数据清洗")):
        return "coding"
    return "general"


# ---------------------------------------------------------------- day0 / score / freeze

def cmd_day0(args) -> int:
    PATHS.ensure()
    from . import token_probe
    installed = token_probe.install(sink=PATHS.usage / "tokens-day0.jsonl")
    _say(f"[1/4] token 埋点：{'已安装' if (installed or token_probe.is_installed()) else '不可用（缺 langchain-openai）'}")

    log = _newest_log()
    if not log:
        _say("[2/4] 未找到历史日志，跳过 trace 导出")
        return 1
    tr = export_from_log(log)
    out = PATHS.experiments / "day0"
    write_trace(tr, out)
    _say(f"[2/4] trace 导出：{out / 'trace.json'}（steps={tr.total_steps}, A={tr.contract_rate:.3f}, "
         f"artifacts={len(tr.artifacts)}）")

    usage_files = list((PATHS.usage).glob("tokens-*.jsonl")) if PATHS.usage.exists() else []
    _say(f"[3/4] token 记录：{len(usage_files)} 个文件"
         + ("（真实 run 之后才会有数据）" if not usage_files else ""))

    if not PATHS.frozen_manifest.exists():
        _say("[4/4] 尚未冻结评分符号：确认代码稳定后执行 `run_adaptive freeze`")
    else:
        v = frozen.verify()
        _say(f"[4/4] Frozen Core：{'完整' if not v else '有 ' + str(len(v)) + ' 处变动'}")
    return 0


def cmd_freeze(args) -> int:
    PATHS.ensure()
    m = frozen.freeze()
    s = frozen.manifest_summary()
    _say(f"已冻结：{m['path'] if 'path' in m else PATHS.frozen_manifest}")
    _say(f"  符号 {s['symbols']} 个（缺失 {s['missing']}）/ 文件 {s['paths']} 个 / env {s['env_keys']} 项")
    return 0


def cmd_verify(args) -> int:
    v = frozen.verify()
    if not v:
        _say("Frozen Core 完整")
        return 0
    _say(f"发现 {len(v)} 处越界：")
    for x in v:
        _say(f"  - {x.kind}: {x.target} (expected={x.expected} actual={x.actual})")
    return 1


def cmd_score(args) -> int:
    log = Path(args.log) if args.log else _newest_log()
    if not log or not Path(log).exists():
        _say("找不到日志")
        return 1
    tr = export_from_log(log)
    mode = "live" if args.live else "replay"
    sc = score_trace(tr, mode=mode, task_id=args.task_id)
    out = Path(args.out) if args.out else PATHS.reports / f"score-{tr.run_id}"
    write_trace(tr, out)
    write_score(sc, out)
    _say(json.dumps(sc.to_dict(), ensure_ascii=False, indent=2))
    _say(f"\n产物：{out}")
    return 0


# ---------------------------------------------------------------- calibrate

def cmd_calibrate(args) -> int:
    PATHS.ensure()
    tasks = _load_tasks(args.tasks_file, limit=args.limit)
    k = args.k
    est = args.avg_tokens * len(tasks) * k
    _say(f"标定：{len(tasks)} 任务 × {k} 次 = {len(tasks) * k} runs"
         + ("（假跑，无成本）" if args.fake else f"，预计 ~{est:,} tokens"))
    if not args.fake and not args.yes:
        _say("真实标定很贵。确认请加 --yes")
        return 2

    root = PATHS.experiments / "calibration" / now_stamp()
    sb = create_sandbox(root / "sandbox", "base", keep=bool(args.keep))
    groups: dict[str, list[float]] = {}
    budget = Budget.from_settings(CFG)
    try:
        for task in tasks:
            vals: list[float] = []
            for rep in range(k):
                ro = run_one(sb, task, rep, "base", root / task["task_id"] / f"r{rep}",
                             cfg=CFG, fake_profile="base" if args.fake else None)
                if ro.ok:
                    vals.append(ro.Q)
                    if ro.tokens:
                        try:
                            budget.spend(ro.tokens)
                        except Exception:
                            pass
                else:
                    _say(f"  ! {task['task_id']} r{rep} 失败：{ro.error[:120]}")
            if vals:
                groups[task["task_id"]] = vals
                _say(f"  {task['task_id']}: n={len(vals)} Q={[round(v, 1) for v in vals]}")
    finally:
        if not args.keep:
            sb.keep = False
            sb.cleanup()

    report = calib.summarize(groups)
    calib.save(report, root / "calibration.json")
    if args.publish:
        calib.save(report)
    _say("")
    _say(f"sd_run={report['sd_run']}  sd_diff_null={report['sd_diff_null']}  "
         f"门槛={report['target']}  {'通过' if report['passed'] else '未通过'}")
    _say(f"80% 把握下：检出 +4 分需 {report['required_n_for_4pts']} 对，"
         f"+8 分需 {report['required_n_for_8pts']} 对")
    _say(f"产物：{root / 'calibration.json'}")
    if not report["passed"]:
        _say("→ 不要加样本量掩盖：回 evaluator 改 rubric 锚点与来源抽取规则")
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------- round

def cmd_round(args) -> int:
    PATHS.ensure()
    problems = validate()
    if problems:
        _say("配置提示：" + "；".join(problems))

    tasks = _load_tasks(args.tasks_file, limit=args.limit)
    round_id = f"r{now_stamp()}"
    out = PATHS.runs / args.task_label / round_id
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. 诊断 ----
    log = Path(args.trace) if args.trace else _newest_log()
    if not log or not Path(log).exists():
        _say("找不到可用于诊断的日志（用 --trace 指定）")
        return 1
    tr = export_from_log(log)
    findings, dsum = diagnosis.from_trace(tr)
    write_trace(tr, out)
    write_json(out / "diagnosis.json", {"findings": [f.to_dict() for f in findings],
                                        "summary": dsum})
    _say(f"[1/4] 诊断：dominant={dsum['dominant']} UNKNOWN={dsum['unknown_ratio']:.0%} "
         f"({dsum['total']} 条信号)")
    if dsum["hint"]:
        _say(f"      ! {dsum['hint']}")

    # ---- 2. 提案 + Gate A ----
    ledger = Ledger()
    cands = proposer.propose(dsum["dominant"], ledger, avg_run_tokens=args.avg_tokens)
    if not cands:
        _say("[2/4] 没有可用候选（假设都已在台账里）")
        write_json(out / "decision.json", {"decision": "SKIP", "reason": "no_candidate"})
        return 0
    cand = cands[0]
    write_json(out / "candidate.json", cand.to_dict())
    violations = frozen.verify()
    leaks = proposer.leakage_hits(cand.diff_text())
    dec_a = gate_a(cand, violations, leaks)
    _say(f"[2/4] 候选 {cand.target}: {cand.old}→{cand.new} / Gate A: {explain(dec_a)}")
    if not dec_a.accepted:
        _write_decision(out, dec_a, {"candidate": cand.to_dict()})
        ledger.update(cand.hyp_fingerprint, "rejected",
                      {"reason": dec_a.reason, "stage": "gate_a"}, notes="Gate A")
        _render_report(out, cand, dec_a, None, dsum)
        return 0

    # ---- 3. Probe ----
    profiles = None
    if args.fake:
        case = args.case
        profiles = {"base": "base",
                    "cand": {"good": "good", "zero": "zero", "worse": "worse"}[case]}
        _say(f"[3/4] 假跑 probe（case={case}）")
    else:
        _say(f"[3/4] 真实 probe：{len(tasks)} 任务 × n={args.n or CFG.n_default}")
    outcome = probe_candidate(cand, tasks, out / "probe", n=args.n,
                              keep_sandbox=bool(args.keep),
                              fake_profiles=profiles)
    cmp = outcome.comparison
    assert cmp is not None
    ev = cmp.evidence()

    # ---- 4. 判定 ----
    if outcome.stopped == "stop_loss":
        dec = Decision("REJECT", Reason.REGRESSION.value,
                       [{"check": "stage1_stop_loss", "pass": False,
                         "mean_delta": round(ev.paired().mean, 4)}],
                       {"note": "Stage 1 明显变差，止损"})
    else:
        dec = gate_b(ev)
    write_json(out / "comparison.json", cmp.to_dict())
    _write_decision(out, dec, {"candidate": cand.to_dict(),
                               "probe": {"n": outcome.n_actual, "tokens": outcome.tokens,
                                         "stopped": outcome.stopped}})
    _say(f"[4/4] {explain(dec)}  n={ev.paired().n} mean(Δ)={ev.paired().mean:+.2f} "
         f"δ={ev.paired().delta_crit:.2f} tokens={outcome.tokens:,}")

    # ---- 台账 / 经验 ----
    status = {"ACCEPT": "accepted", "REJECT": "rejected"}.get(dec.decision, "inconclusive")
    if dec.reason == Reason.WITHIN_NOISE.value:
        status = "inconclusive"
    ledger.update(cand.hyp_fingerprint, status,
                  {"n": ev.paired().n, "dQ": round(ev.paired().mean, 4),
                   "p": round(ev.paired().p_value, 6), "dC": round(ev.dC, 4),
                   "reason": dec.reason}, tasks=[t["task_id"] for t in tasks])
    if dec.accepted:
        Store().create(
            problem=f"{cand.taxonomy}|{getattr(cand, 'target', '-')}|0",
            claim=cand.hypothesis, component=f"L{cand.level}/{cand.target}",
            edit={"patch_hash": cand.candidate_id, "old": cand.old, "new": cand.new},
            n_pairs=ev.paired().n, n_target=CFG.n_default,
            p_value=ev.paired().p_value, regression_rate=ev.regression_rate,
            dQ=round(ev.paired().mean, 4), dC=round(ev.dC, 4),
            positive=[t["task_id"] for t in tasks],
            negative=[k for k, v in cmp.per_task.items()
                      if v["mean_delta"] < -CFG.regression_min_points],
            hypothesis_id=cand.hyp_fingerprint)

    _render_report(out, cand, dec, cmp, dsum, outcome)
    _say(f"产物：{out}")
    return 0 if dec.accepted else 0


def _render_report(out: Path, cand, decision: Decision, cmp, dsum: dict,
                   outcome=None) -> None:
    lines = [f"# 单轮报告 {out.name}", "",
             f"- 时间：{now_stamp()}", f"- 阶段：{CFG.stage}（{STAGE_LABEL.get(CFG.stage, '')}）",
             f"- 诊断主导模式：`{dsum['dominant']}`（UNKNOWN {dsum['unknown_ratio']:.0%}）", "",
             "## 候选", "",
             f"- target：`{getattr(cand, 'target', '-')}`",
             f"- 改动：`{getattr(cand, 'old', '')}` → `{getattr(cand, 'new', '')}`",
             f"- 假设：{getattr(cand, 'hypothesis', '')}", ""]
    if cmp is not None:
        ev = cmp.evidence()
        pr = ev.paired()
        lines += ["## 证据（配对）", "", "```", cmp_summary(cmp), "```", "",
                  f"- 判定：**{decision.decision}**（{decision.reason}）",
                  f"- 行为指标 ΔB 仅供归因，不参与判定", ""]
        if cmp.per_task:
            lines += ["| 任务 | n | mean(Δ) | sd |", "|---|---|---|---|"]
            for task, v in sorted(cmp.per_task.items()):
                lines.append(f"| {task} | {v['n']} | {v['mean_delta']:+.2f} | {v['sd']:.2f} |")
            lines.append("")
    else:
        lines += ["## 判定", "", f"**{decision.decision}**（{decision.reason}）", ""]
    if decision.gate_trace:
        lines += ["## 判据明细", "", "| 检查 | 通过 | 值 |", "|---|---|---|"]
        for row in decision.gate_trace:
            vals = {k: v for k, v in row.items() if k not in ("check", "pass")}
            lines.append(f"| {row['check']} | {'✓' if row['pass'] else '✗'} | `{vals}` |")
        lines.append("")
    if outcome is not None:
        lines += ["## 成本", "", f"- tokens：{outcome.tokens:,}",
                  f"- 实际配对数：{outcome.n_actual} / 计划 {outcome.n_planned}",
                  f"- 停止原因：{outcome.stopped or '无'}", ""]
    write_text(out / "report.md", "\n".join(lines))


def cmd_report(args) -> int:
    runs = sorted([p for p in PATHS.runs.rglob("report.md")], key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not runs:
        _say("还没有轮次报告")
        return 1
    p = runs[0] if not args.round else PATHS.runs / args.round / "report.md"
    _say(p.read_text(encoding="utf-8"))
    _say(f"\n文件：{p}")
    return 0


def cmd_status(args) -> int:
    PATHS.ensure()
    _say(f"阶段：{CFG.stage}（{STAGE_LABEL.get(CFG.stage, '')}）")
    _say(f"阈值：A_FLOOR={CFG.a_floor} A_TOL={CFG.a_tol} R_MAX={CFG.r_max} "
         f"β0={CFG.beta0} β1={CFG.beta1} n={CFG.n_default}")
    _say(f"预算：单轮 {CFG.cost_cap_round:,} / 单候选 {CFG.cost_cap_cand:,} tokens")
    _say(f"Frozen：{frozen.manifest_summary()}")
    cal = calib.load()
    _say(f"标定：{('sd_diff_null=' + str(cal.get('sd_diff_null'))) if cal else '未标定'}")
    _say(f"台账：{Ledger().stats()}")
    _say(f"经验：{len(Store().rows())} 条")
    return 0


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="adaptive_runtime", description="Adaptive Agent Runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest").set_defaults(
        func=lambda a: 0 if selftest_module.selftest()["passed"] else 1)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    sub.add_parser("day0").set_defaults(func=cmd_day0)
    sub.add_parser("status").set_defaults(func=cmd_status)

    p = sub.add_parser("tasks"); psub = p.add_subparsers(dest="sub", required=True)
    pi = psub.add_parser("init"); pi.add_argument("--limit", type=int, default=24)
    pi.set_defaults(func=cmd_tasks_init)

    p = sub.add_parser("score")
    p.add_argument("--log", default=None); p.add_argument("--task-id", default="")
    p.add_argument("--out", default=None); p.add_argument("--live", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("calibrate")
    p.add_argument("--tasks-file", default=None); p.add_argument("--limit", type=int, default=6)
    p.add_argument("--k", type=int, default=CFG.k_calib)
    p.add_argument("--avg-tokens", type=int, default=200_000)
    p.add_argument("--fake", action="store_true"); p.add_argument("--yes", action="store_true")
    p.add_argument("--keep", action="store_true"); p.add_argument("--publish", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("round")
    p.add_argument("--tasks-file", default=None); p.add_argument("--limit", type=int, default=8)
    p.add_argument("--trace", default=None); p.add_argument("--task-label", default="round")
    p.add_argument("--n", type=int, default=None); p.add_argument("--avg-tokens", type=int, default=200_000)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--case", default="good", choices=("good", "zero", "worse"))
    p.add_argument("--keep", action="store_true")
    p.set_defaults(func=cmd_round)

    p = sub.add_parser("report"); p.add_argument("--round", default=None)
    p.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
