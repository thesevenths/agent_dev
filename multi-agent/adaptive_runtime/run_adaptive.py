"""统一入口：日 0 检查、标定、评分、单轮演化、自检、冻结、报告、无人值守闭环。

    python -m adaptive_runtime.run_adaptive selftest
    python -m adaptive_runtime.run_adaptive freeze
    python -m adaptive_runtime.run_adaptive verify
    python -m adaptive_runtime.run_adaptive day0
    python -m adaptive_runtime.run_adaptive tasks init --limit 24
    python -m adaptive_runtime.run_adaptive tasks offline        # 本地数据任务集（不联网）
    python -m adaptive_runtime.run_adaptive calibrate --offline --limit 8 --k 4 --yes --publish
    python -m adaptive_runtime.run_adaptive calibrate --fake                 # 自检用
    python -m adaptive_runtime.run_adaptive score --log log/xxx.log
    python -m adaptive_runtime.run_adaptive round --fake --case good
    python -m adaptive_runtime.run_adaptive round --tasks-file experiments/tasks/replay.json
    python -m adaptive_runtime.run_adaptive report
    python -m adaptive_runtime.run_adaptive evolve          # 探索/剪枝状态
    python -m adaptive_runtime.run_adaptive base show
    python -m adaptive_runtime.run_adaptive base promote --operator <name> --note <理由>
    python -m adaptive_runtime.run_adaptive benchmark --fake
    # 无人值守：自己换基线 + 自己落地 + 自己出具达成判据
    python -m adaptive_runtime.run_adaptive auto --offline --rounds 3 --n 8 --parallel 4 --yes
    python -m adaptive_runtime.run_adaptive auto --verify-only     # 只从产物重算判据（不跑模型）

两个全局开关为什么放在子命令上而不是只靠环境变量：
``--offline`` 是**不烧 Tavily 免费额度**与**降噪**的同一件事（联网内容逐日变化会把
sd(Δ) 抬高到检不出真实改进）；``--parallel`` 是墙钟的唯一把手（单 run ≈ 18 分钟，
而 token 成本不计）。两者都会写进每轮的测量口径，跨轮可比。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import auto as auto_module
from . import bench_tasks as bench_tasks_mod
from . import calibrate as calib
from . import (benchmark, diag_batch, diagnosis, evolution, frozen, harness, offline_tasks,
               proposer, quarantine, regression, trajectory, version)
from .budget import Budget
from .comparator import summarize as cmp_summary
from .evaluator import judge_version, score_trace, write_score
from .experience import Store
from .gate import Decision, Reason, explain, gate_a, gate_b, ledger_status
from .io_util import now_stamp, read_json, write_json, write_text
from .ledger import Ledger
from .probe import create_sandbox, probe_candidate, run_jobs, run_one, run_one_isolated
from .settings import CFG, PATHS, detectability, validate
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


def _apply_runtime_flags(args) -> None:
    """把 ``--offline`` / ``--parallel`` 写回 CFG，并把离线任务集接上缺省路径。

    为什么可以直改 CFG：CFG 是进程内唯一可变配置，且子进程的环境由
    ``probe._env_for_run(cfg.offline)`` 统一注入 —— 不靠调用方自己传。
    """
    if getattr(args, "offline", False):
        CFG.offline = True
        if not getattr(args, "tasks_file", None):
            p = PATHS.tasks / "offline_local.json"
            if not p.exists():
                _say("离线任务集不存在 → 正在生成本地数据与 offline_local.json")
                _say(f"  写出：{offline_tasks.write_tasks_file()}")
            args.tasks_file = str(p)
    if getattr(args, "parallel", None) is not None:
        CFG.max_parallel = max(1, int(args.parallel))
    if getattr(args, "n", None):
        # --n 必须写回 CFG，而不能只当调用参数传下去：``measure_mode()`` 记的是
        # ``n_default``，而每轮 meta 要靠它核对「两次测的是不是同一把尺」。
        # 不写回时落盘的口径会写着 n=8，而实际跑的是 n=3 —— 这条证据是假的。
        CFG.n_default = max(1, int(args.n))


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


def cmd_tasks_offline(args) -> int:
    """生成/刷新**离线**任务集（本地 CSV + 固定快照，全程不联网）。"""
    PATHS.ensure()
    out = offline_tasks.write_tasks_file(force_data=bool(args.force_data))
    data = read_json(out, default=None) or {}
    _say(f"已写出：{out}（{data.get('count')} 个任务，requires_web={data.get('requires_web')}）")
    _say(f"任务族分布：{data.get('families')}")
    _say("跟闭环：所有命令加 --offline（等价于 ADAPTIVE_OFFLINE=1，三个联网工具走本地语料）")
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

    # Base 基线必须在 day0 就钉住：否则后续「对着谁比」会变成隐性变量
    b = version.ensure_initial()
    _say(f"[5/4] Base 基线：{b.tag}（commit {b.base_commit or '未知'}，"
         f"env {len(b.env)} 项）；打 tag：{version.tag_hint()}")
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
    workers = max(1, int(CFG.max_parallel or 1))
    _say(f"调度：并发度={workers}（ADAPTIVE_MAX_PARALLEL=1 时逐个串行）"
         if workers > 1 else "调度：串行（并发度 1）；标定很慢时可设 ADAPTIVE_MAX_PARALLEL")
    try:
        # 一次性把「全部任务 × k 次」交给调度器：标定是同一臂（base）同一份配置，
        # 任务之间无依赖，串行跑纯属墙钟浪费。旧实现的双层 for 在 workers=1 时与下面等价。
        jobs = [("%s#%d" % (task["task_id"], rep),
                 (lambda tk=task, r=rep: run_one_isolated(
                     sb, tk, r, "base", root / tk["task_id"] / f"r{r}",
                     cfg=CFG, fake_profile="base" if args.fake else None)))
                for task in tasks for rep in range(k)]
        finished = run_jobs(jobs, workers)
        for task in tasks:
            vals: list[float] = []
            for rep in range(k):
                ro = finished.get("%s#%d" % (task["task_id"], rep))
                if ro is None:
                    continue
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
    # 默认只吃闭环自己跑出来的那批 run（``source=round_traces``）。
    # 只有两种情况退回人工日志：显式 --trace，或闭环在本机还没产出过任何真实轨迹（首次使用）。
    # 退回时必须在 diagnosis.json 里标成 manual_log：那份日志与本轮探针无因果关系，
    # 报告要能看出这一轮的结论不是自洽产物。（旧实现无条件拿 mtime 最新日志当诊断，
    # 于是「你在 Studio 里问一句」会静默改掉闭环的病口 —— 那不是闭环。）
    dsum: dict = {}
    if args.trace:
        log = Path(args.trace)
    else:
        dirs = diag_batch.recent_trace_dirs()
        if not dirs:
            log = _newest_log()
        else:
            findings, dsum = diag_batch.from_run_dirs(dirs[0], arm="base", cfg=CFG)
            log = None
            usable, why = diag_batch.is_usable(dsum)
            if not usable and not dsum.get("dominant"):
                _say(f"闭环轨迹不可用：{why}（退回人工日志）")
                findings, dsum, log = [], {}, _newest_log()
    if log is not None:
        if not Path(log).exists():
            _say("找不到可用于诊断的日志（用 --trace 指定）")
            return 1
        tr = export_from_log(log, root=PATHS.root)
        # 先用确定性评分给这份 trace 打一次分，再交给诊断层。
        # 顺序不能反：诊断的「质量缺陷通道」消费的是 E/C，而 E/C 只有打分后才有。
        # 缺了这一步，诊断只认 kind=recursion/truncated/exception 这类硬失败，
        # 于是「跑完了但每个数字都没来源」会被判成无失败信号。
        diag_score = score_trace(tr, root=PATHS.root, mode="replay",
                                 task_id=args.task_label, arm="diag")
        write_score(diag_score, out / "diagnosis_score")
        findings, dsum = diagnosis.from_trace(tr, score=diag_score)
        dsum["source"] = "manual_log"
        dsum.setdefault("n_runs", 1)
        write_trace(tr, out)
    write_json(out / "diagnosis.json", {"findings": [f.to_dict() for f in findings],
                                        "summary": dsum})
    _say(f"[1/4] 诊断（源={dsum.get('source','?')}，{dsum.get('n_runs',0)} 个 run）："
         f"dominant={dsum['dominant']} UNKNOWN={dsum['unknown_ratio']:.0%} "
         f"({dsum['total']} 条信号)")
    if dsum.get("source") == "manual_log":
        _say("      ! 本轮诊断读的是人工日志，provenance 不自洽：结论仅供参考，"
             "无人值守跑法请用 benchmark/auto（它只吃本轮轨迹）")
    if dsum["hint"]:
        _say(f"      ! {dsum['hint']}")

    # ---- 2. 提案 + Gate A ----
    ledger = Ledger()
    plan = evolution.evolve_plan(ledger.rows, dsum["dominant"])
    write_json(out / "evolution.json", plan)

    # Task Harness = Base + 本轮检索到的 ≤3 条经验（只读覆盖层，轮末丢弃）
    base = version.ensure_initial()
    exp_store = Store()
    lessons = exp_store.retrieve(f"{dsum['dominant']}|-|0")
    th = harness.assemble(base.tag, dsum["dominant"], lessons)
    harness.write(th, out)
    write_json(out / "base_version.json", base.to_dict())

    if plan["explore"]:
        _say(f"[2/4] exploration 触发：{plan['explore_reason']} → 强制覆盖 "
             f"{'@L'.join([str(x) for x in (plan['forced_pair'] or ('-', '-'))])}")
    if plan["exclude_targets"]:
        _say(f"      dormant 组件（跳过）：{', '.join(plan['exclude_targets'])}")

    cands = proposer.propose(dsum["dominant"], ledger, avg_run_tokens=args.avg_tokens,
                             exclude_targets=plan["exclude_targets"],
                             prefer_pair=plan["forced_pair"])
    # --level 可在单层内独立实测（默认 None = 按搜索空间顺序，L1 成本最低）
    if getattr(args, "level", None):
        want = int(args.level)
        picked = [c for c in cands if c.level == want]
        if not picked:
            _say(f"[2/4] 该失败模式下没有 L{want} 候选（可用层级："
                 f"{sorted({c.level for c in cands})}）")
            write_json(out / "decision.json", {"decision": "SKIP",
                                              "reason": f"no_candidate_L{want}"})
            return 0
        cands = picked
    if not cands:
        _say("[2/4] 没有可用候选（假设都已在台账里）")
        write_json(out / "decision.json", {"decision": "SKIP", "reason": "no_candidate"})
        return 0
    cand = cands[0]
    # 归因三元组（P0 验收 B 组）：对着哪个基线、改了什么东西、用什么尺子测的
    cand.base_commit = base.base_commit
    write_json(out / "candidate.json", cand.to_dict())
    violations = frozen.verify()
    leaks = proposer.leakage_hits(cand.diff_text())
    dec_a = gate_a(cand, violations, leaks)
    _say(f"[2/4] 候选 {cand.target}: {cand.old}→{cand.new} / Gate A: {explain(dec_a)}")
    if not dec_a.accepted:
        _write_decision(out, dec_a, {"candidate": cand.to_dict(),
                                     "judge_version": judge_version()})
        ledger.update(cand.hyp_fingerprint, "rejected",
                      {"reason": dec_a.reason, "stage": "gate_a"}, notes="Gate A")
        quarantine.record(cand.to_dict(), dec_a.reason, "gate_a", round_dir=out)
        _render_report(out, cand, dec_a, None, dsum, plan=plan, base=base)
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

    # Regression 集（只用于否决）：任一被接受的候选都必须在它上面没回归
    reg_ids = {t["task_id"] for t in regression.load()}
    reg = regression.check(cmp, reg_ids)
    write_json(out / "regression_check.json", reg)
    if reg["covered"] and dec.accepted and not reg["ok"]:
        dec = Decision("REJECT", Reason.REGRESSION.value,
                       [{"check": "regression_set", "pass": False,
                         "regression_rate": reg["regression_rate"],
                         "r_max": reg["r_max"], "regressed": reg["regressed"]}],
                       {"note": "在 regression 集上出现显著回归（该集只用于否决）"})
        _say(f"      ! regression 集命中：{reg['regressed']}")
    write_json(out / "comparison.json", cmp.to_dict())
    _write_decision(out, dec, {"candidate": cand.to_dict(),
                               "judge_version": judge_version(),
                               "probe": {"n": outcome.n_actual, "tokens": outcome.tokens,
                                         "stopped": outcome.stopped}})
    _say(f"[4/4] {explain(dec)}  n={ev.paired().n} mean(Δ)={ev.paired().mean:+.2f} "
         f"δ={ev.paired().delta_crit:.2f} tokens={outcome.tokens:,}")

    # ---- 台账 / 经验 ----
    # 假跑（--fake）的轨迹是受控合成数据，**一条都不许进真实台账/经验库**：
    # case=good 的 dQ=38.1 是写死的画像值，一旦落成 accepted，该假设就被
    # ``is_duplicate`` 永久跳过（搜索空间凭空少一格），而经验库会把一条编造的
    # 「已验证」注入后续真实任务的提示 —— 那是闭环最脏的假阳性：自己教会自己信一句假话。
    # （benchmark 早就是这么防的，本轮才补上同样的守卫。）
    status = ledger_status(dec, stopped=outcome.stopped)
    if args.fake:
        _say(f"      假跑（case={args.case}）：台账与经验库均不写入，状态={status}")
    else:
        ledger.update(cand.hyp_fingerprint, status,
                      {"n": ev.paired().n, "dQ": round(ev.paired().mean, 4),
                       "p": round(ev.paired().p_value, 6), "dC": round(ev.dC, 4),
                       "reason": dec.reason, "stopped": outcome.stopped},
                      tasks=[t["task_id"] for t in tasks])
    if dec.accepted and not args.fake:
        Store().create(
            problem=f"{cand.taxonomy}|{getattr(cand, 'target', '-')}|0",
            claim=cand.hypothesis, component=f"L{cand.level}/{cand.target}",
            edit={"patch_hash": cand.patch_hash, "base_commit": cand.base_commit,
                  "old": cand.old, "new": cand.new},
            n_pairs=ev.paired().n, n_target=CFG.n_default,
            p_value=ev.paired().p_value, regression_rate=ev.regression_rate,
            dQ=round(ev.paired().mean, 4), dC=round(ev.dC, 4),
            positive=[t["task_id"] for t in tasks],
            negative=[k for k, v in cmp.per_task.items()
                      if v["mean_delta"] < -CFG.regression_min_points],
            judge_version=judge_version(),
            hypothesis_id=cand.hyp_fingerprint)
    else:
        # 被拒 → 沙箱已删（回滚），但证据必须归档，否则「同一 patch 又提一遍」查不出来。
        # 假跑不写：假轨迹不是真实证据，写进去会把隔离区变成噪声。
        if not args.fake:
            quarantine.record(cand.to_dict(), dec.reason, "gate_b",
                              probe={"n": outcome.n_actual, "tokens": outcome.tokens,
                                     "mean_delta": round(ev.paired().mean, 4),
                                     "delta_crit": round(ev.paired().delta_crit, 4),
                                     "stopped": outcome.stopped},
                              round_dir=out, failures=outcome.failures)
    if dec.accepted and not args.fake:
        # 被接受候选的正面任务并入 regression 集（只追加、永不用于生成假设）
        added = regression.add_from_accepted(tasks, cand.candidate_id)
        if added:
            _say(f"      regression 集新增 {added} 个任务（当前 {len(regression.load())}）")
    write_json(out / "quarantine.json", quarantine.summary())

    _render_report(out, cand, dec, cmp, dsum, outcome, plan=plan, base=base)
    # Incumbent：本轮被接受的候选，只作为比较链留档，轮末不进入 Base
    inc = version.incarnate([dict(cand.to_dict(), decision=dec.decision, reason=dec.reason)]) \
        if dec.accepted else []
    write_json(out / "incumbents.json",
               {"incumbents": inc, "base_tag": base.tag,
                "note": "Incumbent 轮末丢弃，基线永远是 Base，不是上一轮候选"})
    write_json(out / "prune.json", evolution.report(ledger.rows))
    _say(f"产物：{out}")
    return 0


def _render_report(out: Path, cand, decision: Decision, cmp, dsum: dict,
                   outcome=None, plan: dict | None = None, base=None) -> None:
    lines = [f"# 单轮报告 {out.name}", "",
             f"- 时间：{now_stamp()}", f"- 阶段：{CFG.stage}（{STAGE_LABEL.get(CFG.stage, '')}）",
             f"- 基线：`{getattr(base, 'tag', '-')}`（{getattr(base, 'base_commit', '') or 'commit 未知'}）",
             f"- 诊断主导模式：`{dsum['dominant']}`（UNKNOWN {dsum['unknown_ratio']:.0%}）", "",
             "## 候选", "",
             f"- target：`{getattr(cand, 'target', '-')}`（L{getattr(cand, 'level', '-')}）",
             f"- 改动：`{getattr(cand, 'old', '')}` → `{getattr(cand, 'new', '')}`",
             f"- patch_hash：`{getattr(cand, 'patch_hash', '-')}`"
             f" / base_commit：`{getattr(cand, 'base_commit', '') or '-'}`"
             f" / judge_version：`{judge_version()}`",
             f"- 假设：{getattr(cand, 'hypothesis', '')}",
             f"- 来源：{'exploration（强制覆盖冷落组合）' if getattr(cand, 'exploration', False) else 'exploitation（针对主导失败模式）'}",
             ""]
    if plan:
        lines += ["## 演化约束", "",
                  f"- dormant 组件跳过：`{plan.get('exclude_targets') or '无'}`",
                  f"- exploration：{'触发' if plan.get('explore') else '未触发'}"
                  f"（{plan.get('explore_reason', {}).get('reason', '-')}）", ""]
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
    v = version.summary()
    cur = v["current"] or {}
    _say(f"Base：{cur.get('tag', '未建立')}（commit {cur.get('base_commit') or '未知'}）"
         + ("  ← 建议做一次人工 promote 评审" if v["promote_due"] else ""))
    _say(f"台账：{Ledger().stats()}")
    _say(f"经验：{len(Store().rows())} 条")
    ev = evolution.report(Ledger().rows)
    _say(f"演化：层用量 {ev['usage_by_level']} / dormant {ev['dormant'] or '无'} "
         f"/ exploration {'触发' if ev['should_explore'] else '未触发'}")
    _say(f"判分口径：judge_version={judge_version()}")
    _say(f"隔离区：{quarantine.summary()['by_reason'] or '空'}")
    return 0


# ---------------------------------------------------------------- base / evolve

def cmd_base(args) -> int:
    PATHS.ensure()
    if args.base_cmd == "promote":
        pending = Store().promotion_candidates()
        if not pending:
            _say("没有达到 promote 标准的经验（需 ≥2 个任务族验证通过）")
            return 1
        _say(f"待合并经验 {len(pending)} 条：")
        for r in pending:
            _say(f"  - {r.get('component')}: {r.get('claim')}（conf={r.get('confidence')}）")
        if not args.operator:
            _say("\npromote 必须由人工执行：加 --operator <你的名字> --note <理由>")
            return 2
        new = version.promote(pending, operator=args.operator, note=args.note)
        _say(f"\n已 promote 到 {new.tag}（commit {new.base_commit or '未知'}）")
        _say(f"打 tag：{version.tag_hint()}")
        return 0
    s = version.summary()
    if s["current"] is None:
        _say("Base 基线尚未建立，正在建立 base/v1 …")
        version.ensure_initial()
        s = version.summary()
    _say(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark(args) -> int:
    """连续多轮闭环：回答「它到底有没有越跑越好」。"""
    PATHS.ensure()
    problems = validate()
    if problems:
        _say("配置提示：" + "；".join(problems))
    if args.rounds < 2:
        _say("提示：rounds<2 只能看单轮，看不到「累积提升」；建议 ≥3")

    tasks = _load_tasks(args.tasks_file, limit=args.limit)
    fakes = {"base": "base", "cand": args.case} if args.fake else None
    if not args.fake:
        est = args.rounds * args.n * 2 * args.avg_tokens
        _say(f"成本预估：{args.rounds} 轮 × {args.n} 任务 × 2 臂 ≈ "
             f"{est:,} tokens（另加 holdout 的 base 臂）")

    rep = benchmark.run_benchmark(
        tasks, rounds=args.rounds, n=args.n, holdout_ratio=args.holdout,
        level=args.level, trace=args.trace, task_label=args.task_label,
        fake_profiles=fakes, keep_sandbox=args.keep)

    if rep.get("error"):
        _say(f"无法开始：{rep['error']}")
        return 1
    for line in rep["verdict"]:
        _say(f"- {line}")
    _say(f"report：{Path(rep['out_dir']) / 'benchmark.md'}")
    return 0


def cmd_auto(args) -> int:
    """无人值守闭环：跑（或续跑）到 rounds 轮，自己换基线，自己出判据。"""
    PATHS.ensure()
    out = auto_module._dir(args.task_label, args.out_dir or None)

    if args.verify_only:
        state = auto_module.load_state(out)
        if not state:
            _say(f"没有可重算的状态：{out / auto_module.STATE_NAME}（先跑过 auto 才能出判据）")
            return 1
        v = auto_module.write_verdict(out, state)
        _say(_verdict_text(v))
        _say(f"产物：{out / auto_module.VERDICT_NAME}")
        return 0 if v["achieved"] else 1

    tasks = _load_tasks(args.tasks_file, limit=args.limit)
    fakes = {"base": "base", "cand": args.case} if args.fake else None

    # δ 的口径必须与真实跑法一致：自由度是 **train 任务数**，不是 ADAPTIVE_N。
    # 所以 validate() 排在 split() 之后 —— 旧顺序只能拿 n 猜配对数，于是报出
    # 「n=3 ⇒ δ≈37 分，10 分的改进检不出」这种根本不存在的门限。
    tr, ho = benchmark.split(tasks, args.holdout)
    problems = validate(n_tasks=len(tr), n_reps=args.n)
    if problems:
        _say("配置提示：" + "；".join(problems))
    if not args.fake:
        # 这行不受 --yes 影响：它是「本轮能不能得出显著结论」唯一的事前提示，
        # 而无人值守跑法恰恰总是带 --yes。
        _say(detectability(len(tr), args.n)["text"])

    if not args.offline and not args.fake:
        _say("[warn] 未开 --offline：会消耗 Tavily 免费额度，且联网内容逐日变化会抬高噪声。"
             "无人值守跑法几乎总是应当 --offline。")
    if not args.yes and not args.fake:
        from .bench_tasks import profile_for
        # 按口径选画像：离线单 run 实测 ~225 s（并发 4、31 个 run / 1752 s 反推），
        # 拿在线的 1109 s 估会贵一个数量级，把本来能跑完的计划报成“30 小时不可行”——
        # 那是用错的常数否掉了用户的方案。
        prof = profile_for(offline=CFG.offline, task_profile=CFG.task_profile)
        plan = bench_tasks_mod.plan_cost(len(tr), args.n, args.rounds, len(ho),
                                         profile=prof)
        # auto 比 benchmark 多两块：接受后的 holdout **配对**，以及换基线后的 holdout 重测
        extra = len(ho) * CFG.holdout_reps * 2 * args.rounds
        runs = plan.total_runs + extra
        hours = runs * prof.seconds_per_run / 3600.0 / max(1, CFG.max_parallel)
        _say(f"墙钟预估：~{runs} 个 run（train 两臂 {plan.evolve_runs} + holdout "
             f"{runs - plan.evolve_runs}）× {prof.seconds_per_run:.0f}s ÷ 并发 {CFG.max_parallel}"
             f" ≈ {hours:.1f} 小时（token 不计费，但这是真实时间成本）")
        _say("确认要跑请加 --yes；先拿一个计时试点可用 --rounds 1 --limit 6 --n 2")
        _say(f"另外：自动换基线要求已有实测标定（先跑 calibrate --offline --publish），"
             f"否则机器判据会一直 BLOCK。标定文件："
             f"{'已存在' if (PATHS.calibration.exists()) else '不存在'}")
        return 2

    rep = auto_module.run_auto(
        tasks, rounds=args.rounds, n=args.n, holdout_ratio=args.holdout,
        label=args.task_label, out_dir=args.out_dir or None,
        resume=not args.no_resume, verify=not args.no_verify,
        fake_profiles=fakes, keep_sandbox=args.keep,
        on_progress=lambda m: _say(f"  · {m}"))
    if rep.get("error"):
        _say(f"无法开始：{rep['error']}")
        return 1
    report = rep.get("report") or {}
    for line in report.get("verdict") or []:
        _say(f"- {line}")
    v = rep.get("verdict") or {}
    _say("")
    _say(_verdict_text(v))
    _say(f"学习曲线：{out / auto_module.CURVE_NAME}")
    _say(f"判据文档：{out / auto_module.VERDICT_NAME}")
    return 0 if v.get("achieved") else 1


def _verdict_text(v: dict) -> str:
    lines = [f"RRSI 判据：{'已达成' if v.get('achieved') else '未达成'}"
             f"（硬条件 {v.get('passed_blocking')}/{v.get('n_blocking')}）"]
    for c in v.get("criteria") or []:
        mark = "OK  " if c["passed"] else ("FAIL" if c["blocking"] else "warn")
        lines.append(f"  [{mark}] {c['name']}")
    return "\n".join(lines)


def cmd_evolve(args) -> int:
    PATHS.ensure()
    rep = evolution.report(Ledger().rows)
    write_json(PATHS.experiments / "evolution_report.json", rep)
    _say(f"层用量（按被接受计）：{rep['usage_by_level']}")
    _say(f"exploration：{'触发' if rep['should_explore'] else '未触发'}"
         f"（{rep['explore_reason'].get('reason')}）")
    if rep["exploration_pairs"]:
        _say(f"  待覆盖组合：{rep['exploration_pairs'][:6]}")
    if rep["pruned"]:
        _say("剪枝判定（连续 WITHIN_NOISE → 该组件惰性）：")
        for p in rep["pruned"]:
            _say(f"  - {p['target']}（L{p['level']}，连续 {p['within_noise_streak']} 次）"
                 f"：{p['verdict']}")
    else:
        _say("剪枝：暂无组件达到阈值")
    if rep["dormant"]:
        _say(f"当前 dormant：{rep['dormant']}")
    _say(f"产物：{PATHS.experiments / 'evolution_report.json'}")
    return 0


# ---------------------------------------------------------------- regression / trajectory

def cmd_regression(args) -> int:
    PATHS.ensure()
    if args.reg_cmd == "init":
        # 只用「当前 base 已经能通过」的任务当体检项：连 base 都过不了的不能算回归
        data = read_json(PATHS.tasks / "replay.json", default=None) or {}
        pool = [t for t in (data.get("tasks") or [])
                if float(t.get("contract_rate") or 0) >= 0.999]
        if not pool:
            _say("没有 contract_rate=1.0 的历史任务可用；先用 `tasks init` 或补跑日志")
            return 1
        added = regression.add_from_accepted(pool, candidate_id="init")
        cov = regression.coverage()
        _say(f"regression 集：新增 {added}，当前 {cov['count']} 个")
        _say(f"任务族分布：{cov['families']}")
        _say(f"是否符合规格（≥{regression.MIN_TASKS} 任务 / ≥{regression.MIN_FAMILIES} 族 / "
             f"每族 ≥{regression.MIN_PER_FAMILY}）：{'是' if cov['meets_spec'] else '否'}")
        return 0
    s = regression.summary()
    _say(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


def cmd_trajectory(args) -> int:
    PATHS.ensure()
    logs = list(iter_logs(PATHS.log_dir))[: args.limit]
    if not logs:
        _say("没有历史日志")
        return 1
    traces = [export_from_log(p) for p in logs]
    rep = trajectory.contrast(traces)
    rep["by_agent"] = trajectory.by_agent(traces)
    md = trajectory.to_markdown(rep)
    out = PATHS.reports / "trajectory_contrast.md"
    write_text(out, md)
    write_json(PATHS.reports / "trajectory_contrast.json", rep)
    _say(md)
    _say(f"\n产物：{out}（基于 {len(logs)} 份日志）")
    return 0


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="adaptive_runtime",
        description="Adaptive Agent Runtime —— harness 自优化的控制平面（CLI）",
        epilog="退出码：0 正常完成；1 执行失败或自检未通过。"
               "各子命令的详细说明见 docs/adaptive_runtime_verification.md。")
    sub = ap.add_subparsers(dest="cmd", required=True, title="子命令")

    sub.add_parser("selftest", help="判定器自检（含阴性/阳性对照，零模型成本）").set_defaults(
        func=lambda a: 0 if selftest_module.selftest()["passed"] else 1)
    sub.add_parser("freeze", help="生成 Frozen Core 清单（锁定评分链路）").set_defaults(func=cmd_freeze)
    sub.add_parser("verify", help="校验生产文件未被修改").set_defaults(func=cmd_verify)
    sub.add_parser("day0", help="Day-0 初始化：建立 Base 基线并冻结").set_defaults(func=cmd_day0)
    sub.add_parser("status", help="输出阈值、Base、假设台账与演化状态").set_defaults(func=cmd_status)

    p = sub.add_parser("tasks", help="任务集管理"); psub = p.add_subparsers(dest="sub", required=True)
    pi = psub.add_parser("init", help="从历史 run 抽取可用于配对的任务")
    pi.add_argument("--limit", type=int, default=24, help="最多抽取的任务数")
    pi.set_defaults(func=cmd_tasks_init)
    po = psub.add_parser("offline", help="生成本地数据任务集（全程不联网，不烧 Tavily 额度）")
    po.add_argument("--force-data", action="store_true", help="重写本地 CSV 数据集")
    po.set_defaults(func=cmd_tasks_offline)

    p = sub.add_parser("score", help="对单次 run 打分（replay 或 live）")
    p.add_argument("--log", default=None, help="run 日志路径")
    p.add_argument("--task-id", default="", help="任务标识（用于配对聚合）")
    p.add_argument("--out", default=None, help="评分结果输出路径")
    p.add_argument("--live", action="store_true", help="按实际产物打分（默认 replay）")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("calibrate", help="噪声标定：实测 sd(Δ) 并写出 calibration.json")
    p.add_argument("--tasks-file", default=None, help="任务清单路径")
    p.add_argument("--limit", type=int, default=6, help="参与标定的任务数")
    p.add_argument("--k", type=int, default=CFG.k_calib, help="每任务重复次数")
    p.add_argument("--avg-tokens", type=int, default=CFG.avg_tokens_default, help="单 run token 估算值（用于成本预估）")
    p.add_argument("--fake", action="store_true", help="使用受控模拟轨迹，不产生模型调用")
    p.add_argument("--yes", action="store_true", help="跳过成本确认")
    p.add_argument("--keep", action="store_true", help="保留沙箱目录")
    p.add_argument("--publish", action="store_true", help="将结果写入 calibration.json")
    p.add_argument("--offline", action="store_true", help="离线模式（不联网，本地语料）")
    p.add_argument("--parallel", type=int, default=None, help="并发 run 数（默认 1=串行）")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("round", help="执行一轮完整闭环（诊断→提案→探测→判定）")
    p.add_argument("--tasks-file", default=None, help="任务清单路径")
    p.add_argument("--limit", type=int, default=8, help="参与本轮的任务数")
    p.add_argument("--trace", default=None, help="直接指定 trace（跳过任务集）")
    p.add_argument("--task-label", default="round", help="任务标签（写入实验结果目录）")
    p.add_argument("--n", type=int, default=None, help="覆盖默认配对任务数")
    p.add_argument("--avg-tokens", type=int, default=CFG.avg_tokens_default, help="单 run token 估算值")
    p.add_argument("--fake", action="store_true", help="使用受控模拟轨迹，不产生模型调用")
    p.add_argument("--case", default="good", choices=("good", "zero", "worse"),
                   help="模拟轨迹类型：good=真实改进 / zero=零改动 / worse=故意变差")
    p.add_argument("--level", type=int, default=None, choices=(1, 2, 3),
                   help="只在该层选候选（配合 ADAPTIVE_MAX_EDIT_LEVEL 使用）")
    p.add_argument("--keep", action="store_true", help="保留沙箱目录")
    p.add_argument("--offline", action="store_true", help="离线模式（不联网，用本地数据任务集）")
    p.add_argument("--parallel", type=int, default=None, help="并发 run 数（默认 1=串行）")
    p.set_defaults(func=cmd_round)

    p = sub.add_parser("report", help="输出指定轮次的结论文档")
    p.add_argument("--round", default=None, help="轮次 ID（默认取最新）")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("base", help="Base 基线管理"); psub = p.add_subparsers(dest="base_cmd", required=True)
    psub.add_parser("show", help="查看当前 Base 与 env 快照").set_defaults(func=cmd_base)
    pp = psub.add_parser("promote", help="推进 Base（必须人工指定操作者）")
    pp.add_argument("--operator", default=None, help="操作者标识（禁止 system/auto）")
    pp.add_argument("--note", default="", help="推进说明")
    pp.set_defaults(func=cmd_base)

    sub.add_parser("evolve", help="输出探索与淘汰判定（exploration / pruning）").set_defaults(func=cmd_evolve)

    p = sub.add_parser("benchmark", help="多轮闭环：ΔQ 序列与 holdout 平稳性")
    p.add_argument("--tasks-file", default=None, help="任务清单路径")
    p.add_argument("--limit", type=int, default=12, help="可用任务总数上限")
    p.add_argument("--rounds", type=int, default=3, help="闭环轮数")
    p.add_argument("--n", type=int, default=4, help="每轮参与配对的任务数")
    p.add_argument("--holdout", type=float, default=0.3, dest="holdout",
                   help="留出任务比例（用于平稳性检查）")
    p.add_argument("--level", type=int, default=None, choices=(1, 2, 3), help="限定候选层级")
    p.add_argument("--trace", default=None, help="直接指定 trace")
    p.add_argument("--task-label", default="bench", help="任务标签")
    p.add_argument("--avg-tokens", type=int, default=CFG.avg_tokens_default, help="单 run token 估算值")
    p.add_argument("--fake", action="store_true", help="使用受控模拟轨迹，不产生模型调用")
    p.add_argument("--case", default="good", choices=("good", "zero", "worse"), help="模拟轨迹类型")
    p.add_argument("--keep", action="store_true", help="保留沙箱目录")
    p.add_argument("--offline", action="store_true", help="离线模式（不联网，用本地数据任务集）")
    p.add_argument("--parallel", type=int, default=None, help="并发 run 数（默认 1=串行）")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("auto", help="无人值守闭环：自己换基线 + 自己落地 + 自己出达成判据")
    p.add_argument("--tasks-file", default=None, help="任务清单路径（--offline 时默认取离线集）")
    p.add_argument("--limit", type=int, default=18, help="可用任务总数上限")
    p.add_argument("--rounds", type=int, default=3, help="目标总轮数（续跑时算的是累计值）")
    p.add_argument("--n", type=int, default=None, help="每轮参与配对的任务数（默认取 ADAPTIVE_N）")
    p.add_argument("--holdout", type=float, default=0.3, dest="holdout", help="holdout 比例")
    p.add_argument("--task-label", default="auto", help="运行标签（同一 label 可续跑）")
    p.add_argument("--out-dir", default=None, help="运行目录（默认 experiments/auto/<label>）")
    p.add_argument("--offline", action="store_true",
                   help="离线模式：不烧 Tavily 额度，且消除联网内容逐日变化带来的噪声")
    p.add_argument("--parallel", type=int, default=None, help="并发 run 数（墙钟把手，默认 1）")
    p.add_argument("--fake", action="store_true", help="受控模拟轨迹（只验管线，不会真 promote）")
    p.add_argument("--case", default="good", choices=("good", "zero", "worse"), help="模拟轨迹类型")
    p.add_argument("--no-resume", action="store_true", help="不续跑，重新开始一条曲线")
    p.add_argument("--no-verify", action="store_true",
                   help="跳过换基线后的 holdout 重测（不推荐：那就失去了「上线后是否仍更好」的观测）")
    p.add_argument("--verify-only", action="store_true",
                   help="不跑任何模型，只从磁盘产物重算达成判据")
    p.add_argument("--yes", action="store_true", help="跳过墙钟预估确认")
    p.add_argument("--keep", action="store_true", help="保留沙箱目录")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("regression", help="回归集管理（只否决不演化）")
    psub = p.add_subparsers(dest="reg_cmd", required=True)
    psub.add_parser("init", help="由已接受的候选建立回归集").set_defaults(func=cmd_regression)
    psub.add_parser("status", help="查看回归集覆盖情况").set_defaults(func=cmd_regression)

    p = sub.add_parser("trajectory", help="成功/失败轨迹分布对比（弱化版）")
    p.add_argument("--limit", type=int, default=40, help="参与对比的 run 数上限")
    p.set_defaults(func=cmd_trajectory)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    _apply_runtime_flags(args)
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
