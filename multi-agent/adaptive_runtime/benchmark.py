"""连续多轮 benchmark：判断闭环是否持续产生有效改进。

为何单轮 round 不足
--------------------
单轮只能回答「该候选在这一批任务上是否存在显著收益」。真正需要回答的是
「该机制能否让 agent 持续改进」。这需要两项额外证据：

1. **提升曲线**：第 1..k 轮的 ΔQ 形状。若自第 2 轮起始终为 WITHIN_NOISE，
   说明搜索空间已穷举或诊断信号不足 —— 这同样构成结论，且是需要记录的重要结论。
2. **holdout 平稳性**：在**未 promote** 的轮次中，holdout 任务的 base Q 应保持平稳。
   若其自行上升，说明存在泄漏（例如 holdout 任务参与了诊断 / 提案 / 经验写入），
   此时所有「提升」均不可信。该环节最易出现自证偏差，因此必须每轮检测。

诊断输入的来源（这一条改过，理由写在 ``diag_batch`` 里）
------------------------------------------------------
旧实现是 ``log = _newest_log()``：诊断读的是 ``log/`` 里 mtime 最新的那次人工运行，
与本轮探针跑出来的失败毫无关系 —— 那不是闭环，那是「对着一份随机历史开药方」。
现在诊断只吃**本轮自己产出的 base 臂 trace**（第 1 轮先做一次 base 臂 bootstrap 扫描），
并且机器排除 holdout 任务 id。要人工指定日志仍可用 ``--trace``，但会在 summary 里
标成 ``source="manual_log"``，报告里能看出这一轮的 provenance 不是自洽的。

边界说明（哪些仍然不成立）
--------------------------
- 本模块**自己**不 promote：它是「测量环」。无人值守的换基线在 ``auto.py`` +
  ``autopromote.py``，那里把人工门换成一组更硬的机器判据（不再是「必须是人」）。
- 因此 benchmark 的 base Q 只在（人工或 auto）promote 之后才应发生变化；
  未 promote 期间「holdout 不动」才是正确行为。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import diag_batch, diagnosis, evolution, proposer, version
from .comparator import summarize as cmp_summary
from .diagnosis import Taxonomy
from .evaluator import judge_version, score_trace
from .experience import Store
from .gate import Decision, explain, gate_a, gate_b, ledger_status
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
    #: holdout **配对**证据（泛化层）：与 evolve 集分开算，不参与提案
    holdout_mean_delta: float = 0.0
    holdout_pairs: int = 0
    holdout_delta_crit: float = 0.0
    #: 本轮诊断的 provenance：round_traces（自洽）/ manual_log（人工指定）/ bootstrap
    diag_source: str = ""
    diag_runs: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- 诊断与臂跑

def base_sweep(tasks: list[dict], out: str | Path, cfg: Settings, *, reps: int = 1,
               fake_profiles: dict[str, str] | None = None,
               keep_sandbox: bool = False) -> Path:
    """只跑 base 臂，产出一批可诊断的 run 目录（返回其 probe 目录）。

    为什么需要它：闭环第一轮还没有「本轮轨迹」可读，而 ``_newest_log()`` 那份人工日志
    不属于本闭环的证据链。宁可多花一轮 base 臂的机时，也要让第 1 轮的诊断同样来自
    「当前生产 harness 真实跑出来的失败」。

    走 ``run_one_isolated``：开了并发时 bootstrap 扫描同样需要每 run 一份隔离副本。

    并发：这里必须走 ``run_jobs``。旧写法是双层 for，即使 ``ADAPTIVE_MAX_PARALLEL=4``
    也会把 bootstrap 扫描（13 任务 × 2 rep）排成一个个串行 —— 实测每 run ~225 s，
    那就是 97 分钟只为了「让第一轮有诊断可读」，而同一批 run 并发 4 只需 25 分钟。
    探针与 holdout 重测走的是 ``probe_candidate``（本来就并发），于是真正耗时的
    反而不是主体而是脚手架。
    """
    from .probe import create_sandbox, run_jobs, run_one_isolated

    out = Path(out)
    probe_dir = out / "bootstrap"
    sb = create_sandbox(probe_dir / "sandbox-base", "base", keep=keep_sandbox)
    workers = max(1, int(getattr(cfg, "max_parallel", 1) or 1))
    try:
        jobs = [(f"{task['task_id']}#r{rep}",
                 (lambda tk=task, r=rep: run_one_isolated(
                     sb, tk, r, "base",
                     probe_dir / "base" / f"{tk['task_id']}-r{r}",
                     cfg=cfg, fake_profile=(fake_profiles or {}).get("base"))))
                for task in tasks for rep in range(reps)]
        run_jobs(jobs, workers)
    finally:
        if not keep_sandbox:
            sb.keep = False
            sb.cleanup()
    return probe_dir


def paired_holdout(cand, tasks: list[dict], out: str | Path, cfg: Settings, *,
                   reps: int | None = None,
                   fake_profiles: dict[str, str] | None = None,
                   keep_sandbox: bool = False) -> ProbeOutcome:
    """holdout 上跑**配对**探针 —— 「只看 base 均值」不是泛化证据，那是两次不同的噪声。

    自动换基线要求 holdout 提供的是「换一个没参与过诊断/提案的任务集，改动还成立吗」，
    这只有配对（同任务同 rep 的 base/cand）才能回答。单臂均值只能看漂移。
    """
    return probe_candidate(cand, tasks, Path(out) / "holdout",
                           cfg=cfg, n=reps or cfg.holdout_reps,
                           keep_sandbox=keep_sandbox, fake_profiles=fake_profiles,
                           stage1_tasks=max(1, len(tasks)), stage1_reps=1)


# ---------------------------------------------------------------- 主流程

def run_benchmark(tasks: list[dict], *, rounds: int = 3, n: int = 4,
                  holdout_ratio: float = 0.3, level: int | None = None,
                  trace: str | Path | None = None, out_dir: str | Path | None = None,
                  task_label: str = "bench", fake_profiles: dict[str, str] | None = None,
                  keep_sandbox: bool = False,
                  cfg: Settings | None = None,
                  on_accept=None,
                  ledger_path: str | Path | None = None,
                  force_dominant: str | None = None) -> dict:
    """跑 rounds 轮闭环，返回曲线与结论。

    ``on_accept(cand, rr, ctx)``：每轮 ACCEPT 后的回调（``auto.py`` 用它做自动换基线）。
    默认 None ⇒ 本模块继续作为「只测量、不换基线」的环。
    """
    cfg = cfg or CFG
    train, holdout = split(tasks, holdout_ratio)
    if not train:
        return {"error": "train 集为空，无法开始"}

    out = Path(out_dir) if out_dir else (PATHS.experiments / "benchmark" /
                                         f"{task_label}-{now_stamp()}")
    out.mkdir(parents=True, exist_ok=True)
    #: ledger_path 不传时退回全局默认（向后兼容：selftest / 单测仍走全局文件）。
    #: 传则按实验隔离（experiments/auto/<label>/ledger.jsonl），避免跨实验污染，
    #: 且能随 --no-resume 重置 —— 全局 ledger 做不到这两点。
    ledger = Ledger(ledger_path) if ledger_path else Ledger()
    store = Store()
    results: list[RoundResult] = []
    base_before = version.current()
    judge = judge_version(cfg)
    holdout_ids = {t["task_id"] for t in holdout}
    #: 下一轮的诊断输入 = 「当前生产 harness 真实跑出来的那一臂」的 run 目录。
    #: 为什么带 arm：一旦本轮自动 promote+落地，生产代码就等于本轮的 cand 臂，
    #: 再读 base 臂诊断就是在给「已经修好的病」开药。
    diag_src: list[tuple[Path, str]] = []

    for r in range(1, rounds + 1):
        rr = RoundResult(round_no=r)

        # ---- 诊断（只吃本轮/上一轮自己跑出来的 base 臂 trace）----
        if trace:
            log = Path(trace)
            if not log.exists():
                rr.notes.append(f"--trace 指定的日志不存在：{log}")
                results.append(rr)
                continue
            tr = export_from_log(log, root=PATHS.root)
            _s = score_trace(tr, root=PATHS.root, mode="replay",
                             task_id=task_label, arm="diag")
            _findings, dsum = diagnosis.from_trace(tr, score=_s)
            dsum["source"] = "manual_log"
            dsum["n_runs"] = 1
            rr.diag_source = "manual_log"
            rr.diag_runs = 1
        else:
            if not diag_src:
                diag_src = [(base_sweep(train[:max(1, min(4, len(train)))],
                                        out / "round0", cfg,
                                        fake_profiles=fake_profiles,
                                        keep_sandbox=keep_sandbox), "base")]
                rr.notes.append("第 1 轮：用 base 臂 bootstrap 扫描作为诊断输入（不读人工日志）")
            findings_all: list = []
            summary_acc: dict = {}
            for dd, arm in diag_src:
                fnd, sm = diag_batch.from_run_dirs(dd, arm=arm, cfg=cfg,
                                                   exclude_task_ids=holdout_ids)
                findings_all.extend(fnd)
                summary_acc = dict(sm, arm=arm)
            dsum = diagnosis.summarize(findings_all, cfg)
            dsum.update({k: v for k, v in summary_acc.items()
                         if k not in ("counts", "total", "dominant", "unknown_ratio", "hint")})
            diag_batch.write_diagnosis(out / f"round{r}", findings_all, dsum, cfg)
            rr.diag_source = str(dsum.get("source"))
            rr.diag_runs = int(dsum.get("n_runs") or 0)
        rr.dominant = force_dominant or dsum["dominant"]
        rr.unknown_ratio = float(dsum.get("unknown_ratio") or 0.0)
        if dsum.get("hint"):
            rr.notes.append(dsum["hint"])

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
            if not fake_profiles:
                # 状态由 ledger_status 定：Gate A 里 COST_CAP 只是「这轮买不起」，
                # 不是证伪（详见该函数 docstring），写死 rejected 会把假设永久烧掉。
                ledger.update(cand.hyp_fingerprint, ledger_status(dec_a),
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

        # 模拟模式仅用于管线自检，其产物为受控轨迹，不得写入台账与经验库：
        # 否则真实轮次会将这些假设视为「已尝试」（ledger 去重会跳过它们），
        # 经验库也会把无效证据检索进 Task Harness。
        is_fake = bool(fake_profiles)
        if dec.accepted and po.stopped != "stop_loss" and not is_fake:
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
        if not is_fake:
            # 状态必须按「这次拒绝是不是证伪」分类，而不是按 ACCEPT/REJECT 二分：
            # 超时导致的 INCOMPLETE_EVIDENCE 与 Stage 1 止损都还没对这个假设下任何结论，
            # 记成 rejected 就等于「一次测量事故永久否决一个改动」，闭环会在几轮内
            # 把整个搜索空间烧完，然后每轮 SKIP。见 gate.ledger_status。
            ledger.update(cand.hyp_fingerprint,
                          ledger_status(dec, stopped=po.stopped),
                          {"n": pr.n, "dQ": round(pr.mean, 4), "p": round(pr.p_value, 6),
                           "round": r, "reason": rr.reason,
                           "stopped": po.stopped},
                          tasks=[t["task_id"] for t in train])
        else:
            rr.notes.append("fake 模式：不写台账/经验库")

        # ---- holdout：配对证据 + 均值漂移 ----
        # 旧实现只跑 base 臂（只能看「有没有捣鬼」），但自动换基线需要的是
        # 「泛化层证据」，那必须是同任务同 rep 的配对差。
        holdout_cmp = None
        if holdout and dec.accepted and po.stopped != "stop_loss":
            try:
                ho = paired_holdout(cand, holdout, out / f"round{r}", cfg,
                                    fake_profiles=fake_profiles, keep_sandbox=keep_sandbox)
                holdout_cmp = ho.comparison
                if holdout_cmp is not None:
                    hev = holdout_cmp.evidence(cfg)
                    hpr = hev.paired(cfg.alpha)
                    rr.holdout_mean_delta = round(hpr.mean, 4)
                    rr.holdout_delta_crit = round(hpr.delta_crit, 4)
                    rr.holdout_pairs = len(holdout_cmp.pairs)
                    rr.holdout_base_q = round(_mean([p.base_Q for p in holdout_cmp.pairs]), 4)
                    rr.holdout_n = len(holdout_cmp.per_task)
                    rr.tokens += ho.tokens
                    if ho.stopped:
                        rr.notes.append(f"holdout 探针提前停止：{ho.stopped}")
            except Exception as e:
                rr.notes.append(f"holdout 配对探针失败：{type(e).__name__}: {e}")
        elif holdout:
            # 未被接受的候选不必跑泛化层（贵）；但漂移仍需一个观测点 ⇒ 只跑 base 臂
            hq = _holdout_base_quality(holdout, out / f"round{r}", cfg,
                                       fake_profiles=fake_profiles)
            rr.holdout_base_q, rr.holdout_n = hq[0], hq[1]

        # ---- 回归集（与 round 同口径；benchmark 自己不否决，只交给 on_accept/policy）----
        from . import regression as reg_mod
        reg_ids = {t["task_id"] for t in reg_mod.load()}
        reg = reg_mod.check(po.comparison, reg_ids)
        write_json(out / f"round{r}" / "regression_check.json", reg)

        # ---- 接受后的出口：benchmark 默认什么都不做（只测量），auto 在这里换基线 ----
        if on_accept is not None and dec.accepted and po.stopped != "stop_loss" and not is_fake:
            ctx = {"cfg": cfg, "round_no": r, "out_dir": out / f"round{r}",
                   "comparison": po.comparison, "evidence": ev, "paired": pr,
                   "holdout_comparison": holdout_cmp, "regression": reg,
                   "stopped": po.stopped, "train": train, "holdout": holdout,
                   "decision": dec, "judge_version": judge}
            try:
                res = on_accept(cand, rr, ctx) or {}
            except Exception as e:                  # 回调失败不能让整条曲线崩掉
                res = {"error": f"{type(e).__name__}: {e}"}
                rr.notes.append(f"on_accept 异常：{res['error']}")
            if res.get("promoted") and res.get("landed"):
                # 生产 harness 已经等于本轮 cand 臂 ⇒ 下一轮诊断改读 cand 臂
                diag_src = [(Path(po.out_dir), "cand")]
                rr.notes.append(f"已自动换基线并落地：{res.get('tag')}")
            elif res.get("promoted"):
                diag_src = [(Path(po.out_dir), "cand")]
                rr.notes.append("已换基线但未落地（land_changes=0）：版本升了而行为没变")
            else:
                diag_src = [(Path(po.out_dir), "base")]
        else:
            diag_src = [(Path(po.out_dir), "base")]
        cleanup(po)
        results.append(rr)

    base_after = version.current()
    report = _summarize(results, holdout, base_before, base_after, judge)
    #: 曲线同时进返回值与磁盘：调用方（``auto``）要把它累进学习曲线，
    #: 只返 out_dir 会让调用方去重读自己刚让别人写的文件，多一步可失败的 IO。
    report["curve"] = [r.to_dict() for r in results]
    write_json(out / "benchmark.json",
               {"curve": report["curve"], **report,
                "train_ids": [t["task_id"] for t in train],
                "holdout_ids": [t["task_id"] for t in holdout]})
    (out / "benchmark.md").write_text(_render(report, results), encoding="utf-8")
    report["out_dir"] = str(out)
    return report


def _holdout_base_quality(tasks: list[dict], out: Path, cfg: Settings,
                          fake_profiles: dict[str, str] | None) -> tuple[float, int]:
    """只跑 base 臂，用于观察「未 promote 的轮次里 holdout 有没有自己动」。

    同样走 ``run_jobs``：那是几个 holdout 任务 × 一个 run，串行就是白等。
    """
    from .probe import run_jobs, run_one_isolated
    from .probe import create_sandbox

    sb = create_sandbox(out / "sandbox-holdout-base", "base", keep=False)
    workers = max(1, int(getattr(cfg, "max_parallel", 1) or 1))
    jobs = [(task["task_id"],
             (lambda tk=task: run_one_isolated(
                 sb, tk, 0, "base", out / "holdout-base" / f"{tk['task_id']}-r0",
                 cfg=cfg, fake_profile=(fake_profiles or {}).get("base"))))
            for task in tasks]
    res = run_jobs(jobs, workers)
    qs = [float(ro.Q) for ro in res.values() if getattr(ro, "ok", False)]
    sb.keep = False
    sb.cleanup()          # 旧实现没删沙箱：benchmark 跑完会在 experiments 下留一堆副本
    return (_mean(qs), len(qs))


def _summarize(results: list[RoundResult], holdout: list[dict],
               base_before, base_after, judge: str) -> dict:
    accepted = [r for r in results if r.decision == "ACCEPT"]
    rejected = [r for r in results if r.decision == "REJECT"]
    skipped = [r for r in results if r.decision == "SKIP"]
    deltas = [r.mean_delta for r in results if r.n_tasks]
    holdout_qs = [r.holdout_base_q for r in results if r.holdout_n]
    holdout_deltas = [r.holdout_mean_delta for r in results if r.holdout_pairs]
    diag_sources = sorted({r.diag_source for r in results if r.diag_source})

    promoted = bool(base_before and base_after and
                    base_before.version != base_after.version)
    drift = 0.0
    if len(holdout_qs) >= 2:
        drift = holdout_qs[-1] - holdout_qs[0]

    verdict = []
    verdict.append(f"跑完 {len(results)} 轮：ACCEPT {len(accepted)} / REJECT {len(rejected)} "
                   f"/ SKIP {len(skipped)}")
    if deltas:
        verdict.append(f"ΔQ 序列（evolve 集）：{['%+.1f' % d for d in deltas]}")
    if holdout_deltas:
        verdict.append(f"ΔQ 序列（holdout 配对）：{['%+.1f' % d for d in holdout_deltas]}")
    if diag_sources:
        verdict.append("诊断输入来源：" + ", ".join(f"`{s}`" for s in diag_sources)
                       + ("" if diag_sources == ["round_traces"] else
                          "  ← 含 manual_log：那一轮的 provenance 不自洽，结论应降级"))
    if accepted:
        verdict.append("至少有一个改动真正过了噪声门 —— 闭环能产出有效结论")
    elif results:
        verdict.append("没有任何改动过门：要么搜索空间已穷举，要么诊断信号不足（看 UNKNOWN 占比）")
    if holdout and not promoted:
        if abs(drift) <= CFG.regression_min_points:
            verdict.append(f"holdout 平稳（drift={drift:+.2f}），未 promote 期间没有泄漏或漂移")
        else:
            verdict.append(f"[warn] holdout 漂移 {drift:+.2f} 且版本未 promote —— "
                           "存在泄漏或非确定性干扰，本轮结论不可信")
    if promoted:
        how = "自动 promote" if any("自动换基线" in n for r in results for n in r.notes) \
            else "人工 promote"
        verdict.append(f"base 版本 {base_before.version}→{base_after.version}（{how}），"
                       "此时 holdout 上升才算泛化证据")

    return {"judge_version": judge, "rounds": len(results),
            "accepted": len(accepted), "rejected": len(rejected), "skipped": len(skipped),
            "delta_series": [round(d, 4) for d in deltas],
            "holdout_series": [round(q, 2) for q in holdout_qs],
            "holdout_delta_series": [round(d, 4) for d in holdout_deltas],
            "holdout_drift": round(drift, 4),
            "diag_sources": diag_sources,
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
              "| 轮 | dominant | 候选 | L | 决策 | n | ΔQ | δ | p | holdout ΔQ | 诊断源 | tokens |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r.round_no} | {r.dominant or '-'} | `{r.target or '-'}` | "
                     f"{r.level or '-'} | {r.decision}/{r.reason} | {r.n_tasks} | "
                     f"{r.mean_delta:+.2f} | {r.delta_crit:.2f} | {r.p_value:.3f} | "
                     f"{(('%+.2f' % r.holdout_mean_delta) + f'/{r.holdout_pairs}') if r.holdout_pairs else '-'} | "
                     f"{r.diag_source or '-'}({r.diag_runs}) | {r.tokens:,} |")
    for r in results:
        if r.notes:
            lines.append(f"\n- 第 {r.round_no} 轮备注：{'; '.join(r.notes)}")
    lines += ["", "## 怎么读这份报告", "",
              "1. **ΔQ 序列**才是「有没有在变强」的证据。全为正且逐步过门 = 有效；"
              "全在噪声内 = 搜索空间或诊断信号已经不够了。",
              "2. **holdout ΔQ** 是唯一能算「泛化」的一列：它没参与诊断也没参与提案。"
              "未 promote 时 holdout 均值应平稳，若它自己往上走，先查泄漏。",
              "3. **单轮 ACCEPT 不等于累积提升**。累积要靠 promote 换基线（人工或 auto），"
              "再重跑本 benchmark 对比。",
              "4. **诊断源必须是 `round_traces`**。出现 `manual_log` 说明那一轮读的是别人跑的日志，"
              "它的 dominant 与本轮探针无因果关系。"]
    return "\n".join(lines) + "\n"
