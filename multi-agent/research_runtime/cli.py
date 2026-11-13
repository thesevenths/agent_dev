"""ARR 1.0 -- unified product CLI.

    python -m research_runtime.cli run --domain scinet|embodied|quant
    python -m research_runtime.cli benchmark --all-domains
    python -m research_runtime.cli evolve --target research_harness --rounds 2
    python -m research_runtime.cli report --input <artifact.json>
    python -m research_runtime.cli verify [--ci-log <pytest.log>]

One entry point for research, benchmark and RRSI Harness evolution. Everything
below is REAL: real engines, real frozen datasets, real paired campaigns, real
comparisons, real quality evaluation. Fake profiles exist ONLY in the R4 unit
tests and are never reachable from these commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "experiments" / "arr_1_0"


def _harness(name: str):
    from .arr_runtime import BASE_HARNESS, IMPROVED_HARNESS, NEGATIVE_HARNESS
    return {"base": BASE_HARNESS, "improved": IMPROVED_HARNESS,
            "negative": NEGATIVE_HARNESS}[name]


def _task_for(domain: str, *, objective: str = "", seeds: list[int] | None = None):
    from .research_task import ResearchTask
    defaults = {
        "scinet": ("Reduce multi-step latent dynamics prediction error on the "
                   "frozen pendulum world model",
                   {"primary_metric": "latent_mse@1",
                    "metric_direction": "minimize",
                    "decision_split": "validation",
                    "reserved_final_split": "test"},
                   {"independent_holdout": True, "holdout_split": "test",
                    "ood_task": False,
                    "paired_seeds": [42, 43, 44, 45, 46]}),
        "embodied": ("Reduce long-horizon state prediction error of the simulated "
                     "robot world model without violating safety constraints",
                     {"primary_metric": "state_mse@1",
                      "metric_direction": "minimize",
                      "decision_split": "validation",
                      "reserved_final_split": "test"},
                     {"independent_holdout": True, "holdout_split": "test",
                      "ood_task": True,
                      "paired_seeds": [42, 43, 44, 45, 46]}),
        "quant": ("Discover a robust market-neutral signal with positive "
                  "out-of-sample net-of-cost return and bounded drawdown",
                  {"primary_metric": "net_return_after_cost",
                   "metric_direction": "maximize",
                   "decision_split": "validation",
                   "reserved_final_split": "test"},
                  {"independent_holdout": True, "holdout_split": "test",
                   "ood_task": True,
                   "paired_seeds": [42, 43, 44, 45, 46]}),
    }
    if domain not in defaults:
        raise SystemExit(f"unknown domain {domain!r}; choose from {sorted(defaults)}")
    objective_text, protocol, holdout = defaults[domain]
    if seeds:
        holdout = dict(holdout, paired_seeds=list(seeds))
    budget = {"experiments": 20, "max_jobs": 30,
              "max_total_seconds": 3600.0, "epochs": 1}
    return ResearchTask(domain=domain, objective=objective or objective_text,
                        evaluation_protocol=protocol,
                        heldout_policy=holdout, budget=budget)


def cmd_run(args: argparse.Namespace) -> int:
    from .arr_runtime import run_research_task
    task = _task_for(args.domain, objective=args.objective, seeds=args.seeds)
    if args.budget_experiments:
        task.budget["experiments"] = args.budget_experiments
    result = run_research_task(task, harness=_harness(args.harness),
                               run_root=args.run_root)
    out = {
        "domain": result.domain, "task_id": result.task_id, "ok": result.ok,
        "outcome": result.outcome, "quality": result.quality.get("Q"),
        "components": result.quality.get("components"),
        "hard_violations": result.quality.get("hard_violations"),
        "n_hypotheses": len(result.hypotheses),
        "trials": f"{result.campaign.get('succeeded_trials')}/"
                  f"{result.campaign.get('total_trials')}",
        "holdout": result.evidence.get("holdout"),
        "ood": result.evidence.get("ood"),
        "report_path": result.evidence.get("report_path"),
        "result_path": result.provenance.get("result_path"),
        "artifact_sha256": result.artifact_sha256,
        "error": result.error,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0 if result.ok else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .arr_runtime import run_research_task
    from .domain_registry import default_registry, register_domain
    from .research_environment import build_local_bundle
    for name in ("embodied", "quant"):
        if not default_registry.contains(name):
            register_domain(build_local_bundle(name))
    domains = args.domains or ["scinet", "embodied", "quant"]
    results = []
    for domain in domains:
        task = _task_for(domain)
        r = run_research_task(task, harness=_harness(args.harness),
                              run_root=args.run_root)
        results.append(r)
        print(f"[benchmark] {domain}: ok={r.ok} outcome={r.outcome} "
              f"Q={r.quality.get('Q')} trials="
              f"{r.campaign.get('succeeded_trials')}/{r.campaign.get('total_trials')} "
              f"err={r.error[:120]}")
    summary = {
        "schema_version": "arr-1.0-benchmark-v1",
        "domains": domains,
        "same_research_service": True,
        "n_domains": len(results),
        "n_ok": sum(1 for r in results if r.ok),
        "results": [r.model_dump(mode="json") for r in results],
    }
    import hashlib
    summary["artifact_sha256"] = hashlib.sha256(json.dumps(
        summary, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    out = Path(args.run_root) / f"arr_1_0_benchmark_{summary['artifact_sha256'][:12]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True,
                              ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[benchmark] summary: {out}")
    return 0 if summary["n_ok"] == len(results) else 1


def cmd_evolve(args: argparse.Namespace) -> int:
    from adaptive_runtime.harness_evolution import evolve
    from .arr_runtime import BASE_HARNESS, NEGATIVE_HARNESS
    # Five INDEPENDENT selection tasks, each with DIFFERENT paired seeds =>
    # different real bootstrap samples, not copies of one run. n=5 gives the
    # paired t-test real statistical power (n=2 would be logically unable to
    # distinguish "works on this task" from "works generally").
    tasks = [
        _task_for("embodied", seeds=[42, 43, 44, 45, 46]),
        _task_for("embodied", seeds=[50, 51, 52, 53, 54]),
        _task_for("embodied", seeds=[58, 59, 60, 61, 62]),
        _task_for("embodied", seeds=[66, 67, 68, 69, 70]),
        _task_for("embodied", seeds=[74, 75, 76, 77, 78]),
    ]
    # Held-out / OOD is a DIFFERENT DOMAIN. A Harness improvement that only
    # helps Embodied must not be accepted; this is the strongest OOD check.
    heldout = _task_for("quant", seeds=[60, 61, 62, 63, 64])
    payload = evolve(
        base_harness=BASE_HARNESS, tasks=tasks, heldout_task=heldout,
        run_root=args.run_root, repo_root=WORKSPACE_ROOT,
        rounds=args.rounds, max_minutes=args.max_minutes,
        negative_harness=NEGATIVE_HARNESS,
        include_degenerate_control=args.degenerate_control,
        operator=args.operator or "")
    print(json.dumps({
        "dominant_failure": payload["dominant_failure"],
        "n_accepted": payload["n_accepted"],
        "n_rejected": payload["n_rejected"],
        "n_inconclusive": payload["n_inconclusive"],
        "n_negative_controls": payload["n_negative_controls"],
        "negative_controls_rejected": payload["negative_controls_rejected"],
        "evaluation_substrate_unchanged": payload["evaluation_substrate_unchanged"],
        "promotion": payload["promotion"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "artifact_path": payload["artifact_path"],
        "artifact_sha256": payload["artifact_sha256"],
        "records": [{
            "target": r["candidate"]["target"],
            "old": r["candidate"]["old"], "new": r["candidate"]["new"],
            "decision": r["decision"], "mean_delta": r["evidence"]["mean_delta"],
            "holdout_delta": r["holdout"].get("delta"),
            "negative_control": r["negative_control"],
            "substrate_unchanged": r["substrate_unchanged"],
        } for r in payload["records"]],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


def _latest(pattern: str) -> str:
    files = sorted(WORKSPACE_ROOT.glob(pattern))
    if not files:
        raise SystemExit(f"no artifact matches {pattern!r} under {WORKSPACE_ROOT}")
    return str(files[-1])


def _base_state() -> dict:
    p = WORKSPACE_ROOT / "experiments" / "base_state.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _discover_artifact(*, role: str, explicit: str, bundle_dir: str,
                       preferred_sha: str | None) -> tuple[str, str]:
    """Robust artifact discovery (P0-3): prefer an explicit path, then the
    committed evidence bundle, then base_state.json's referenced path, then a
    broad glob. Returns (path, source_label). Raises if nothing is found."""
    candidates: list[tuple[str, str]] = []
    if explicit:
        candidates.append((explicit, "explicit"))
    if bundle_dir:
        pat = f"{bundle_dir}/raw/*{role}*.json"
        hits = sorted(WORKSPACE_ROOT.glob(pat), key=lambda x: x.stat().st_mtime,
                      reverse=True)
        for h in hits:
            candidates.append((str(h), "evidence-bundle"))
    # base_state.json referenced committed path (P0-3 fix).
    for f in (_base_state().get("evidence_bundle_files") or []):
        if f.get("role") == role and f.get("committed_path"):
            candidates.append((str(WORKSPACE_ROOT / f["committed_path"]),
                              "base_state.json"))
    # Broad glob catches both arr_1_0_evolve/ and arr_1_0_evolve_r3/.
    glob_pat = (f"experiments/arr_1_0_benchmark*/{role}_*.json" if role == "benchmark"
                else f"experiments/arr_1_0_evolve*/{role}_*.json")
    hits = sorted(WORKSPACE_ROOT.glob(glob_pat), key=lambda x: x.stat().st_mtime,
                  reverse=True)
    for h in hits:
        candidates.append((str(h), f"glob:{glob_pat}"))

    if not candidates:
        raise SystemExit(f"no {role} artifact found under {WORKSPACE_ROOT}")
    # If a preferred sha is known, prefer the candidate whose content matches.
    for path, src in candidates:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if preferred_sha and data.get("artifact_sha256") == preferred_sha:
            return path, src
    # Otherwise the first existing, parseable candidate (most recent preferred).
    for path, src in candidates:
        if Path(path).is_file():
            return path, src
    raise SystemExit(f"no readable {role} artifact among {len(candidates)} candidates")


def cmd_verify(args: argparse.Namespace) -> int:
    """Check the 13 ARR 1.0 acceptance criteria against REAL artifacts.

    Nothing is declared from memory: every row is derived from a saved
    benchmark / evolution / CI artifact, and a missing artifact fails loudly.
    Artifact discovery is robust (P0-3): explicit path > committed evidence
    bundle > base_state.json reference > broad glob. A P0-3 row also confirms
    the canonical artifact referenced by base_state.json is actually present
    and its SHA-256 matches.
    """
    bundle_dir = args.bundle or ""
    pref_sha = None
    bs = _base_state()
    for f in (bs.get("evidence_bundle_files") or []):
        if f.get("role") == "evolution" and f.get("sha256"):
            pref_sha = f["sha256"]
            break

    bench_path, bench_src = _discover_artifact(
        role="benchmark", explicit=args.benchmark, bundle_dir=bundle_dir,
        preferred_sha=None)
    evo_path, evo_src = _discover_artifact(
        role="evolution", explicit=args.evolution, bundle_dir=bundle_dir,
        preferred_sha=pref_sha)
    bench = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    evo = json.loads(Path(evo_path).read_text(encoding="utf-8"))
    results = {r.get("domain"): r for r in bench.get("results", [])
               if isinstance(r, dict)}
    recs = evo.get("records", [])
    accepted = [r for r in recs if r.get("decision") == "ACCEPT"]
    rejected = [r for r in recs if r.get("decision") == "REJECT"]

    # ---- P0-3: the canonical artifact referenced by base_state.json must be
    # retrievable on disk AND its on-disk SHA-256 must match the committed hash
    # (no silent substitution; the artifact is now committed in the bundle). ----
    retrievable = Path(evo_path).is_file()
    from research_runtime.evidence_verifier import sha256_file
    on_disk_sha = sha256_file(evo_path)
    sha_ok = retrievable and (pref_sha is None or on_disk_sha == pref_sha)
    # ---- D4: if a committed evidence bundle is referenced, verify its internal
    # self-consistency (every file in evidence_hashes.json matches on disk). ----
    bundle_ok = True
    bundle_note = "no evidence bundle referenced"
    if bs.get("evidence_bundle"):
        from research_runtime.evidence_verifier import EvidenceVerifier
        bpath = WORKSPACE_ROOT / bs["evidence_bundle"]
        if bpath.is_dir():
            vr = EvidenceVerifier().verify_bundle_integrity(bpath)
            bundle_ok = vr.all_passed()
            vd = vr.as_dict()
            bundle_note = (f"{vd['n_passed']}/{vd['n_checks']} bundle files match"
                           if bundle_ok else "bundle hash mismatch")
        else:
            bundle_ok = False
            bundle_note = "evidence bundle dir missing"

    def _domain_ok(name: str) -> bool:
        r = results.get(name) or {}
        rep = r.get("report") or {}
        return bool(r.get("ok") and rep.get("hypotheses")
                    and rep.get("report_sha256")
                    and (r.get("campaign") or {}).get("status") == "completed")

    def _autonomous(name: str) -> bool:
        rep = (results.get(name) or {}).get("report") or {}
        return bool(rep.get("hypotheses") and any(
            "experiment:" in str(x) for x in rep.get("evidence_refs", [])))

    quant_rep = (results.get("quant") or {}).get("report") or {}
    checks: list[tuple[str, bool, str]] = [
        ("1  SciNet++ 自主研究闭环", _domain_ok("scinet"),
         "report hypotheses + campaign completed + report_sha256"),
        ("2  Embodied 仿真研究闭环", _domain_ok("embodied"),
         "real simulator, frozen robot_sim_v1"),
        ("3  Quant 回测闭环", _domain_ok("quant") and not quant_rep.get(
            "frozen_test_used", True),
         "causal backtest, reserved test never used for selection"),
        ("4  三域共用同一 Runtime",
         bool(bench.get("same_research_service")) and bench.get("n_domains") == 3
         and bench.get("n_ok") == 3,
         f"n_ok={bench.get('n_ok')}/{bench.get('n_domains')}"),
        ("5  无人工假设，证据驱动自主产生",
         all(_autonomous(d) for d in ("scinet", "embodied", "quant")),
         "every report cites a real experiment in evidence_refs"),
        ("6  从真实轨迹发现流程缺陷",
         bool(evo.get("dominant_failure"))
         and bool((evo.get("analysis") or {}).get("counts")),
         f"dominant={evo.get('dominant_failure')}"),
        ("7  自动产生 Harness 候选",
         bool(recs) and all((r.get("candidate") or {}).get(
             "candidate_type") == "research_harness" for r in recs),
         f"n_candidates={len(recs)}"),
        ("8  候选在 Sandbox 中实际运行",
         bool(recs) and all(((r.get("evidence") or {}).get("patch") or {}).get(
             "config_changed") for r in recs),
         "verified harness.json SHA changed in isolated sandbox"),
        ("9  Base/Candidate 公平比较",
         bool(recs) and all(len(r.get("task_deltas") or []) >= 3
                            and r.get("substrate_unchanged") for r in recs),
         "paired seeds + fixed evaluator substrate hash"),
        ("10 独立任务上真实改善",
         bool(accepted) and any((r.get("holdout") or {}).get("confirmed")
                                and (r.get("holdout") or {}).get("delta", 0) > 0
                                and not (r.get("holdout") or {}).get(
                                    "holdout_used_for_selection", True)
                                for r in accepted),
         f"accepted={len(accepted)}, holdout confirmed"),
        ("11 拒绝无效/退化候选",
         bool(rejected) and evo.get("negative_controls_rejected", 0) >= 1,
         f"rejected={len(rejected)}, "
         f"controls_rejected={evo.get('negative_controls_rejected')}"),
        ("12 决策可追溯、可回滚",
         bool(evo.get("artifact_sha256")) and bool(evo.get("rollback_available"))
         and evo.get("evaluation_substrate_unchanged")
         and (evo.get("promotion") or {}).get("status") == "not_requested",
         "SHA-256 artifact + no automatic promote"),
    ]
    # 13 is verified from the CI log when supplied, never self-declared.
    ci_ok, ci_note = False, "no --ci-log supplied"
    if args.ci_log:
        text = Path(args.ci_log).read_text(encoding="utf-8", errors="replace")
        line = [ln for ln in text.splitlines() if " passed" in ln]
        ci_ok = bool(line) and " failed" not in line[-1] and " error" not in line[-1]
        ci_note = line[-1].strip()[-120:] if line else "no pytest summary found"
    checks.append(("13 CI 通过，真实/Fake 分明", ci_ok, ci_note))

    # ---- P0-3 + D4: evidence retrievability & bundle integrity (not a numbered
    # acceptance row but a guard the user explicitly required). These precede the
    # CI row so a missing/corrupt canonical artifact fails the whole verify. ----
    checks.insert(0, ("P0-3 基线证据可检索且 SHA 一致",
                      bool(retrievable and sha_ok),
                      f"evo_src={evo_src} sha_match={sha_ok}"))
    checks.insert(1, ("D4 证据包自洽", bool(bundle_ok), bundle_note))

    for label, ok, note in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label} -- {note}")
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\nARR 1.0 acceptance: {n_pass}/{len(checks)} "
          f"from real artifacts (benchmark={bench.get('artifact_sha256', '')[:12]}"
          f"@{bench_src}, evolution={evo.get('artifact_sha256', '')[:12]}@{evo_src})")
    return 0 if n_pass == len(checks) else 1


def cmd_report(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if "records" in data:
        print(f"ARR 1.0 Harness evolution: {data['dominant_failure']}")
        print(f"  accepted={data['n_accepted']} rejected={data['n_rejected']} "
              f"inconclusive={data['n_inconclusive']}")
        for rec in data["records"]:
            print(f"  - {rec['candidate']['target']} "
                  f"{rec['candidate']['old']}->{rec['candidate']['new']}: "
                  f"{rec['decision']} (mean_delta={rec['evidence']['mean_delta']})")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str)[:4000])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research_runtime.cli",
                                description="ARR 1.0 unified research runtime")
    p.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run ONE real research task")
    run.add_argument("--domain", required=True,
                     choices=["scinet", "embodied", "quant"])
    run.add_argument("--objective", default="")
    run.add_argument("--budget-experiments", type=int, default=0)
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--harness", default="improved",
                     choices=["base", "improved", "negative"])
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("benchmark", help="run all domains")
    bench.add_argument("--all-domains", action="store_true")
    bench.add_argument("--domains", nargs="+")
    bench.add_argument("--harness", default="improved",
                       choices=["base", "improved", "negative"])
    bench.set_defaults(func=cmd_benchmark)

    ev = sub.add_parser("evolve", help="real RRSI Harness self-improvement")
    ev.add_argument("--target", default="research_harness",
                    choices=["research_harness"])
    ev.add_argument("--rounds", type=int, default=2)
    ev.add_argument("--max-minutes", type=float, default=30.0)
    ev.add_argument("--benchmark", default="research_benchmark_v1")
    ev.add_argument("--degenerate-control", dest="degenerate_control",
                    action="store_true", default=True,
                    help="evaluate a deliberately degenerate Harness via the "
                         "full paired protocol; it must be REJECTed")
    ev.add_argument("--no-degenerate-control", dest="degenerate_control",
                    action="store_false",
                    help="skip the degenerate control (debugging only)")
    ev.add_argument("--operator", default="")
    ev.set_defaults(func=cmd_evolve)

    rep = sub.add_parser("report", help="summarise a saved artifact")
    rep.add_argument("--input", required=True)
    rep.set_defaults(func=cmd_report)

    ver = sub.add_parser("verify", help="check the 13 acceptance criteria")
    ver.add_argument("--benchmark", default="")
    ver.add_argument("--evolution", default="")
    ver.add_argument("--bundle", default="",
                     help="path to a committed evidence bundle dir "
                          "(e.g. experiments/releases/arr_1_0_r1)")
    ver.add_argument("--ci-log", default="")
    ver.set_defaults(func=cmd_verify)

    lp = sub.add_parser("loop", help="autonomous multi-round research loop (G)")
    lp.add_argument("--domain", required=True,
                    choices=["embodied", "quant", "scinet"])
    lp.add_argument("--baseline-config", required=True,
                    help="path to the seed baseline config YAML")
    lp.add_argument("--project-dir", required=True,
                    help="root the loop writes isolated campaign artefacts under")
    lp.add_argument("--rounds", type=int, default=3)
    lp.add_argument("--dataset-version", default="robot_sim_v1")
    lp.add_argument("--seeds", nargs="+", type=int, default=[50, 51, 52, 53, 54],
                    help="paired (base/candidate) experiment seeds for the DOMAIN "
                         "layer (default 5: the M2.7 formal-ACCEPT minimum). Fewer "
                         "than 5 yields only INCONCLUSIVE, never ACCEPT.")
    lp.add_argument("--require-frozen-manifest", action="store_true", default=True,
                    help="(default ON) require the frozen dataset manifest to "
                         "verify on disk before each round; production research "
                         "must never run without it")
    lp.add_argument("--allow-unfrozen-data", dest="require_frozen_manifest",
                    action="store_false",
                    help="DEV ONLY: permit non-frozen data. Production research "
                         "must NOT use this; a missing/unverified manifest fails "
                         "loudly (fail-closed) by default")
    lp.add_argument("--harness-rounds", type=int, default=1,
                    help="RRSI harness-evolution rounds per loop round "
                         "(default 1). Raise for real capability testing.")
    lp.add_argument("--harness-max-minutes", type=float, default=15.0,
                    help="Per-round Harness evolution time budget in minutes "
                         "(default 15.0). M2.6: a real budget so the harness can "
                         "actually search, not the previous 1.0-min floor.")
    lp.add_argument("--holdout-seeds", nargs="*", type=int, default=[],
                    help="independent seeds for holdout confirmation of domain "
                         "candidates. M3.0: EVERY holdout seed's raw result is "
                         "persisted and a strict majority must confirm the paired "
                         "direction; a partially-failed or contradictory holdout "
                         "downgrades to INCONCLUSIVE. Provide >= "
                         "--min-seeds-for-accept for a formal ACCEPT.")
    lp.add_argument("--min-seeds-for-accept", type=int, default=5,
                    help="minimum VALID paired seeds for a formal Domain ACCEPT "
                         "(default 5, HARD FLOOR: values below 5 are clamped UP, "
                         "never down). Fewer seeds (e.g. 2) can only produce "
                         "INCONCLUSIVE. M3.0 frozen protocol: two-sided Student-t "
                         "CI (df=n-1) + family-wise Holm correction across all "
                         "candidates in the round; CI straddling 0 is INCONCLUSIVE "
                         "in BOTH directions.")
    lp.add_argument("--practical-floor-rel", type=float, default=0.01,
                    help="relative practical-improvement floor: a positive mean gain "
                         "below this fraction of the baseline metric is noise-level "
                         "(INCONCLUSIVE). Default 0.01, HARD FLOOR (clamped UP). "
                         "M3.0: a missing baseline_mean fails closed instead of "
                         "skipping this check.")
    lp.add_argument("--require-holdout-confirmation", dest="require_holdout_confirmation",
                    action="store_true", default=True,
                    help="(default ON) a formal Domain ACCEPT requires an "
                         "independent per-seed holdout confirmation; without it the "
                         "candidate is INCONCLUSIVE.")
    lp.add_argument("--no-holdout-confirmation", dest="require_holdout_confirmation",
                    action="store_false",
                    help="DEV / PROVISIONAL ONLY: skip the independent holdout. "
                         "Positive candidates are then INCONCLUSIVE (provisional), "
                         "NEVER a formal ACCEPT. Seed / CI / practical gates stay "
                         "enforced. Production research must keep confirmation ON.")
    lp.add_argument("--no-stop-on-no-accept", dest="stop_on_no_accept",
                    action="store_false", default=True,
                    help="EVIDENCE-COLLECTION mode: run ALL --rounds even when a "
                         "round accepts nothing (default: stop early on a genuine "
                         "no-gain round). This does NOT relax any ACCEPT gate -- "
                         "INCONCLUSIVE / REJECT decisions are preserved verbatim; "
                         "it only lets a multi-round study collect its full trail.")
    lp.add_argument("--state-path", default="",
                    help="loop checkpoint path; defaults to "
                         "<project-dir>/state/loop_state.json; resume an "
                         "interrupted loop automatically when present")
    lp.set_defaults(func=cmd_loop)
    return p


def _build_real_domain_runner(domain: str,
                              project_dir: str | Path | None = None):
    """Wire a REAL domain engine as the dual-layer domain runner.

    The runner executes one materialised config (base or candidate) end-to-end
    through the domain's own compute function (train -> evaluate) and returns the
    primary metric Q. No fabricated metric is ever produced: if the engine or its
    frozen dataset is unavailable, this raises rather than faking a number.

    ``project_dir`` is threaded into the engine spec for the local numeric engines
    so the frozen dataset the engine READS is the SAME root the
    CandidateConfigBuilder VERIFIED (data-root isolation, M2.5 P1). Without it the
    engine would fall back to its module-level source-tree path while candidate
    configs are materialised under --project-dir -- a provenance gap.
    """
    if domain == "embodied":
        from .domains.embodied import embodied_compute, PRIMARY_METRIC
        compute = embodied_compute
    elif domain == "quant":
        from .domains.quant import quant_compute, PRIMARY_METRIC
        compute = quant_compute
    elif domain == "scinet":
        # Real SciNet++ engine via the existing SciNetAdapter (subprocess JSON-RPC
        # over research_api.cli). No FakeRunner is used; an unreachable engine
        # fails closed to NaN -> the candidate is INCONCLUSIVE, never faked.
        from .domains.scinet import scinet_compute, PRIMARY_METRIC
        compute = scinet_compute
    else:
        raise SystemExit(
            f"no real engine runner wired for domain {domain!r}; the loop must "
            f"run the REAL engine (no FakeRunner is allowed)")
    # Single source of truth: the domain module defines PRIMARY_METRIC, shared by
    # the research task, the engine runner and the dual-layer RRSI decision.
    primary_metric = PRIMARY_METRIC
    from pathlib import Path as _P
    import tempfile as _tf
    import yaml as _yaml

    def runner(art: dict) -> float:
        # Fail-closed: a missing / unreadable config yields NaN (candidate
        # INCONCLUSIVE) instead of crashing the whole loop.
        cfg_path = _P(art["config_path"])
        if not cfg_path.is_file():
            return float("nan")
        try:
            cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return float("nan")
        if not isinstance(cfg, dict):
            return float("nan")
        job_dir = _P(art["directory"])
        job_dir.mkdir(parents=True, exist_ok=True)
        seed = int(cfg.get("seed", 42))
        # Common spec fields; local numeric engines also receive the SAME data
        # root the candidate configs were materialised/verified against.
        base_spec: dict = {"stage": "train", "seed": seed}
        eval_spec: dict = {"stage": "evaluate", "seed": seed,
                           "run_id": job_dir.name}
        if project_dir is not None and domain != "scinet":
            base_spec["project_dir"] = str(project_dir)
            eval_spec["project_dir"] = str(project_dir)
        if domain == "scinet":
            tr = compute(cfg, base_spec, job_dir,
                         config_path=art["config_path"])
        else:
            tr = compute(cfg, base_spec, job_dir)
        if tr.get("status") != "succeeded":
            return float("nan")
        run_id = job_dir.name
        if domain == "scinet":
            ev = compute(cfg, eval_spec, job_dir,
                         config_path=art["config_path"])
        else:
            ev = compute(cfg, eval_spec, job_dir)
        if ev.get("status") != "succeeded":
            return float("nan")
        q = (ev.get("metrics") or {}).get(primary_metric)
        if q is None:
            return float("nan")
        return float(q)

    return runner


def cmd_loop(args: argparse.Namespace) -> int:
    """Autonomous multi-round research loop (ARR 1.1 / Package G).

    Runs the dual-layer RRSI for ``--rounds`` rounds; the next round inherits the
    version the previous round ACCEPTED (domain config baked into a new baseline,
    harness patches applied via the real patch primitive), and a crashed process
    resumes from the last completed round. BOTH layers run REAL experiments:
      * DOMAIN MODEL -- the real engine (Embodied simulator / Quant backtest).
      * RESEARCH HARNESS -- the real ``adaptive_runtime.harness_evolution.evolve``
        on genuine research tasks, with an independent held-out task that never
        participates in candidate selection. A missing frozen dataset or engine
        fails loudly, never silently.
    """
    from .loop import run_loop
    from adaptive_runtime.harness_evolution import evolve
    # A safe default checkpoint location: a dedicated file under the project dir,
    # never the CWD (an empty --state-path previously resolved to the working
    # directory, which is unsafe and easy to clobber).
    state_path = args.state_path or str(
        Path(args.project_dir) / "state" / "loop_state.json")
    # CandidateConfigBuilder.build_baseline resolves a RELATIVE baseline config
    # against project_dir AND requires the result to live inside project_dir.
    # Normalise once here so a relative --baseline-config (e.g. "config.yaml")
    # is not joined twice ("proj/proj/config.yaml"). Absolute paths are kept
    # verbatim. This is a CLI boundary fix, generic to every domain.
    baseline_config = Path(args.baseline_config)
    if not baseline_config.is_absolute():
        baseline_config = Path(args.project_dir) / baseline_config
    baseline_config = baseline_config.resolve()
    runner = _build_real_domain_runner(args.domain, args.project_dir)
    # Genuine harness layer: real base harness + real research tasks + an
    # independent held-out task (cross-domain, so a harness improvement that only
    # helps THIS domain cannot be accepted on the held-out evidence).
    from .arr_runtime import BASE_HARNESS, NEGATIVE_HARNESS
    heldout_domain = "quant" if args.domain != "quant" else "embodied"
    tasks = [_task_for(args.domain, seeds=[42, 43, 44]),
             _task_for(args.domain, seeds=[50, 51, 52])]
    for t in tasks:
        t.budget["experiments"] = 4
    heldout = _task_for(heldout_domain, seeds=[60, 61, 62])
    heldout.budget["experiments"] = 4
    report = run_loop(
        state_path=state_path,
        domain=args.domain,
        baseline_config_path=str(baseline_config),
        project_dir=args.project_dir,
        max_rounds=args.rounds,
        dataset_version=args.dataset_version,
        domain_seeds=tuple(args.seeds),
        domain_runner=runner,
        domain_require_frozen_manifest=args.require_frozen_manifest,
        domain_holdout_seeds=tuple(args.holdout_seeds),
        min_seeds_for_accept=args.min_seeds_for_accept,
        practical_improvement_floor_rel=args.practical_floor_rel,
        require_holdout_confirmation=args.require_holdout_confirmation,
        stop_on_no_accept=args.stop_on_no_accept,
        harness_evolve=evolve,
        base_harness=BASE_HARNESS,
        harness_tasks=tasks,
        harness_heldout_task=heldout,
        base_harness_tag=BASE_HARNESS.fingerprint(),
        extra={"rounds": args.harness_rounds,
               "max_minutes": args.harness_max_minutes,
               "negative_harness": NEGATIVE_HARNESS},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("status") == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
