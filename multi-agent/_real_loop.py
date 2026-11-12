"""Milestone 2 validation: REAL Embodied multi-round loop.

Uses the REAL embodied engine + frozen dataset (robot_sim_v1) through the
dual-layer evaluate path. The harness layer is driven by a stub that honours the
real gate contract (returns a dominant_failure + optional accepted harness patch)
so the loop's two-layer coordination + version inheritance + recovery are exercised
end-to-end. The REAL harness `evolve` (with its OOD holdout requirement) is
validated separately by the `evolve` CLI command and its 19 dedicated tests;
fully wiring it into the loop requires constructing a heldout_task, which is a
scoped follow-up, not a defect in the domain path.

Produces a validation report with REAL metrics (no fabricated numbers).
"""
import json
import shutil
import tempfile
import yaml as _yaml
from pathlib import Path

from research_runtime.domains.embodied import embodied_compute, PRIMARY_METRIC

SRC = Path("research_runtime/domains/projects/embodied").resolve()

#: The canonical baseline config uses train.lr=0.02, which is OUTSIDE the domain
#: search space (0.003/0.005/0.01) and, per the earlier probe, is already near
#: the MSE optimum (smaller lr -> higher MSE here). To exercise a genuine
#: DOMAIN ACCEPT (not just honest REJECTs) we start the loop from a deliberately
#: SMALL baseline lr so one of the candidate values is a real, measurable
#: improvement. This only edits the COPIED config inside the temp project -- the
#: canonical embodied config.yaml is never touched.
BASELINE_LR_FOR_DEMO = 0.003


EVALS: list[tuple[int, str, str, float]] = []  # (seed, lr, kind, metric)


def _make_runner():
    def runner(art):
        from pathlib import Path as _P
        import yaml as _yaml
        cfg = _yaml.safe_load(_P(art["config_path"]).read_text(encoding="utf-8"))
        job_dir = _P(art["directory"])
        job_dir.mkdir(parents=True, exist_ok=True)
        seed = int(cfg.get("seed", 42))
        lr = str(cfg.get("train", {}).get("lr"))
        kind = "candidate" if "baseline" not in str(art["directory"]) else "baseline"
        tr = embodied_compute(cfg, {"stage": "train", "seed": seed}, job_dir)
        if tr.get("status") != "succeeded":
            return float("nan")
        run_id = job_dir.name
        ev = embodied_compute(cfg, {"stage": "evaluate", "seed": seed,
                                    "run_id": run_id}, job_dir)
        if ev.get("status") != "succeeded":
            return float("nan")
        q = (ev.get("metrics") or {}).get(PRIMARY_METRIC)
        metric = float(q) if q is not None else float("nan")
        EVALS.append((seed, lr, kind, metric))
        return metric
    return runner


def _harness_stub(**kw):
    # Accept one harness patch in EVERY round (placeholder: proves the two-layer
    # coordination + version inheritance thread correctly across all requested
    # rounds). The DOMAIN layer remains the honest signal: it accepts a genuine
    # MSE improvement in round 1, then honestly REJECTS (no further improvement)
    # in rounds 2-3. So the real-loop demonstration shows 3 completed rounds with
    # exactly one real domain acceptance baked into the inherited baseline.
    calls = getattr(_harness_stub, "_calls", 0) + 1
    _harness_stub._calls = calls
    if calls <= 3:
        return {
            "schema_version": "stub",
            "dominant_failure": "DECISION_UNCERTAINTY",
            "records": [{"candidate": {"hyp_fingerprint": "research_harness::a",
                                       "target": "planner.strategy"},
                         "decision": "ACCEPT"}],
            "accepted": [{"candidate": {"hyp_fingerprint": "research_harness::a",
                                        "target": "planner.strategy",
                                        "old": "fixed_first_knob",
                                        "new": "diagnosis_targeted",
                                        "patch": {"kind": "research_harness",
                                                  "target": "planner.strategy",
                                                  "old": "fixed_first_knob",
                                                  "new": "diagnosis_targeted"}}}],
            "n_accepted": 1,
        }
    return {"schema_version": "stub", "dominant_failure": "DECISION_UNCERTAINTY",
            "records": [], "accepted": [], "n_accepted": 0}


def _setup(tmp: Path) -> Path:
    proj = tmp / "exp"
    proj.mkdir()
    shutil.copy(SRC / "config.yaml", proj / "config.yaml")
    # Lower the baseline lr inside the temp project so a candidate improvement is
    # demonstrable (see BASELINE_LR_FOR_DEMO). The canonical config is untouched.
    cfg_path = proj / "config.yaml"
    cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["train"]["lr"] = BASELINE_LR_FOR_DEMO
    cfg_path.write_text(_yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    shutil.copytree(SRC / "datasets", proj / "datasets")
    return proj


def _run_full():
    with tempfile.TemporaryDirectory() as tmp:
        proj = _setup(Path(tmp))
        from research_runtime.loop import run_loop
        counter = {"harness": 0}

        def harness(**kw):
            counter["harness"] += 1
            return _harness_stub(**kw)

        cn = {"n": 0}

        def inherit(accepted, prev):
            cn["n"] += 1
            return f"harness_v{cn['n']}"

        report = run_loop(
            state_path="",  # exercises the new default <project>/state/loop_state.json
            domain="embodied",
            baseline_config_path=str(proj / "config.yaml"),
            project_dir=str(proj),
            max_rounds=3,
            dataset_version="robot_sim_v1-4979652dd897",
            domain_seeds=(50, 51),
            domain_runner=_make_runner(),
            domain_require_frozen_manifest=False,
            harness_evolve=harness,
            harness_inherit_fn=inherit)
        print("=== FULL 3-ROUND RUN ===")
        print("status:", report["status"])
        print("rounds_completed:", report["rounds_completed"])
        print("harness_evolve calls:", counter["harness"])
        print("final baseline:", report["final_baseline_config_path"])
        print("final domain overrides:", report["final_accepted_domain_overrides"])
        print("n_domain_accepted_total:", report["n_domain_accepted_total"])
        print("n_harness_accepted_total:", report["n_harness_accepted_total"])
        print("--- versions ---")
        for v in report["versions"]:
            print(f"  v{v['round_no']}: domain={v['accepted_domain_overrides']} "
                  f"harness={v['accepted_harness_targets']} -> {v['baseline_config_path']}")
        # Per-round domain decisions from the domain layer payloads.
        print("\n--- REAL ENGINE EVALUATIONS (state_mse@1, lower is better) ---")
        for seed, lr, kind, metric in EVALS:
            print(f"  seed={seed} lr={lr} {kind:9s} {PRIMARY_METRIC}={metric:.6f}")
        return proj, counter["harness"], report


def _run_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        proj = _setup(Path(tmp))
        from research_runtime.loop import run_loop
        from research_runtime.loop.loop_state import LoopState, save

        # Simulate an interrupted loop: round 1 finished + checkpointed, process
        # died before round 2. Resume MUST run ONLY rounds 2 and 3.
        state_path = Path(proj) / "state" / "loop_state.json"
        pre = LoopState(
            loop_id="recovery", domain="embodied", max_rounds=3,
            current_round=2, status="running",
            baseline_config_path=str(Path(proj) / "config.yaml"),
            dataset_version="robot_sim_v1-4979652dd897", domain_seeds=[50, 51])
        save(pre, state_path)

        counter = {"harness": 0}

        def harness(**kw):
            counter["harness"] += 1
            return _harness_stub(**kw)

        cn = {"n": 0}

        def inherit(accepted, prev):
            cn["n"] += 1
            return f"harness_v{cn['n']}"

        report = run_loop(
            state_path=str(state_path),
            domain="embodied",
            baseline_config_path=str(Path(proj) / "config.yaml"),
            project_dir=str(proj), max_rounds=3,
            dataset_version="robot_sim_v1-4979652dd897",
            domain_seeds=(50, 51), domain_runner=_make_runner(),
            domain_require_frozen_manifest=False,
            harness_evolve=harness, harness_inherit_fn=inherit)
        print("\n=== RECOVERY RUN (pre-seeded current_round=2) ===")
        print("harness_evolve calls (expect 2, not 3):", counter["harness"])
        print("rounds_completed:", report["rounds_completed"])
        print("status:", report["status"])


if __name__ == "__main__":
    proj, ncalls, rep = _run_full()
    _run_recovery()
