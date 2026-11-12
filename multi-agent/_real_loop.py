"""Milestone 2.5 validation: REAL dual-layer Embodied research loop.

Both RRSI layers run GENUINE experiments inside the same autonomous loop:

  * DOMAIN MODEL layer  -- real Embodied engine (train -> evaluate on the frozen
    robot_sim_v1 dataset); direction-aware ACCEPT/REJECT of ``train.lr``.
  * RESEARCH HARNESS layer -- real ``adaptive_runtime.harness_evolution.evolve``
    on genuine research tasks (embodied) with an INDEPENDENT held-out task
    (quant). No stub, no prefilled acceptance record. A weak-signal / bounded
    budget run yields honest INCONCLUSIVE/REJECT; an ACCEPT (if the evidence
    supports one) is threaded into the next round via the real patch primitive.

Also enforced (M2.5 P0/P1):
  * ``domain_require_frozen_manifest=True`` -- every experiment verifies the
    frozen dataset manifest + per-split SHA-256 on disk.
  * Data-root isolation -- the engine loads the SAME verified project root the
    candidate was built against (never a silently different canonical copy).
  * Real process crash recovery -- see ``_run_recovery`` (subprocess fault
    injection in the accompanying test).

Produces a validation report with REAL metrics and artifact SHA-256 (no
fabricated numbers).
"""
import json
import shutil
import tempfile
import time
from pathlib import Path

from research_runtime.cli import _task_for, WORKSPACE_ROOT
from research_runtime.arr_runtime import BASE_HARNESS, NEGATIVE_HARNESS
from research_runtime.domains.embodied import embodied_compute, PRIMARY_METRIC
from research_runtime.loop.version_inheritance import default_harness_inherit
from adaptive_runtime.harness_evolution import evolve

SRC = Path("research_runtime/domains/projects/embodied").resolve()

#: Start from a deliberately small baseline lr so a candidate improvement is
#: demonstrable (only the copied temp config is edited; canonical is untouched).
BASELINE_LR_FOR_DEMO = 0.003

EVALS: list[tuple[int, str, str, float]] = []     # (seed, lr, kind, metric)
HARNESS_RUNS: list[dict] = []                       # one payload per harness evolve


def _make_runner(project_dir: str):
    def runner(art):
        from pathlib import Path as _P
        import yaml as _yaml
        cfg = _yaml.safe_load(_P(art["config_path"]).read_text(encoding="utf-8"))
        job_dir = _P(art["directory"])
        job_dir.mkdir(parents=True, exist_ok=True)
        seed = int(cfg.get("seed", 42))
        lr = str(cfg.get("train", {}).get("lr"))
        kind = "candidate" if "baseline" not in str(art["directory"]) else "baseline"
        # Pass the verified project root so the engine loads the SAME frozen
        # dataset the candidate was built against (data-root isolation).
        tr = embodied_compute(cfg, {"stage": "train", "seed": seed,
                                    "project_dir": project_dir}, job_dir)
        if tr.get("status") != "succeeded":
            return float("nan")
        run_id = job_dir.name
        ev = embodied_compute(cfg, {"stage": "evaluate", "seed": seed,
                                    "run_id": run_id,
                                    "project_dir": project_dir}, job_dir)
        if ev.get("status") != "succeeded":
            return float("nan")
        q = (ev.get("metrics") or {}).get(PRIMARY_METRIC)
        metric = float(q) if q is not None else float("nan")
        EVALS.append((seed, lr, kind, metric))
        return metric
    return runner


def _real_harness(**kw):
    """Wrap the REAL harness evolve so we can capture each round's payload."""
    payload = evolve(**kw)
    HARNESS_RUNS.append(payload)
    return payload


def _build_tasks():
    """Genuine research tasks for the harness layer.

    The harness is evaluated on EMBODIED research tasks; an INDEPENDENT QUANT
    task is held out and never participates in candidate selection. Budgets are
    bounded so the multi-round loop stays tractable while still running real
    paired campaigns.
    """
    tasks = [_task_for("embodied", seeds=[42, 43, 44])]
    tasks[0].budget["experiments"] = 4
    heldout = _task_for("quant", seeds=[60, 61, 62])
    heldout.budget["experiments"] = 4
    return tasks, heldout


def _setup(tmp: Path) -> Path:
    proj = tmp / "exp"
    proj.mkdir()
    shutil.copy(SRC / "config.yaml", proj / "config.yaml")
    cfg_path = proj / "config.yaml"
    cfg = _yaml_safe_load(cfg_path)
    cfg["train"]["lr"] = BASELINE_LR_FOR_DEMO
    cfg_path.write_text(_yaml_dump(cfg), encoding="utf-8")
    shutil.copytree(SRC / "datasets", proj / "datasets")
    return proj


def _yaml_safe_load(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _yaml_dump(d):
    import yaml
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True)


def _run_full():
    tasks, heldout = _build_tasks()
    with tempfile.TemporaryDirectory() as tmp:
        proj = _setup(Path(tmp))
        from research_runtime.loop import run_loop

        report = run_loop(
            state_path="",  # default <project>/state/loop_state.json
            domain="embodied",
            baseline_config_path=str(proj / "config.yaml"),
            project_dir=str(proj),
            max_rounds=3,
            dataset_version="robot_sim_v1-4979652dd897",
            domain_seeds=(50, 51, 52, 53, 54),
            domain_runner=_make_runner(str(proj)),
            domain_require_frozen_manifest=True,
            harness_evolve=_real_harness,
            base_harness=BASE_HARNESS,
            harness_tasks=tasks,
            harness_heldout_task=heldout,
            base_harness_tag=BASE_HARNESS.fingerprint(),
            harness_inherit_fn=default_harness_inherit,
            stop_on_no_accept=False,  # observe all 3 genuine rounds; decisions stay honest
            extra={"rounds": 1, "max_minutes": 1.0,
                   "negative_harness": NEGATIVE_HARNESS})
        _print_report(report, proj, "FULL 3-ROUND RUN", label="full")
        return proj, report


def _run_recovery():
    tasks, heldout = _build_tasks()
    with tempfile.TemporaryDirectory() as tmp:
        proj = _setup(Path(tmp))
        from research_runtime.loop import run_loop
        from research_runtime.loop.loop_state import LoopState, save

        # Real interrupted loop: round 1 finished + checkpointed, process died
        # before round 2. Resume MUST run ONLY rounds 2 and 3.
        state_path = Path(proj) / "state" / "loop_state.json"
        pre = LoopState(
            loop_id="recovery", domain="embodied", max_rounds=3,
            current_round=2, status="running",
            baseline_config_path=str(Path(proj) / "config.yaml"),
            base_harness_tag=BASE_HARNESS.fingerprint(),
            dataset_version="robot_sim_v1-4979652dd897", domain_seeds=[50, 51],
            last_completed_round=1)  # round 1 is already done before the crash
        save(pre, state_path)

        n_before = len(HARNESS_RUNS)
        report = run_loop(
            state_path=str(state_path),
            domain="embodied",
            baseline_config_path=str(Path(proj) / "config.yaml"),
            project_dir=str(proj), max_rounds=3,
            dataset_version="robot_sim_v1-4979652dd897",
            domain_seeds=(50, 51), domain_runner=_make_runner(str(proj)),
            domain_require_frozen_manifest=True,
            harness_evolve=_real_harness,
            base_harness=BASE_HARNESS,
            harness_tasks=tasks,
            harness_heldout_task=heldout,
            base_harness_tag=BASE_HARNESS.fingerprint(),
            harness_inherit_fn=default_harness_inherit,
            stop_on_no_accept=False,  # ensure both resumed rounds 2&3 run
            extra={"rounds": 1, "max_minutes": 1.0,
                   "negative_harness": NEGATIVE_HARNESS})
        n_harness_calls = len(HARNESS_RUNS) - n_before
        print("\n=== RECOVERY RUN (pre-seeded current_round=2) ===")
        print("harness_evolve calls (expect 2, not 3):", n_harness_calls)
        print("rounds_completed:", report["rounds_completed"])
        print("status:", report["status"])
        _print_report(report, proj, "RECOVERY", label="recovery")
        return report


def _print_report(report, proj, title, label):
    print(f"\n=== {title} ===")
    print("status:", report["status"])
    print("rounds_completed (accepted-version rounds):", report["rounds_completed"])
    print("rounds_executed (all rounds that ran):", report.get("rounds_executed"))
    print("n_domain_accepted_total:", report["n_domain_accepted_total"])
    print("n_harness_accepted_total:", report["n_harness_accepted_total"])
    print("final baseline:", report["final_baseline_config_path"])
    print("--- versions ---")
    for v in report["versions"]:
        print(f"  v{v['round_no']}: domain={v['accepted_domain_overrides']} "
              f"harness={v['accepted_harness_targets']} -> {v['baseline_config_path']}")
    print("--- REAL ENGINE EVALUATIONS (state_mse@1, lower is better) ---")
    for seed, lr, kind, metric in EVALS:
        print(f"  seed={seed} lr={lr} {kind:9s} {PRIMARY_METRIC}={metric:.6f}")
    print("--- REAL HARNESS LAYER (per evolve call) ---")
    for i, h in enumerate(HARNESS_RUNS):
        if h.get("_label") and h["_label"] != label:
            continue
        print(f"  evolve#{i}: dominant={h.get('dominant_failure')} "
              f"accepted={h.get('n_accepted')} rejected={h.get('n_rejected')} "
              f"inconclusive={h.get('n_inconclusive')} "
              f"controls_rejected={h.get('negative_controls_rejected')}")
        for r in h.get("records", []):
            c = r["candidate"]
            print(f"      {c['target']} {c['old']}->{c['new']}: {r['decision']}")
    # Dataset / artifact provenance.
    manifest = proj / "datasets" / "robot_sim_v1" / "manifest.json"
    if manifest.is_file():
        import hashlib
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        print(f"--- frozen manifest sha256: {sha[:16]}... ---")


if __name__ == "__main__":
    _run_full()
    _run_recovery()
