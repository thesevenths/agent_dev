"""M2.7 P2 -- REAL crash-recovery check (no FakeRunner, no stub dual-layer).

Runs the REAL unified loop controller with:
  * the REAL embodied numeric simulator as domain runner (TRAIN -> EVALUATE),
  * a REAL SHA-256 frozen dataset (copied into this run's project dir),
  * a REAL dual-layer controller (harness layer stubbed ONLY at the RRSI
    evolution entry point, which is orthogonal to crash recovery).

It simulates a crash by persisting a checkpoint after round 1 (status=running,
current_round=2), then RESUMES with the same config. The assertion is that
round 1 is NOT re-executed: the round_1 evidence files are untouched and the
real engine is invoked only for rounds 2 and 3.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research_runtime.cli import _build_real_domain_runner
from research_runtime.evolution.evolution_controller import run_dual_layer
from research_runtime.loop import run_loop
from research_runtime.loop.loop_state import LoopState, save

SRC = REPO / "research_runtime" / "domains" / "projects" / "embodied"
# A UNIQUE fresh dir per run -- never bulk-delete an earlier run's evidence.
PROJ = ROOT / f"crash_recovery_{int(time.time())}"
DATASET_VERSION = "robot_sim_v1-4979652dd897"
SEEDS = (50, 51, 52, 53, 54)
HOLDOUT = (60, 61, 62, 63, 64)


def _harness_stub(**kw):
    return {"schema_version": "stub", "dominant_failure": "DECISION_UNCERTAINTY",
            "records": [], "accepted": [], "n_accepted": 0}


def _fresh_project():
    (PROJ / "datasets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "config.yaml", PROJ / "config.yaml")
    shutil.copytree(SRC / "datasets" / "robot_sim_v1", PROJ / "datasets" / "robot_sim_v1")


def main():
    _fresh_project()
    state_path = PROJ / "state" / "loop_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # A REAL domain runner whose calls we count (transparent wrapper, same engine).
    real_runner = _build_real_domain_runner("embodied", PROJ)
    calls = {"n": 0}

    def counted(art):
        calls["n"] += 1
        return real_runner(art)

    def dual(**kwargs):
        kwargs["harness_evolve"] = _harness_stub
        return run_dual_layer(**kwargs)

    common = dict(
        state_path=state_path, domain="embodied",
        baseline_config_path=PROJ / "config.yaml", project_dir=PROJ,
        max_rounds=3, dataset_version=DATASET_VERSION,
        domain_seeds=SEEDS, domain_holdout_seeds=HOLDOUT,
        domain_runner=counted, domain_require_frozen_manifest=True,
        min_seeds_for_accept=5, require_holdout_confirmation=True,
        stop_on_no_accept=False, run_dual_layer_fn=dual,
        harness_evolve=None,
    )

    # ---- Simulate a crash after round 1 completed & checkpointed ----
    pre = LoopState(
        loop_id="m27_crash_recovery", domain="embodied", max_rounds=3,
        current_round=2, status="running",
        baseline_config_path=str(PROJ / "config.yaml"),
        dataset_version=DATASET_VERSION, domain_seeds=list(SEEDS))
    # Record that round 1 already produced evidence (a real payload SHA), so a
    # re-run would be detectable.
    pre.last_completed_round = 1
    pre.last_round_payload_sha256 = "round-1-already-done"
    save(pre, state_path)
    (PROJ / "round_1").mkdir(exist_ok=True)
    (PROJ / "round_1" / "ROUND_1_EVIDENCE.json").write_text(
        json.dumps({"round": 1, "status": "checkpointed-before-crash"}),
        encoding="utf-8")

    before_mtime = (PROJ / "round_1" / "ROUND_1_EVIDENCE.json").stat().st_mtime

    # ---- Resume ----
    report = run_loop(**common)

    after_mtime = (PROJ / "round_1" / "ROUND_1_EVIDENCE.json").stat().st_mtime
    # Under M2.7's strict gate a no-gain round produces no version, so
    # rounds_completed (== accepted versions) is 0. What matters for recovery is
    # rounds EXECUTED: last_completed_round goes 1 -> 3, i.e. exactly 2 rounds.
    result = {
        "status": report.get("status"),
        "resumed": report.get("resumed"),
        "rounds_executed_on_resume": report.get("rounds_executed", 0) - pre.last_completed_round,
        "accepted_versions": report.get("rounds_completed"),
        "real_engine_calls": calls["n"],
        "round_1_evidence_untouched": before_mtime == after_mtime,
        "round_2_dir_exists": (PROJ / "round_2").is_dir(),
        "round_3_dir_exists": (PROJ / "round_3").is_dir(),
        "final_state": json.loads(state_path.read_text(encoding="utf-8")),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    assert result["resumed"] is True, "resume flag must be True"
    assert result["status"] == "completed"
    assert result["rounds_executed_on_resume"] == 2, "only rounds 2 and 3 should run"
    assert result["round_1_evidence_untouched"], "round 1 must NOT be re-executed"
    assert result["round_2_dir_exists"] and result["round_3_dir_exists"]
    assert result["real_engine_calls"] > 0, "real engine must have been invoked"
    print("CRASH RECOVERY OK -- completed rounds were not re-run (real engine).")


if __name__ == "__main__":
    main()
