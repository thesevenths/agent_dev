"""M3.2 -- REAL Embodied validation under the M3.0 statistical gate.

Two independent checks, BOTH on the REAL frozen ``robot_sim_v1`` dataset via the
production ``embodied_compute`` (pure-stdlib train -> multi-step rollout; no
FakeRunner, no gymnasium/mujoco/lerobot -- those were a FALSE blocker from a
stale memory; see the acceptance report).

  1. GATE-DRIVEN candidates: call ``run_dual_layer(domain="embodied", ...)`` with a
     STUBBED research-harness layer (the harness layer is orthogonal to the
     DOMAIN statistical gate being validated). This reuses the EXACT M3.0 gate
     path used by the M3.0 SciNet++ 10+10 re-validation: Student-t CI + Holm +
     bidirectional + fail-closed + per-seed holdout.
  2. TARGETED "REAL model fix": the embodied domain's own genuine intervention
     ``model.drift_correction`` (a TRAIN-ONLY residual correction that reduces
     long-horizon error, per research_runtime/domains/embodied.py). Evaluated
     through the SAME statistics primitives (evaluate_domain_candidate ->
     domain_family_significance -> _decide_domain) so the verdict is produced by
     the identical gate code, not a re-implementation.

Seeds: 10 PAIRED (50-59) + 10 INDEPENDENT holdout (60-69), min_seeds_for_accept
raised from the hard floor 5 to 10 to satisfy the M3.0 / ChatGPT Phase-0 rule 6
(>=10 paired + >=10 holdout for a formal improvement).

Historical M2.7 / M3.0 artifacts are NOT read or written; output goes to
M3_2_runs/ only.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research_runtime.domains.embodied import (  # noqa: E402
    PROJECT_DIR, read_dataset_version,
)
from research_runtime.cli import _build_real_domain_runner  # noqa: E402
from research_runtime.evolution.evolution_controller import (  # noqa: E402
    run_dual_layer, domain_family_significance, _decide_domain,
    CandidateStatus, DEFAULT_ALPHA,
)
from research_runtime.evolution.domain_evolver import (  # noqa: E402
    evaluate_domain_candidate, DomainModelCandidate, CandidateType,
)

DATASET_VERSION = read_dataset_version()           # robot_sim_v1-4979652dd897
BASE_CFG = str(PROJECT_DIR / "config.yaml")
PROJ = PROJECT_DIR
SEEDS = tuple(range(50, 60))                        # 10 PAIRED seeds
HOLDOUT = tuple(range(60, 70))                      # 10 INDEPENDENT holdout seeds
RUN_ROOT = REPO / "M3_2_runs" / "embodied_10seed"
OUT = REPO / "M3_2_runs" / "embodied_10seed_result.json"
MIN_SEEDS = 10                                      # raise hard floor 5 -> 10


def _harness_stub(**kw):
    """Research-harness layer only; the DOMAIN layer below is 100% REAL engine.
    Mirrors the M3.0 SciNet++ driver: the harness layer is orthogonal to the
    DOMAIN statistical gate being validated."""
    return {
        "schema_version": "stub-harness-layer",
        "dominant_failure": "DECISION_UNCERTAINTY",
        "records": [], "accepted": [], "n_accepted": 0,
    }


def _verify_frozen_data() -> dict:
    root = PROJECT_DIR / "datasets" / "robot_sim_v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    for split, info in (manifest.get("splits") or {}).items():
        p = root / info["file"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != info["sha256"]:
            bad.append(split)
    return {
        "dataset_version": manifest.get("dataset_version"),
        "splits": sorted((manifest.get("splits") or {}).keys()),
        "sha_mismatch": bad,
    }


def _decide(rec: dict) -> str:
    return _decide_domain(rec, min_seeds_for_accept=MIN_SEEDS,
                          require_holdout_confirmation=True).value


def _rec_row(r: dict) -> dict:
    st = r.get("statistics") or {}
    return {
        "target": r.get("target"), "new_value": r.get("new"),
        "n_valid": r.get("n_valid"), "n_invalid": r.get("n_invalid"),
        "mean_improvement": r.get("mean_improvement"),
        "std_improvement": r.get("std_improvement"),
        "baseline_mean": r.get("baseline_mean"),
        "holdout_improvement": r.get("holdout_improvement"),
        "holdout_n_valid": r.get("holdout_n_valid"),
        "holdout_n_seeds": r.get("holdout_n_seeds"),
        "holdout_n_invalid": r.get("holdout_n_invalid"),
        "holdout_direction_fraction": r.get("holdout_direction_fraction"),
        "holdout_improvements": r.get("holdout_improvements"),
        "statistics": r.get("statistics"),
        "decision": r.get("decision"),
    }


def main() -> dict:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = _verify_frozen_data()
    assert frozen["dataset_version"] == DATASET_VERSION, frozen
    assert not frozen["sha_mismatch"], "frozen dataset SHA mismatch -> fail closed"
    assert Path(BASE_CFG).is_file(), BASE_CFG

    runner = _build_real_domain_runner("embodied", PROJ)
    t0 = time.time()
    print(f"START real Embodied 10+10 validation | {time.strftime('%H:%M:%S')}",
          flush=True)
    print(f"  project_dir={PROJ}", flush=True)
    print(f"  dataset_version={DATASET_VERSION}", flush=True)
    print(f"  paired_seeds={SEEDS} holdout_seeds={HOLDOUT} min_seeds={MIN_SEEDS}",
          flush=True)

    # ---- (1) GATE-DRIVEN candidates via the EXACT M3.0 dual-layer gate ----
    payload = run_dual_layer(
        base_harness=None, tasks=[], heldout_task=None,
        run_root=RUN_ROOT / "dual_layer", repo_root=REPO, rounds=1,
        max_minutes=1.0, domain="embodied", domain_base_config=BASE_CFG,
        project_dir=str(PROJ), dataset_version=DATASET_VERSION,
        domain_seeds=SEEDS, domain_runner=runner,
        domain_require_frozen_manifest=True, domain_holdout_seeds=HOLDOUT,
        harness_evolve=_harness_stub, min_seeds_for_accept=MIN_SEEDS,
        require_holdout_confirmation=True)
    dom = payload["domain_layer"]
    gate_rows = [_rec_row(r) for r in dom["records"]]

    # ---- (2) TARGETED genuine model fix: model.drift_correction=0.75 ----
    cand = DomainModelCandidate(
        candidate_id="", domain="embodied", target="model.drift_correction",
        old=None, new=0.75, taxonomy="dynamics_error_accumulation",
        hypothesis="TRAIN-ONLY residual drift correction conditioned on sign of x;"
                  " genuinely reduces long-horizon rollout error accumulation.")
    rec = evaluate_domain_candidate(
        domain="embodied", candidate=cand, base_config=BASE_CFG,
        project_dir=str(PROJ), campaign_id="m3_2_drift", seeds=SEEDS,
        dataset_version=DATASET_VERSION, runner=runner,
        require_frozen_manifest=True, holdout_seeds=HOLDOUT)
    holm = domain_family_significance([rec])
    rec["statistics"] = {
        "protocol": "M3.0-student-t-holm",
        "family_alpha": DEFAULT_ALPHA,
        "p_value": round(holm[0]["p_value"], 9),
        "holm_effective_alpha": holm[0]["effective_alpha"],
        "holm_significant": holm[0]["significant"],
        "method": holm[0]["method"], "rank": holm[0]["rank"],
        "n_compared": holm[0]["n_compared"],
    }
    rec["decision"] = _decide(rec)
    drift_row = _rec_row(rec)

    elapsed = time.time() - t0
    out = {
        "protocol": "M3.0-student-t-holm",
        "domain": "embodied",
        "engine": "REAL embodied_compute (pure-stdlib train + multi-step rollout "
                  "on frozen robot_sim_v1; NO gymnasium/mujoco/lerobot)",
        "domain_runner_is_real_engine": True,
        "harness_layer_stubbed": True,
        "harness_stub_rationale": (
            "M3.2 validates the DOMAIN statistical gate; the research-harness "
            "layer is orthogonal. No FakeRunner is used for the domain layer."),
        "frozen_dataset": frozen,
        "paired_seeds": list(SEEDS), "holdout_seeds": list(HOLDOUT),
        "min_seeds_for_accept": MIN_SEEDS,
        "elapsed_seconds": round(elapsed, 2),
        "statistics_protocol": dom.get("statistics_protocol"),
        "gate_driven": {
            "n_accepted": dom["n_accepted"], "n_rejected": dom["n_rejected"],
            "n_inconclusive": dom["n_inconclusive"], "records": gate_rows,
        },
        "targeted_model_fix": {
            "target": "model.drift_correction", "new_value": 0.75,
            "decision": drift_row["decision"], "record": drift_row,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")
    print(f"\nDONE elapsed={elapsed:.1f}s", flush=True)
    print(f"[gate-driven] accept={dom['n_accepted']} reject={dom['n_rejected']} "
          f"inconclusive={dom['n_inconclusive']}", flush=True)
    for r in gate_rows:
        st = r.get("statistics") or {}
        print(f"  target={r['target']!s:<22} new={r['new_value']!s:<8} "
              f"n_valid={r['n_valid']} mean={r['mean_improvement']} "
              f"p={st.get('p_value')} holm_sig={st.get('holm_significant')} "
              f"ho_frac={r['holdout_direction_fraction']} -> {r['decision']}",
              flush=True)
    print(f"[targeted] model.drift_correction=0.75 -> {drift_row['decision']} "
          f"(p={drift_row['statistics'].get('p_value')} "
          f"holm_sig={drift_row['statistics'].get('holm_significant')} "
          f"ho_frac={drift_row['holdout_direction_fraction']})", flush=True)
    return out


if __name__ == "__main__":
    main()
