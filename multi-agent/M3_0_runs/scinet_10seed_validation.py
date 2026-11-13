"""M3.0 Phase 0 rule 6 -- REAL 10+10 seed re-validation of the M2.7 improvement.

Runs the REAL SciNet++ engine (SciNetAdapter -> research_api.cli subprocess,
torch) through the REAL unified dual-layer controller, on the REAL frozen
pendulum dataset, with 10 paired seeds and 10 INDEPENDENT holdout seeds.

No FakeRunner for the domain layer: ``domain_runner`` is the production
``cli._build_real_domain_runner``. ONLY the RESEARCH_HARNESS evolution entry
point is stubbed, because M3.0 validates the DOMAIN statistical gate and the
harness layer is orthogonal (and its baseline collection spawns many extra
engine jobs). This is stated explicitly so the evidence cannot be overclaimed.

Historical M2.7 artifacts are NOT read or written by this run; its output goes
to M3_0_runs/ only.
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

from research_runtime.cli import _build_real_domain_runner  # noqa: E402
from research_runtime.evolution.evolution_controller import (  # noqa: E402
    run_dual_layer,
)

PROJ = (REPO / "M2_7_runs" / "scinet").resolve()
CFG = (PROJ / "configs" / "pendulum_smoke.yaml").resolve()
DATASET_VERSION = "pendulum-a0e03386be89"
SEEDS = tuple(range(50, 60))        # 10 PAIRED seeds
HOLDOUT = tuple(range(60, 70))      # 10 INDEPENDENT holdout seeds
RUN_ROOT = REPO / "M3_0_runs" / "scinet_10seed"
OUT = REPO / "M3_0_runs" / "scinet_10seed_result.json"


def _harness_stub(**kw):
    """Harness layer only -- domain layer below is 100% REAL engine."""
    return {
        "schema_version": "stub-harness-layer",
        "dominant_failure": "DECISION_UNCERTAINTY",
        "records": [], "accepted": [], "n_accepted": 0,
    }


def _verify_frozen_data() -> dict:
    root = PROJ / "datasets" / "pendulum_smoke_v1"
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


def main() -> dict:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = _verify_frozen_data()
    assert frozen["dataset_version"] == DATASET_VERSION, frozen
    assert not frozen["sha_mismatch"], "frozen dataset SHA mismatch -> fail closed"
    assert CFG.is_file(), CFG

    runner = _build_real_domain_runner("scinet", PROJ)
    t0 = time.time()
    print(f"START real SciNet++ 10+10 validation | {time.strftime('%H:%M:%S')}",
          flush=True)
    print(f"  project_dir={PROJ}", flush=True)
    print(f"  paired_seeds={SEEDS} holdout_seeds={HOLDOUT}", flush=True)
    print(f"  frozen={frozen}", flush=True)

    payload = run_dual_layer(
        base_harness=None, tasks=[], heldout_task=None,
        run_root=RUN_ROOT, repo_root=REPO, rounds=1, max_minutes=1.0,
        domain="scinet", domain_base_config=str(CFG),
        project_dir=str(PROJ), dataset_version=DATASET_VERSION,
        domain_seeds=SEEDS, domain_holdout_seeds=HOLDOUT,
        domain_require_frozen_manifest=True,
        domain_runner=runner, harness_evolve=_harness_stub)
    elapsed = time.time() - t0

    dom = payload["domain_layer"]
    rows = []
    for r in dom["records"]:
        rows.append({
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
        })
    out = {
        "protocol": "M3.0-student-t-holm",
        "domain": "scinet",
        "engine": "REAL SciNet++ (SciNetAdapter / research_api.cli / torch)",
        "domain_runner_is_real_engine": True,
        "harness_layer_stubbed": True,
        "harness_stub_rationale": (
            "M3.0 validates the DOMAIN statistical gate; the harness layer is "
            "orthogonal. No FakeRunner is used for the domain layer."),
        "frozen_dataset": frozen,
        "paired_seeds": list(SEEDS), "holdout_seeds": list(HOLDOUT),
        "elapsed_seconds": round(elapsed, 2),
        "statistics_protocol": dom.get("statistics_protocol"),
        "n_accepted": dom["n_accepted"], "n_rejected": dom["n_rejected"],
        "n_inconclusive": dom["n_inconclusive"],
        "records": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")
    print(f"\nDONE elapsed={elapsed:.1f}s "
          f"accept={dom['n_accepted']} reject={dom['n_rejected']} "
          f"inconclusive={dom['n_inconclusive']}", flush=True)
    for r in rows:
        st = r.get("statistics") or {}
        print(f"  lr={r['new_value']!s:<7} n_valid={r['n_valid']} "
              f"mean={r['mean_improvement']} std={r['std_improvement']} "
              f"p={st.get('p_value')} holm_sig={st.get('holm_significant')} "
              f"ho_frac={r['holdout_direction_fraction']} "
              f"-> {r['decision']}", flush=True)
    return out


if __name__ == "__main__":
    main()
