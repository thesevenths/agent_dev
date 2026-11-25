"""
scripts.compute_stratified_split_v2
===================================
P1-B1.1 / P7: apply the genuine stratified allocator to the historical 20-episode
V2 pilot and EMIT A COVERAGE REPORT.

This does NOT overwrite the historical pilot. The original (random) split recorded
in ``manifest.json`` is preserved unchanged. The stratified assignment is written
to a separate artifact so the gap between "what we have" and "what genuine
stratification would require" is explicit and machine-verifiable.

The report is the input the P1-B2 collector will consume to produce a truly
stratified dataset (with far more episodes than 20, so the strata are actually
coverable).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.stratified_split import (  # noqa: E402
    StratifiedSplitter, build_episode_metadata,
)
import yaml  # noqa: E402

V2 = ROOT / "datasets" / "controlled_pendulum_v2_pilot"
MAN = V2 / "manifest.json"
OUT = V2 / "stratified_split_report.json"


def main() -> int:
    man = json.loads(MAN.read_text(encoding="utf-8"))
    episodes = build_episode_metadata(man)
    # The seed is declared in the config, not the manifest. Fall back to a fixed
    # deterministic value if the config cannot be resolved.
    seed = 20261112
    try:
        cfg_path = ROOT / str(man.get("config_file", "")).replace("\\", "/")
        if cfg_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            seed = int(cfg["experiment"]["seed"])
    except Exception:
        pass

    # The historical config has no `stratification` block; use the documented
    # default bins. P1-B2 will declare these explicitly in its YAML.
    splitter = StratifiedSplitter()
    result = splitter.allocate(episodes, master_seed=int(seed))
    dist = splitter.verify_distribution(result)

    report = {
        "source_manifest": "datasets/controlled_pendulum_v2_pilot/manifest.json",
        "note": (
            "Genuine stratified allocation over the 20-episode HISTORICAL pilot. "
            "The historical random split in manifest.json is PRESERVED; this "
            "report documents what real stratification would assign and, crucially, "
            "where the 20-episode pilot CANNOT cover strata (BLOCKED/INCONCLUSIVE). "
            "It is NOT a claim that the historical data is stratified."
        ),
        "stratification_dimensions": [
            "action_profile", "mass_bin", "length_bin", "damping_bin",
            "theta0_bin", "omega0_bin", "regime",
        ],
        "requested_ratios": splitter.cfg.requested_ratios,
        "min_per_cell": splitter.cfg.min_per_cell,
        "status": result.status,
        "global_counts": result.global_counts,
        "requested_counts": result.requested_counts,
        "ratio_errors": result.ratio_errors,
        "distribution_verification": dist,
        "n_episodes": result.n_episodes,
        "n_strata": len(result.coverage),
        "n_blocked_strata": len(result.blocked_strata),
        "n_inconclusive_strata": len(result.inconclusive_strata),
        "n_unachievable_strata": len(result.unachievable_strata),
        "blocked_strata": result.blocked_strata,
        "inconclusive_strata": result.inconclusive_strata,
        "unachievable_strata": result.unachievable_strata,
        "assignments": result.assignments,
        "coverage": result.coverage,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 70)
    print(f"P1-B1.1 / P7 stratified allocation over historical 20-episode pilot")
    print("=" * 70)
    print(f"status                : {result.status}")
    print(f"global_counts        : {result.global_counts}")
    print(f"requested_counts     : {result.requested_counts}")
    print(f"ratio_errors         : { {k: round(v, 4) for k, v in result.ratio_errors.items()} }")
    print(f"n_strata             : {len(result.coverage)}")
    print(f"blocked_strata       : {len(result.blocked_strata)}")
    print(f"inconclusive_strata  : {len(result.inconclusive_strata)}")
    print(f"unachievable_strata  : {len(result.unachievable_strata)}")
    print(f"report               : {OUT}")
    print("\nPer-stratum coverage (status :: n -> counts):")
    for key, c in sorted(result.coverage.items()):
        print(f"  {c['status']:11s} {key}  n={c['n']} counts={c['counts']}")
    # This is expected to be BLOCKED/INCONCLUSIVE with only 20 episodes -- that
    # is the honest finding, not a failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
