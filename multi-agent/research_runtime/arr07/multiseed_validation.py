"""ARR-07: paired multi-seed validation of the best agent-discovered config.

SciNet++ `seed` controls model init; data is FROZEN, so paired seeds are valid.
Candidate comes from the real LeJEPA autonomous study (iter1, +25% MSE).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..generic.research_contract import paired_seed_verdict
from . import core, scinet_backend

# Clean, scientifically valid config from the deterministic sweep
# (ARR-07-PARAM-SWEEP: lr=0.01 gave +51.9% MSE, +0.384 cosine, single seed).
# Uses the DEFAULT structure (dim=4, hidden[32,32]) - no absurd over-parameterisation.
CANDIDATE = {"train.lr": 0.01}

METRIC = "latent_mse@1"


def validate(root: str | Path, seeds: tuple[int, ...] = (41, 42, 43),
             candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    (root / "raw_data" / "scinet").mkdir(parents=True, exist_ok=True)
    cand = candidate or CANDIDATE
    base_pairs, arm_pairs, rows = [], [], []
    for s in seeds:
        b = scinet_backend.run_scinet_experiment(
            experiment_id=f"ms_s{s}_base", task_id="multiseed", seed=s,
            overrides={}, root=root, protocol_hash=core.sha256_json("arr07-ms-v1"))
        a = scinet_backend.run_scinet_experiment(
            experiment_id=f"ms_s{s}_cand", task_id="multiseed", seed=s,
            overrides=cand, root=root, protocol_hash=core.sha256_json("arr07-ms-v1"))
        ok = bool(b.get("ok") and a.get("ok"))
        bm = am = bc = ac = None
        if ok:
            bm = float(b["metrics"][METRIC]); am = float(a["metrics"][METRIC])
            bc = float(b["metrics"]["cosine@1"]); ac = float(a["metrics"]["cosine@1"])
            base_pairs.append((bm, bc)); arm_pairs.append((am, ac))
        rows.append({"seed": s, "engine_ok": ok,
                     "baseline_mse": bm, "candidate_mse": am,
                     "baseline_cosine": bc, "candidate_cosine": ac,
                     "relative_gain": round((bm - am) / bm, 6) if bm else None,
                     "cosine_delta": round(ac - bc, 6) if bc is not None else None,
                     "baseline_error": b.get("error", ""),
                     "candidate_error": a.get("error", "")})
    ci = paired_seed_verdict(base_pairs, arm_pairs, direction="min") if base_pairs \
        else {"ok": False, "reason": "no valid pairs"}
    out = {"n_seeds": len(seeds), "n_valid_pairs": len(base_pairs),
           "candidate": cand, "metric": METRIC, "rows": rows,
           "paired_ci": ci,
           "conclusion": ("CONFIRMED positive MSE gain" if ci.get("significantly_positive")
                          else "NOT confirmed: CI includes zero or no valid pairs")}
    (root / "multiseed_result.json").write_text(json.dumps(
        out, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return out


__all__ = ["CANDIDATE", "validate"]

if __name__ == "__main__":
    import sys
    seeds = tuple(int(x) for x in sys.argv[1:]) or (41, 42, 43)
    r = validate(Path("experiments/campaigns/ARR-07-MULTISEED"), seeds=seeds)
    print(json.dumps({"n_valid": r["n_valid_pairs"],
                      "conclusion": r["conclusion"], "ci": r["paired_ci"]},
                     indent=2, ensure_ascii=False))
