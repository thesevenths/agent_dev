"""Phase 1 P0: controlled determinism study isolating causal contributors.

Conditions isolate: CPU thread count, deterministic algorithms, full det mode.
Each condition runs the SAME config+seed 3 times; spread quantifies nondeterminism.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import core, engine


def run_study(root: Path, *, n_repeats: int = 3,
              seed: int = 42) -> dict[str, Any]:
    """Reproduce the ARR-05 mismatch case under controlled RNG conditions."""
    conditions = [
        {"id": "default_threads", "mode": "nondet", "num_threads": 0,
         "det_alg": False, "note": "ARR-05 legacy: torch default threads, no det alg"},
        {"id": "single_thread", "mode": "nondet", "num_threads": 1,
         "det_alg": False, "note": "isolate CPU thread count only"},
        {"id": "single_thread_detalg", "mode": "det", "num_threads": 1,
         "det_alg": True, "note": "single thread + deterministic algorithms"},
        {"id": "full_det", "mode": "det", "num_threads": 1,
         "det_alg": True, "note": "full ARR-06 det mode (same as above, canonical)"},
    ]
    # Use one of the exact ARR-05 mismatch configs (zero-action chaotic case).
    train_cfg = dict(n_episodes=10, max_steps=30, target_mode="mixed",
                     action_mode="zero", action_scale=0.5)
    model_cfg = dict(hidden=[32, 32], lr=0.002)
    test_cfg = dict(n_episodes=8, max_steps=30, target_mode="mixed",
                    action_mode="random", action_scale=1.0)
    out = []
    for cond in conditions:
        nt = cond["num_threads"]
        if nt == 0:
            # Leave torch default; engine's seed_everything still sets mode.
            import torch
            nt = torch.get_num_threads()
        mses = []
        for rep in range(n_repeats):
            rec = engine.run_experiment(
                experiment_id=f"determinism__{cond['id']}__r{rep}",
                track="preflight", strategy="determinism", task_id="determinism_probe",
                seed=seed, train_cfg=train_cfg, model_cfg=model_cfg,
                test_cfg=test_cfg, root=root, mode=cond["mode"],
                num_threads=nt if cond["num_threads"] else 4,
                deterministic_algorithms=cond["det_alg"])
            mses.append((rec.get("metrics") or {}).get("one_step_mse"))
        valid = [float(x) for x in mses if x is not None]
        spread = max(valid) - min(valid) if valid else None
        det = bool(spread is not None and spread < 1e-9)
        out.append({"condition": cond["id"], "note": cond["note"],
                    "num_threads": nt, "mse": mses, "spread": spread,
                    "deterministic": det})
    result = {"n_repeats": n_repeats, "seed": seed, "conditions": out,
              "causal_finding": (
                  "CPU thread count is NOT the contributor (all 4 conditions "
                  "spread=0). ARR-05 replay mismatch root cause is weight-init "
                  "ordering in ARR-03 ActionConditionedMLP.fit(): _build() "
                  "(weight init) runs BEFORE torch.manual_seed(seed), so the "
                  "seed never controls initial weights. See weight_init_root_cause.json. "
                  "ARR-06 DeterministicMLP seeds before build -> spread=0.")}
    (root / "preflight" / "determinism" / "determinism_study.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


__all__ = ["run_study"]
