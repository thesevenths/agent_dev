"""Mechanism scouting for M3 Reacher regimes (design phase, not formal campaign)."""
import json
from pathlib import Path

from research_runtime.arr_research_capability_03.reacher import run_arm

OUT = Path("ARR_03_M3_SCOUT")
OUT.mkdir(exist_ok=True)

# Standard held-out test distribution (evaluator-only, fixed across arms).
TEST = dict(n_episodes=6, max_steps=30, target_mode="mixed",
            action_mode="random", action_scale=1.0)

# Each regime: (failure_train_cfg, oracle_train_cfg, model_failure, model_oracle)
# The intervention differs per mechanism; test set is identical for both.
REGIMES = {
    "E01_capacity": (
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(hidden=[4], lr=0.002), dict(hidden=[64, 64], lr=0.002)),
    "E02_coverage": (
        dict(n_episodes=2, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(n_episodes=16, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
    "E03_shift_target": (
        dict(n_episodes=8, max_steps=30, target_mode="center", action_mode="random", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
    "E04_shift_action": (
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=0.2),
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
    "E05_optim": (
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.3), dict(hidden=[32, 32], lr=0.001)),
    "E06_action_cond": (
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="zero", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed", action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
}

out = {}
for rid, (tf, to, mf, mo) in REGIMES.items():
    rd = OUT / rid
    f = run_arm(arm="failure", regime_id=rid, seed=42, train_cfg=tf,
                test_cfg=TEST, model_cfg=mf, out_dir=rd / "failure", epochs=40)
    o = run_arm(arm="oracle", regime_id=rid, seed=42, train_cfg=to,
                test_cfg=TEST, model_cfg=mo, out_dir=rd / "oracle", epochs=40)
    fm = f.metrics.get("one_step_mse")
    om = o.metrics.get("one_step_mse")
    imp = (fm - om) / max(abs(fm), 1e-9) if (fm and om) else None
    row = {"failure_ok": f.ok, "oracle_ok": o.ok,
           "failure_mse": fm, "oracle_mse": om,
           "rel_improvement": imp,
           "failure_err": f.error[:200], "oracle_err": o.error[:200]}
    out[rid] = row
    print(rid, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in row.items()}), flush=True)

(OUT / "scout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("DONE", flush=True)
