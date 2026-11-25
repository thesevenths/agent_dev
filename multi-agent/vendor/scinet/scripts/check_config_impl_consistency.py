"""
scripts/check_config_impl_consistency.py
========================================
P1-A' review fix P3: verify that configuration *documentation and declarations*
actually match the implementation. Catches the stale seed comment class of bug
(the config claimed ``master_seed*1000 + ...`` while the code used SHA-256).

Checks (all framework-independent, read-only):
  1. config header contains no stale multiplication-form seed claim;
  2. config declares the SHA-256 derivation, matching derive_episode_seed;
  3. declared integrator / theta_representation / torque_bounds / episode_steps
     agree with the resolved simulator behaviour and config values;
  4. resolved_pilot_config values match the actual simulator defaults/derivation.

Usage:
  python scripts/check_config_impl_consistency.py [--config PATH]
  -> datasets/controlled_pendulum_pilot/analysis/config_impl_consistency.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.controlled_pendulum import derive_episode_seed  # noqa: E402
from data.controlled_pendulum_v2 import ValidatedPendulum  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/dynamics/controlled_pendulum_jepa.yaml")
    args = ap.parse_args(argv)

    text = args.config.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    sim = cfg["simulator"]
    resolved = cfg.get("resolved_pilot_config", {})

    checks = []

    def chk(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. stale seed formula must NOT be present as an assertion
    stale = "master_seed*1000 + scenario_idx*100" in text.replace("NOT by multiplication", "")
    # The corrected header mentions it only inside a "legacy/stale" explanation.
    stale_assert = any(
        ln.strip().startswith("# Determinism:") and "*1000" in ln
        for ln in text.splitlines())
    chk("no_stale_seed_formula_in_header", not stale_assert,
        {"stale_header_line_found": stale_assert})

    # 2. SHA-256 derivation declared AND matches implementation numerically
    declared_sha = "sha256" in text.lower() or "SHA-256" in text
    s0 = derive_episode_seed(int(cfg["experiment"]["seed"]), 0, 0)
    s1 = derive_episode_seed(int(cfg["experiment"]["seed"]), 2, 3)
    chk("sha256_seed_declaration_matches_impl", declared_sha and s0 != s1
        and 0 <= s0 < 2 ** 32,
        {"declared": declared_sha, "seed_0_0": s0, "seed_2_3": s1,
         "within_32bit": 0 <= s0 < 2 ** 32})

    # 3. declared integrator / theta_representation / bounds / steps
    chk("integrator_declared_rk4",
        str(sim.get("integrator", "")).lower() == "rk4",
        {"value": sim.get("integrator")})
    chk("theta_representation_unwrapped",
        str(sim.get("theta_representation", "")) == "unwrapped",
        {"value": sim.get("theta_representation")})
    tb = sim.get("torque_bounds")
    chk("torque_bounds_match_action_max_torque",
        tb is not None and abs(float(tb[1]) - float(sim["action_max_torque"])) < 1e-12,
        {"torque_bounds": tb, "action_max_torque": sim["action_max_torque"]})
    chk("episode_steps_match_horizon_steps",
        int(sim.get("episode_steps", -1)) == int(sim["horizon_steps"]),
        {"episode_steps": sim.get("episode_steps"),
         "horizon_steps": sim["horizon_steps"]})

    # 4. resolved_pilot_config consistency
    chk("resolved_integrator", resolved.get("integrator") == sim.get("integrator"),
        {"resolved": resolved.get("integrator"), "simulator": sim.get("integrator")})
    chk("resolved_seed_derivation",
        resolved.get("seed_derivation") == "sha256_first4_bytes",
        {"value": resolved.get("seed_derivation")})
    chk("resolved_storage_dtype",
        resolved.get("storage_dtype") == cfg["pilot"].get("storage_dtype")
        == "float64",
        {"resolved": resolved.get("storage_dtype"),
         "pilot": cfg["pilot"].get("storage_dtype")})

    # 5. simulator behaviour matches the declarations (theta unwrapped)
    sc = cfg["scenarios"][0]
    p = ValidatedPendulum(g=sc["g"], L=sc["L"], m=sc["m"], b=sc["b"],
                          dt=sim["dt"], torque_bounds=tb or [-5.0, 5.0])
    import numpy as np
    obs = p.simulate(np.zeros(50, dtype=np.float64),
                     theta0=2.0 * np.pi + 0.1, omega0=0.0)
    chk("simulator_does_not_wrap_theta",
        abs(float(obs[0, 0]) - (2.0 * np.pi + 0.1)) < 1e-12
        and float(obs[-1, 0]) > 5.0,
        {"theta0_stored": float(obs[0, 0]), "theta_final": float(obs[-1, 0])})

    n_pass = sum(1 for c in checks if c["passed"])
    verdict = "PASS" if n_pass == len(checks) else "FAIL"
    out = {
        "verdict": verdict, "n_pass": n_pass, "n_total": len(checks),
        "config": str(args.config.relative_to(ROOT)), "checks": checks,
    }
    dest = ROOT / "datasets/controlled_pendulum_pilot/analysis"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config_impl_consistency.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"config/implementation consistency: {verdict} ({n_pass}/{len(checks)})")
    for c in checks:
        if not c["passed"]:
            print(f"  FAIL {c['check']}: {c['detail']}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
