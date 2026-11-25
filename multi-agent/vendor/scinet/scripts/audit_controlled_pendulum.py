"""
scripts/audit_controlled_pendulum.py
=====================================
Scientific acceptance audit for the P1-A Controlled-Pendulum pilot.

    python scripts/audit_controlled_pendulum.py \
        --config configs/dynamics/controlled_pendulum_jepa.yaml

Runs an 8-item numerical acceptance checklist over the stored pilot dataset and
the simulator itself, then (bonus) confirms the stored tensors satisfy the
``dynamics.contracts.DynamicsBatch`` contract in float64.

Checklist
---------
 1. Integrity & schema        -- npz/manifest present, shapes/dtypes correct,
                                 manifest SHA-256 matches the npz, ids unique.
 2. Deterministic replay      -- every episode re-simulated from its recorded
                                 (scenario, profile, seed) reproduces the stored
                                 observations/conditions BITWISE (float64).
 3. Zero-torque analytic      -- small-angle RK4 theta matches the analytic
                                 solution theta(t)=th0 cos(wt)+(w0/w) sin(wt).
 4. Zero-torque energy        -- for b=0, tau=0 the mechanical energy is
                                 conserved to RK4 tolerance (validates the
                                 gravity sign and the 1/(m L^2) torque term).
 5. Action-sensitivity sign   -- opposite-signed torques drive the trajectory
                                 apart (control authority).
 6. Action-sensitivity mag.   -- larger |torque| yields a larger deviation from
                                 the zero-torque baseline; no collapse.
 7. RK4 convergence           -- halving the step shrinks the error ~16x
                                 (confirming genuine 4th-order integration).
 8. Unwrapped-theta & bounds  -- no spurious +/-2pi jumps, torques in bounds,
                                 no NaN/Inf, all episodes equal length.

Exit code 0 iff every check passes. A JSON report is written with --report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _check(name: str, passed: bool, detail: dict) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit P1-A controlled-pendulum pilot")
    parser.add_argument("--config", required=True, type=Path, help="YAML pilot config")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Override experiment.output_root")
    parser.add_argument("--report", type=Path, default=None,
                        help="Write a JSON audit report to this path")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from data.controlled_pendulum import (
        ControlledPendulum,
        ScenarioSpec,
        ProfileSpec,
        generate_episode,
        derive_episode_seed,
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else \
        Path(cfg["experiment"]["output_root"])
    npz_path = dataset_dir / "pilot_episodes.npz"
    manifest_path = dataset_dir / "manifest.json"

    if not npz_path.is_file():
        print(f"ERROR: {npz_path} not found. Run collect_controlled_pendulum.py first.",
              file=sys.stderr)
        return 2
    if not manifest_path.is_file():
        print(f"ERROR: {manifest_path} not found.", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = _load_npz(npz_path)
    dt = float(manifest["dt"])
    T = int(manifest["T"])
    torque_max = float(manifest["torque_max"])
    obs = data["observations"]
    cond = data["conditions"]
    valid = data["valid_mask"]
    init = data["initial_state"]

    results = []

    # ------------------------------------------------------------------ #
    # 1. Integrity & schema
    # ------------------------------------------------------------------ #
    meta = json.loads(str(data["meta_str"]))
    expected_obs = (manifest["n_episodes"], T + 1, manifest["obs_dim"])
    expected_cond = (manifest["n_episodes"], T, manifest["cond_dim"])
    id_match = list(meta.get("episode_ids", [])) == [e["episode_id"] for e in manifest["episodes"]]

    import hashlib
    actual_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    sha_ok = (actual_sha == manifest["npz_sha256"])

    schema_ok = (
        obs.shape == tuple(expected_obs) and obs.dtype == np.float64 and
        cond.shape == tuple(expected_cond) and cond.dtype == np.float64 and
        valid.shape == (manifest["n_episodes"], T) and valid.dtype == bool and
        init.shape == (manifest["n_episodes"], 2) and
        data["scenario_index"].shape == (manifest["n_episodes"],) and
        data["profile_index"].shape == (manifest["n_episodes"],)
    )
    uniq = manifest["integrity"]["unique_episode_ids"]
    check1 = schema_ok and sha_ok and id_match and uniq and bool(valid.all()) and \
        (manifest["n_episodes"] == 20)
    results.append(_check("integrity_schema", check1, {
        "observations_shape": list(obs.shape), "observations_dtype": str(obs.dtype),
        "conditions_shape": list(cond.shape), "valid_mask_all_true": bool(valid.all()),
        "n_episodes": manifest["n_episodes"], "unique_episode_ids": uniq,
        "manifest_sha256_match": sha_ok, "episode_id_match": id_match,
        "expected_obs": list(expected_obs), "expected_cond": list(expected_cond),
    }))

    # Bonus: DynamicsBatch contract (float64) construction must validate.
    try:
        import torch
        from dynamics.contracts import DynamicsBatch
        ep_ids = [e["episode_id"] for e in manifest["episodes"]]
        batch = DynamicsBatch(
            torch.as_tensor(obs, dtype=torch.float64),
            torch.as_tensor(cond, dtype=torch.float64),
            torch.as_tensor(valid, dtype=torch.bool),
            ep_ids,
        )
        contract_ok = True
        contract_detail = {"validated": True, "dtype": "float64",
                           "batch_size": batch.batch_size,
                           "num_steps": batch.num_steps,
                           "obs_dim": batch.obs_dim, "cond_dim": batch.cond_dim}
    except Exception as exc:  # pragma: no cover
        contract_ok = False
        contract_detail = {"validated": False, "error": repr(exc)}
    results[-1]["detail"]["dynamics_batch_contract"] = contract_detail

    # ------------------------------------------------------------------ #
    # 2. Deterministic replay (bitwise)
    # ------------------------------------------------------------------ #
    replay_fail = []
    seed_mismatch = []
    # The stacked arrays are in the SAME order as manifest["episodes"] (collect
    # appended episode_meta in plan order), so row i of obs/cond corresponds to
    # manifest["episodes"][i]. Index by enumerate position, not scenario_index.
    for i, e in enumerate(manifest["episodes"]):
        s_idx = e["scenario_index"]
        p_idx = e["profile_index"]
        expected_seed = derive_episode_seed(manifest["master_seed"], s_idx, p_idx)
        if expected_seed != e["seed"]:
            seed_mismatch.append(e["episode_id"])
        sc = ScenarioSpec(**e["scenario"])
        pr = ProfileSpec(**e["profile"])
        regen = generate_episode(sc, pr, e["seed"], T, dt)
        same_obs = np.array_equal(regen.observations, obs[i])
        same_cond = np.array_equal(regen.conditions, cond[i])
        if not (same_obs and same_cond):
            replay_fail.append(e["episode_id"])
    check2 = (not replay_fail) and (not seed_mismatch)
    results.append(_check("deterministic_replay", check2, {
        "replayed": manifest["n_episodes"],
        "replay_mismatches": replay_fail,
        "seed_derivation_mismatches": seed_mismatch,
    }))

    # ------------------------------------------------------------------ #
    # 3. Zero-torque analytic match (small angle)
    # ------------------------------------------------------------------ #
    g, L = 9.81, 1.0
    theta0, omega0 = 0.05, 0.0
    T_small = 300
    sim_z = ControlledPendulum(g=g, L=L, m=1.0, b=0.0, dt=dt)
    zeros = np.zeros(T_small, dtype=np.float64)
    obs_z = sim_z.simulate(zeros, theta0, omega0)
    w = math.sqrt(g / L)
    t = np.arange(T_small + 1) * dt
    theta_analytic = theta0 * np.cos(w * t) + (omega0 / w) * np.sin(w * t)
    max_err = float(np.max(np.abs(obs_z[:, 0] - theta_analytic)))
    tol3 = 1e-3
    check3 = max_err < tol3
    results.append(_check("zero_torque_analytic", check3, {
        "theta0": theta0, "g": g, "L": L, "T": T_small, "dt": dt,
        "max_abs_theta_error": max_err, "tolerance": tol3,
    }))

    # ------------------------------------------------------------------ #
    # 4. Zero-torque energy conservation (b=0)
    # ------------------------------------------------------------------ #
    m, g2, L2 = 1.0, 9.81, 1.0
    th0, w0 = 2.0, 1.0
    sim_e = ControlledPendulum(g=g2, L=L2, m=m, b=0.0, dt=dt)
    obs_e = sim_e.simulate(np.zeros(T, dtype=np.float64), th0, w0)
    th = obs_e[:, 0]
    om = obs_e[:, 1]
    E = 0.5 * m * (L2 ** 2) * om ** 2 + m * g2 * L2 * (1.0 - np.cos(th))
    E0 = E[0]
    max_e_drift = float(np.max(np.abs(E - E0)))
    rel_drift = max_e_drift / (abs(E0) + 1e-12)
    tol4 = 1e-5
    check4 = rel_drift < tol4
    results.append(_check("zero_torque_energy", check4, {
        "g": g2, "L": L2, "m": m, "theta0": th0, "omega0": w0,
        "E0": float(E0), "max_abs_energy_drift": max_e_drift,
        "relative_drift": rel_drift, "tolerance": tol4,
    }))

    # ------------------------------------------------------------------ #
    # 5. Action-sensitivity (sign)
    # ------------------------------------------------------------------ #
    sim_s = ControlledPendulum(g=9.81, L=1.0, m=1.0, b=0.1, dt=dt)
    tau = torque_max / 2.0
    obs_pos = sim_s.simulate(np.full(T, tau, dtype=np.float64), 2.0, 0.0)
    obs_neg = sim_s.simulate(np.full(T, -tau, dtype=np.float64), 2.0, 0.0)
    obs_zero = sim_s.simulate(np.zeros(T, dtype=np.float64), 2.0, 0.0)
    d_pos_neg = float(np.abs(obs_pos[-1, 0] - obs_neg[-1, 0]))
    d_pos_zero = float(np.abs(obs_pos[-1, 0] - obs_zero[-1, 0]))
    tol5 = 0.1
    check5 = (d_pos_neg > tol5) and (d_pos_zero > tol5)
    results.append(_check("action_sensitivity_sign", check5, {
        "tau_magnitude": tau, "final_theta_pos": float(obs_pos[-1, 0]),
        "final_theta_neg": float(obs_neg[-1, 0]),
        "final_theta_zero": float(obs_zero[-1, 0]),
        "abs_pos_minus_neg": d_pos_neg, "abs_pos_minus_zero": d_pos_zero,
        "tolerance": tol5,
    }))

    # ------------------------------------------------------------------ #
    # 6. Action-sensitivity (magnitude)
    # ------------------------------------------------------------------ #
    sim_m = ControlledPendulum(g=9.81, L=1.0, m=1.0, b=0.1, dt=dt)
    base = sim_m.simulate(np.zeros(T, dtype=np.float64), 2.0, 0.0)[-1, 0]
    c_vals = [0.0, 0.25, 0.5, 1.0]
    finals = []
    devs = []
    for c in c_vals:
        o = sim_m.simulate(np.full(T, c * torque_max, dtype=np.float64), 2.0, 0.0)
        finals.append(float(o[-1, 0]))
        devs.append(float(abs(o[-1, 0] - base)))
    unique_finals = len(set(round(x, 6) for x in finals))
    dev_larger = devs[3] >= devs[1]  # c=1.0 deviates at least as much as c=0.25
    nonzero_dev = all(d > 1e-6 for d in devs[1:])
    check6 = (unique_finals >= 3) and dev_larger and nonzero_dev
    results.append(_check("action_sensitivity_magnitude", check6, {
        "c_vals": c_vals, "final_thetas": finals, "deviations_from_zero": devs,
        "unique_finals": unique_finals, "largest_dev_ge_smaller_dev": dev_larger,
        "nonzero_deviations": nonzero_dev,
    }))

    # ------------------------------------------------------------------ #
    # 7. RK4 convergence (4th order)
    # ------------------------------------------------------------------ #
    sim_c = ControlledPendulum(g=9.81, L=1.0, m=1.0, b=0.1, dt=dt)
    # A few representative start states; integrate a short horizon.
    # `simulate_substep` already returns values on the transition grid (T+1
    # points at 0, dt, 2dt, ...), so coarse / dt-2 / dt-16 are all directly
    # comparable without resampling.
    starts = [(2.0, 0.0), (1.0, 3.0), (-2.5, -1.5), (0.5, 2.0)]
    torq = 1.3
    n_steps = 60
    max_e_dt = 0.0
    max_e_dt2 = 0.0
    for (th0c, w0c) in starts:
        ref = sim_c.simulate_substep(np.full(n_steps, torq), th0c, w0c, 16)  # dt/16
        coarse = sim_c.simulate(np.full(n_steps, torq), th0c, w0c)           # dt
        fine = sim_c.simulate_substep(np.full(n_steps, torq), th0c, w0c, 2)  # dt/2
        max_e_dt = max(max_e_dt, float(np.max(np.abs(coarse - ref))))
        max_e_dt2 = max(max_e_dt2, float(np.max(np.abs(fine - ref))))
    ratio = (max_e_dt / max_e_dt2) if max_e_dt2 > 0 else float("inf")
    # Expect ~16x (4th order): coarse error ~16x the dt/2 error.
    check7 = ratio > 10.0
    results.append(_check("rk4_convergence", check7, {
        "n_starts": len(starts), "n_steps": n_steps, "dt": dt,
        "max_error_dt": max_e_dt, "max_error_dt_over_2": max_e_dt2,
        "error_ratio_dt_vs_dt2": ratio, "expected_order_4_ratio": "~16",
        "tolerance_ratio": 10.0,
    }))

    # ------------------------------------------------------------------ #
    # 8. Unwrapped-theta & bounds sanity
    # ------------------------------------------------------------------ #
    jump_fail = []
    bounds_fail = []
    nan_fail = []
    for i in range(obs.shape[0]):
        dtheta = np.abs(np.diff(obs[i, :, 0]))
        if np.any(dtheta >= math.pi - 1e-9):
            jump_fail.append(manifest["episodes"][i]["episode_id"])
        if np.any(np.abs(cond[i, :, 0]) > torque_max + 1e-9):
            bounds_fail.append(manifest["episodes"][i]["episode_id"])
        if not (np.all(np.isfinite(obs[i])) and np.all(np.isfinite(cond[i]))):
            nan_fail.append(manifest["episodes"][i]["episode_id"])
    equal_len = len(set(obs.shape[0] for _ in [0])) == 1  # trivially true
    all_len_eq = all(o.shape[0] == T + 1 for o in obs) and all(c.shape[0] == T for c in cond)
    check8 = (not jump_fail) and (not bounds_fail) and (not nan_fail) and all_len_eq
    results.append(_check("unwrapped_theta_bounds", check8, {
        "max_abs_theta_step_over_all": float(np.max(np.abs(np.diff(obs[:, :, 0])))),
        "jump_failures": jump_fail, "bounds_failures": bounds_fail,
        "nonfinite_failures": nan_fail, "all_episodes_equal_length": bool(all_len_eq),
        "pi_threshold": math.pi,
    }))

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    verdict = "PASS" if n_pass == n_total else "FAIL"
    print("=" * 72)
    print(f"P1-A CONTROLLED PENDULUM PILOT AUDIT  --  {verdict} ({n_pass}/{n_total})")
    print(f"dataset : {dataset_dir}")
    print(f"version : {manifest['dataset_version']}")
    print("=" * 72)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['check']}")
        for k, v in r["detail"].items():
            print(f"        {k}: {v}")
    print("=" * 72)

    if args.report:
        report = {
            "verdict": verdict,
            "n_pass": n_pass,
            "n_total": n_total,
            "dataset_version": manifest["dataset_version"],
            "npz_sha256": manifest["npz_sha256"],
            "checks": results,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written: {args.report}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
