"""
scripts/audit_controlled_pendulum_v2.py
=======================================
P1-B1 / Phase 5: INDEPENDENT acceptance audit of the v2 pilot (10 checks).

Read-only w.r.t. the dataset; writes evidence JSON into the v2 analysis dir.

Checks
------
  1. exact replay using the PERSISTED conditions array directly
  2. independent DOP853 (scipy, ODE re-implemented) comparison
  3. energy-balance identity dE/dt = tau*omega - b*I*omega^2
  4. parameter and action bounds validation
  5. per-step finite-state checks
  6. regime classification (preregistered)
  7. initial-state diversity
  8. episode-level split isolation
  9. SHA-256 integrity (npz + per-episode)
 10. report-number traceability (delegated to verify_p1a_reports v2)

Also runs the Q-C short_rod stress-test at dt=0.005 with the SAME physical
duration and torque waveform as the historical v1 S1_step episode.

Usage:
  python scripts/audit_controlled_pendulum_v2.py
  -> datasets/controlled_pendulum_v2_pilot/{audit_report.json, analysis/*.json}
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.integrate import solve_ivp  # noqa: E402
from data.controlled_pendulum import (  # noqa: E402
    ControlledPendulum, ProfileSpec, ScenarioSpec, generate_torque_sequence,
)
from data.controlled_pendulum_v2 import (  # noqa: E402
    ValidatedPendulum, classify_regime, generate_torque_sequence_v2,
)

V2 = ROOT / "datasets" / "controlled_pendulum_v2_pilot"
V1 = ROOT / "datasets" / "controlled_pendulum_pilot"
CFG = ROOT / "configs" / "dynamics" / "controlled_pendulum_v2_pilot.yaml"


def energy(obs, p):
    I = p["m"] * p["L"] * p["L"]
    th, om = obs[:, 0], obs[:, 1]
    return 0.5 * I * om ** 2 + p["m"] * p["g"] * p["L"] * (1 - np.cos(th))


def scipy_reference(p, torques, theta0, omega0, dt):
    g, L, m, b = p["g"], p["L"], p["m"], p["b"]
    I = m * L * L

    def f(t, y, tau):
        return [y[1], -(g / L) * math.sin(y[0]) - b * y[1] + tau / I]

    y = np.array([float(theta0), float(omega0)], dtype=np.float64)
    out = [y.copy()]
    for tau in torques:
        sol = solve_ivp(f, (0.0, dt), y, args=(float(tau),),
                        method="DOP853", rtol=1e-12, atol=1e-14)
        y = sol.y[:, -1]
        out.append(y.copy())
    return np.asarray(out, dtype=np.float64)


def load():
    with np.load(V2 / "pilot_episodes.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    man = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    return data, man, cfg


def independent_regime_audit(obs, man, cfg):
    """Independently re-derive every episode's regime from RAW observations.

    This is NOT a format-only check. It:
      1. loads the preregistered ``regime_criteria`` from the YAML config and
         validates it is complete and well-formed (missing/invalid -> fail);
      2. cross-checks that the manifest's stored ``regime_criteria`` equals the
         YAML (config/implementation consistency);
      3. recomputes each episode's regime from ``obs[i]`` + its physical
         parameters using the YAML thresholds -- never trusting the stored label;
      4. recomputes aggregate regime counts and compares them to the manifest;
      5. reports per-episode label mismatches.

    Returns ``(passed, detail)``.
    """
    from data.controlled_pendulum_v2 import (
        _validate_regime_criteria, classify_regime,
    )

    eps = man["episodes"]
    # 1. load + validate preregistered criteria from YAML
    try:
        criteria = cfg["regime_criteria"]
        _validate_regime_criteria(criteria)
    except Exception as exc:  # missing/invalid criterion must fail
        return False, {
            "passed": False,
            "error": f"regime_criteria invalid or missing in config: {exc}",
            "criteria_present": "regime_criteria" in cfg,
        }

    # 2. config consistency: YAML must equal manifest-stored resolved config
    stored_criteria = man.get("resolved_config", {}).get("regime_criteria")
    config_consistent = (stored_criteria == criteria)

    # 3 + 4. recompute per-episode and aggregate from raw observations
    recomputed_counts = {"bounded_oscillation": 0,
                         "transitional": 0,
                         "continuous_rotation": 0}
    label_mismatches = []
    metric_mismatches = []
    for i, e in enumerate(eps):
        pp = e["physical_parameters"]
        sc = ScenarioSpec(
            name=e["episode_id"], g=pp["g"], L=pp["L"], m=pp["m"], b=pp["b"],
            theta0=float(e["initial_state"][0]), omega0=float(e["initial_state"][1]))
        r = classify_regime(obs[i], sc, criteria=criteria)
        recomputed_counts[r["regime"]] += 1
        if r["regime"] != e["regime"]:
            label_mismatches.append({
                "episode_id": e["episode_id"],
                "stored": e["regime"], "recomputed": r["regime"]})
        rd = e.get("regime_detail", {})
        for mk in ("winding_number", "theta_range_rad", "omega_sign_changes"):
            if mk in rd and abs(float(rd[mk]) - float(r[mk])) > 1e-9:
                metric_mismatches.append({
                    "episode_id": e["episode_id"], "metric": mk,
                    "stored": rd[mk], "recomputed": r[mk]})

    stored_counts = man.get("regime_counts", {})
    counts_match = (recomputed_counts == stored_counts)
    passed = (not label_mismatches) and counts_match and config_consistent
    return passed, {
        "passed": passed,
        "criteria_present": True,
        "config_consistent_with_yaml": config_consistent,
        "recomputed_counts": recomputed_counts,
        "manifest_counts": stored_counts,
        "counts_match": counts_match,
        "n_label_mismatches": len(label_mismatches),
        "label_mismatches": label_mismatches[:20],
        "n_metric_mismatches": len(metric_mismatches),
        "criteria_used": criteria,
        "note": ("Regime recomputed independently from raw observations using "
                 "YAML regime_criteria; not a stored-label format check."),
    }


def main() -> int:
    data, man, cfg = load()
    obs, cond, valid = data["observations"], data["conditions"], data["valid_mask"]
    dt = float(man["dt"])
    T = int(man["T"])
    tb = man["resolved_config"]["torque_bounds"]
    eps = man["episodes"]

    results = []

    def add(name, passed, detail):
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. exact replay with the PERSISTED conditions array ------------------ #
    mism = []
    for i, e in enumerate(eps):
        pp = e["physical_parameters"]
        p = dict(g=pp["g"], L=pp["L"], m=pp["m"], b=pp["b"])
        pen = ValidatedPendulum(**p, dt=dt, torque_bounds=tb)
        # IMPORTANT: use stored cond[i,:,0] verbatim, not regenerated actions.
        o = pen.simulate(cond[i, :, 0], *e["initial_state"])
        if not np.array_equal(o, obs[i]):
            mism.append(e["episode_id"])
    add("exact_replay_from_persisted_conditions",
        not mism, {"n": len(eps), "mismatches": mism,
                   "max_abs_error": 0.0 if not mism else None})

    # 2. independent DOP853 ------------------------------------------------- #
    xcheck = []
    for i, e in enumerate(eps):
        pp = e["physical_parameters"]
        p = dict(g=pp["g"], L=pp["L"], m=pp["m"], b=pp["b"])
        ref = scipy_reference(p, cond[i, :, 0], *e["initial_state"], dt)
        xcheck.append({"episode_id": e["episode_id"],
                       "max_abs_state_error": float(np.max(np.abs(ref - obs[i])))})
    max_x = max(x["max_abs_state_error"] for x in xcheck)
    add("independent_dop853_crosscheck", max_x < 1e-4,
        {"max_abs_state_error_over_all": max_x,
         "tolerance": 1e-4, "per_episode": xcheck})

    # 3. energy balance ------------------------------------------------------ #
    ebal = []
    for i, e in enumerate(eps):
        pp = e["physical_parameters"]
        p = dict(g=pp["g"], L=pp["L"], m=pp["m"], b=pp["b"])
        I, b = pp["m"] * pp["L"] ** 2, pp["b"]
        E = energy(obs[i], p)
        om_l, om_r = obs[i, :-1, 1], obs[i, 1:, 1]
        tq = cond[i, :, 0]
        dE_pred = 0.5 * dt * ((tq * om_l - b * I * om_l ** 2)
                              + (tq * om_r - b * I * om_r ** 2))
        dE_obs = np.diff(E)
        scale = max(float(np.max(np.abs(dE_obs))), 1e-12)
        ebal.append({"episode_id": e["episode_id"],
                     "rel_residual": float(np.max(np.abs(dE_obs - dE_pred)) / scale)})
    max_e = max(x["rel_residual"] for x in ebal)
    add("energy_balance_identity", max_e < 1e-2,
        {"identity": "dE/dt = tau*omega - b*I*omega^2",
         "max_rel_residual": max_e, "tolerance": 1e-2, "per_episode": ebal})

    # 4. parameter + action bounds ------------------------------------------- #
    bad_p, bad_a = [], []
    lo, hi = tb
    d = cfg["distributions"]
    for i, e in enumerate(eps):
        pp = e["physical_parameters"]
        if not (d["mass_range"][0] - 1e-9 <= pp["m"] <= d["mass_range"][1] + 1e-9
                and d["length_range"][0] - 1e-9 <= pp["L"] <= d["length_range"][1] + 1e-9
                and d["damping_rate_range"][0] - 1e-9 <= pp["b"] <= d["damping_rate_range"][1] + 1e-9
                and abs(pp["g"] - d["gravity_g"]) < 1e-9
                and pp["m"] > 0 and pp["L"] > 0 and pp["b"] >= 0):
            bad_p.append(e["episode_id"])
        t = cond[i, :, 0]
        if not (np.all(t >= lo - 1e-9) and np.all(t <= hi + 1e-9)):
            bad_a.append(e["episode_id"])
    add("parameter_and_action_bounds", not bad_p and not bad_a,
        {"parameter_violations": bad_p, "action_violations": bad_a,
         "torque_bounds": tb, "train_ranges": {
             k: d[k] for k in ("mass_range", "length_range",
                               "damping_rate_range", "theta0_range", "omega0_range")}})

    # 5. per-step finite ------------------------------------------------------ #
    finite = bool(np.all(np.isfinite(obs)) and np.all(np.isfinite(cond)))
    max_dth = float(np.max(np.abs(np.diff(obs[:, :, 0], axis=1))))
    add("per_step_finite_state", finite and max_dth < math.pi,
        {"all_finite": finite, "max_abs_delta_theta_per_step": max_dth,
         "pi_threshold": math.pi})

    # 6. INDEPENDENT regime audit (recompute from raw observations + YAML) --- #
    reg_pass, reg_detail = independent_regime_audit(obs, man, cfg)
    add("independent_regime_audit", reg_pass, reg_detail)

    # 7. initial-state diversity --------------------------------------------- #
    th0 = np.array([e["initial_state"][0] for e in eps])
    om0 = np.array([e["initial_state"][1] for e in eps])
    uniq_th = len(np.unique(np.round(th0, 6)))
    uniq_om = len(np.unique(np.round(om0, 6)))
    add("initial_state_diversity", uniq_th >= 15 and uniq_om >= 15
        and float(np.std(th0)) > 0.5,
        {"distinct_theta0": uniq_th, "distinct_omega0": uniq_om,
         "theta0_range_observed": [float(th0.min()), float(th0.max())],
         "omega0_range_observed": [float(om0.min()), float(om0.max())],
         "theta0_std": float(np.std(th0)), "n": len(eps),
         "v1_baseline": "single point (theta0=2.0, omega0=0.0)"})

    # 8. episode-level split isolation --------------------------------------- #
    splits = {s: [e["episode_id"] for e in eps if e["split"] == s]
              for s in ("train", "validation", "test")}
    sets = {k: set(v) for k, v in splits.items()}
    overlap = (sets["train"] & sets["validation"]) | \
              (sets["train"] & sets["test"]) | \
              (sets["validation"] & sets["test"])
    from dynamics.contracts import dynamics_batch_from_windows
    win = {}
    for s, ids in splits.items():
        idxs = [i for i, e in enumerate(eps) if e["episode_id"] in sets[s]]
        w = dynamics_batch_from_windows(
            [obs[i] for i in idxs], [cond[i] for i in idxs],
            episode_ids=[eps[i]["episode_id"] for i in idxs],
            window_len=11, stride=5)
        win[s] = {"n_episodes": len(idxs), "n_windows": int(w.batch_size),
                  "all_sources_in_split": bool(set(w.episode_ids) <= sets[s]),
                  "all_valid": bool(w.valid_mask.all().item())}
    add("episode_level_split_isolation",
        not overlap and all(v["all_sources_in_split"] for v in win.values())
        and all(v["n_episodes"] > 0 for v in win.values()),
        {"counts": {k: len(v) for k, v in splits.items()},
         "overlaps": sorted(overlap), "windows": win})

    # 9. SHA-256 integrity ---------------------------------------------------- #
    npz_sha = hashlib.sha256((V2 / "pilot_episodes.npz").read_bytes()).hexdigest()
    ep_fail = []
    for i, e in enumerate(eps):
        h = hashlib.sha256(obs[i].tobytes() + cond[i].tobytes()).hexdigest()
        if h != e["episode_sha256"]:
            ep_fail.append(e["episode_id"])
    add("sha256_integrity", npz_sha == man["npz_sha256"] and not ep_fail,
        {"npz_match": npz_sha == man["npz_sha256"], "npz_sha256": npz_sha,
         "episode_hash_failures": ep_fail, "git_commit": man["git_commit"]})

    # 10. report-number traceability ------------------------------------------ #
    # Verify the FOUR P1-A'/P1-B1 reports against the v2 evidence (strict
    # JSONPath + per-line anchor + full numeric comparison), AND confirm the
    # hardened v1-report verifier still passes.
    sys.path.insert(0, str(ROOT / "scripts"))
    import verify_v2_reports as V2V
    import verify_p1a_reports as V1V
    rep2, ev2 = V2V.load_inputs()
    vr2 = V2V.verify(rep2, ev2)
    rep1, ev1 = V1V.load_inputs()
    vr1 = V1V.verify(rep1, ev1)
    add("report_number_traceability_v2",
        vr2["verdict"] == "PASS" and vr1["verdict"] == "PASS",
        {"v2_reports": {"verdict": vr2["verdict"],
                        "n_pass": vr2["n_pass"], "n_total": vr2["n_total"]},
         "v1_reports": {"verdict": vr1["verdict"],
                        "n_pass": vr1["n_pass"], "n_total": vr1["n_total"]},
         "all_cases_numerically_compared":
             vr2["strictness"]["all_cases_numerically_compared"]})

    # ---- Q-C: short_rod stress test at dt=0.005 (v1 pilot, READ ONLY) ----- #
    with np.load(V1 / "pilot_episodes.npz", allow_pickle=False) as z:
        v1_cond = z["conditions"]
    v1_man = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    s1 = next(e for e in v1_man["episodes"] if e["episode_id"] == "S1_step")
    p1 = s1["scenario"]
    tor_v1 = v1_cond[v1_man["episodes"].index(s1), :, 0]
    dt_fine = float(cfg["stress_test"]["dt"])
    factor = int(round(float(v1_man["dt"]) / dt_fine))
    fine = ValidatedPendulum(g=p1["g"], L=p1["L"], m=p1["m"], b=p1["b"],
                             dt=dt_fine, torque_bounds=[-5.0, 5.0])
    obs_fine = fine.simulate(np.repeat(tor_v1, factor), p1["theta0"], p1["omega0"])
    obs_fine_on_grid = obs_fine[::factor]
    ref_fine = scipy_reference(p1, tor_v1, p1["theta0"], p1["omega0"],
                               float(v1_man["dt"]))
    obs_coarse = _v1_obs(v1_man, s1)
    err_coarse = float(np.max(np.abs(obs_coarse - ref_fine)))
    err_fine = float(np.max(np.abs(obs_fine_on_grid - ref_fine)))
    stress = {
        "source_episode": "S1_step",
        "original_dt": float(v1_man["dt"]), "original_steps": int(v1_man["T"]),
        "stress_dt": dt_fine, "stress_internal_steps": int(v1_man["T"]) * factor,
        "physical_duration_s": float(v1_man["dt"]) * int(v1_man["T"]),
        "preserves_duration": True,
        "torque_waveform": "np.repeat(original, 4) -- identical physical input",
        "max_abs_error_dt_002": err_coarse,
        "max_abs_error_dt_0005": err_fine,
        "error_reduction": err_coarse / err_fine if err_fine > 0 else None,
    }

    n_pass = sum(1 for r in results if r["passed"])
    verdict = "PASS" if n_pass == len(results) else "FAIL"
    out = {
        "verdict": verdict, "n_pass": n_pass, "n_total": len(results),
        "dataset_version": man["dataset_version"],
        "npz_sha256": man["npz_sha256"], "git_commit": man["git_commit"],
        "checks": results, "stress_test_q_c": stress,
    }
    (V2 / "audit_report.json").write_text(json.dumps(out, indent=2),
                                          encoding="utf-8")
    print("=" * 70)
    print(f"P1-B1 V2 PILOT AUDIT -- {verdict} ({n_pass}/{len(results)})")
    print("=" * 70)
    for r in results:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['check']}")
    print(f"Q-C stress: dt=0.02 err={err_coarse:.3e} -> "
          f"dt=0.005 err={err_fine:.3e} (x{stress['error_reduction']:.1f})")
    print(f"report: {V2 / 'audit_report.json'}")
    return 0 if verdict == "PASS" else 1


def _v1_obs(v1_man, s1):
    """Load the v1 S1_step observations (read-only)."""
    with np.load(V1 / "pilot_episodes.npz", allow_pickle=False) as z:
        obs = z["observations"]
    return obs[v1_man["episodes"].index(s1)]


if __name__ == "__main__":
    sys.exit(main())
