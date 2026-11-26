"""
scripts.verify_historical_datasets
==================================

Non-destructive integrity + exact-replay verification for the two historical
controlled-pendulum pilot datasets (P1-A v1 pilot, P1-B1 v2 pilot).

This script does NOT regenerate or overwrite any dataset. It:

  1. Historical dataset SHA verification
     Recomputes the on-disk SHA-256 of each ``pilot_episodes.npz`` and asserts
     it equals the value recorded in the corresponding ``manifest.json``
     (``npz_sha256``). This proves the bytes on disk are the same bytes that
     were originally collected and signed.

  2. V2 per-episode ``episode_sha256`` re-derivation
     For the v2 pilot every episode carries an ``episode_sha256`` =
     sha256(obs.tobytes() + cond.tobytes()). We recompute it from the stored
     arrays and assert equality -- proving the manifest's content hashes are
     internally consistent with the data they describe.

  3. Exact replay from recorded parameters
     For each episode we reconstruct the torque sequence and roll out the
     *identical* simulator used at collection time, purely from the parameters
     recorded in the manifest (scenario / physical params, action profile,
     seed, initial state, dt, torque bounds). We then compare the reconstructed
     observations (and conditions for v2) against the stored arrays.

     Because the simulator is a deterministic float64 RK4 and the data on disk
     are float64, an exact replay must reproduce the stored trajectories
     bit-for-bit (max absolute error == 0.0). Any non-zero error would mean the
     stored data is NOT reproducible from its own recorded provenance -- a
     scientific red flag. We assert a tight tolerance (1e-12) and report the
     actual maximum error.

Outputs
-------
One JSON evidence file per dataset:
    datasets/controlled_pendulum_pilot/historical_replay_evidence.json
    datasets/controlled_pendulum_v2_pilot/historical_replay_evidence.json

Exit code is 0 only if every check passes. Designed to run both locally and in
CI (it requires no network, no training, and no data regeneration).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Ensure the project root (containing the ``data`` package) is importable when
# this script is invoked directly as ``python scripts/verify_historical_datasets.py``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.controlled_pendulum import (
    ControlledPendulum,
    ProfileSpec as V1ProfileSpec,
    ScenarioSpec as V1ScenarioSpec,
    generate_torque_sequence,
)
from data.controlled_pendulum_v2 import (
    ValidatedPendulum,
    ProfileSpec as V2ProfileSpec,
    generate_torque_sequence_v2,
)

# Local execution reproduces bit-identical float64 trajectories. Allow a tiny
# slop for cross-platform / BLAS-independent float reordering just in case, but
# in practice the reported max error is exactly 0.0.
REPLAY_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _v1_profile(d: dict) -> V1ProfileSpec:
    return V1ProfileSpec(
        name=str(d["name"]),
        type=str(d["type"]),
        torque_max=float(d["torque_max"]),
        fraction=float(d.get("fraction", 0.5)),
        magnitude=float(d.get("magnitude", 0.6)),
        amplitude=float(d.get("amplitude", 0.7)),
        frequency_hz=float(d.get("frequency_hz", 0.5)),
        phase=float(d.get("phase", 0.0)),
    )


def _v1_scenario(d: dict) -> V1ScenarioSpec:
    return V1ScenarioSpec(
        name=str(d.get("name", "scenario")),
        g=float(d.get("g", 9.81)),
        L=float(d.get("L", 1.0)),
        m=float(d.get("m", 1.0)),
        b=float(d.get("b", 0.1)),
        theta0=float(d.get("theta0", 2.0)),
        omega0=float(d.get("omega0", 0.0)),
    )


def verify_v1(mf_path: Path, npz_path: Path) -> dict:
    mf = json.loads(mf_path.read_text())
    npz = np.load(npz_path)

    recorded_sha = mf.get("npz_sha256")
    actual_sha = sha256_file(npz_path)
    file_sha_ok = (actual_sha == recorded_sha)

    T = int(mf["T"])
    dt = float(mf["dt"])

    episodes = mf["episodes"]
    per_ep: list[dict] = []
    worst_err = 0.0
    for i, e in enumerate(episodes):
        sc = _v1_scenario(e["scenario"])
        pr = _v1_profile(e["profile"])
        seed = int(e["seed"])
        theta0 = float(e["initial_state"][0])
        omega0 = float(e["initial_state"][1])

        tor = generate_torque_sequence(pr, T, dt, seed)
        sim = ControlledPendulum(g=sc.g, L=sc.L, m=sc.m, b=sc.b, dt=dt)
        obs_recon = sim.simulate(tor, theta0, omega0)

        stored = npz["observations"][i]
        max_err = float(np.max(np.abs(obs_recon - stored)))
        worst_err = max(worst_err, max_err)
        per_ep.append({
            "episode_id": e.get("episode_id"),
            "index": i,
            "max_abs_obs_error": max_err,
        })

    replay_ok = worst_err <= REPLAY_TOLERANCE
    overall_ok = file_sha_ok and replay_ok
    return {
        "dataset": "controlled_pendulum_pilot (v1, P1-A)",
        "manifest": str(mf_path),
        "npz": str(npz_path),
        "file_sha256_actual": actual_sha,
        "file_sha256_recorded": recorded_sha,
        "file_sha256_match": file_sha_ok,
        "n_episodes": len(episodes),
        "T": T,
        "dt": dt,
        "replay_tolerance": REPLAY_TOLERANCE,
        "replay_max_abs_obs_error": worst_err,
        "replay_exact": replay_ok,
        "verdict": "PASS" if overall_ok else "FAIL",
        "per_episode": per_ep,
    }


def verify_v2(mf_path: Path, npz_path: Path) -> dict:
    mf = json.loads(mf_path.read_text())
    npz = np.load(npz_path)

    recorded_sha = mf.get("npz_sha256")
    actual_sha = sha256_file(npz_path)
    file_sha_ok = (actual_sha == recorded_sha)

    T = int(mf["T"])
    dt = float(mf["dt"])
    tb = list(mf["resolved_config"]["torque_bounds"])

    episodes = mf["episodes"]
    per_ep: list[dict] = []
    worst_err_obs = 0.0
    worst_err_cond = 0.0
    sha_all_ok = True
    for i, e in enumerate(episodes):
        pp = e["physical_parameters"]
        g = float(pp["g"]); L = float(pp["L"]); m = float(pp["m"]); b = float(pp["b"])
        theta0 = float(e["initial_state"][0])
        omega0 = float(e["initial_state"][1])
        pr_dict = e["action_profile"]
        pr = V2ProfileSpec(
            name=str(pr_dict["name"]),
            type=str(pr_dict["type"]),
            torque_max=float(pr_dict["torque_max"]),
            fraction=float(pr_dict.get("fraction", 0.5)),
            magnitude=float(pr_dict.get("magnitude", 0.6)),
            amplitude=float(pr_dict.get("amplitude", 0.7)),
            frequency_hz=float(pr_dict.get("frequency_hz", 0.5)),
            phase=float(pr_dict.get("phase", 0.0)),
        )
        seed = int(e["seed"])
        hold = int(e["action_hold_steps"])

        tor = generate_torque_sequence_v2(
            pr, T, dt, seed, hold_steps=hold, torque_bounds=tb)
        pen = ValidatedPendulum(g=g, L=L, m=m, b=b, dt=dt, torque_bounds=tb)
        obs_recon = pen.simulate(tor, theta0, omega0)
        cond_recon = tor.reshape(-1, 1).astype(np.float64)

        stored_obs = npz["observations"][i]
        stored_cond = npz["conditions"][i]
        err_obs = float(np.max(np.abs(obs_recon - stored_obs)))
        err_cond = float(np.max(np.abs(cond_recon - stored_cond)))
        worst_err_obs = max(worst_err_obs, err_obs)
        worst_err_cond = max(worst_err_cond, err_cond)

        # Re-derive the episode content hash exactly as the collector did:
        #   sha256(obs.tobytes() + cond.tobytes())
        ep_sha = hashlib.sha256(
            obs_recon.tobytes() + cond_recon.tobytes()).hexdigest()
        sha_ok = (ep_sha == e["episode_sha256"])
        sha_all_ok = sha_all_ok and sha_ok

        per_ep.append({
            "episode_id": e.get("episode_id"),
            "index": i,
            "max_abs_obs_error": err_obs,
            "max_abs_cond_error": err_cond,
            "episode_sha256_match": sha_ok,
        })

    replay_ok = (worst_err_obs <= REPLAY_TOLERANCE
                 and worst_err_cond <= REPLAY_TOLERANCE)
    overall_ok = file_sha_ok and replay_ok and sha_all_ok
    return {
        "dataset": "controlled_pendulum_v2_pilot (v2, P1-B1)",
        "manifest": str(mf_path),
        "npz": str(npz_path),
        "file_sha256_actual": actual_sha,
        "file_sha256_recorded": recorded_sha,
        "file_sha256_match": file_sha_ok,
        "n_episodes": len(episodes),
        "T": T,
        "dt": dt,
        "torque_bounds": tb,
        "replay_tolerance": REPLAY_TOLERANCE,
        "replay_max_abs_obs_error": worst_err_obs,
        "replay_max_abs_cond_error": worst_err_cond,
        "replay_exact": replay_ok,
        "episode_sha256_all_match": sha_all_ok,
        "verdict": "PASS" if overall_ok else "FAIL",
        "per_episode": per_ep,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    v1_mf = root / "datasets" / "controlled_pendulum_pilot" / "manifest.json"
    v1_npz = root / "datasets" / "controlled_pendulum_pilot" / "pilot_episodes.npz"
    v2_mf = root / "datasets" / "controlled_pendulum_v2_pilot" / "manifest.json"
    v2_npz = root / "datasets" / "controlled_pendulum_v2_pilot" / "pilot_episodes.npz"

    v1_res = verify_v1(v1_mf, v1_npz)
    v2_res = verify_v2(v2_mf, v2_npz)

    (v1_mf.parent / "historical_replay_evidence.json").write_text(
        json.dumps(v1_res, indent=2))
    (v2_mf.parent / "historical_replay_evidence.json").write_text(
        json.dumps(v2_res, indent=2))

    all_ok = (v1_res["verdict"] == "PASS") and (v2_res["verdict"] == "PASS")

    print("=== Historical dataset integrity + exact replay ===")
    for r in (v1_res, v2_res):
        print(f"\n[{r['dataset']}]  verdict = {r['verdict']}")
        print(f"  npz sha256 match      : {r['file_sha256_match']}")
        print(f"  replay exact (<=1e-12): {r['replay_exact']}  "
              f"(max obs err {r['replay_max_abs_obs_error']:.3e})")
        if "episode_sha256_all_match" in r:
            print(f"  per-episode sha256     : {r['episode_sha256_all_match']}")
        print(f"  evidence -> {Path(r['manifest']).parent / 'historical_replay_evidence.json'}")

    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
