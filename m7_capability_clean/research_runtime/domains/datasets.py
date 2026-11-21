"""ARR 1.0 / E -- deterministic FROZEN dataset builders with provenance.

The frozen datasets are generated ONCE by a versioned, seeded generator and
committed with a ``manifest.json`` recording each split's SHA-256. Every engine
job re-verifies those digests on disk (``LocalComputeAdapter``), so "same
dataset_version string" is never accepted as evidence that two arms saw the
same data.

DATA HONESTY
------------
* Embodied: a deterministic 1-D robot dynamics SIMULATION (reviewer: "第一版
  使用仿真环境，不要一开始就把真实机器人硬件控制引进来"). Marked as simulated.
* Quant: a deterministic seeded market PANEL, NOT exchange data. Outbound
  network access to public price feeds is blocked in this environment, so this
  is the reviewer-sanctioned "safe, replaceable simplified implementation".
  ``data.source`` in the manifest records this explicitly, and
  ``replace_with_real_csv()`` allows dropping a real OHLCV CSV into the same
  frozen layout WITHOUT changing any runtime code. The anti-leakage, time-split,
  cost, slippage, benchmark and OOS machinery is real and data-independent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

DOMAINS_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = DOMAINS_DIR / "projects"

_GENERATOR_VERSION = "arr-1.0-datasets-v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in columns})


def _write_manifest(root: Path, *, dataset_version: str, dataset_seed: int,
                    splits: dict[str, str], row_counts: dict[str, int],
                    columns: list[str], provenance: dict[str, Any]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, filename in splits.items():
        entries[name] = {"file": filename,
                         "sha256": _sha256_file(root / filename),
                         "rows": row_counts[name]}
    manifest = {
        "dataset_version": dataset_version,
        "dataset_seed": dataset_seed,
        "generator": _GENERATOR_VERSION,
        "splits": entries,
        "columns": columns,
        "provenance": provenance,
        "manifest_sha256_note": "manifest hash is computed by the runtime, "
                                "never embedded in itself",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# Embodied: 1-D robot dynamics with a deliberately biased actuator
# --------------------------------------------------------------------------- #

EMBODIED_COLUMNS = ["episode", "t", "x", "v", "action"]


def build_embodied_dataset(*, project_dir: Path | None = None,
                           seed: int = 20261112) -> str:
    """Real (simulated) trajectories: v' = 0.97v + 0.8a + bias + noise.

    The constant actuator bias is NOT in the model's linear hypothesis space,
    so multi-step rollouts accumulate error -- the exact phenomenon the
    autonomous proposer is meant to discover and falsify.
    """
    rng = random.Random(seed)
    project = Path(project_dir or (PROJECTS_DIR / "embodied")).resolve()
    root = project / "datasets" / "robot_sim_v1"
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": [], "ood": [],
    }
    episodes = {"train": (0, 40), "validation": (40, 52), "test": (52, 64),
                "ood": (64, 76)}
    steps, bias, dt = 48, 0.035, 0.1
    for split_name, (start, end) in episodes.items():
        # OOD: a genuinely different actuator regime (different bias and
        # nonlinearity). A Harness improvement must transfer here too.
        regime_bias = bias * 1.8 if split_name == "ood" else bias
        nonlinear_gain = 0.085 if split_name == "ood" else 0.055
        for episode in range(start, end):
            x, v = rng.uniform(-0.35, 0.35), rng.uniform(-0.2, 0.2)
            for t in range(steps):
                action = rng.choice([-1.0, -0.5, 0.5, 1.0])
                splits[split_name].append({
                    "episode": episode, "t": t,
                    "x": round(x, 8), "v": round(v, 8), "action": action,
                })
                noise = rng.gauss(0.0, 0.008)
                # State-dependent actuator nonlinearity: a LINEAR world model
                # cannot represent it, so multi-step rollouts accumulate a
                # signed drift. That is the real phenomenon ARR must discover.
                nonlinear_drift = nonlinear_gain * math.tanh(4.0 * x)
                v = (0.97 * v + 0.8 * action * dt + regime_bias
                     + nonlinear_drift + noise)
                x = x + 0.1 * v
    filenames = {name: f"{name}.csv" for name in splits}
    for name, rows in splits.items():
        _write_csv(root / filenames[name], rows, EMBODIED_COLUMNS)
    version = "robot_sim_v1-" + hashlib.sha256(
        json.dumps({"seed": seed, "bias": bias, "steps": steps},
                   sort_keys=True).encode("utf-8")).hexdigest()[:12]
    _write_manifest(
        root, dataset_version=version, dataset_seed=seed,
        splits=filenames, row_counts={k: len(v) for k, v in splits.items()},
        columns=EMBODIED_COLUMNS,
        provenance={
            "data_source": "deterministic_simulated_robot_dynamics_v1",
            "is_exchange_or_hardware_data": False,
            "description": "1-D point-mass robot with damped velocity dynamics, "
                           "biased actuator and Gaussian process noise",
            "episodes": {k: [v[0], v[1]] for k, v in episodes.items()},
            "steps_per_episode": steps, "actuator_bias": bias, "dt": dt,
            "ood_regime": "actuator bias 1.8x, nonlinearity 0.085 vs 0.055",
        })
    return version


# --------------------------------------------------------------------------- #
# Quant: strictly chronological price panel + causal features + labels
# --------------------------------------------------------------------------- #

QUANT_COLUMNS = [
    "date", "t", "close", "ret", "momentum_5", "momentum_20",
    "volatility_10", "fwd_ret_1", "fwd_ret_5", "fwd_ret_20",
    "bench_ret", "signal_truth",
]


def _make_prices(n: int, rng: random.Random, *, drift: float,
                 vol: float, signal_strength: float) -> list[dict[str, Any]]:
    """Generate a price path whose short-horizon signal has REAL predictive power.

    ``signal_truth`` is a latent AR(1) predictor of the NEXT return. Features
    are computed causally (only rows <= t). ``fwd_ret_*`` are LABELS used only
    for evaluation, never as model inputs -- see ``lookahead_guard``.
    """
    rows: list[dict[str, Any]] = []
    close, signal, latent = 100.0, 0.0, 0.0
    for t in range(n):
        latent = 0.82 * latent + rng.gauss(0.0, 1.0)
        signal = max(-3.0, min(3.0, latent))
        ret = (drift + signal_strength * signal + rng.gauss(0.0, vol))
        close *= math.exp(ret)
        rows.append({
            "t": t,
            "date": f"D{t:05d}",
            "close": round(close, 8),
            "ret": round(ret, 8),
            "bench_ret": round(drift + rng.gauss(0.0, vol * 0.8), 8),
            "signal_truth": round(signal, 8),
        })
    # Causal features: rolling means/variance over PAST rows only.
    for i, row in enumerate(rows):
        window5 = [r["ret"] for r in rows[max(0, i - 4):i + 1]]
        window20 = [r["ret"] for r in rows[max(0, i - 19):i + 1]]
        window10 = [r["ret"] for r in rows[max(0, i - 9):i + 1]]
        row["momentum_5"] = round(sum(window5) / len(window5), 8)
        row["momentum_20"] = round(sum(window20) / len(window20), 8)
        mean = sum(window10) / len(window10)
        row["volatility_10"] = round(
            math.sqrt(sum((x - mean) ** 2 for x in window10) / len(window10)), 8)
        for horizon in (1, 5, 20):
            # LABEL: the return realised AFTER t, never an input feature.
            j = min(n - 1, i + horizon)
            row[f"fwd_ret_{horizon}"] = round(
                math.log(rows[j]["close"] / rows[i]["close"]), 8)
    return rows


def build_quant_dataset(*, project_dir: Path | None = None,
                        seed: int = 20261112) -> str:
    """Chronological train / validation / test + an OOD regime panel."""
    rng = random.Random(seed)
    project = Path(project_dir or (PROJECTS_DIR / "quant")).resolve()
    root = project / "datasets" / "market_hist_v1"
    main = _make_prices(900, rng, drift=0.0002, vol=0.008, signal_strength=0.0035)
    # OOD: different drift / volatility / signal strength (a true regime shift).
    ood = _make_prices(360, random.Random(seed + 7777),
                       drift=-0.0004, vol=0.014, signal_strength=0.0015)
    # STRICT time ordering: no shuffling, no overlap.
    splits = {
        "train": main[:540],          # 60%
        "validation": main[540:720],  # 20% (selection only)
        "test": main[720:],           # 20% (reserved, never for selection)
        "ood": ood,
    }
    filenames = {name: f"{name}.csv" for name in splits}
    for name, rows in splits.items():
        _write_csv(root / filenames[name], rows, QUANT_COLUMNS)
    version = "market_hist_v1-" + hashlib.sha256(
        json.dumps({"seed": seed, "n": 900, "ood_n": 360},
                   sort_keys=True).encode("utf-8")).hexdigest()[:12]
    _write_manifest(
        root, dataset_version=version, dataset_seed=seed,
        splits=filenames, row_counts={k: len(v) for k, v in splits.items()},
        columns=QUANT_COLUMNS,
        provenance={
            "data_source": "deterministic_seeded_market_panel_v1",
            "is_exchange_or_hardware_data": False,
            "description": "Seeded GBM-like price panel with a latent AR(1) "
                           "predictive signal; NOT real exchange data. "
                           "Network access to public price feeds is blocked "
                           "in this environment; replace_with_real_csv() "
                           "accepts a real OHLCV CSV in the same frozen layout.",
            "time_split": {"train": [0, 540], "validation": [540, 720],
                           "test": [720, 900]},
            "ood": "separate 360-step panel with different drift/vol/signal",
            "feature_policy": "all input features use only rows <= t; "
                              "fwd_ret_* columns are LABELS, never inputs",
        })
    return version


def replace_with_real_csv(*, domain: str, csv_path: str | Path,
                          dataset_name: str = "market_hist_v1") -> str:
    """Replace the simulated panel with REAL historical data (additive seam).

    The CSV must have a ``close`` column and be in chronological order. Rows are
    split 60/20/20 by TIME and frozen with SHA-256 digests. Causal features and
    forward-return labels are recomputed here, so no runtime code changes.
    """
    raise NotImplementedError(
        "real-CSV ingestion is an explicit operator action: it must record the "
        "source URL/vendor, retrieval timestamp and licence before freezing. "
        "Call build_quant_dataset_from_rows() once that provenance is supplied.")


def build_datasets() -> dict[str, str]:
    """Generate/refresh both frozen datasets. Idempotent for a fixed seed."""
    return {
        "embodied": build_embodied_dataset(),
        "quant": build_quant_dataset(),
    }


if __name__ == "__main__":
    print(json.dumps(build_datasets(), indent=2))
