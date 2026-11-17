"""M7 Domain Pack (embodied): controlled config templates.

Every arm changes at most ONE whitelisted knob from the frozen baseline. The
capacity-control arm is mandatory: an improvement must not be attributed to
the research mechanism when extra training epochs alone can explain it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ....harness_rsi.identity import sha256_file
from .dataset_registry import baseline_config

ARMS: dict[str, dict[str, Any]] = {
    "baseline": {"overrides": {}, "purpose": "frozen control"},
    "action_conditioned": {
        "overrides": {"model.drift_correction": 0.75},
        "purpose": "action-conditioned residual dynamics correction"},
    "multi_step_horizon": {
        "overrides": {"model.rollout_horizon": 6},
        "purpose": "longer open-loop planning horizon"},
    "capacity_control": {
        "overrides": {"train.epochs": 120},
        "purpose": "confound control: extra optimisation budget"},
    "lr_control": {
        "overrides": {"train.lr": 0.01},
        "purpose": "alternative-explanation control"},
}


def _set_dotted(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def build_config(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm}")
    config = baseline_config()
    for key, value in ARMS[arm]["overrides"].items():
        _set_dotted(config, key, value)
    return config


def make_configs(out_dir: str | Path) -> dict[str, Any]:
    root = (Path(out_dir) / "configs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    configs: dict[str, Any] = {}
    for arm in ARMS:
        path = root / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(build_config(arm), sort_keys=True),
                        encoding="utf-8")
        configs[arm] = {"path": str(path), "sha256": sha256_file(path),
                        "overrides": ARMS[arm]["overrides"],
                        "purpose": ARMS[arm]["purpose"]}
    return {"version": "m7_embodied_configs_v1", "configs": configs,
            "single_knob_per_arm": True,
            "capacity_control_present": "capacity_control" in configs}


__all__ = ["ARMS", "build_config", "make_configs"]
