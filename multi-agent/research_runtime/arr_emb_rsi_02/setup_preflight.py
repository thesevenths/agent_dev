"""ARR_EMB_RSI_02 preflight: freeze datasets and validate Oracle sensitivity."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from research_runtime.arr_emb_rsi_02.adapter import WorldModelAdapter
from research_runtime.arr_emb_rsi_02.benchmark import GENERATORS, build_regimes
from research_runtime.settings import DEFAULT_SCINET_DIR
from research_runtime.arr_emb_rsi_01.adapter import sha256_file

OUT = Path("ARR_EMB_RSI_02_PREFLIGHT")


def _freeze(adapter: WorldModelAdapter, name: str, out_rel: str,
            overrides: dict, fixed_test: Path | None = None) -> dict:
    cfg_path, _ = adapter.build_config(
        run=type("R", (), {"failure_id": name, "arm": "freeze",
                           "seed": 42})(),
        overrides=overrides)
    info = adapter.adapter.freeze_dataset(config=cfg_path, out=out_rel)
    out = DEFAULT_SCINET_DIR / out_rel
    if fixed_test:
        shutil.copy2(fixed_test / "test.npz", out / "test.npz")
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["splits"]["test"]["sha256"] = sha256_file(out / "test.npz")
        manifest["splits"]["test"]["fixed_test_source"] = str(
            fixed_test / "test.npz")
        manifest_path.write_text(json.dumps(manifest, indent=2,
                                            sort_keys=True), encoding="utf-8")
    return info


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for gen, spec in GENERATORS.items():
        a = WorldModelAdapter(run_root=OUT / gen / "freeze",
                              base_config=spec["config"], max_jobs=2,
                              max_seconds=1800)
        common = dict(spec["common"])
        common["data.t_max"] = spec["t_max"]
        main = _freeze(a, f"{gen}_main", spec["dataset"], common)
        ood = dict(common)
        ood["data.t_max"] = spec["ood_t_max"]
        ood_ds = _freeze(a, f"{gen}_ood", spec["ood_dataset"], ood)
        cov16 = dict(common)
        cov16["data.train_samples"] = 4
        cov16["train.batch_size"] = 4
        cov16_ds = _freeze(a, f"{gen}_cov4", spec["cov4_dataset"], cov16,
                           fixed_test=DEFAULT_SCINET_DIR / spec["dataset"])
        cov64 = dict(common)
        cov64["data.train_samples"] = 64
        cov64_ds = _freeze(a, f"{gen}_cov64", spec["cov64_dataset"], cov64,
                           fixed_test=DEFAULT_SCINET_DIR / spec["dataset"])
        datasets[gen] = {"main": main.get("dataset_version"),
                         "ood": ood_ds.get("dataset_version"),
                         "cov4": cov16_ds.get("dataset_version"),
                         "cov64": cov64_ds.get("dataset_version")}
    result = {"datasets": datasets, "oracle": {}}
    for regime in build_regimes():
        spec = GENERATORS[regime.generator]
        adapter = WorldModelAdapter(
            run_root=OUT / "runs" / regime.regime_id.replace("::", "__"),
            base_config=spec["config"], max_jobs=6, max_seconds=3600)
        ood_ov = None
        if regime.mechanism == "E03":
            ood_ov = {"data.frozen_dir": regime.ood_dataset,
                      "data.t_max": spec["ood_t_max"]}
        for arm in ("base", "oracle"):
            overrides = dict(regime.common)
            overrides.update(regime.failure_override if arm == "base"
                             else regime.oracle_override)
            r = adapter.run_arm(arm=arm, failure_id=regime.regime_id.split("::")[0],
                                seed=42, overrides=overrides,
                                ood_overrides=ood_ov, run_suffix=f"preflight_{arm}")
            key = f"{regime.regime_id}::{arm}"
            result["oracle"][key] = {"ok": r.ok, "metrics": r.metrics,
                                     "ood_metrics": r.ood_metrics,
                                     "config_sha256": r.config_sha256,
                                     "error": r.error[:300]}
        print("done", regime.regime_id, flush=True)
    (OUT / "preflight.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
