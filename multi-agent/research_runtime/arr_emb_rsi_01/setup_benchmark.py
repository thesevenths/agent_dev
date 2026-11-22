"""Setup real, hash-verified failure mechanisms for ARR_EMB_RSI_01."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from research_runtime.arr_emb_rsi_01.adapter import WorldModelAdapter, sha256_file
from research_runtime.arr_emb_rsi_01.integrity import sha256_json
from research_runtime.settings import DEFAULT_SCINET_DIR

BASE = {
    "E01": {"model.predictor_hidden": [8, 8]},
    "E02": {"data.train_samples": 16},
    "E03": {"model.predictor_hidden": [8, 8]},
    "E04": {"jepa.rollout_steps": [1]},
}
ORACLE = {
    "E01": {"model.predictor_hidden": [16, 16]},
    "E02": {"data.train_samples": 64},
    "E03": {"model.predictor_hidden": [16, 16]},
    "E04": {"jepa.rollout_steps": [1, 2, 3]},
}


def _build_config(adapter: WorldModelAdapter, name: str,
                  overrides: dict, frozen_dir: str | None = None) -> str:
    cfg = adapter._load_base()
    for key, value in overrides.items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    if frozen_dir:
        cfg["data"]["frozen_dir"] = frozen_dir
    path = adapter.config_root / f"setup_{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return str(path)


def _freeze_coverage(adapter: WorldModelAdapter, samples: int,
                     test_source: Path) -> str:
    out_name = f"datasets/arr_emb_rsi_coverage_{samples}"
    out = DEFAULT_SCINET_DIR / out_name
    cfg_path = _build_config(adapter, f"freeze_cov_{samples}",
                             {"data.train_samples": samples}, out_name)
    info = adapter.adapter.freeze_dataset(config=cfg_path, out=out_name)
    # Keep the SAME hidden test artifact as the baseline smoke dataset.
    shutil.copy2(test_source / "test.npz", out / "test.npz")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["test"]["sha256"] = sha256_file(out / "test.npz")
    manifest["splits"]["test"]["inherited_from"] = str(test_source / "test.npz")
    manifest["p0_2_fixed_test_note"] = "hidden test kept fixed across coverage arms"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    return out_name


def run(out: str = "ARR_EMB_RSI_01_BENCH_SETUP") -> dict:
    adapter = WorldModelAdapter(run_root=out, max_jobs=24, max_seconds=5400)
    base_test = DEFAULT_SCINET_DIR / "datasets/pendulum_smoke_v1"
    cov16 = _freeze_coverage(adapter, 16, base_test)
    cov64 = _freeze_coverage(adapter, 64, base_test)
    result = {"coverage_datasets": {16: cov16, 64: cov64}, "verification": {}}
    for fid in ("E01", "E02", "E03", "E04"):
        for arm in ("failure", "oracle"):
            ov = dict(BASE[fid] if arm == "failure" else ORACLE[fid])
            ood = None
            if fid == "E02":
                cov = cov16 if arm == "failure" else cov64
                ov["data.frozen_dir"] = cov
            if fid == "E03":
                ood = {"data.frozen_dir": "datasets/pendulum_smoke_ood_v1",
                       "data.t_max": 14.0}
            r = adapter.run_arm(arm=arm, failure_id=fid, seed=42,
                                overrides=ov, ood_overrides=ood,
                                run_suffix=f"setup_{arm}")
            result["verification"][f"{fid}_{arm}"] = {
                "ok": r.ok, "metrics": r.metrics, "ood_metrics": r.ood_metrics,
                "config_sha256": r.config_sha256,
                "artifact_hashes": r.artifact_hashes, "error": r.error[:200]}
    Path(out, "benchmark_setup.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False,
                   default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
