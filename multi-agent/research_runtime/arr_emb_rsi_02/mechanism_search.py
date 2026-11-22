"""Search valid Oracle-sensitive E02/E04 injections across generators."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from research_runtime.arr_emb_rsi_01.adapter import sha256_file
from research_runtime.arr_emb_rsi_02.adapter import WorldModelAdapter
from research_runtime.arr_emb_rsi_02.benchmark import GENERATORS
from research_runtime.settings import DEFAULT_SCINET_DIR

OUT = Path("ARR_EMB_RSI_02_MECHANISM_SEARCH")
OUT.mkdir(parents=True, exist_ok=True)


def freeze(a, name, out_rel, ov, fixed_test):
    cfg, _ = a.build_config(
        run=type("R", (), {"failure_id": name, "arm": "freeze", "seed": 42})(),
        overrides=ov)
    info = a.adapter.freeze_dataset(config=cfg, out=out_rel)
    out = DEFAULT_SCINET_DIR / out_rel
    shutil.copy2(fixed_test / "test.npz", out / "test.npz")
    mpath = out / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["splits"]["test"]["sha256"] = sha256_file(out / "test.npz")
    mpath.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    return info


res = {}
for gen, spec in GENERATORS.items():
    a = WorldModelAdapter(run_root=OUT / gen / "freeze",
                          base_config=spec["config"], max_jobs=16,
                          max_seconds=5400)
    common = dict(spec["common"])
    common["data.t_max"] = spec["t_max"]
    cov = dict(common)
    cov["data.train_samples"] = 4
    cov["train.batch_size"] = 4
    cov_ds = f"datasets/arr_emb_rsi_02/{gen}_cov4"
    freeze(a, f"{gen}_cov4", cov_ds, cov,
           DEFAULT_SCINET_DIR / spec["dataset"])
    cases = {
        "E02_cov4": dict(common, **{"data.frozen_dir": cov_ds,
                                    "data.train_samples": 4,
                                    "train.batch_size": 4}),
        "E02_cov64": dict(common, **{"data.frozen_dir": spec["cov64_dataset"],
                                     "data.train_samples": 64,
                                     "train.batch_size": 4}),
        "E04_lr02": dict(common, **{"train.lr": 0.2}),
        "E04_lr0008": dict(common, **{"train.lr": 0.0008}),
    }
    res[gen] = {}
    for name, ov in cases.items():
        r = a.run_arm(arm=name, failure_id=gen, seed=42, overrides=ov,
                      run_suffix=name)
        v = (r.metrics or {}).get("latent_mse@1")
        res[gen][name] = {"ok": r.ok, "mse1": v,
                          "error": r.error[:200]}
        print(gen, name, res[gen][name], flush=True)
(OUT / "search.json").write_text(json.dumps(
    res, indent=2, sort_keys=True, default=str), encoding="utf-8")
