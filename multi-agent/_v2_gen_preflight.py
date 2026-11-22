import json
from pathlib import Path

from research_runtime.arr_emb_rsi_02.adapter import WorldModelAdapter

out = Path("ARR_EMB_RSI_02_GENERATOR_PREFLIGHT")
res = {}
for gen, cfg in [("newton", "configs/newton.yaml"),
                 ("double_pendulum", "configs/double_pendulum.yaml")]:
    a = WorldModelAdapter(run_root=out / gen, base_config=cfg,
                          max_jobs=4, max_seconds=1800)
    dataset = f"datasets/arr_emb_rsi_02/{gen}_v1"
    freeze_cfg, _ = a.build_config(
        run=type("R", (), {"failure_id": gen, "arm": "freeze", "seed": 42})(),
        overrides={"data.train_samples": 64, "data.test_samples": 16,
                   "data.validation_samples": 16, "data.adapt_samples": 16,
                   "device": "cpu"})
    info = a.adapter.freeze_dataset(config=freeze_cfg, out=dataset)
    main_ov = {"data.frozen_dir": dataset, "data.train_samples": 64,
               "data.test_samples": 16, "data.validation_samples": 16,
               "data.adapt_samples": 16, "device": "cpu",
               "train.epochs": 1, "train.batch_size": 16}
    r = a.run_arm(arm="base", failure_id=gen, seed=42,
                  overrides=main_ov, run_suffix="gen_preflight")
    res[gen] = {"dataset_version": info.get("dataset_version"), "ok": r.ok,
                "metrics": r.metrics, "error": r.error[:500]}
    print(gen, json.dumps(res[gen]), flush=True)
(out / "generator_preflight.json").write_text(
    json.dumps(res, indent=2, default=str), encoding="utf-8")
