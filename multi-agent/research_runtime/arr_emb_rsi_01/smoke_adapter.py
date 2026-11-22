"""Real-engine smoke test for the embodied world-model adapter."""
from __future__ import annotations

import json
from pathlib import Path

from research_runtime.arr_emb_rsi_01.adapter import WorldModelAdapter


def run(out: str = "ARR_EMB_RSI_01_ADAPTER_SMOKE") -> dict:
    adapter = WorldModelAdapter(run_root=out, max_jobs=6, max_seconds=1800.0)
    base = {"model.predictor_hidden": [16, 16]}
    cand = {"model.predictor_hidden": [64, 64]}
    runs = [
        adapter.run_arm(arm="base", failure_id="E01", seed=42,
                        overrides=base, run_suffix="smoke_base"),
        adapter.run_arm(arm="oracle", failure_id="E01", seed=42,
                        overrides=cand, run_suffix="smoke_oracle"),
    ]
    payload = {
        "smoke": "real_engine",
        "runs": [r.to_dict() for r in runs],
        "all_ok": all(r.ok for r in runs),
        "metrics": [r.metrics for r in runs],
    }
    Path(out, "smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                   default=str), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
