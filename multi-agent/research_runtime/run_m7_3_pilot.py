"""Run the M7.3 Research Memory pilot with four controlled arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_runtime.capability_bench.memory_bench import (compute_metrics,
                                                            run_benchmark)
from research_runtime.discovery.llm_agent import LLMAgentAdapter, is_available
from research_runtime.harness_rsi.identity import freeze_json, git_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", default="M7_3_MEMORY")
    ap.add_argument("--out-dir", default="M7_3_MEMORY/pilot_run")
    args = ap.parse_args()
    if not is_available():
        raise SystemExit("REAL_LLM_BLOCKED")
    root = Path(args.benchmark_dir)
    out = Path(args.out_dir)
    manifest = json.loads((root / "benchmark_manifest.json").read_text(
        encoding="utf-8"))
    agent = LLMAgentAdapter(max_retries=2, timeout_s=90.0)
    executed = run_benchmark(root=root, agent=agent, out_dir=out,
                             run_id="m7_3_memory_pilot")
    metrics = compute_metrics(root=root, results_dir=out,
                              manifest=manifest)
    freeze_json(metrics, out / "metrics.json")
    freeze_json({"adapter": "real_llm", "model_id": agent.cfg["model"],
                 "provider": agent._provider(), "n_calls": len(agent.calls),
                 "source": git_fingerprint(Path(__file__).resolve().parents[1])},
                out / "run_manifest.json")
    print(json.dumps({"verdict": metrics["verdict"],
                      "n_results": executed["n_results"],
                      "per_arm": metrics["per_arm"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
