"""Run the M7.2 Scientific Judgment pilot with the real LLM agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_runtime.capability_bench.scientific_judgment import (
    compute_metrics, load_cases, load_ground_truth, run_benchmark, select_pilot)
from research_runtime.discovery.llm_agent import LLMAgentAdapter, is_available
from research_runtime.harness_rsi.identity import freeze_json, git_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", default="M7_2_BENCHMARK")
    ap.add_argument("--out-dir", default="M7_2_BENCHMARK/pilot_run")
    ap.add_argument("--per-category", type=int, default=2)
    args = ap.parse_args()
    if not is_available():
        raise SystemExit("REAL_LLM_BLOCKED")
    bench = Path(args.benchmark_dir)
    root = Path(args.out_dir)
    manifest = json.loads((bench / "benchmark_manifest.json").read_text(
        encoding="utf-8"))
    cases = select_pilot(load_cases(bench), args.per_category)
    agent = LLMAgentAdapter(max_retries=2, timeout_s=90.0)
    result = run_benchmark(cases=cases, agent=agent, out_dir=root,
                           run_id="m7_2_scientific_judgment_pilot")
    metrics = compute_metrics(
        results_dir=root, ground_truth=load_ground_truth(bench),
        benchmark_manifest=manifest)
    freeze_json(metrics, root / "metrics.json")
    freeze_json({"adapter": "real_llm", "model_id": agent.cfg["model"],
                 "provider": agent._provider(), "n_calls": len(agent.calls),
                 "source": git_fingerprint(Path(__file__).resolve().parents[1])},
                root / "run_manifest.json")
    print(json.dumps({"n_cases": result["n_cases"],
                      "verdict": metrics["verdict"],
                      "n_scored": metrics["n_scored"],
                      "root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
