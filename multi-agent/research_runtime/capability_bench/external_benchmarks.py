"""ARR v1.0 WP-08: external benchmark adapters + coverage matrix.

Design contract:
  * Adapter preserves the EXTERNAL benchmark's own task definitions and
    scoring rules — it NEVER rewrites a test set to make ARR pass.
  * No benchmark result is reported unless a real dataset file and a real
    evaluation run exist. Access/licence unknown -> status BLOCKED/NOT_RUN.
  * An adapter EXISTING is not evidence that ARR PASSED the benchmark.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

EXTERNAL_BENCHMARKS = (
    "ScienceAgentBench", "FIRE-Bench", "DISCERN", "ResearchClawBench",
    "SCI-AGENT-EXAM", "SciAgentArena",
)


class BenchmarkAdapter:
    """Base contract. Subclasses implement load/evaluate against REAL data."""

    name = "base"
    requires_license = True

    def __init__(self, *, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def available(self) -> dict[str, Any]:
        tasks = self.root / "tasks.jsonl"
        return {"ok": tasks.is_file(),
                "reason": "" if tasks.is_file() else "benchmark_data_not_present",
                "path": str(tasks)}

    def load_tasks(self) -> list[dict[str, Any]]:
        import json
        tasks = self.root / "tasks.jsonl"
        if not tasks.is_file():
            return []
        return [json.loads(line) for line in tasks.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def evaluate(self, *, agent: Any, limit: int = 0) -> dict[str, Any]:
        """Run the benchmark's OWN scoring; never fabricate a result."""
        avail = self.available()
        if not avail["ok"]:
            return {"status": "BLOCKED", "reason": avail["reason"],
                    "benchmark": self.name, "n_tasks": 0}
        tasks = self.load_tasks()
        if limit:
            tasks = tasks[:limit]
        return {"status": "NOT_RUN", "reason": "no_evaluation_implemented_for_adapter",
                "benchmark": self.name, "n_tasks": len(tasks)}


class GenericJsonlBenchmarkAdapter(BenchmarkAdapter):
    """A minimal real adapter for any JSONL benchmark with {id,question,answer}."""

    def evaluate(self, *, agent: Any, limit: int = 0) -> dict[str, Any]:
        avail = self.available()
        if not avail["ok"]:
            return {"status": "BLOCKED", "reason": avail["reason"],
                    "benchmark": self.name, "n_tasks": 0}
        tasks = self.load_tasks()
        if limit:
            tasks = tasks[:int(limit)]
        if agent is None:
            return {"status": "BLOCKED", "reason": "no_agent_provided",
                    "benchmark": self.name, "n_tasks": len(tasks)}
        correct = 0
        for t in tasks:
            try:
                traj = agent.call(prompt=str(t.get("question", "")),
                                  response_format_json=False, max_tokens=256)
                if str(t.get("answer", "")).strip().lower() in str(
                        getattr(traj, "content", "")).lower():
                    correct += 1
            except Exception:  # noqa: BLE001 - record as failure, never fabricate
                continue
        return {"status": "EVALUATED", "benchmark": self.name,
                "n_tasks": len(tasks), "n_correct": correct,
                "accuracy": round(correct / len(tasks), 4) if tasks else None,
                "note": "uses benchmark's own id/question/answer fields"}


def coverage_matrix(*, base_dir: str | Path) -> dict[str, Any]:
    """Report REAL availability of each external benchmark adapter."""
    base = Path(base_dir)
    rows = []
    for name in EXTERNAL_BENCHMARKS:
        adapter = GenericJsonlBenchmarkAdapter(
            root=base / name.replace(" ", "_").replace("-", "_"))
        avail = adapter.available()
        rows.append({"benchmark": name,
                     "adapter": "GenericJsonlBenchmarkAdapter",
                     "data_present": avail["ok"],
                     "status": "AVAILABLE" if avail["ok"] else "BLOCKED",
                     "reason": avail.get("reason", ""),
                     "claim": "adapter_only_not_pass"})
    return {"version": "arr_v1_benchmark_coverage_v1", "rows": rows,
            "note": "adapter availability is NOT benchmark PASS; "
                    "licensing/access must be verified separately"}


__all__ = ["EXTERNAL_BENCHMARKS", "BenchmarkAdapter",
           "GenericJsonlBenchmarkAdapter", "coverage_matrix"]
