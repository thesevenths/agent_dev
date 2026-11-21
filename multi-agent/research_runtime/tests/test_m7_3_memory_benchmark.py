"""M7.3 benchmark isolation and metric tests."""
from __future__ import annotations

from research_runtime.capability_bench.memory_bench import compute_metrics


def test_memory_benchmark_ground_truth_is_isolated():
    from pathlib import Path
    import json
    for p in Path("M7_3_MEMORY/scenarios").glob("*.json"):
        text = p.read_text(encoding="utf-8")
        assert "relevant_memory_ids" not in text
        assert "expected_repeat_failure" not in text
    snap = json.loads(Path("M7_3_MEMORY/memory_snapshots/s01.json").read_text(
        encoding="utf-8"))
    assert len(snap["arm_on"]) == len(snap["arm_shuffled"]) == \
        len(snap["arm_equal_token"]) == 900


def test_memory_pilot_verdict_is_inconclusive():
    import json
    from pathlib import Path
    manifest = json.loads(Path("M7_3_MEMORY/benchmark_manifest.json").read_text(
        encoding="utf-8"))
    result = compute_metrics(root="M7_3_MEMORY",
                             results_dir="M7_3_MEMORY/pilot_run",
                             manifest=manifest)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["per_arm"]["ON"]["n"] == 10
