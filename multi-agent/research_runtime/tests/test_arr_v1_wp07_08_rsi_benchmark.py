"""ARR v1.0 WP-07/08 tests: method registry + external benchmark honesty."""
from __future__ import annotations

import json

import pytest

from research_runtime.capability_bench.external_benchmarks import (
    EXTERNAL_BENCHMARKS, GenericJsonlBenchmarkAdapter, coverage_matrix)
from research_runtime.harness_rsi.method_registry import (
    MethodRegistry, MethodRegistryError)
from research_runtime.research_store.db import ResearchDatabase


def _registry(tmp_path):
    return MethodRegistry(db=ResearchDatabase(tmp_path / "r.sqlite"),
                          root=tmp_path / "methods")


def test_method_registry_requires_failure_evidence_for_promoted(tmp_path):
    reg = _registry(tmp_path)
    with pytest.raises(MethodRegistryError):
        reg.register(method_id="m1", version="v1",
                     component_config={"hypothesis_critic": "strict"},
                     status="PROMOTED")  # no source_failure_refs


def test_method_registry_records_and_verifies_sha(tmp_path):
    reg = _registry(tmp_path)
    r = reg.register(
        method_id="m1", version="v1",
        component_config={"hypothesis_critic": "strict"},
        source_failure_refs=["failure_ev_1", "failure_ev_2"],
        diagnosis={"category": "EVIDENCE_GROUNDING_WEAK"},
        expected_improvement="fewer unsupported claims",
        falsification_criteria=["holdout delta < MME"],
        status="EVALUATED")
    assert r["method_sha"]
    v = reg.verify(r["method_sha"])
    assert v["ok"] and v["status"] == "EVALUATED"
    assert len(reg.list()) >= 1


def test_method_registry_rejects_fabricated_sha(tmp_path):
    reg = _registry(tmp_path)
    assert reg.verify("0" * 64)["ok"] is False


def test_method_sha_drift_detected(tmp_path):
    reg = _registry(tmp_path)
    r = reg.register(method_id="m1", version="v1",
                     component_config={"a": 1}, status="PROPOSED")
    p = r["artifact_path"]
    obj = json.loads(open(p, encoding="utf-8").read())
    obj["component_config"]["a"] = 999
    open(p, "w", encoding="utf-8").write(json.dumps(obj))
    assert reg.verify(r["method_sha"])["reason"] == "artifact_sha_drift"


def test_external_benchmark_blocked_without_data(tmp_path):
    a = GenericJsonlBenchmarkAdapter(
        root=tmp_path / "ScienceAgentBench")
    out = a.evaluate(agent=object())
    assert out["status"] == "BLOCKED"
    assert out["n_tasks"] == 0


def test_external_benchmark_runs_real_data(tmp_path):
    root = tmp_path / "DISCERN"
    root.mkdir()
    (root / "tasks.jsonl").write_text(
        json.dumps({"id": "t1", "question": "2+2?", "answer": "4"}) + "\n",
        encoding="utf-8")

    class FakeAgent:
        def call(self, **kw):
            import types
            return types.SimpleNamespace(content="The answer is 4")

    a = GenericJsonlBenchmarkAdapter(root=root)
    out = a.evaluate(agent=FakeAgent())
    assert out["status"] == "EVALUATED"
    assert out["n_tasks"] == 1 and out["n_correct"] == 1
    assert out["accuracy"] == 1.0


def test_coverage_matrix_reports_all_six(tmp_path):
    cm = coverage_matrix(base_dir=tmp_path)
    names = {r["benchmark"] for r in cm["rows"]}
    assert names == set(EXTERNAL_BENCHMARKS)
    assert all(r["status"] == "BLOCKED" for r in cm["rows"])  # no data
    assert all(r["claim"] == "adapter_only_not_pass" for r in cm["rows"])
