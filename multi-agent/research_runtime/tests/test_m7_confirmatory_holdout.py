"""M6.1-fix-A: independent confirmation stage tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_runtime.harness_rsi.benchmark_v2 import (
    HoldoutAccessVerifier, build_benchmark_v2,
)
from research_runtime.harness_rsi.confirmatory import (
    ConfirmationError, freeze_candidate_for_selection, run_confirmatory_holdout,
)
from research_runtime.harness_rsi.identity import freeze_json


@pytest.fixture()
def setup(tmp_path):
    spec = build_benchmark_v2(tmp_path / "bench")
    verifier = HoldoutAccessVerifier(spec, tmp_path / "access.jsonl")
    frozen = freeze_candidate_for_selection(
        candidate_sha="a" * 64, selection_result_sha="b" * 64,
        prereg_sha="c" * 64, out_path=tmp_path / "frozen.json")
    return spec, verifier, frozen, tmp_path


def _runner_factory(root):
    def runner(task, arm):
        p = root / f"{task.task_id}_{arm}.json"
        frozen = freeze_json({"task": task.task_id, "arm": arm,
                              "metrics": {"q": 0.9 if arm == "candidate" else 0.5}}, p)
        return {"ok": True, "metrics": frozen and {"q": 0.9},
                "result_sha256": frozen["sha256"], "artifact_path": str(p)}
    return runner


def test_holdout_only_after_frozen_candidate(setup):
    spec, verifier, frozen, tmp = setup
    r = run_confirmatory_holdout(
        spec=spec, verifier=verifier, frozen=frozen,
        experiment_id="e1", runner=_runner_factory(tmp / "results"),
        out_dir=tmp / "confirm", holdout_task_ids=[t.task_id for t in
                                                  spec.by_split("holdout")[:2]])
    assert r["ok"] and r["n_tasks"] == 2
    ev = json.loads(Path(r["evidence_path"]).read_text(encoding="utf-8"))
    assert ev["generator_never_saw_holdout"] is True
    assert len(ev["base"]) == len(ev["candidate"]) == 2


def test_holdout_reuse_fails_closed(setup):
    spec, verifier, frozen, tmp = setup
    ids = [t.task_id for t in spec.by_split("holdout")[:2]]
    run_confirmatory_holdout(spec=spec, verifier=verifier, frozen=frozen,
                             experiment_id="e2", runner=_runner_factory(
                                 tmp / "r1"), out_dir=tmp / "c1",
                             holdout_task_ids=ids)
    with pytest.raises(ConfirmationError):
        run_confirmatory_holdout(spec=spec, verifier=verifier, frozen=frozen,
                                 experiment_id="e2", runner=_runner_factory(
                                     tmp / "r2"), out_dir=tmp / "c2",
                                 holdout_task_ids=ids)


def test_holdout_result_sha_mismatch_fails(setup):
    spec, verifier, frozen, tmp = setup
    def bad_runner(task, arm):
        p = tmp / f"bad_{task.task_id}_{arm}.json"
        freeze_json({"x": 1}, p)
        return {"ok": True, "metrics": {}, "result_sha256": "0" * 64,
                "artifact_path": str(p)}
    with pytest.raises(ConfirmationError):
        run_confirmatory_holdout(spec=spec, verifier=verifier, frozen=frozen,
                                 experiment_id="e3", runner=bad_runner,
                                 out_dir=tmp / "c3", holdout_task_ids=[
                                     t.task_id for t in spec.by_split(
                                         "holdout")[:1]])
