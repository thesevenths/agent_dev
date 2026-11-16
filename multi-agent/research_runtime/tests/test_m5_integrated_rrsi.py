"""M5-INTEGRATED -- RRSI facade serialization + holdout evidence builder (ENGINE-FREE)."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_domain_facade_generate_uses_dataclass_to_dict():
    from research_runtime.rrsi.domain_model_evolver import generate
    cands = generate(domain="embodied", dominant_failure="DECISION_UNCERTAINTY",
                     limit=1)
    assert cands and isinstance(cands[0], dict)
    assert cands[0]["target"] == "train.lr"
    assert cands[0]["new"] in (0.003, 0.005, 0.01)


def test_harness_facade_import_path_and_record_serialization():
    import research_runtime.rrsi.harness_evolver as he
    # module imported cleanly -> the wrong evolution_spaces import is fixed
    assert he.RESEARCH_HARNESS_LAYER == "RESEARCH_HARNESS"
    from adaptive_runtime.harness_evolution import EvolutionRecord
    r = EvolutionRecord(round_no=1, dominant="x", candidate={})
    assert isinstance(r.to_dict(), dict)


def test_build_holdout_evidence_from_real_manifest(tmp_path):
    from research_runtime.rrsi import (
        build_holdout_evidence, verify_holdout_evidence,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"dataset_version":"v1"}', encoding="utf-8")
    # P0-E: two REAL result files on disk; their SHAs must be bound verbatim.
    result_files = []
    for seed in (60, 61):
        rp = tmp_path / f"result_{seed}.json"
        rp.write_text(f'{{"seed":{seed},"Q":0.001}}', encoding="utf-8")
        result_files.append(str(rp))
    ev = build_holdout_evidence(
        manifest_path=str(manifest), split_id="test",
        selection_split_id="validation",
        holdout_seeds=[60, 61], selection_seeds=[42, 43],
        candidate_sha="a" * 64, parent_sha="b" * 64,
        result_files=result_files,
        holdout_metrics={"holdout_improvement": 0.001},
        direction_fraction=1.0)
    sha = "a" * 64
    import hashlib
    msha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    real_shas = [hashlib.sha256(Path(f).read_bytes()).hexdigest()
                 for f in result_files]
    assert ev["dataset_manifest_sha"] == msha
    assert ev["result_shas"] == real_shas        # real artifacts, not derived
    assert ev["independent"] is True
    v = verify_holdout_evidence(ev, expected_candidate_sha=sha,
                                expected_parent_sha="b" * 64,
                                manifest_path=str(manifest))
    assert v["verified"], v["reasons"]
    # A fabricated evidence object with no real result files must FAIL.
    bad = dict(ev, result_shas=[hashlib.sha256(b"fabricated").hexdigest()] * 2)
    v2 = verify_holdout_evidence(
        bad, expected_candidate_sha=sha, expected_parent_sha="b" * 64,
        artifact_reader=lambda e: {"ok": False, "reasons": ["result_sha_not_on_disk"]})
    assert not v2["verified"]


def test_coordinator_outcome_carries_candidate_identity(tmp_path):
    from research_runtime.rrsi import (
        RRSICoordinator, EvolutionRegistry, DOMAIN_MODEL_LAYER,
    )
    from research_runtime.research_store.db import ResearchDatabase
    coord = RRSICoordinator(
        EvolutionRegistry(db=ResearchDatabase(tmp_path / "e.sqlite")))
    out = coord.run(
        experiment_id="E", dominant_failure="x",
        evidence={"changed_layers": [DOMAIN_MODEL_LAYER]},
        generator=lambda **k: [{"candidate_id": "cid1"}],
        evaluator=lambda **k: {"decision": "REJECTED",
                               "holdout_confirmed": True},
        layer=DOMAIN_MODEL_LAYER)
    o = out["outcomes"][0]
    assert o["candidate_id"] == "cid1" and o["candidate_sha"]
