"""M7 Domain Pack: engine-free identity, isolation and evidence integrity."""
from __future__ import annotations

import json

import pytest

from research_runtime.discovery.domain_packs import DomainPack, load_pack
from research_runtime.discovery.domain_packs.embodied_world_model import evaluator
from research_runtime.harness_rsi.identity import freeze_json, sha256_file


def test_pack_implements_generic_protocol_and_has_no_family_overlap(tmp_path):
    pack = load_pack("embodied_world_model")
    assert isinstance(pack, DomainPack)
    ident = pack.dataset_identity()
    assert ident["family_isolation"]["ok"]
    assert ident["family_isolation"]["seed_is_not_family"]
    assert set(ident["family_manifest"]["splits"]) == {
        "train", "validation", "test", "ood"}
    configs = pack.make_configs(tmp_path)
    assert configs["single_knob_per_arm"]
    assert configs["capacity_control_present"]
    for arm, meta in configs["configs"].items():
        assert len(meta["overrides"]) <= 1
        assert meta["path"].startswith(str(tmp_path.resolve()))


def test_evaluator_rejects_tampered_result_sha(tmp_path):
    result_path = tmp_path / "result.json"
    payload = {"status": "succeeded", "metrics": {
        "state_mse@1": 0.1, "task_success_rate": 1.0,
        "safety_violations": 0.0}}
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    train_path = tmp_path / "train_result.json"
    train_path.write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    summary = {"version": "fake", "arms": ["baseline"], "seeds": [42],
               "splits": ["validation"], "results": {
                   "baseline|s42": {"status": "succeeded",
                                    "train": {"status": "succeeded",
                                              "result_path": str(train_path),
                                              "result_sha256": sha256_file(train_path)},
                                    "evaluations": {"validation": {
                                        "status": "succeeded",
                                        "metrics": payload["metrics"],
                                        "result_path": str(result_path),
                                        "result_sha256": sha256_file(result_path)}}}}}
    frozen = freeze_json(summary, tmp_path / "engine_results.json")
    assert evaluator.verify_result_shas({"summary": summary})["ok"]
    result_path.write_text(json.dumps({**payload, "metrics": {
        **payload["metrics"], "state_mse@1": 0.01}}), encoding="utf-8")
    check = evaluator.verify_result_shas({"summary": summary})
    assert not check["ok"]
    assert check["problems"]
    assert len(frozen["sha256"]) == 64
