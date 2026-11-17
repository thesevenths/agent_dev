"""M7 G12: coordinator state machine, gates, budget, resume and holdout."""
from __future__ import annotations

import hashlib
import json

import pytest

from research_runtime.discovery.discovery_coordinator import DiscoveryCoordinator
from research_runtime.discovery.literature.base import Paper
from research_runtime.harness_rsi.identity import freeze_json, sha256_file
from research_runtime.research_store.db import ResearchDatabase


class FakeProvider:
    name = "fake_offline"

    def __init__(self):
        self.events = []

    def search(self, query, *, limit, budget):
        self.events.append({"query": query.query_id, "status": "ok"})
        return [Paper(
            paper_id=f"p{i}", doi=f"10.1000/m7{i}",
            title=f"Action-conditioned world model study {i}",
            authors=["Author A"], year=2019 + i,
            abstract=("world model action conditioned dynamics multi-step "
                      "prediction generalization unseen dynamics"),
            source="fake_offline", url="", citations=i, references=[],
            retrieved_at="2026-01-01", query_id=query.query_id,
            content_sha="a" * 64, metadata_sha="b" * 64,
            full_text_available=False) for i in range(6)]


class FakeTrajectory:
    trajectory_id = "fake_traj_m71"
    prompt_sha256 = "a" * 64
    model_id = "fake-model"
    ok = True
    content = json.dumps({
        "H1": {"mechanism": "action conditioned dynamics",
               "predictions": ["state_mse improves on holdout"],
               "falsifying_observations": ["state_mse does not improve"]},
        "H2": {"mechanism": "data conditioning explanation",
               "predictions": ["state_mse improves with matched data"],
               "falsifying_observations": ["state_mse vanishes with matched data"]},
        "H3": {"mechanism": "in-distribution only effect",
               "predictions": ["state_mse improves in-distribution"],
               "falsifying_observations": ["state_mse improves out of distribution"]},
        "H0": {"mechanism": "no systematic difference",
               "predictions": ["state_mse unchanged"],
               "falsifying_observations": ["state_mse improves beyond MME"]}})

    def to_dict(self):
        return {"provider": "fake", "model_id": "fake-model",
                "content": self.content, "ok": True}


class FakeAgent:
    def call(self, **kwargs):
        return FakeTrajectory()


def fake_executor(*, experiment_id, arms, seeds, splits, out_dir):
    root = __import__("pathlib").Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in arms:
        for seed in seeds:
            base_value = 1.0
            if arm == "action_conditioned":
                base_value = 0.5
            elif arm == "capacity_control" or arm == "lr_control":
                base_value = 0.9
            train_payload = {"experiment_id": experiment_id, "arm": arm,
                             "seed": seed, "stage": "train", "status": "succeeded",
                             "metrics": {"train_mse": 0.01},
                             "elapsed_seconds": 0.01}
            train_path = root / f"{arm}_s{seed}_train.json"
            train_path.write_text(json.dumps(train_payload, sort_keys=True),
                                  encoding="utf-8")
            entries = {}
            for split in splits:
                payload = {
                    "experiment_id": experiment_id, "arm": arm, "seed": seed,
                    "stage": "evaluate", "status": "succeeded", "split": split,
                    "metrics": {"state_mse@1": base_value,
                                "task_success_rate": 1.0,
                                "safety_violations": 0.0},
                    "elapsed_seconds": 0.02,
                    "dataset_version": "fake", "engine": "fake_offline"}
                path = root / f"{arm}_s{seed}_{split}.json"
                path.write_text(json.dumps(payload, sort_keys=True),
                                encoding="utf-8")
                entries[split] = {"status": "succeeded",
                                  "metrics": payload["metrics"],
                                  "result_path": str(path),
                                  "result_sha256": sha256_file(path),
                                  "run_id": f"{arm}_{seed}", "job_id": path.stem,
                                  "evaluation_split": split,
                                  "elapsed_seconds": 0.02}
            results[f"{arm}|s{seed}"] = {
                "status": "succeeded",
                "train": {"status": "succeeded",
                          "result_path": str(train_path),
                          "result_sha256": sha256_file(train_path),
                          "elapsed_seconds": 0.01, "metrics": {}},
                "evaluations": entries}
    summary = {"version": "fake_results_v1", "experiment_id": experiment_id,
               "arms": list(arms), "seeds": list(seeds), "splits": list(splits),
               "results": results}
    frozen = freeze_json(summary, root / "engine_results.json")
    return {"ok": True, "summary_sha256": frozen["sha256"],
            "path": frozen["path"]}


def make_coordinator(tmp_path, **overrides):
    kw = dict(run_id="m7_test_run", db=ResearchDatabase(tmp_path / "r.sqlite"),
              out_dir=tmp_path / "discovery", providers=[FakeProvider()],
              agent=FakeAgent(), executor=fake_executor,
              approval=lambda protocol_sha: {
                  "protocol_sha": protocol_sha, "approver": "human-operator",
                  "decision": "APPROVED"},
              pilot_effects={"ep1": 0.12, "ep2": 0.10,
                             "ep3": 0.14, "ep4": 0.11})
    kw.update(overrides)
    return DiscoveryCoordinator(**kw)


def test_engine_free_coordinator_end_to_end(tmp_path):
    c = make_coordinator(tmp_path)
    result = c.run()
    assert result["ok"], result
    assert result["status"] == "COMPLETED"
    assert result["scientific_verdict"] == "REPLICATED"
    assert c.runtime.verify_journal(c.run_id)["ok"]
    assert set(c.state["stages"]) == set(
        __import__("research_runtime.discovery.discovery_coordinator",
                   fromlist=["STAGES"]).STAGES)
    assert c.state["stages"]["HUMAN_APPROVAL"]["status"] == "DONE"
    holdout = c.root / "claims" / "holdout_access.json"
    assert holdout.is_file()
    ledger = json.loads(holdout.read_text(encoding="utf-8"))
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["purpose"] == "final_holdout_evaluation"
    assert len(c.runtime.reservations(c.run_id)) == 2


def test_missing_human_approval_blocks_before_execution(tmp_path):
    c = make_coordinator(tmp_path, approval=None)
    result = c.run()
    assert not result["ok"]
    assert result["status"] == "BLOCKED"
    assert result["stage"] == "HUMAN_APPROVAL"
    assert not (c.root / "experiments" / "selection").exists()


def test_budget_exhaustion_blocks_replication(tmp_path):
    c = make_coordinator(tmp_path, global_budget=1)
    result = c.run()
    assert not result["ok"]
    assert result["stage"] == "REPLICATION"
    assert "budget_exhausted" in result["reason"]
    assert c.runtime.verify_journal(c.run_id)["ok"]


def test_resume_is_idempotent(tmp_path):
    c = make_coordinator(tmp_path)
    first = c.run()
    sha1 = c.state["state_file_sha"]
    c2 = make_coordinator(tmp_path)
    second = c2.run()
    assert second["resumed"]
    assert second["status"] == "COMPLETED"
    assert c2.state["state_file_sha"] == sha1
    assert len(c2.runtime.reservations(c2.run_id)) == 2


def test_missing_pilot_effects_force_inconclusive_verdict(tmp_path):
    c = make_coordinator(tmp_path, pilot_effects=None)
    result = c.run()
    assert result["ok"] and result["status"] == "COMPLETED"
    assert result["scientific_verdict"] == "INCONCLUSIVE"
    assert c.state["power_gate"] == "NOT_ASSESSED"
    reservations = c.runtime.reservations(c.run_id)
    assert all(r["state"] == "consumed" for r in reservations)


class FakeLoadedHarness:
    harness_sha = "b" * 64
    component_shas = {"hypothesis_critic": "c" * 64}
    parent_sha = "d" * 64
    activation_id = "act_m71"


class FakeBindings:
    def __init__(self):
        self.bound = []

    def bind(self, *, campaign_id, domain, loaded):
        self.bound.append((campaign_id, domain, loaded.harness_sha))


class FakeFormalLoader:
    def __init__(self):
        self.bindings = FakeBindings()
        self.calls = []

    def resolve_for_campaign(self, *, campaign_id, domain):
        self.calls.append((campaign_id, domain))
        return FakeLoadedHarness()


def test_formal_loader_binding_and_reservations_are_consumed(tmp_path):
    loader = FakeFormalLoader()
    c = make_coordinator(tmp_path, formal_loader=loader)
    result = c.run()
    assert result["ok"] and result["status"] == "COMPLETED"
    assert c.state["runtime_integration"] == "FORMAL_HARNESS_LOADER"
    assert len(loader.calls) == 2
    assert all(sha == "b" * 64 for _, _, sha in loader.bindings.bound)
    reservations = c.runtime.reservations(c.run_id)
    assert len(reservations) == 2
    assert all(r["state"] == "consumed" and r["outcome"] == "succeeded"
               for r in reservations)
