"""M3.4 -- engine-free tests for the UnifiedResearchScheduler.

These tests NEVER touch a real engine: ``run_dual_layer`` is monkeypatched with a
surrogate that returns a realistic dual-layer payload, so we verify the
coordination logic (isolation, aggregation, promotion, idempotent resume,
evidence index) without torch / simulators / the SciNet++ subprocess.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import yaml

from research_runtime.scheduler import (
    DomainSpec, UnifiedResearchScheduler, _harness_stub,
    _manifest_provenance, _sha256_file, collect_domain_provenance,
)


def _fake_dual_layer(*, domain, domain_base_config, project_dir, dataset_version,
                     domain_seeds, run_root, domain_campaign_id,
                     ood_config_path=None, **_kw):
    """Surrogate for run_dual_layer: returns a realistic domain_layer payload."""
    # Two candidates: one ACCEPTED, one REJECTED. mean_improvement sign follows
    # the metric direction already folded into the value (positive = better).
    if domain == "multi_accept":
        records = [
            {"target": "train.lr", "new": "0.003", "decision": "ACCEPTED",
             "mean_improvement": 0.010, "holdout_direction_fraction": 1.0,
             "ood_decision": "ACCEPTED", "ood_dataset_version": "x-ood"},
            {"target": "train.lr", "new": "0.01", "decision": "ACCEPTED",
             "mean_improvement": 0.030, "holdout_direction_fraction": 1.0,
             "ood_decision": "ACCEPTED", "ood_dataset_version": "x-ood"},
        ]
    elif domain == "reject":
        records = [
            {"target": "train.lr", "new": "0.05", "decision": "REJECTED",
             "mean_improvement": -0.004, "holdout_direction_fraction": 0.2,
             "ood_decision": "REJECTED"},
        ]
    else:
        records = [
            {"target": "train.lr", "new": "0.005", "decision": "ACCEPTED",
             "mean_improvement": 0.012, "holdout_direction_fraction": 1.0,
             "ood_decision": "ACCEPTED", "ood_dataset_version": "x-ood"},
        ]
    return {
        "schema_version": "arr-1.1-dual-layer-v1",
        "domain_layer": {
            "domain": domain,
            "statistics_protocol": {"name": "M3.0-student-t-holm"},
            "records": records,
            "n_accepted": sum(1 for r in records if r["decision"] == "ACCEPTED"),
            "n_rejected": sum(1 for r in records if r["decision"] == "REJECTED"),
            "n_inconclusive": sum(1 for r in records
                                  if r["decision"] == "INCONCLUSIVE"),
            "ood_config_path": ood_config_path,
            "ood_n_accepted": (sum(1 for r in records
                                   if r.get("ood_decision") == "ACCEPTED")
                               if ood_config_path else None),
        },
    }


def _specs(run_id="r1"):
    return [
        DomainSpec(domain="scinet", base_config="/p/scinet.yaml",
                  dataset_version="id-v1", paired_seeds=(50, 51, 52),
                  holdout_seeds=(60, 61), project_dir="/p/scinet",
                  metric_direction="minimize", ood_config_path="/p/ood.yaml"),
        DomainSpec(domain="embodied", base_config="/p/emb.yaml",
                  dataset_version="robot_sim_v1", paired_seeds=(50, 51, 52),
                  project_dir="/p/embodied", metric_direction="minimize"),
        DomainSpec(domain="quant", base_config="/p/q.yaml",
                  dataset_version="q-v1", paired_seeds=(50, 51, 52),
                  project_dir="/p/quant", metric_direction="maximize",
                  synthetic_data=True),
    ]


class TestSchedulerIsolation:
    def test_unique_campaign_id_and_per_domain_run_root(self, tmp_path):
        calls = []
        with mock.patch("research_runtime.scheduler.run_dual_layer",
                        side_effect=lambda **kw: (calls.append(kw) or
                                                  _fake_dual_layer(**kw))):
            sched = UnifiedResearchScheduler(
                run_id="r1", run_root=tmp_path / "run", repo_root=tmp_path,
                domains=_specs(), build_runner=lambda *a, **k: None)
            sched.run()
        # Three domains -> three calls, each isolated.
        assert len(calls) == 3
        by_domain = {c["domain"]: c for c in calls}
        for spec in _specs():
            c = by_domain[spec.domain]
            assert c["run_root"] == str(tmp_path / "run" / spec.domain)
            assert c["domain_campaign_id"] == f"r1_{spec.domain}"
        # scinet carries the OOD config; others do not.
        assert by_domain["scinet"]["ood_config_path"] == "/p/ood.yaml"
        assert by_domain["embodied"]["ood_config_path"] is None


class TestSchedulerAggregation:
    def test_per_domain_summary_populated(self, tmp_path):
        with mock.patch("research_runtime.scheduler.run_dual_layer",
                        side_effect=_fake_dual_layer):
            sched = UnifiedResearchScheduler(
                run_id="r1", run_root=tmp_path / "run", repo_root=tmp_path,
                domains=_specs(), build_runner=lambda *a, **k: None)
            out = sched.run()
        assert out["status"] == "completed"
        assert set(out["per_domain"]) == {"scinet", "embodied", "quant"}
        sc = out["per_domain"]["scinet"]
        assert sc["n_accepted"] == 1 and sc["n_rejected"] == 0
        assert sc["campaign_id"] == "r1_scinet"
        assert Path(sc["registry_path"]).name == "candidate_registry.jsonl"
        assert "scinet" in sc["registry_path"]
        # quant is flagged synthetic in the evidence index.
        assert out["evidence_index"]["domains"]["quant"]["synthetic_data"] is True


class TestPromotionDeterministic:
    def test_picks_largest_improvement_not_execution_order(self):
        recs = _fake_dual_layer(domain="multi_accept",
                                domain_base_config="x", project_dir="x",
                                dataset_version="x", domain_seeds=(1, 2),
                                run_root="x", domain_campaign_id="c")[
            "domain_layer"]["records"]
        spec = DomainSpec(domain="multi_accept", base_config="x",
                         dataset_version="x", paired_seeds=(1, 2),
                         metric_direction="minimize")
        promo = UnifiedResearchScheduler._select_promotion(spec, recs)
        # 0.01 and 0.03 are both better-than-baseline for a minimize metric
        # (sign folded in: improvement magnitude is what we rank). The larger
        # absolute improvement (0.03) wins, regardless of record order.
        assert promo["chosen"]["new"] == "0.01"
        assert promo["candidates_considered"] == 2

    def test_no_accepted_returns_none(self):
        recs = _fake_dual_layer(domain="reject", domain_base_config="x",
                                project_dir="x", dataset_version="x",
                                domain_seeds=(1, 2), run_root="x",
                                domain_campaign_id="c")["domain_layer"]["records"]
        spec = DomainSpec(domain="reject", base_config="x", dataset_version="x",
                         paired_seeds=(1, 2), metric_direction="minimize")
        promo = UnifiedResearchScheduler._select_promotion(spec, recs)
        assert promo["chosen"] is None


class TestSchedulerResume:
    def test_second_run_skips_completed_domains(self, tmp_path):
        calls = []
        with mock.patch("research_runtime.scheduler.run_dual_layer",
                        side_effect=lambda **kw: (calls.append(kw) or
                                                  _fake_dual_layer(**kw))):
            sched1 = UnifiedResearchScheduler(
                run_id="r1", run_root=tmp_path / "run", repo_root=tmp_path,
                domains=_specs(), build_runner=lambda *a, **k: None)
            sched1.run()
            n_after_first = len(calls)
            assert n_after_first == 3
            # Re-run on the SAME run_root: completed domains must be skipped.
            sched2 = UnifiedResearchScheduler(
                run_id="r1", run_root=tmp_path / "run", repo_root=tmp_path,
                domains=_specs(), build_runner=lambda *a, **k: None)
            out2 = sched2.run()
        # No additional engine calls; all three resumed from disk.
        assert len(calls) == n_after_first
        assert set(out2["skipped_resumed"]) == {"scinet", "embodied", "quant"}
        assert out2["completed"] == []


class TestEvidenceIndexAndState:
    def test_files_written_with_expected_keys(self, tmp_path):
        with mock.patch("research_runtime.scheduler.run_dual_layer",
                        side_effect=_fake_dual_layer):
            sched = UnifiedResearchScheduler(
                run_id="r1", run_root=tmp_path / "run", repo_root=tmp_path,
                domains=_specs(), build_runner=lambda *a, **k: None)
            out = sched.run()
        state_path = tmp_path / "run" / "scheduler_state.json"
        ev_path = tmp_path / "run" / "evidence_index.json"
        assert state_path.is_file() and ev_path.is_file()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        assert state["run_id"] == "r1" and state["n_domains"] == 3
        assert state["completed"] == ["scinet", "embodied", "quant"]
        assert ev["run_id"] == "r1"
        assert set(ev["domains"]) == {"scinet", "embodied", "quant"}
        # Quant synthetic flag travels into the evidence index.
        assert ev["domains"]["quant"]["synthetic_data"] is True


class TestHarnessStub:
    def test_stub_harness_returns_no_accepted(self):
        out = _harness_stub(base_harness=None, tasks=[], heldout_task=None)
        assert out["n_accepted"] == 0
        assert out["dominant_failure"] == "DECISION_UNCERTAINTY"


class TestDecisionCasingRegression:
    """Regression: domain_layer records carry LOWER-CASE verdict strings.

    A previous bug compared them against upper-case constants ("ACCEPTED" /
    "REJECTED" / "INCONCLUSIVE"), so every count collapsed to 0 and promotion
    was never selected. The real E2E (arr_m34_001) exposed this. These tests
    lock the casing-normalised behaviour.
    """

    def test_lowercase_decisions_are_counted(self, tmp_path):
        recs = [
            {"target": "train.lr", "new": "0.003", "decision": "accepted",
             "mean_improvement": 0.010, "holdout_direction_fraction": 1.0,
             "ood_decision": "accepted"},
            {"target": "train.lr", "new": "0.01", "decision": "rejected",
             "mean_improvement": -0.004, "holdout_direction_fraction": 0.2,
             "ood_decision": "rejected"},
            {"target": "train.lr", "new": "0.02", "decision": "inconclusive",
             "mean_improvement": 0.001, "holdout_direction_fraction": 1.0,
             "ood_decision": "inconclusive"},
        ]
        spec = DomainSpec(domain="scinet", base_config="x",
                          dataset_version="x", paired_seeds=(1, 2),
                          metric_direction="minimize")
        summary = UnifiedResearchScheduler._summarize_domain(
            spec, {"domain_layer": {"records": recs,
                                    "statistics_protocol": {"name": "M3.0"}}})
        assert summary["n_accepted"] == 1
        assert summary["n_rejected"] == 1
        assert summary["n_inconclusive"] == 1

    def test_lowercase_accepted_is_eligible_for_promotion(self):
        recs = [
            {"target": "train.lr", "new": "0.003", "decision": "accepted",
             "mean_improvement": 0.010, "holdout_direction_fraction": 1.0,
             "ood_decision": "accepted"},
            {"target": "train.lr", "new": "0.01", "decision": "accepted",
             "mean_improvement": 0.030, "holdout_direction_fraction": 1.0,
             "ood_decision": "accepted"},
        ]
        spec = DomainSpec(domain="scinet", base_config="x",
                          dataset_version="x", paired_seeds=(1, 2),
                          metric_direction="minimize")
        promo = UnifiedResearchScheduler._select_promotion(spec, recs)
        assert promo["chosen"]["new"] == "0.01"
        assert promo["candidates_considered"] == 2

    def test_evidence_index_keys_by_fingerprint_not_target(self, tmp_path):
        # Three candidates share target "train.lr"; keying by target would
        # collapse them to one entry. Keying by hyp_fingerprint preserves all.
        recs = [
            {"hyp_fingerprint": "d::train.lr|None->0.003|X",
             "target": "train.lr", "new": "0.003", "decision": "rejected",
             "mean_improvement": -0.002, "holdout_direction_fraction": 0.0},
            {"hyp_fingerprint": "d::train.lr|None->0.005|X",
             "target": "train.lr", "new": "0.005", "decision": "rejected",
             "mean_improvement": -0.001, "holdout_direction_fraction": 0.0},
            {"hyp_fingerprint": "d::train.lr|None->0.01|X",
             "target": "train.lr", "new": "0.01", "decision": "rejected",
             "mean_improvement": -0.0009, "holdout_direction_fraction": 0.0},
        ]
        spec = DomainSpec(domain="embodied", base_config="x",
                          dataset_version="x", paired_seeds=(1, 2),
                          metric_direction="minimize")
        summary = UnifiedResearchScheduler._summarize_domain(
            spec, {"domain_layer": {"records": recs,
                                    "statistics_protocol": {"name": "M3.0"}}})
        state = {"run_id": "r1"}
        ev = UnifiedResearchScheduler._build_evidence_index(
            None, state, {"embodied": summary})
        assert len(ev["domains"]["embodied"]["decisions"]) == 3
        assert ev["domains"]["embodied"]["n_rejected"] == 3


# --------------------------------------------------------------------------- #
# P0-2 -- OOD provenance traceability.
#
# These tests build REAL files in tmp_path (frozen manifest + engine-written
# metric artifacts) and assert the collector reads them. No value is mocked as
# "already known": the expected hashes are computed from those same files.
# --------------------------------------------------------------------------- #

def _make_frozen_data(tmp_path: Path, name: str, version: str):
    """A minimal but structurally real frozen dataset + SHA-pinned manifest."""
    cfg = tmp_path / "configs" / f"{name}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump({
        "experiment": name,
        "data": {"frozen_dir": f"datasets/{name}",
                 "dataset_seed": 20261114, "splits": ["train", "validation"]},
    }), encoding="utf-8")
    ddir = tmp_path / "datasets" / name
    ddir.mkdir(parents=True)
    manifest = {
        "dataset_version": version,
        "config_path": str(cfg),
        "config_sha256": _sha256_file(cfg),
        "schema_version": 1,
        "splits": {
            "train": {"file": "train.csv", "sha256": hashlib.sha256(
                b"train-bytes").hexdigest(), "samples": 10},
            "validation": {"file": "validation.csv", "sha256": hashlib.sha256(
                b"validation-bytes").hexdigest(), "samples": 5},
        },
    }
    mpath = ddir / "manifest.json"
    mpath.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return cfg, mpath


def _make_metric_artifact(path: Path, *, version, checkpoint, run_id,
                          config_path=None, metrics=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "checkpoint_sha256": checkpoint,
        "dataset_version": version,
        "metrics": metrics if metrics is not None else {"latent_mse@1": 0.0123},
        "run_id": run_id,
        "split": "validation",
    }
    if config_path is not None:
        doc["config_path"] = str(config_path)
    path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    return path


class TestOODProvenance:
    def test_complete_chain_from_real_artifacts(self, tmp_path):
        """All five P0-2 facts must be traceable and match the files on disk."""
        id_cfg, id_mpath = _make_frozen_data(tmp_path, "id_set", "id-v1")
        ood_cfg, ood_mpath = _make_frozen_data(tmp_path, "ood_set", "ood-v2")
        project = tmp_path / "project"
        campaign = project / "outputs" / "research_campaigns" / "r1_scinet"
        ckpt = hashlib.sha256(b"ckpt").hexdigest()
        ood_art = _make_metric_artifact(
            campaign / "cand_seed_50" / "ood_metrics.json",
            version="ood-v2", checkpoint=ckpt, run_id="run-ood-001",
            config_path=ood_cfg, metrics={"latent_mse@1": 0.0099})
        _make_metric_artifact(
            campaign / "cand_seed_50" / "backtest_metrics.json",
            version="id-v1", checkpoint=ckpt, run_id="run-ood-001")

        spec = DomainSpec(domain="scinet", base_config=id_cfg,
                          dataset_version="id-v1", paired_seeds=(50,),
                          project_dir=project, ood_config_path=ood_cfg)
        prov = collect_domain_provenance(
            spec=spec, dr_root=tmp_path / "run" / "scinet",
            campaign_id="r1_scinet")

        assert prov["status"] == "complete", prov["missing"]
        assert prov["ood_dataset"]["dataset_version"] == "ood-v2"
        # manifest SHA is the hash of the REAL manifest file, not a claim.
        assert prov["ood_dataset"]["manifest"]["manifest_sha256"] == _sha256_file(ood_mpath)
        assert prov["ood_dataset"]["manifest"]["config_sha256"] == _sha256_file(ood_cfg)
        assert prov["ood_dataset"]["manifest"]["split_count"] == 2
        assert prov["id_dataset"]["manifest"]["manifest_sha256"] == _sha256_file(id_mpath)
        assert prov["checkpoint_sha256"]["ood"] == [ckpt]
        assert prov["evaluation_run_ids"]["ood"] == ["run-ood-001"]
        assert prov["ood_metrics"]["n_with_metrics"] == 1
        assert prov["ood_metrics"]["artifacts"][0]["metrics"] == {"latent_mse@1": 0.0099}
        assert prov["ood_metrics"]["artifacts"][0]["file_sha256"] == _sha256_file(ood_art)
        # ID artifacts must NOT be mislabelled as OOD.
        assert prov["checkpoint_sha256"]["id"] == [ckpt]
        id_arts = [a for a in prov["artifact_files"] if a["axis"] == "id"]
        assert len(id_arts) == 1 and id_arts[0]["path"].endswith("backtest_metrics.json")

    def test_axis_classified_by_content_not_filename(self, tmp_path):
        """An engine that names its OOD file differently is still detected."""
        id_cfg, _ = _make_frozen_data(tmp_path, "id_set", "id-v1")
        ood_cfg, _ = _make_frozen_data(tmp_path, "ood_set", "ood-v2")
        project = tmp_path / "project"
        campaign = project / "outputs" / "research_campaigns" / "r1_scinet"
        _make_metric_artifact(campaign / "c" / "evaluation_result_metrics.json",
                              version="ood-v2", checkpoint="c" * 64,
                              run_id="run-9", config_path=ood_cfg)
        spec = DomainSpec(domain="scinet", base_config=id_cfg,
                          dataset_version="id-v1", paired_seeds=(50,),
                          project_dir=project, ood_config_path=ood_cfg)
        prov = collect_domain_provenance(
            spec=spec, dr_root=tmp_path / "run", campaign_id="r1_scinet")
        assert prov["ood_metrics"]["n_artifacts"] == 1
        assert prov["ood_metrics"]["artifacts"][0]["axis_basis"] == "config_path"

    def test_missing_evidence_is_reported_not_fabricated(self, tmp_path):
        """OOD requested but no artifacts -> incomplete, with explicit missing."""
        id_cfg, _ = _make_frozen_data(tmp_path, "id_set", "id-v1")
        ood_cfg, _ = _make_frozen_data(tmp_path, "ood_set", "ood-v2")
        spec = DomainSpec(domain="scinet", base_config=id_cfg,
                          dataset_version="id-v1", paired_seeds=(50,),
                          project_dir=tmp_path / "project",
                          ood_config_path=ood_cfg)
        prov = collect_domain_provenance(
            spec=spec, dr_root=tmp_path / "run", campaign_id="r1_scinet")
        assert prov["status"] == "incomplete"
        assert "ood_checkpoint_sha256" in prov["missing"]
        assert "ood_evaluation_run_ids" in prov["missing"]
        assert "ood_metrics" in prov["missing"]
        assert prov["checkpoint_sha256"]["ood"] == []
        assert prov["ood_dataset"]["dataset_versions"] == ["ood-v2"]

    def test_no_ood_axis_is_not_applicable(self, tmp_path):
        id_cfg, _ = _make_frozen_data(tmp_path, "id_set", "id-v1")
        spec = DomainSpec(domain="embodied", base_config=id_cfg,
                          dataset_version="id-v1", paired_seeds=(50,),
                          project_dir=tmp_path / "project")
        prov = collect_domain_provenance(
            spec=spec, dr_root=tmp_path / "run", campaign_id="r1_embodied")
        assert prov["status"] == "not_applicable"
        assert prov["missing"] == []
        assert prov["ood_requested"] is False

    def test_evidence_index_exposes_provenance_and_raw_ood_metrics(self, tmp_path):
        """The scheduler must publish the chain, not merely compute it."""
        id_cfg, id_mpath = _make_frozen_data(tmp_path, "id_set", "id-v1")
        ood_cfg, _ = _make_frozen_data(tmp_path, "ood_set", "ood-v2")
        project = tmp_path / "project"
        campaign = project / "outputs" / "research_campaigns" / "r1_scinet"
        ckpt = hashlib.sha256(b"ckpt2").hexdigest()
        _make_metric_artifact(campaign / "c" / "ood_metrics.json",
                              version="ood-v2", checkpoint=ckpt,
                              run_id="run-xyz", config_path=ood_cfg,
                              metrics={"latent_mse@1": 0.0077})
        specs = [DomainSpec(domain="scinet", base_config=id_cfg,
                            dataset_version="id-v1", paired_seeds=(50,),
                            project_dir=project, ood_config_path=ood_cfg)]
        with mock.patch("research_runtime.scheduler.run_dual_layer",
                        side_effect=_fake_dual_layer):
            sched = UnifiedResearchScheduler(
                run_id="r1", run_root=tmp_path / "run", repo_root=tmp_path,
                domains=specs, build_runner=lambda *a, **k: None)
            out = sched.run()
        entry = out["evidence_index"]["domains"]["scinet"]
        assert entry["provenance_status"] == "complete"
        assert entry["provenance_missing"] == []
        assert entry["ood_provenance"]["ood_dataset"]["dataset_version"] == "ood-v2"
        assert entry["ood_provenance"]["ood_dataset"]["manifest"]["manifest_sha256"] \
            == _sha256_file(tmp_path / "datasets" / "ood_set" / "manifest.json")
        assert entry["ood_provenance"]["checkpoint_sha256"]["ood"] == [ckpt]
        assert entry["ood_provenance"]["evaluation_run_ids"]["ood"] == ["run-xyz"]
        assert entry["ood_provenance"]["ood_metrics"]["artifacts"][0]["metrics"] \
            == {"latent_mse@1": 0.0077}
        # Raw gate-produced OOD metrics are surfaced per decision, not invented.
        d = next(iter(entry["decisions"].values()))
        assert "ood_decision" in d and "mean_improvement" in d
        # summary must carry the chain too (used by resume / re-aggregation).
        assert out["per_domain"]["scinet"]["provenance"]["status"] == "complete"
        assert entry["ood_dataset_version"] == "ood-v2"

    def test_legacy_summary_without_provenance_is_explicit(self):
        """Old evidence still validates; absence is labelled, never faked."""
        summary = {"domain": "scinet", "status": "completed",
                   "dataset_version": "id-v1", "records": []}
        ev = UnifiedResearchScheduler._build_evidence_index(
            None, {"run_id": "old"}, {"scinet": summary})
        entry = ev["domains"]["scinet"]
        assert entry["provenance_status"] == "not_collected"
        assert entry["ood_provenance"] is None

