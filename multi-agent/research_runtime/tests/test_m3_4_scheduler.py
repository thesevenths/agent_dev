"""M3.4 -- engine-free tests for the UnifiedResearchScheduler.

These tests NEVER touch a real engine: ``run_dual_layer`` is monkeypatched with a
surrogate that returns a realistic dual-layer payload, so we verify the
coordination logic (isolation, aggregation, promotion, idempotent resume,
evidence index) without torch / simulators / the SciNet++ subprocess.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from research_runtime.scheduler import (
    DomainSpec, UnifiedResearchScheduler, _harness_stub,
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

