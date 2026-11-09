"""P1-05 REAL end-to-end: 5 paired seeds x 2 arms x 2 jobs = 20 real jobs.

Runs the genuine SciNet++ CLI/worker (no mocks), then runs the REAL comparator
over the persisted SQLite campaign. Success means the comparison is complete,
traceable and rule-consistent -- NOT that the candidate beat the baseline.
In particular, no decision threshold is tuned to obtain an acceptance.

Enable by leaving the default; skip with ``RESEARCH_SKIP_E2E=1``.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

import yaml

from research_runtime.adapters import SciNetAdapter
from research_runtime.campaign import CampaignRunSpec, CampaignStore
from research_runtime.campaign.scinet import (
    build_scinet_campaign_runner,
    scinet_campaign_policy,
)
from research_runtime.comparison import (
    ComparisonReportStore,
    ComparisonStoreError,
    content_sha256,
)
from research_runtime.comparison.scinet import (
    SCINET_DEFAULT_GUARDRAILS,
    build_scinet_comparator,
)
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.schemas import CandidateExperiment, ExperimentPlan
from tests.rt_tests.campaign_fakes import NoEvidence

SCINET_ROOT = Path(os.getenv("SCINET_DIR", r"E:\LLM\SciNet++"))
MANIFEST = SCINET_ROOT / "datasets" / "pendulum_smoke_v1" / "manifest.json"
CONFIG = "configs/pendulum_smoke.yaml"
PRIMARY_METRIC = "latent_mse@1"
SEEDS = [42, 43, 44, 45, 46]
VALID_DECISIONS = {"accepted_for_confirmation", "rejected", "inconclusive"}


@unittest.skipIf(os.environ.get("RESEARCH_SKIP_E2E") == "1",
                 "RESEARCH_SKIP_E2E=1")
@unittest.skipUnless(MANIFEST.is_file() and
                     (SCINET_ROOT / "research_api" / "cli.py").is_file(),
                     "SciNet++ checkout / frozen smoke dataset unavailable")
class ComparisonIntegrationTests(unittest.TestCase):
    def test_real_five_seed_campaign_then_comparison(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        dataset_version = manifest["dataset_version"]
        campaign_id = "cmp_e2e_" + uuid.uuid4().hex[:8]

        plan = ExperimentPlan(
            plan_id="plan_cmp_e2e", objective_id="obj_cmp_e2e",
            baseline_experiment_id="baseline_cmp_e2e",
            dataset_version=dataset_version,
            primary_metric=PRIMARY_METRIC,
            metric_direction="minimize",
            paired_seeds=list(SEEDS),
            guardrails=SCINET_DEFAULT_GUARDRAILS,
            candidates=[CandidateExperiment(
                candidate_id="cand_lr", hypothesis_id="H_lr",
                config_overrides={"train.lr": 0.0008},
                seeds=list(SEEDS), dataset_version=dataset_version,
            )],
        )
        run_spec = CampaignRunSpec(
            campaign_id=campaign_id,
            objective_id=plan.objective_id, plan_id=plan.plan_id,
            baseline_config_path=CONFIG, dataset_version=dataset_version,
            evaluation_split="validation", job_timeout_s=300,
            max_jobs=30, max_total_seconds=3600,
            approved_hypothesis_ids=["H_lr"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = CampaignStore(Path(tmp) / "campaigns.sqlite")
            adapter = SciNetAdapter()
            manager = ExperimentManager(adapter, evidence_store=NoEvidence())
            policy = scinet_campaign_policy(epochs=1, timeout_per_job_s=300,
                                             max_jobs=30)
            runner = build_scinet_campaign_runner(
                adapter=adapter, manager=manager, store=store, policy=policy)

            record = runner.run(plan, run_spec)

            # ---- execution facts ---------------------------------------
            self.assertEqual(
                record.status, "completed",
                f"trials: {[(t.trial_id, t.status, t.error_message) for t in record.trials]}")
            self.assertEqual(len(record.trials), 10)
            self.assertEqual(record.completed_trials, 10)
            for trial in record.trials:
                self.assertEqual(trial.evaluation_split, "validation")
                self.assertEqual(trial.dataset_version, dataset_version)
                self.assertTrue(trial.train_run_id)
                self.assertTrue(trial.evaluation_job_id)

            # ---- real data identity, not a shared label ------------------
            cfg_root = runner.campaign_root / campaign_id
            cfg_b = yaml.safe_load(
                (cfg_root / "baseline_seed_42" / "config.yaml").read_text(
                    encoding="utf-8"))
            cfg_c = yaml.safe_load(
                (cfg_root / "cand_lr_seed_42" / "config.yaml").read_text(
                    encoding="utf-8"))
            self.assertEqual(cfg_b["data"], cfg_c["data"])
            self.assertEqual(cfg_b["seed"], cfg_c["seed"])
            self.assertNotEqual(cfg_b["train"]["lr"], cfg_c["train"]["lr"])

            # ---- REAL comparison over persisted SQLite state --------------
            report_store = ComparisonReportStore(runner.campaign_root)
            comparator = build_scinet_comparator(
                campaign_store=store, report_store=report_store,
                project_dir=SCINET_ROOT)
            reports = comparator.compare(campaign_id)
            self.assertEqual(len(reports), 1)
            report = reports[0]
            self.assertEqual(comparator.event_errors, [])

            self.assertEqual(report.campaign_id, campaign_id)
            self.assertEqual(report.candidate_id, "cand_lr")
            self.assertEqual(report.n_planned, 5)
            self.assertEqual(report.n_pairs, 5)
            self.assertEqual(report.missing_pairs, [])
            self.assertIn(report.decision, VALID_DECISIONS)
            self.assertEqual(report.metric_direction, "minimize")
            self.assertEqual(report.delta_convention, "positive_is_better")
            self.assertEqual(len(report.paired_observations), 5)
            self.assertEqual(len(report.evidence_refs), 10)
            self.assertTrue(all(o["baseline_train_run_id"]
                                for o in report.paired_observations))
            self.assertTrue(all(o["candidate_train_run_id"]
                                for o in report.paired_observations))
            self.assertNotEqual(report.baseline_config_sha256,
                                report.candidate_config_sha256)

            # Guardrail and frozen-data evidence came from the real engine.
            self.assertTrue(report.guardrail_results)
            self.assertEqual(report.guardrail_results[0]["metric"], "cosine@1")
            self.assertIsNotNone(report.dataset_split_hashes)
            for split in ("train", "validation", "adapt", "test",
                          "manifest.json"):
                self.assertIn(split, report.dataset_split_hashes)
            self.assertEqual(report.dataset_version, dataset_version)
            self.assertEqual(report.evaluation_split, "validation")

            # ---- immutable, hash-traceable report --------------------------
            self.assertEqual(report.report_sha256, content_sha256(report))
            artifact = (report_store.list_reports(campaign_id)[0])
            report_path = (cfg_root / "comparisons" / artifact["file"])
            self.assertTrue(report_path.is_file())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["report_sha256"], report.report_sha256)
            self.assertNotIn("promoted", payload)
            self.assertNotIn("winner", json.dumps(payload).lower())
            loaded = report_store.load(campaign_id, report.comparison_id)
            self.assertEqual(loaded.model_dump(), report.model_dump())
            before = report_path.read_bytes()
            again = comparator.compare(campaign_id)[0]
            self.assertEqual(again.comparison_id, report.comparison_id)
            self.assertEqual(report_path.read_bytes(), before)
            self.assertEqual(len(report_store.list_reports(campaign_id)), 1)

            events = [e["event_type"] for e in store.list_events(campaign_id)]
            self.assertIn("comparison_completed", events)

            # Tampering with the stored report is detected on reload.
            report_path.write_bytes(before + b" ")
            with self.assertRaises(ComparisonStoreError):
                report_store.load(campaign_id, report.comparison_id)
            report_path.write_bytes(before)

            print(f"\n[E2E] campaign={campaign_id} decision={report.decision} "
                  f"rule={report.decision_rule} "
                  f"mean_delta={report.mean_delta:.6f} "
                  f"ci95={report.confidence_interval_95} "
                  f"guardrail={report.guardrail_results[0]['status']}")


if __name__ == "__main__":
    unittest.main()
