"""P1-04 REAL end-to-end: one baseline + one candidate + one paired seed.

Runs the genuine SciNet++ CLI/worker (no mocks): 2 trials x 2 jobs = 4 jobs.
Success means the campaign executed and recorded trustworthy facts -- NOT
that the candidate beat the baseline. That comparison is P1-05's job.

Enable by leaving the default; skip with ``RESEARCH_SKIP_E2E=1``.
"""
from __future__ import annotations

import json
import math
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
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.schemas import CandidateExperiment, ExperimentPlan, ExperimentSpec
from tests.rt_tests.campaign_fakes import NoEvidence

SCINET_ROOT = Path(os.getenv("SCINET_DIR", r"E:\LLM\SciNet++"))
MANIFEST = SCINET_ROOT / "datasets" / "pendulum_smoke_v1" / "manifest.json"
CONFIG = "configs/pendulum_smoke.yaml"
PRIMARY_METRIC = "latent_mse@1"


@unittest.skipIf(os.environ.get("RESEARCH_SKIP_E2E") == "1",
                 "RESEARCH_SKIP_E2E=1")
@unittest.skipUnless(MANIFEST.is_file() and
                     (SCINET_ROOT / "research_api" / "cli.py").is_file(),
                     "SciNet++ checkout / frozen smoke dataset unavailable")
class CampaignIntegrationTests(unittest.TestCase):
    def test_real_campaign_train_validation_and_manifest(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        dataset_version = manifest["dataset_version"]
        campaign_id = "campaign_e2e_" + uuid.uuid4().hex[:8]

        plan = ExperimentPlan(
            plan_id="plan_e2e", objective_id="obj_e2e",
            baseline_experiment_id="baseline_e2e",
            dataset_version=dataset_version,
            primary_metric=PRIMARY_METRIC,
            paired_seeds=[42],
            candidates=[CandidateExperiment(
                candidate_id="cand_lr",
                hypothesis_id="H_lr",
                config_overrides={"train.lr": 0.0008},
                seeds=[42],
                dataset_version=dataset_version,
            )],
        )
        spec = CampaignRunSpec(
            campaign_id=campaign_id,
            objective_id=plan.objective_id,
            plan_id=plan.plan_id,
            baseline_config_path=CONFIG,
            dataset_version=dataset_version,
            evaluation_split="validation",
            job_timeout_s=300,
            max_jobs=10,
            max_total_seconds=1800,
            approved_hypothesis_ids=["H_lr"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = CampaignStore(Path(tmp) / "campaigns.sqlite")
            adapter = SciNetAdapter()
            manager = ExperimentManager(adapter, evidence_store=NoEvidence())
            policy = scinet_campaign_policy(
                epochs=1, timeout_per_job_s=300, max_jobs=10)
            runner = build_scinet_campaign_runner(
                adapter=adapter, manager=manager, store=store, policy=policy)

            record = runner.run(plan, spec)

            # ---- execution facts ---------------------------------------
            self.assertEqual(record.status, "completed",
                             f"trials: {[(t.trial_id, t.status, t.error_message) for t in record.trials]}")
            self.assertEqual(len(record.trials), 2)
            self.assertEqual(record.completed_trials, 2)
            baseline = record.trial("baseline_seed_42")
            candidate = record.trial("cand_lr_seed_42")
            self.assertIsNotNone(baseline)
            self.assertIsNotNone(candidate)
            self.assertNotEqual(baseline.config_sha256, candidate.config_sha256)

            for trial in (baseline, candidate):
                self.assertEqual(trial.status, "succeeded")
                self.assertEqual(trial.evaluation_split, "validation")
                self.assertEqual(trial.dataset_version, dataset_version)
                self.assertTrue(trial.train_job_id)
                self.assertTrue(trial.evaluation_job_id)
                self.assertTrue(trial.train_run_id)
                self.assertNotEqual(trial.train_job_id, trial.evaluation_job_id)
                eval_spec = ExperimentSpec.model_validate(trial.evaluation_spec)
                self.assertEqual(eval_spec.run_id, trial.train_run_id,
                                 "validation MUST reuse the trained checkpoint")
                self.assertEqual(eval_spec.split, "validation")
                value = trial.validation_metrics.get(PRIMARY_METRIC)
                self.assertIsNotNone(value)
                self.assertTrue(math.isfinite(float(value)))
                # Each job owns its own result file; no stage overwrites another.
                self.assertNotEqual(trial.train_result_path,
                                    trial.evaluation_result_path)
                self.assertTrue(Path(trial.train_result_path).is_file())
                self.assertTrue(Path(trial.evaluation_result_path).is_file())

            # ---- data identity is REAL, not a shared label ---------------
            cfg_b = yaml.safe_load((runner.campaign_root / campaign_id
                                    / "baseline_seed_42" / "config.yaml")
                                   .read_text(encoding="utf-8"))
            cfg_c = yaml.safe_load((runner.campaign_root / campaign_id
                                    / "cand_lr_seed_42" / "config.yaml")
                                   .read_text(encoding="utf-8"))
            self.assertEqual(cfg_b["data"], cfg_c["data"])
            self.assertEqual(cfg_b["seed"], cfg_c["seed"])
            self.assertNotEqual(cfg_b["train"]["lr"], cfg_c["train"]["lr"])
            self.assertEqual(cfg_b["data"]["frozen_dir"],
                             "datasets/pendulum_smoke_v1")

            # ---- manifest reloadable, facts only --------------------------
            result_path = Path(record.final_manifest_path)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["summary"]["total_trials"], 2)
            self.assertEqual(payload["summary"]["succeeded_trials"], 2)
            self.assertEqual(set(payload["arms"]), {"baseline", "cand_lr"})
            self.assertEqual(payload["evaluation_split"], "validation")
            self.assertNotIn("winner", json.dumps(payload).lower())
            events = (result_path.parent / "campaign_events.jsonl")
            self.assertTrue(events.is_file())
            self.assertGreater(len(events.read_text(encoding="utf-8").splitlines()), 5)
            self.assertEqual(runner.get_status(campaign_id)["status"],
                             "completed")


if __name__ == "__main__":
    unittest.main()
