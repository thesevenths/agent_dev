"""P1-04: campaign lifecycle, preflight, pairing, isolation and manifest."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from research_runtime.campaign import CampaignStore
from research_runtime.campaign.exceptions import (
    CampaignPreflightError,
    CampaignStoreError,
)
from research_runtime.schemas import ExperimentSpec
from tests.rt_tests.campaign_fakes import (
    CampaignTestBase,
    DATASET_VERSION,
    PRIMARY_METRIC,
)


class FailingStore(CampaignStore):
    """Simulate a broken DB: no state write may be followed by a submit."""

    def save_trial(self, record, trial, *, event_type, payload=None):
        raise CampaignStoreError("simulated sqlite write failure")


class CampaignRunnerTests(CampaignTestBase):
    # ------------------------------------------------------------ happy path
    def test_full_campaign_executes_every_paired_trial(self):
        runner, adapter, store = self.runner()
        plan = self.plan(seeds=(42, 43))
        spec = self.run_spec(plan)
        record = runner.run(plan, spec)

        self.assertEqual(record.status, "completed")
        self.assertEqual(len(record.trials), 4)
        self.assertTrue(all(t.status == "succeeded" for t in record.trials))
        self.assertEqual(len(adapter.submits), 8)      # 4 trials x 2 jobs
        self.assertEqual(len(set(adapter.submits)), 8)  # deterministic + unique
        self.assertEqual(record.completed_trials, 4)
        self.assertEqual(record.failed_trials, 0)
        self.assertTrue(Path(record.final_manifest_path).is_file())
        for name in ("campaign.json", "campaign_events.jsonl",
                     "campaign_result.json"):
            self.assertTrue((self.campaign_dir() / name).is_file())

    def test_manifest_contains_facts_not_decisions(self):
        runner, _, _ = self.runner()
        plan = self.plan(seeds=(42, 43))
        record = runner.run(plan, self.run_spec(plan))
        payload = json.loads(Path(record.final_manifest_path).read_text(
            encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["summary"], {
            "total_trials": 4, "succeeded_trials": 4,
            "failed_trials": 0, "skipped_trials": 0})
        self.assertEqual(set(payload["arms"]), {"baseline", "cand_0"})
        self.assertEqual(set(payload["arms"]["baseline"]),
                         {"seed_42", "seed_43"})
        text = json.dumps(payload).lower()
        for forbidden in ("winner", "mean_delta", "paired_deltas",
                          "accepted_for_confirmation", "promoted"):
            self.assertNotIn(forbidden, text)
        self.assertIn("p1-05", payload["scope_note"].lower())

    def test_every_trial_selects_on_validation_with_paired_data(self):
        runner, _, _ = self.runner()
        plan = self.plan(seeds=(42, 43))
        record = runner.run(plan, self.run_spec(plan))
        for trial in record.trials:
            self.assertEqual(trial.evaluation_split, "validation")
            self.assertEqual(trial.dataset_version, DATASET_VERSION)
            eval_spec = ExperimentSpec.model_validate(trial.evaluation_spec)
            self.assertEqual(eval_spec.split, "validation")
            self.assertEqual(eval_spec.run_id, trial.train_run_id)
            self.assertEqual(eval_spec.dataset_version, DATASET_VERSION)
            self.assertEqual(trial.validation_metrics[PRIMARY_METRIC], 0.02)

    def test_configs_are_isolated_and_share_one_dataset(self):
        runner, _, _ = self.runner()
        plan = self.plan(seeds=(42, 43))
        record = runner.run(plan, self.run_spec(plan))
        configs = {}
        for trial in record.trials:
            path = self.campaign_dir() / trial.trial_id / "config.yaml"
            self.assertTrue(path.is_file())
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            configs[trial.trial_id] = cfg
            self.assertEqual(cfg["seed"], trial.seed)
            self.assertEqual(cfg["data"]["dataset_seed"], 7)
        baseline_data = configs["baseline_seed_42"]["data"]
        for cfg in configs.values():
            self.assertEqual(cfg["data"], baseline_data)
        self.assertEqual(configs["baseline_seed_42"]["model"]["latent_dim"], 4)
        self.assertEqual(configs["cand_0_seed_42"]["model"]["latent_dim"], 8)
        self.assertNotEqual(record.trial("baseline_seed_42").config_sha256,
                            record.trial("cand_0_seed_42").config_sha256)
        self.assertEqual(len({t.config_path for t in record.trials}), 4)

    # -------------------------------------------------------------- pairing
    def test_baseline_failure_skips_its_candidates_but_next_seed_continues(self):
        adapter = self.adapter(fail_experiments=("baseline_seed_42",))
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan(seeds=(42, 43))
        record = runner.run(plan, self.run_spec(plan))

        self.assertEqual(record.status, "completed_with_failures")
        self.assertEqual(record.trial("baseline_seed_42").status, "failed")
        skipped = record.trial("cand_0_seed_42")
        self.assertEqual(skipped.status, "skipped")
        self.assertIn("baseline_seed_42", skipped.skip_reason or "")
        self.assertEqual(record.trial("baseline_seed_43").status, "succeeded")
        self.assertEqual(record.trial("cand_0_seed_43").status, "succeeded")
        # Failed baseline spawned ONLY its train job; the unpaired candidate
        # spawned none; the other seed ran to completion (2 + 2 jobs).
        self.assertEqual(len(adapter.submits), 5)

    # ------------------------------------------------------------ preflight
    def test_preflight_rejects_unapproved_hypothesis(self):
        runner, adapter, _ = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan, approved_hypothesis_ids=[])
        with self.assertRaises(CampaignPreflightError):
            runner.run(plan, spec)
        self.assertEqual(adapter.submits, [])

    def test_preflight_rejects_reserved_test_split(self):
        runner, adapter, _ = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan, evaluation_split="test")
        with self.assertRaises(CampaignPreflightError):
            runner.run(plan, spec)
        self.assertEqual(adapter.submits, [])

    def test_preflight_rejects_job_budget_overflow(self):
        runner, adapter, _ = self.runner()
        plan = self.plan(seeds=(42, 43), n_candidates=2)
        spec = self.run_spec(plan, max_jobs=1)
        with self.assertRaises(CampaignPreflightError):
            runner.run(plan, spec)
        self.assertEqual(adapter.submits, [])

    def test_preflight_rejects_parallel_execution(self):
        runner, adapter, _ = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan, max_parallel=2)
        with self.assertRaises(CampaignPreflightError):
            runner.run(plan, spec)
        self.assertEqual(adapter.submits, [])

    def test_preflight_rejects_tampered_frozen_split(self):
        (self.frozen / "train.npz").write_bytes(b"tampered", )
        runner, adapter, _ = self.runner()
        plan = self.plan()
        with self.assertRaisesRegex(CampaignPreflightError, "frozen dataset|SHA"):
            runner.run(plan, self.run_spec(plan))
        self.assertEqual(adapter.submits, [])

    def test_preflight_rejects_unhealthy_adapter(self):
        runner, adapter, _ = self.runner(self.adapter(unhealthy=True))
        plan = self.plan()
        with self.assertRaises(CampaignPreflightError):
            runner.run(plan, self.run_spec(plan))
        self.assertEqual(adapter.submits, [])

    # ----------------------------------------------------------- idempotence
    def test_duplicate_run_is_idempotent(self):
        runner, adapter, store = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan)
        first = runner.run(plan, spec)
        before = list(adapter.submits)
        second = runner.run(plan, spec)
        self.assertEqual(second.status, first.status)
        self.assertEqual(adapter.submits, before)
        self.assertIn("duplicate_run_ignored",
                      [e["event_type"] for e in store.list_events(spec.campaign_id)])

    def test_cancel_stops_further_submissions(self):
        adapter = self.adapter(wait_status="timeout")
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        paused = runner.run(plan, spec)
        self.assertEqual(paused.status, "paused")
        cancelled = runner.cancel(spec.campaign_id)
        self.assertEqual(cancelled.status, "cancelled")
        before = list(adapter.submits)
        adapter.opts["wait_status"] = None
        resumed = runner.resume(spec.campaign_id)
        self.assertEqual(resumed.status, "cancelled")
        self.assertEqual(adapter.submits, before)

    def test_get_status_is_read_only(self):
        runner, adapter, _ = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan)
        runner.run(plan, spec)
        before = list(adapter.submits)
        status = runner.get_status(spec.campaign_id)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["summary"]["succeeded_trials"], 2)
        self.assertEqual(adapter.submits, before)

    def test_store_write_failure_submits_no_job(self):
        adapter = self.adapter()
        store = FailingStore(self.db_path)
        policy = self.policy()
        from research_runtime.campaign import CampaignRunner
        runner = CampaignRunner(self.manager(adapter), self.builder(),
                                store, policy)
        plan = self.plan()
        with self.assertRaises(CampaignStoreError):
            runner.run(plan, self.run_spec(plan))
        self.assertEqual(adapter.submits, [])

    def test_event_log_covers_every_transition(self):
        runner, _, store = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan)
        runner.run(plan, spec)
        events = [e["event_type"] for e in store.list_events(spec.campaign_id)]
        for required in ("campaign_created", "config_ready",
                         "train_spec_persisted", "train_submitted",
                         "train_succeeded", "evaluation_spec_persisted",
                         "evaluation_submitted", "trial_succeeded",
                         "campaign_finalized"):
            self.assertIn(required, events)
        self.assertEqual(events[0], "campaign_created")
        self.assertEqual(events[-1], "campaign_finalized")

    # ---------------------------------------------------------- generic-ness
    def test_production_modules_have_no_engine_specific_keys(self):
        """Engine field names belong ONLY to campaign/scinet.py (domain wiring)."""
        root = Path(__file__).resolve().parents[2] / "research_runtime" / "campaign"
        forbidden = ("predictor_hidden", "latent_dim", "ema_momentum",
                     "pendulum", "latent_mse", "cosine@1", "pytorch")
        for path in sorted(root.glob("*.py")):
            if path.name == "scinet.py":
                continue
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text,
                                 f"{path.name} hard-codes {token!r}")


if __name__ == "__main__":
    unittest.main()
