"""P1-04: crash recovery. The three risks that matter most.

1. Recovery must RESUME the existing job, not train a second time.
2. Validation must reuse the ORIGINAL train run_id / checkpoint.
3. Data identity must be verified on disk (frozen manifest + split hashes),
   including when a config directory is reused after a restart.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from research_runtime.campaign.scheduler import make_job_id
from research_runtime.schemas import ExperimentSpec
from tests.rt_tests.campaign_fakes import CampaignTestBase, DATASET_VERSION


class CampaignResumeTests(CampaignTestBase):
    def _restart(self, adapter, *, policy=None):
        """Brand-new manager + runner: NO in-memory state may survive."""
        return self.runner(adapter=adapter, policy=policy,
                           store=self.store())[0]

    # ------------------------------------------------------- core recovery
    def test_resume_does_not_submit_duplicate_train_job(self):
        adapter = self.adapter(wait_status="timeout")
        runner, _, store = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        paused = runner.run(plan, spec)

        self.assertEqual(paused.status, "paused")
        first_submits = list(adapter.submits)
        self.assertEqual(len(first_submits), 1)
        self.assertIn("campaign_paused",
                      [e["event_type"] for e in store.list_events(spec.campaign_id)])

        adapter.opts["wait_status"] = None
        restarted = self._restart(adapter)
        record = restarted.resume(spec.campaign_id)
        self.assertEqual(record.status, "completed")
        self.assertEqual(adapter.submits[0], first_submits[0])
        self.assertEqual(len(adapter.submits), len(set(adapter.submits)))
        self.assertEqual(len(adapter.submits), 4)   # exactly 2 trials x 2 jobs

    def test_completed_jobs_are_never_reexecuted(self):
        runner, adapter, _ = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan)
        runner.run(plan, spec)
        before = list(adapter.submits)
        record = self._restart(adapter).resume(spec.campaign_id)
        self.assertEqual(record.status, "completed")
        self.assertEqual(adapter.submits, before)
        self.assertEqual(record.completed_trials, 2)

    def test_restart_collects_old_jobs_with_persisted_specs(self):
        """ExperimentManager._specs is gone; collect_result must get a spec."""
        adapter = self.adapter(wait_status_seq=["succeeded", "timeout"])
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        paused = runner.run(plan, spec)
        self.assertEqual(paused.status, "paused")
        baseline = paused.trial("baseline_seed_42")
        self.assertEqual(baseline.status, "evaluating")
        self.assertIsNotNone(baseline.train_result_path or baseline.train_run_id)

        adapter.opts["wait_status_seq"] = []
        adapter.collect_calls.clear()
        restarted = self._restart(adapter)
        record = restarted.resume(spec.campaign_id)
        self.assertEqual(record.status, "completed")
        # The restarted process re-collected the OLD train job with the spec
        # that was persisted BEFORE the crash -- never ``_specs`` lookup.
        collected = dict(adapter.collect_calls)
        self.assertIn(baseline.train_job_id, collected)
        self.assertIsInstance(collected[baseline.train_job_id], ExperimentSpec)
        self.assertEqual(collected[baseline.train_job_id].budget["job_id"],
                         baseline.train_job_id)

    def test_validation_after_restart_reuses_the_original_run_id(self):
        adapter = self.adapter(wait_status_seq=["succeeded", "timeout"])
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        paused = runner.run(plan, spec)
        original_run_id = paused.trial("baseline_seed_42").train_run_id

        adapter.opts["wait_status_seq"] = []
        record = self._restart(adapter).resume(spec.campaign_id)
        baseline = record.trial("baseline_seed_42")
        eval_spec = ExperimentSpec.model_validate(baseline.evaluation_spec)
        self.assertEqual(baseline.train_run_id, original_run_id)
        self.assertEqual(eval_spec.run_id, original_run_id)
        self.assertEqual(eval_spec.split, "validation")

    def test_job_ids_are_identical_before_and_after_restart(self):
        adapter = self.adapter(wait_status="timeout")
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        paused = runner.run(plan, spec)
        before = {t.trial_id: (t.train_job_id, t.evaluation_job_id)
                  for t in paused.trials}

        adapter.opts["wait_status"] = None
        resumed = self._restart(adapter).resume(spec.campaign_id)
        for trial in resumed.trials:
            old_train, old_eval = before[trial.trial_id]
            # A trial that had not started before the crash has no persisted
            # job id yet; after resume it gets exactly the deterministic one.
            if old_train is not None:
                self.assertEqual(trial.train_job_id, old_train)
            if old_eval is not None:
                self.assertEqual(trial.evaluation_job_id, old_eval)
            self.assertEqual(
                trial.train_job_id,
                make_job_id(spec.campaign_id, trial.trial_id, "train"))
            self.assertEqual(
                trial.evaluation_job_id,
                make_job_id(spec.campaign_id, trial.trial_id, "evaluate"))

    # ------------------------------------------------------ config integrity
    def test_resume_reuses_existing_configs_without_overwriting(self):
        adapter = self.adapter(wait_status="timeout")
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        paused = runner.run(plan, spec)
        hashes = {t.trial_id: t.config_sha256 for t in paused.trials}
        config_files = {p: p.read_bytes()
                        for p in self.campaign_dir().glob("*/config.yaml")}

        adapter.opts["wait_status"] = None
        record = self._restart(adapter).resume(spec.campaign_id)
        self.assertEqual(record.status, "completed")
        self.assertEqual({t.trial_id: t.config_sha256 for t in record.trials},
                         hashes)
        for path, content in config_files.items():
            self.assertEqual(path.read_bytes(), content)

    def test_resume_rejects_tampered_config(self):
        adapter = self.adapter(wait_status="timeout")
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        runner.run(plan, spec)
        target = self.campaign_dir() / "baseline_seed_42" / "config.yaml"
        target.write_text(target.read_text(encoding="utf-8")
                          + "\n# tampered after crash\n", encoding="utf-8")
        before = list(adapter.submits)

        record = self._restart(adapter).resume(spec.campaign_id)
        self.assertEqual(record.status, "failed")
        self.assertEqual(adapter.submits, before)

    def test_resume_is_idempotent_for_terminal_campaigns(self):
        runner, adapter, _ = self.runner()
        plan = self.plan()
        spec = self.run_spec(plan)
        runner.run(plan, spec)
        before = list(adapter.submits)
        restarted = self._restart(adapter)
        self.assertEqual(restarted.resume(spec.campaign_id).status, "completed")
        self.assertEqual(restarted.resume(spec.campaign_id).status, "completed")
        self.assertEqual(adapter.submits, before)

    def test_run_with_resume_flag_delegates_to_existing_campaign(self):
        adapter = self.adapter(wait_status="timeout")
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        runner.run(plan, spec)
        adapter.opts["wait_status"] = None
        spec2 = self.run_spec(plan, resume=True)
        record = self._restart(adapter).run(plan, spec2)
        self.assertEqual(record.status, "completed")
        self.assertEqual(len(adapter.submits), 4)

    def test_resume_rejects_tampered_frozen_dataset(self):
        """Data identity is re-verified on disk after a restart, not just
        compared as a dataset_version string in JSON."""
        adapter = self.adapter(wait_status="timeout")
        runner, _, _ = self.runner(adapter=adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        runner.run(plan, spec)
        before = list(adapter.submits)
        (self.frozen / "validation.npz").write_bytes(b"changed after crash")

        record = self._restart(adapter).resume(spec.campaign_id)
        self.assertEqual(record.status, "failed")
        self.assertEqual(adapter.submits, before)


if __name__ == "__main__":
    unittest.main()
