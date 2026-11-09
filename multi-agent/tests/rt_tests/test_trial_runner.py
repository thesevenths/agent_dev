"""P1-04: one trial's train -> validation contract, including integrity gates."""
from __future__ import annotations

import unittest

from research_runtime.campaign import CampaignRecord, CampaignRunSpec, TrialRecord
from research_runtime.campaign.exceptions import RecoverableCampaignError
from research_runtime.campaign.trial_runner import TrialRunner
from research_runtime.schemas import ExperimentSpec
from tests.rt_tests.campaign_fakes import (
    BASE_YAML,
    CampaignTestBase,
    DATASET_VERSION,
    PRIMARY_METRIC,
)


class TrialRunnerTests(CampaignTestBase):
    def _seed(self, *, policy=None, **adapter_opts):
        adapter = self.adapter(**adapter_opts)
        store = self.store()
        policy = policy or self.policy()
        manager = self.manager(adapter)
        plan = self.plan()
        spec = self.run_spec(plan)
        artifact = self.builder().build_baseline(
            base_config="configs/base.yaml",
            campaign_id=spec.campaign_id,
            seed=42,
            dataset_version=DATASET_VERSION,
        )
        record = CampaignRecord(
            campaign_id=spec.campaign_id,
            plan_id=plan.plan_id,
            objective_id=plan.objective_id,
            status="running",
            dataset_version=DATASET_VERSION,
            evaluation_split="validation",
            plan=plan.model_dump(mode="json"),
            run_spec=spec.model_dump(mode="json"),
            policy_snapshot=policy.snapshot(),
        )
        store.create_campaign(record)
        trial = TrialRecord(
            trial_id="baseline_seed_42", campaign_id=spec.campaign_id,
            arm_type="baseline", arm_id="baseline", seed=42,
            config_path=str(artifact.config_path),
            config_relpath=artifact.config_relpath,
            config_sha256=artifact.candidate_config_sha256,
            baseline_config_sha256=artifact.baseline_config_sha256,
            dataset_version=DATASET_VERSION,
            evaluation_split="validation",
            primary_metric=PRIMARY_METRIC,
            status="config_ready",
        )
        store.save_trial(record, trial, event_type="config_ready")
        runner = TrialRunner(manager, policy, store)
        return runner, record, trial, spec, adapter, store

    # ------------------------------------------------------------- happy path
    def test_train_and_validation_share_the_same_run_id(self):
        runner, record, trial, spec, adapter, _ = self._seed()
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "succeeded")

        train_spec = ExperimentSpec.model_validate(trial.train_spec)
        eval_spec = ExperimentSpec.model_validate(trial.evaluation_spec)
        self.assertEqual(eval_spec.run_id, trial.train_run_id)
        self.assertEqual(eval_spec.parent_run_id, trial.train_run_id)
        self.assertEqual(eval_spec.parent_job_id, trial.train_job_id)
        self.assertEqual(eval_spec.stage, "evaluate")
        self.assertEqual(eval_spec.split, "validation")
        self.assertEqual(eval_spec.config_path, train_spec.config_path)
        self.assertNotEqual(trial.train_job_id, trial.evaluation_job_id)
        self.assertEqual(trial.validation_metrics[PRIMARY_METRIC], 0.02)
        self.assertEqual(result.evaluation_split, "validation")
        self.assertEqual(len(adapter.submits), 2)

    def test_collect_result_always_receives_the_persisted_spec(self):
        """No dependence on ExperimentManager._specs (process-local memory)."""
        runner, record, trial, spec, adapter, _ = self._seed()
        runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(len(adapter.collect_calls), 2)
        for job_id, passed_spec in adapter.collect_calls:
            self.assertIsInstance(passed_spec, ExperimentSpec)
            self.assertEqual(passed_spec.budget["job_id"], job_id)

    def test_terminal_trial_is_never_reexecuted(self):
        runner, record, trial, spec, adapter, _ = self._seed()
        trial.status = "succeeded"
        runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(adapter.submits, [])

    # ------------------------------------------------------------ integrity
    def test_engine_reporting_test_split_is_rejected(self):
        runner, record, trial, spec, adapter, _ = self._seed(actual_split="test")
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "integrity_error")
        self.assertIn("test", result.error_message or "")
        self.assertEqual(len(adapter.submits), 2)   # train ran; eval was refused

    def test_dataset_version_mismatch_fails_before_evaluation(self):
        runner, record, trial, spec, adapter, _ = self._seed(
            dataset_version="ds-v2")
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "integrity_error")
        self.assertIn("dataset_version mismatch", result.error_message or "")
        self.assertEqual(len(adapter.submits), 1)

    def test_missing_checkpoint_blocks_evaluation(self):
        runner, record, trial, spec, adapter, _ = self._seed(
            missing_checkpoint=True)
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(adapter.submits), 1)
        self.assertIn("checkpoint", result.error_message or "")

    def test_missing_run_id_blocks_evaluation(self):
        runner, record, trial, spec, adapter, _ = self._seed(
            missing_run_id=True)
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(adapter.submits), 1)

    def test_wrong_evaluation_run_id_is_rejected(self):
        runner, record, trial, spec, _, _ = self._seed(actual_run_id="run_other")
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "integrity_error")
        self.assertIn("checkpoint reuse is unproven", result.error_message or "")

    def test_missing_primary_metric_is_rejected(self):
        runner, record, trial, spec, _, _ = self._seed(metrics={})
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertIn("required metric", result.error_message or "")

    def test_non_finite_metric_is_rejected(self):
        runner, record, trial, spec, _, _ = self._seed(
            metrics={PRIMARY_METRIC: float("nan")})
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")

    def test_required_artifact_gate(self):
        policy = self.policy(required_artifacts=("gold_standard.npy",))
        runner, record, trial, spec, _, _ = self._seed(policy=policy)
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertIn("gold_standard.npy", result.error_message or "")

    def test_train_failure_is_a_complete_record(self):
        runner, record, trial, spec, adapter, _ = self._seed(
            train_status="failed")
        result = runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "train_failed")
        self.assertTrue(result.error_message)
        self.assertEqual(len(adapter.submits), 1)

    # ---------------------------------------------------------- recoverable
    def test_wait_timeout_pauses_without_failing_or_resubmitting(self):
        runner, record, trial, spec, adapter, _ = self._seed(
            wait_status="timeout")
        with self.assertRaises(RecoverableCampaignError):
            runner.execute(trial, record=record, run_spec=spec)
        self.assertNotEqual(trial.status, "failed")
        self.assertEqual(len(adapter.submits), 1)   # no evaluation, no retry
        self.assertIsNone(trial.evaluation_job_id)

    def test_undecidable_job_lookup_pauses_before_any_submit(self):
        runner, record, trial, spec, adapter, _ = self._seed(probe_error=True)
        with self.assertRaises(RecoverableCampaignError):
            runner.execute(trial, record=record, run_spec=spec)
        self.assertEqual(adapter.submits, [])


if __name__ == "__main__":
    unittest.main()
