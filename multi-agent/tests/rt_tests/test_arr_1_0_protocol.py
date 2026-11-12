"""ARR 1.0 / Package D+E regression tests (CI-visible, REAL local engines).

These tests run the actual numerical simulators/backtester -- no FakeRunner,
no mocked metrics. The frozen datasets are committed with SHA-256 manifests, so
the same checks run on Linux / macOS / Windows in CI.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_runtime.arr_runtime import (
    BASE_HARNESS, IMPROVED_HARNESS, NEGATIVE_HARNESS,
    ResearchHarness, make_harness_generate, run_research_task,
)
from research_runtime.research_quality import (
    QUALITY_FROZEN_FIELDS, QualityEvidence, QualityWeights,
    embodied_quality_evaluator, evaluate_research_quality,
    quant_quality_evaluator, scinet_quality_evaluator,
)
from research_runtime.research_task import ResearchTask
from adaptive_runtime.evolution_spaces import (
    DOMAIN_SEARCH_SPACE, HARNESS_SEARCH_SPACE, CandidateType,
    candidate_type_of, is_frozen, namespaced_fingerprint, space_for,
)


def _evidenced_report(decision: str) -> dict:
    return {
        "objective_id": "obj_t", "outcome": decision,
        "report_sha256": "b" * 64, "promotion": "not_requested",
        "frozen_test_used": False,
        "evidence_refs": [f"experiment:exp_{i}:result:j{i}" for i in range(2)],
        "experience_refs": ["exp_x"],
        "hypotheses": [
            {"hypothesis_id": "h1", "content_sha256": "a" * 64, "statement": "s1"},
            {"hypothesis_id": "h2", "content_sha256": "c" * 64, "statement": "s2"},
        ],
        "reviews": [
            {"review_id": "r1", "hypothesis_sha256": "a" * 64, "passed": True},
            {"review_id": "r2", "hypothesis_sha256": "c" * 64, "passed": True},
        ],
        "plans": [{"plan_id": "p", "plan_sha256": "d" * 64, "version": 1}],
        "approvals": [{"approval_id": "a", "resource_sha256": "e" * 64,
                       "source": "human", "status": "granted"}],
        "campaigns": [{"campaign_id": "c", "status": "completed"}],
        "comparisons": [
            {"candidate_id": "cand_A", "decision": decision,
             "decision_rule": "r", "n_pairs": 5, "mean_delta": -0.02,
             "guardrail_violations": []},
            {"candidate_id": "cand_B", "decision": decision,
             "decision_rule": "r", "n_pairs": 5, "mean_delta": -0.01,
             "guardrail_violations": []},
        ],
    }


def _evidence() -> QualityEvidence:
    comp = {
        "baseline_config_sha256": "1" * 64, "candidate_config_sha256": "2" * 64,
        "dataset_version": "frozen-v1", "evaluation_split": "validation",
        "n_pairs": 5, "n_planned": 5, "missing_pairs": [],
        "dataset_split_hashes": {"train": "3" * 64},
        "guardrail_violations": [],
        "paired_observations": [{"seed": s, "delta": -0.02} for s in range(5)],
    }
    return QualityEvidence(
        report=_evidenced_report("rejected"), comparison_reports=[comp],
        experiment_results=[{"metrics": {"cosine@1": 0.59,
                                         "latent_mse@1": 0.024,
                                         "physics_consistency": 0.0}}],
        telemetry={"missing_runs": 0, "timeout_runs": 0,
                   "total_trials": 15, "succeeded_trials": 15,
                   "jobs_used": 30, "elapsed_seconds": 100.0,
                   "budget": {"max_jobs": 30, "max_total_seconds": 3600.0},
                   "ood": {"confirmed": True}},
        holdout={"evaluated": True, "holdout_deltas": [-0.02] * 3,
                 "mean_holdout_delta": -0.02,
                 "holdout_direction_agreement": 1.0,
                 "holdout_used_for_decision": False},
    )


class TestUnifiedTaskProtocol(unittest.TestCase):
    def test_task_is_domain_agnostic_and_validated(self):
        t = ResearchTask(domain="embodied", objective="reduce rollout error")
        self.assertTrue(t.objective_id.startswith("obj_"))
        self.assertEqual(t.to_objective().domain, "embodied")
        self.assertEqual(len(t.fingerprint()), 64)
        self.assertEqual(t.evaluation_protocol["decision_split"], "validation")
        self.assertEqual(t.heldout_policy["holdout_split"], "test")

    def test_reserved_split_cannot_be_used_for_selection(self):
        with self.assertRaises(ValueError):
            ResearchTask(domain="quant", objective="x",
                         evaluation_protocol={"decision_split": "test",
                                              "reserved_final_split": "test"})
        with self.assertRaises(ValueError):
            ResearchTask(domain="quant", objective="x",
                         heldout_policy={"paired_seeds": [1, 1],
                                         "holdout_split": "test"})


class TestQualityProtocol(unittest.TestCase):
    def test_falsification_is_positive_and_bare_acceptance_is_not(self):
        ev = _evidence()
        rejected = evaluate_research_quality(ev, domain="scinet")
        bare = evaluate_research_quality(
            QualityEvidence(report={"comparisons": [
                {"decision": "accepted_for_confirmation"} for _ in range(3)]}),
            domain="scinet")
        self.assertGreater(rejected.Q, bare.Q)
        self.assertGreater(rejected.components["D"], 0.70)
        self.assertTrue(bare.hard_violations)
        self.assertFalse(bare.gate_eligible)

    def test_missing_evidence_degrades_every_dimension(self):
        full = evaluate_research_quality(_evidence(), domain="scinet")
        thin = evaluate_research_quality(
            QualityEvidence(report=_evidenced_report("rejected")),
            domain="scinet")
        self.assertGreater(full.components["E"], thin.components["E"])
        self.assertGreater(full.components["R"], thin.components["R"])
        self.assertGreater(full.Q, thin.Q)

    def test_domain_rules_are_non_compensable(self):
        ev = _evidence()
        ev.experiment_results = [{"metrics": {"lookahead_leakage": 1.0,
                                              "max_drawdown": 0.1,
                                              "net_return_after_cost": 0.1}}]
        score = quant_quality_evaluator().evaluate(ev)
        self.assertTrue(score.hard_violations)
        self.assertFalse(score.gate_eligible)
        self.assertLessEqual(score.Q, 0.35)

    def test_domains_have_distinct_injected_rules(self):
        self.assertNotEqual(scinet_quality_evaluator().name,
                            quant_quality_evaluator().name)
        self.assertNotEqual(embodied_quality_evaluator().weights.as_dict(),
                            quant_quality_evaluator().weights.as_dict())
        self.assertIn("min_evidence_integrity", QUALITY_FROZEN_FIELDS)
        with self.assertRaises(ValueError):
            QualityWeights(evidence=-1.0)


class TestEvolutionSpaces(unittest.TestCase):
    def test_two_spaces_are_disjoint(self):
        d = space_for(CandidateType.DOMAIN_MODEL)
        h = space_for(CandidateType.RESEARCH_HARNESS)
        self.assertNotEqual(d.optimization_object, h.optimization_object)
        self.assertFalse(set(DOMAIN_SEARCH_SPACE) & set(HARNESS_SEARCH_SPACE))
        self.assertIn("train.lr", DOMAIN_SEARCH_SPACE)   # Package C preserved
        self.assertIn("planner.strategy", HARNESS_SEARCH_SPACE)

    def test_fingerprints_are_namespaced(self):
        for kind in CandidateType:
            fp = namespaced_fingerprint(kind, target="k", old="a", new="b",
                                        taxonomy="T")
            self.assertEqual(candidate_type_of(fp), kind)

    def test_evaluation_substrate_is_frozen_for_both(self):
        for kind in CandidateType:
            for target in ("quality_evaluator", "evaluation.weights",
                           "gate_a", "scoring", "EVALUATION_SPLIT"):
                self.assertTrue(is_frozen(kind, target), (kind, target))


class TestHarnessBoundary(unittest.TestCase):
    def test_harness_cannot_declare_evaluation_fields(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            replace(BASE_HARNESS, planner_strategy="invalid").validate()
        self.assertNotIn("primary_metric", IMPROVED_HARNESS.to_dict())
        self.assertNotIn("weights", IMPROVED_HARNESS.to_dict())
        self.assertNotEqual(BASE_HARNESS.fingerprint(),
                            IMPROVED_HARNESS.fingerprint())

    def test_planner_is_evidence_grounded_and_whitelisted(self):
        class _Policy:
            allowed_config_keys = ("model.drift_correction", "train.lr")
            intervention_policy = {"dynamics_error_accumulation":
                                   {"model.drift_correction": 0.75}}

        prompt = ('{"source_experiment_id":"exp_real",'
                  '"metrics":{"state_mse@1":0.01,"state_mse@5":0.09},'
                  '"failure_reports":[{"category":"dynamics_error_accumulation"}]}')
        generate = make_harness_generate(IMPROVED_HARNESS, _Policy(),
                                         baseline_config={"train": {"lr": 0.02}})
        items = generate(prompt)
        self.assertEqual(len(items), 2)  # competing hypotheses
        for item in items:
            self.assertTrue(any("exp_real" in r for r in item["evidence_refs"]))
            self.assertTrue(set(item["intervention"]).issubset(
                set(_Policy.allowed_config_keys)))
            self.assertTrue(item["falsification_criteria"])

    def test_negative_harness_is_inert(self):
        class _Policy:
            allowed_config_keys = ("train.lr", "model.drift_correction")
            intervention_policy = {}

        prompt = ('{"source_experiment_id":"exp_real",'
                  '"metrics":{"state_mse@1":0.01,"state_mse@5":0.09},'
                  '"failure_reports":[]}')
        items = make_harness_generate(
            NEGATIVE_HARNESS, _Policy(),
            baseline_config={"train": {"lr": 0.02}})(prompt)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["intervention"], {"train.lr": 0.02})


class TestRealLocalDomains(unittest.TestCase):
    """No mocks: exercise the real local engines and the frozen datasets."""

    def test_frozen_manifests_verify_on_disk(self):
        from research_runtime.domains.local_engine import read_frozen_manifest
        import yaml
        for name in ("embodied", "quant"):
            root = (Path(__file__).resolve().parents[2] / "research_runtime"
                    / "domains" / "projects" / name)
            cfg = yaml.safe_load((root / "config.yaml").read_text())
            manifest = read_frozen_manifest(root, cfg)
            self.assertTrue(manifest["dataset_version"])
            self.assertIn("train", manifest["splits"])
            self.assertIn("test", manifest["splits"])
            self.assertIn("ood", manifest["splits"])
            self.assertFalse(manifest["provenance"]["is_exchange_or_hardware_data"])

    def test_real_embodied_train_and_rollout(self):
        from research_runtime.domains.embodied import PROJECT_DIR, embodied_compute
        from research_runtime.domains.local_engine import LocalComputeAdapter
        from research_runtime.schemas import ExperimentSpec
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LocalComputeAdapter(
                name="embodied_test", project_dir=PROJECT_DIR,
                compute=embodied_compute, job_root=Path(tmp) / "jobs")
            cfg = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text())
            train = ExperimentSpec(
                experiment_id="t1", objective_id="o1", stage="train",
                config_path="config.yaml", budget={"job_id": "j_train"}, seed=42)
            jid = adapter.submit(train)
            tr = adapter.collect_result(jid, train)
            self.assertEqual(tr.status, "succeeded")
            self.assertTrue(tr.checkpoint["sha256"])
            ev = train.with_run_id(str(tr.run_id), stage="evaluate",
                                   split="validation").model_copy(
                update={"budget": {"job_id": "j_eval"}})
            jid2 = adapter.submit(ev)
            res = adapter.collect_result(jid2, ev)
            self.assertEqual(res.status, "succeeded")
            self.assertEqual(res.evaluation_split, "validation")
            self.assertEqual(res.run_id, tr.run_id)
            for k in (1, 2, 3, 4, 5):
                self.assertIn(f"state_mse@{k}", res.metrics)
            self.assertGreater(res.metrics["state_mse@5"],
                               res.metrics["state_mse@1"])  # real accumulation
            self.assertEqual(res.metrics["safety_violations"], 0.0)

    def test_real_quant_backtest_has_no_lookahead(self):
        from research_runtime.domains.quant import PROJECT_DIR, quant_compute
        from research_runtime.domains.local_engine import LocalComputeAdapter
        from research_runtime.schemas import ExperimentSpec
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LocalComputeAdapter(
                name="quant_test", project_dir=PROJECT_DIR,
                compute=quant_compute, job_root=Path(tmp) / "jobs")
            cfg = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text())
            train = ExperimentSpec(
                experiment_id="q1", objective_id="o1", stage="train",
                config_path="config.yaml", budget={"job_id": "qj_train"}, seed=42)
            jid = adapter.submit(train)
            tr = adapter.collect_result(jid, train)
            self.assertEqual(tr.status, "succeeded")
            ev = train.with_run_id(str(tr.run_id), stage="evaluate",
                                   split="test").model_copy(
                update={"budget": {"job_id": "qj_eval"}})
            res = adapter.collect_result(adapter.submit(ev), ev)
            self.assertEqual(res.status, "succeeded")
            self.assertEqual(res.evaluation_split, "test")
            self.assertEqual(res.metrics["lookahead_leakage"], 0.0)
            self.assertLessEqual(res.metrics["max_drawdown"], 0.60)
            self.assertIn("pred_score@20", res.metrics)

    def test_real_end_to_end_research_task_in_ci(self):
        """CI-visible FULL inner loop on a REAL local engine (no FakeRunner).

        Autonomous hypotheses -> Critic -> plan -> approval -> real paired
        campaign -> comparison -> immutable report -> holdout -> quality.
        Two paired seeds keep this ~6 s so it runs on every OS/Python job.
        """
        from research_runtime.research_task import ResearchTask
        t = ResearchTask(
            domain="embodied", objective="CI real autonomous research loop",
            evaluation_protocol={"primary_metric": "state_mse@1",
                                 "metric_direction": "minimize",
                                 "decision_split": "validation",
                                 "reserved_final_split": "test"},
            heldout_policy={"independent_holdout": True,
                            "holdout_split": "test", "ood_task": True,
                            "paired_seeds": [42, 43]},
            budget={"experiments": 8, "max_jobs": 12,
                    "max_total_seconds": 300.0, "epochs": 1})
        with tempfile.TemporaryDirectory() as tmp:
            result = run_research_task(t, harness=IMPROVED_HARNESS,
                                       run_root=Path(tmp))
        self.assertTrue(result.ok, result.error)
        self.assertGreaterEqual(len(result.hypotheses), 1)
        # Autonomous hypotheses are grounded in the REAL baseline experiment,
        # not human-authored: the immutable report carries the evidence refs.
        report = result.report or {}
        self.assertTrue(report.get("evidence_refs"))
        self.assertTrue(any("experiment:" in r
                            for r in report["evidence_refs"]))
        self.assertTrue(all(h.get("content_sha256")
                            for h in result.hypotheses))
        self.assertTrue(all(r.get("passed")
                            for r in report.get("reviews", []) if r))
        self.assertGreaterEqual(result.campaign.get("succeeded_trials", 0), 4)
        self.assertEqual(result.campaign.get("status"), "completed")
        self.assertGreater(result.quality.get("Q", 0), 0.0)
        self.assertFalse(result.quality.get("hard_violations"))
        self.assertFalse(result.evidence["holdout"].get(
            "holdout_used_for_decision", True))
        self.assertFalse(report.get("frozen_test_used", True))
        self.assertTrue(result.artifact_sha256)

    def test_all_domains_register_into_one_registry(self):
        from research_runtime.domain_registry import default_registry
        from research_runtime.research_environment import register_all_domains
        with tempfile.TemporaryDirectory() as tmp:
            names = register_all_domains(data_dir=Path(tmp),
                                         include_scinet=False)
            self.assertEqual(names, ["embodied", "quant"])
            for name in names:
                bundle = default_registry.get(name)
                self.assertIsNotNone(bundle.quality_evaluator)
                self.assertIsNotNone(bundle.metric_interpreter)
                self.assertTrue(callable(bundle.adapter_factory))
                self.assertTrue(callable(bundle.campaign_runner_factory))
                self.assertTrue(callable(bundle.comparator_factory))


if __name__ == "__main__":
    unittest.main()
