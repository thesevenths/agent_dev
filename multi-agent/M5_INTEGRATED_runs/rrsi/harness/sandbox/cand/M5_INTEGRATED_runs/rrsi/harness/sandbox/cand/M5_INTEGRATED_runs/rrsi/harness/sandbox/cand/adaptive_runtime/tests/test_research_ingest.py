"""ARR 0.1 / R4: tests for the research_runtime -> RRSI bridge.

These prove the two subsystems are now connected through the REAL RRSI
primitives (probe.Sandbox isolation, gate_a/gate_b non-compensable gates,
ledger.Ledger + experience.Store record-keeping, version.BaseVersion
attribution) using deterministic --fake profiles -- no network, no model.

The central safety invariant is asserted: the candidate is evaluated in an
isolated sandbox and the LIVE research runtime is never imported or mutated by
this module (it receives the report as plain data and writes only to the
injected ledger / experience / sandbox paths).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_runtime import gate, research_ingest as bridge
from adaptive_runtime.gate import Decision, explain
from adaptive_runtime.ledger import Ledger
from adaptive_runtime.proposer import Candidate
from adaptive_runtime.settings import Settings


def _report(outcome: str = "inconclusive", n_inconclusive: int = 3) -> dict:
    return {
        "objective_id": "obj_p106",
        "outcome": outcome,
        "comparisons": [{"decision": "inconclusive"} for _ in range(n_inconclusive)],
    }


def _evidenced(decision: str) -> tuple[dict, dict]:
    """A report with REAL structural evidence, for a given verdict."""
    h_sha = "a" * 64
    report = {
        "objective_id": "obj_evidenced", "outcome": decision,
        "report_sha256": "b" * 64,
        "domain": "scinet",
        "evidence_refs": [f"experiment:exp_{i}:result:j{i}" for i in range(2)],
        "experience_refs": ["exp_abc"],
        "promotion": "not_requested",
        "frozen_test_used": False,
        "hypotheses": [
            {"hypothesis_id": "h1", "content_sha256": h_sha, "statement": "s1"},
            {"hypothesis_id": "h2", "content_sha256": "c" * 64, "statement": "s2"},
        ],
        "reviews": [
            {"review_id": "r1", "hypothesis_id": "h1",
             "hypothesis_sha256": h_sha, "passed": True},
            {"review_id": "r2", "hypothesis_id": "h2",
             "hypothesis_sha256": "c" * 64, "passed": True},
        ],
        "plans": [{"plan_id": "p1", "plan_sha256": "d" * 64, "version": 1}],
        "approvals": [{"approval_id": "ap1", "resource_sha256": "e" * 64,
                       "source": "human", "status": "granted"}],
        "campaigns": [{"campaign_id": "c1", "status": "completed"}],
        "comparisons": [
            {"comparison_id": "cmp1", "candidate_id": "cand_A",
             "decision": decision, "decision_rule": "practical_effect_confirmed",
             "n_pairs": 5, "mean_delta": -0.02, "guardrail_violations": []},
            {"comparison_id": "cmp2", "candidate_id": "cand_B",
             "decision": decision, "decision_rule": "confidence_interval_regression",
             "n_pairs": 5, "mean_delta": -0.01, "guardrail_violations": []},
        ],
    }
    comp = {
        "comparison_id": "cmp1", "candidate_id": "cand_A",
        "baseline_config_sha256": "1" * 64, "candidate_config_sha256": "2" * 64,
        "dataset_version": "frozen-v1", "evaluation_split": "validation",
        "n_pairs": 5, "n_planned": 5, "missing_pairs": [],
        "dataset_split_hashes": {"train": "3" * 64, "validation": "4" * 64},
        "guardrail_violations": [],
        "paired_observations": [{"seed": s, "delta": -0.02} for s in range(5)],
    }
    evidence = {
        "comparison_reports": [comp],
        "experiment_results": [{"metrics": {"cosine@1": 0.59,
                                            "latent_mse@1": 0.024,
                                            "physics_consistency": 0.0}}],
        "telemetry": {"missing_runs": 0, "timeout_runs": 0,
                      "total_trials": 15, "succeeded_trials": 15,
                      "jobs_used": 30, "elapsed_seconds": 120.0,
                      "budget": {"max_jobs": 30, "max_total_seconds": 3600.0},
                      "ood": {"confirmed": True}},
        "holdout": {"holdout_deltas": [-0.02] * 3,
                    "holdout_direction_agreement": 1.0,
                    "holdout_used_for_decision": False},
    }
    return report, evidence


class TestExportTrace(unittest.TestCase):
    def test_quality_is_not_acceptance_rate(self):
        """D1: the old `accepted/n` scalar is gone; inconclusive is not 0.0."""
        trace = bridge.export_research_trace(_report())
        self.assertEqual(trace["objective_id"], "obj_p106")
        self.assertEqual(trace["n_decisions"], 3)
        self.assertEqual(trace["accepted_rate"], 0.0)
        self.assertNotEqual(trace["quality_score"], 0.0)
        self.assertLess(trace["quality_score"], 0.35)  # hard constraint caps it
        self.assertTrue(trace["hard_violations"])
        self.assertFalse(trace["gate_eligible"])
        self.assertIn("components", trace["quality"])
        self.assertEqual(trace["dominant_signal"], "DECISION_UNCERTAINTY")

    def test_bare_acceptance_is_not_a_perfect_score(self):
        """D1 Goodhart guard: ACCEPT without evidence cannot score 1.0."""
        report = {
            "objective_id": "o2", "outcome": "accepted_for_confirmation",
            "comparisons": [{"decision": "accepted_for_confirmation"},
                            {"decision": "accepted_for_confirmation"}],
        }
        trace = bridge.export_research_trace(report)
        self.assertLess(trace["quality_score"], 0.35)
        self.assertEqual(trace["accepted_rate"], 1.0)
        self.assertEqual(trace["dominant_signal"], "ROBUST_DECISION")

    def test_falsification_with_evidence_scores_high(self):
        """D1 core: rejecting a wrong hypothesis is a valuable discovery."""
        rep_rejected, ev = _evidenced("rejected")
        rep_accepted, _ = _evidenced("accepted_for_confirmation")
        rejected = bridge.export_research_trace(rep_rejected, domain="scinet",
                                                evidence=ev)
        accepted = bridge.export_research_trace(rep_accepted, domain="scinet",
                                                evidence=ev)
        self.assertGreaterEqual(rejected["quality_score"],
                                accepted["quality_score"])
        self.assertGreater(rejected["quality"]["components"]["D"], 0.7)
        self.assertFalse(rejected["hard_violations"])
        self.assertTrue(rejected["gate_eligible"])

    def test_missing_evidence_never_defaults_to_pass(self):
        """E/R dimensions must DEGRADE when provenance is absent."""
        rep, ev = _evidenced("rejected")
        full = bridge.export_research_trace(rep, domain="scinet", evidence=ev)
        bare = bridge.export_research_trace(rep, domain="scinet")
        self.assertGreater(full["quality"]["components"]["E"],
                           bare["quality"]["components"]["E"])
        self.assertGreater(full["quality_score"], bare["quality_score"])
        self.assertTrue(bare["hard_violations"])


class TestProposeResearchCandidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.tmp.name) / "ledger.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_candidate_not_frozen_not_inert(self):
        cand = bridge.propose_research_candidate("DECISION_UNCERTAINTY",
                                                 self.ledger, cfg=Settings())
        self.assertIsInstance(cand, Candidate)
        self.assertNotIn(cand.target, bridge.RESEARCH_FROZEN)
        self.assertEqual(cand.level, 1)
        self.assertEqual(cand.edits, 1)
        self.assertEqual(cand.est_cost, 0)
        # gate_a must pass for this candidate (not inert, within budget).
        dec = gate.gate_a(cand, cfg=Settings())
        self.assertTrue(dec.accepted, explain(dec))

    def test_skips_already_decided_fingerprint(self):
        cand1 = bridge.propose_research_candidate("DECISION_UNCERTAINTY",
                                                  self.ledger, cfg=Settings())
        # Record the same hypothesis as already "proposed" -> is_duplicate blocks
        # re-issuing it; propose must move to the next adjacent value.
        self.ledger.create(target=cand1.target, old=cand1.old, new=cand1.new,
                           taxonomy=cand1.taxonomy, claim=cand1.hypothesis,
                           tasks=[])
        cand2 = bridge.propose_research_candidate("DECISION_UNCERTAINTY",
                                                  self.ledger, cfg=Settings())
        self.assertNotEqual(cand2.hyp_fingerprint, cand1.hyp_fingerprint)


class TestEvaluateAndEvidence(unittest.TestCase):
    def test_deterministic_good_zero_worse(self):
        base = bridge.evaluate_research_runtime("base", "good")
        good = bridge.evaluate_research_runtime("cand", "good")
        zero = bridge.evaluate_research_runtime("cand", "zero")
        worse = bridge.evaluate_research_runtime("cand", "worse")
        self.assertGreater(good.Q, base.Q)
        self.assertAlmostEqual(zero.Q, base.Q)
        self.assertLess(worse.Q, base.Q)

    def test_evidence_deltas(self):
        base = bridge.evaluate_research_runtime("base", "good")
        cand = bridge.evaluate_research_runtime("cand", "good")
        ev = bridge.build_research_evidence(base, cand)
        self.assertEqual(len(ev.task_deltas), 4)
        self.assertAlmostEqual(ev.A_base, 0.60)
        self.assertAlmostEqual(ev.A_cand, 0.79)


class TestIngestGateChain(unittest.TestCase):
    """The outer loop must produce the right ledger fate per deterministic case."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _ingest(self, case: str) -> bridge.IngestResult:
        return bridge.ingest(
            _report(), ledger_path=self.root / "ledger.jsonl",
            experience_path=self.root / "exp.jsonl",
            sandbox_root=self.root / case, fake_case=case)

    def test_good_is_validated(self):
        res = self._ingest("good")
        self.assertTrue(res.decision_a.startswith("ACCEPT"))
        self.assertEqual(res.decision_b, "ACCEPT(VALIDATED)")
        self.assertEqual(res.ledger_status, "accepted")

    def test_zero_is_inconclusive(self):
        res = self._ingest("zero")
        self.assertTrue(res.decision_a.startswith("ACCEPT"))
        # explain() appends " @ <first_fail_check>" for a REJECT decision.
        self.assertTrue(res.decision_b.startswith("REJECT(within_noise)"))
        # within_noise is NOT a refuting reason -> inconclusive, never rejected.
        self.assertEqual(res.ledger_status, "inconclusive")

    def test_worse_is_rejected(self):
        res = self._ingest("worse")
        self.assertTrue(res.decision_a.startswith("ACCEPT"))
        # explain() appends " @ <first_fail_check>" for a REJECT decision.
        self.assertTrue(res.decision_b.startswith("REJECT(regression)"))
        # regression IS a refuting reason -> rejected.
        self.assertEqual(res.ledger_status, "rejected")

    def test_live_runtime_untouched_and_isolated(self):
        # All three writes land under the injected tmp paths, never under the
        # live adaptive_runtime PATHS; and the candidate is evaluated in a
        # sandbox directory distinct from the repo root.
        res = self._ingest("good")
        self.assertTrue(res.live_runtime_untouched)
        self.assertTrue(res.sandbox_path.startswith(str(self.root / "good")))
        self.assertTrue((self.root / "ledger.jsonl").exists())
        self.assertTrue((self.root / "exp.jsonl").exists())
        # The experience record really landed.
        from adaptive_runtime.experience import Store
        rows = Store(self.root / "exp.jsonl").rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["problem"], "RESEARCH_RUNTIME|DECISION_UNCERTAINTY")


if __name__ == "__main__":
    unittest.main()
