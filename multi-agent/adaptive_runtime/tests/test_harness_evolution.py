"""ARR 1.0 / Package F tests: failure analysis, candidate generation, sandbox
patch application and the operator promotion gate.

No FakeRunner is used: these are unit tests of the OUTER loop's *mechanisms*.
The real Base/Candidate experiments are exercised by the CLI and saved as
artifacts under experiments/arr_1_0_evolve/.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adaptive_runtime.evolution_spaces import CandidateType, is_frozen
from adaptive_runtime.harness_evolution import (
    EVALUATION_SUBSTRATE, HarnessCandidate, apply_harness_patch,
    degenerate_control_candidate, evaluation_substrate_snapshot,
    generate_harness_candidates, promote_harness,
)
from adaptive_runtime.ledger import Ledger
from adaptive_runtime.research_failure_analyzer import (
    analyze_many, analyze_task_result,
)
from research_runtime.arr_runtime import BASE_HARNESS, IMPROVED_HARNESS

WORKSPACE = Path(__file__).resolve().parents[2]


def _task_result(decision: str = "inconclusive", *, ok: bool = True) -> dict:
    return {
        "task_id": "task_unit", "domain": "embodied", "ok": ok,
        "error": "", "outcome": decision,
        "report": {"campaigns": [{"campaign_id": "c", "status": "completed"}]},
        "quality": {
            "Q": 0.70,
            "components": {"E": 1.0, "R": 1.0, "G": 0.5, "D": 0.4},
            "checks": {
                "D.hypotheses_are_evidence_grounded": False,
                "D.competing_interventions_tested": False,
                "G.holdout_confirmed_improvement": False,
            },
            "hard_violations": [],
        },
        "telemetry": {"missing_runs": 0, "timeout_runs": 0,
                      "total_trials": 10, "succeeded_trials": 10,
                      "critic_rejections": [], "proposer_skip_reasons": [],
                      "budget": {"max_total_seconds": 3600}},
        "evidence": {"holdout": {"evaluated": True, "mean_holdout_delta": 0.0},
                     "ood": {"evaluated": False}},
        "provenance": {"harness": BASE_HARNESS.to_dict(),
                       "result_path": "task_result.json"},
        "hypotheses": [{"hypothesis_id": "h1"}],
        "comparisons": [{"decision": decision, "mean_delta": 0.0}],
    }


class TestFailureAnalyzer(unittest.TestCase):
    def test_diagnoses_research_process_not_model(self):
        analysis = analyze_task_result(_task_result())
        categories = {f.category for f in analysis.findings}
        self.assertIn("HYPOTHESIS_QUALITY_FAILURE", categories)
        self.assertIn("EXPERIMENT_DESIGN_FAILURE", categories)
        self.assertIn("MEMORY_FAILURE", categories)
        self.assertTrue(analysis.source_sha256)
        for finding in analysis.findings:
            self.assertTrue(finding.evidence)
            self.assertTrue(finding.suggested_target)

    def test_aggregate_dominant_is_evidence_weighted(self):
        out = analyze_many([_task_result(), _task_result()])
        self.assertEqual(out["n_tasks"], 2)
        self.assertTrue(out["dominant"])
        self.assertGreater(sum(out["counts"].values()), 0)

    def test_failed_run_is_tool_usage_failure(self):
        out = analyze_task_result(_task_result(ok=False))
        self.assertIn("TOOL_USAGE_FAILURE",
                      {f.category for f in out.findings})


class TestCandidateGeneration(unittest.TestCase):
    def test_candidates_are_single_knob_and_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "l.jsonl")
            cands = generate_harness_candidates(
                dominant="EXPERIMENT_DESIGN_FAILURE",
                base_harness=BASE_HARNESS, ledger=ledger, limit=4)
            self.assertGreaterEqual(len(cands), 3)
            for c in cands:
                self.assertEqual(c.level, 1)
                self.assertEqual(c.edits, 1)
                self.assertIn(c.target, (
                    "planner.strategy", "planner.hypothesis_prompt",
                    "memory.experience_top_k", "workflow.max_candidates",
                    "workflow.campaign_rounds",
                    "reviewer.require_falsification"))
                self.assertEqual(c.candidate_type,
                                 CandidateType.RESEARCH_HARNESS.value)
                self.assertTrue(c.hyp_fingerprint.startswith(
                    CandidateType.RESEARCH_HARNESS.value + "::"))

    def test_decided_fingerprints_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "l.jsonl")
            first = generate_harness_candidates(
                dominant="EXPERIMENT_DESIGN_FAILURE",
                base_harness=BASE_HARNESS, ledger=ledger, limit=4)
            for c in first:
                ledger.update(c.hyp_fingerprint, "rejected", evidence={})
            second = generate_harness_candidates(
                dominant="EXPERIMENT_DESIGN_FAILURE",
                base_harness=BASE_HARNESS, ledger=ledger, limit=4)
            overlap = {c.hyp_fingerprint for c in first} & {
                c.hyp_fingerprint for c in second}
            self.assertFalse(overlap)


class TestDegenerateControl(unittest.TestCase):
    """Acceptance #11: the loop must be able to REJECT a bad Harness."""

    def test_control_is_labelled_and_not_frozen(self):
        c = degenerate_control_candidate(
            base_harness=BASE_HARNESS,
            dominant="HYPOTHESIS_QUALITY_FAILURE")
        self.assertEqual(c.taxonomy, "NEGATIVE_CONTROL")
        self.assertEqual(c.target, "planner.strategy")
        self.assertEqual(c.new, "repeat_refuted")
        self.assertNotEqual(c.old, c.new)
        self.assertTrue(c.hyp_fingerprint.startswith(
            CandidateType.RESEARCH_HARNESS.value + "::"))
        self.assertFalse(is_frozen(CandidateType.RESEARCH_HARNESS, c.target))

    def test_gate_b_rejects_a_real_regression(self):
        from adaptive_runtime import gate
        from adaptive_runtime.settings import Settings
        deltas = [-0.02] * 5
        ev = gate.Evidence(
            deltas=deltas, task_deltas=deltas, A_base=1.0, A_cand=1.0,
            dC=0.0, regression_rate=1.0, n_tasks=5)
        dec = gate.gate_b(ev, cfg=Settings())
        self.assertFalse(dec.accepted)
        self.assertIn("REJECT", gate.explain(dec))


class TestSandboxPatch(unittest.TestCase):
    def test_patch_is_real_and_changes_harness(self):
        cand = HarnessCandidate(
            candidate_id="harness_cand_test", target="planner.strategy",
            old="fixed_first_knob", new="diagnosis_targeted",
            taxonomy="EXPERIMENT_DESIGN_FAILURE",
            patch={"kind": "research_harness", "target": "planner.strategy",
                   "old": "fixed_first_knob", "new": "diagnosis_targeted"},
            hyp_fingerprint="research_harness::test")
        with tempfile.TemporaryDirectory() as tmp:
            tiny_root = Path(tmp) / "tiny_root"
            tiny_root.mkdir()
            (tiny_root / "marker.txt").write_text("isolated workspace")
            patched, report = apply_harness_patch(
                BASE_HARNESS, cand, sandbox_root=Path(tmp) / "sb",
                repo_root=tiny_root)
            self.assertEqual(patched.planner_strategy, "diagnosis_targeted")
            self.assertTrue(report["config_changed"])
            self.assertEqual(report["frozen_introduced"], [])
            payload = json.loads(Path(report["harness_path"]).read_text())
            self.assertEqual(payload["harness"]["planner_strategy"],
                             "diagnosis_targeted")
            self.assertTrue((Path(report["sandbox_path"]) /
                             "harness_sha256.txt").is_file())

    def test_frozen_target_is_refused(self):
        cand = HarnessCandidate(
            candidate_id="bad", target="quality_evaluator", old="a", new="b",
            taxonomy="T", patch={}, hyp_fingerprint="x")
        with tempfile.TemporaryDirectory() as tmp:
            tiny_root = Path(tmp) / "tiny_root"
            tiny_root.mkdir()
            with self.assertRaises(ValueError):
                apply_harness_patch(BASE_HARNESS, cand,
                                   sandbox_root=Path(tmp) / "sb",
                                   repo_root=tiny_root)

    def test_evaluation_substrate_snapshot_is_stable(self):
        before = evaluation_substrate_snapshot(WORKSPACE)
        after = evaluation_substrate_snapshot(WORKSPACE)
        self.assertEqual(before, after)
        self.assertEqual(set(before), set(EVALUATION_SUBSTRATE))
        self.assertTrue(all(v != "MISSING" for v in before.values()))


class TestOperatorGate(unittest.TestCase):
    def test_promotion_refuses_system_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({
                "n_accepted": 1, "accepted": [{"candidate": {}}],
                "evaluation_substrate_unchanged": True,
                "artifact_sha256": "abc"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                promote_harness(artifact_path=path, operator="system")

    def test_promotion_refuses_no_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({
                "n_accepted": 0, "accepted": [],
                "evaluation_substrate_unchanged": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                promote_harness(artifact_path=path, operator="human-alice")

    def test_improved_harness_has_no_evaluation_fields(self):
        keys = set(IMPROVED_HARNESS.to_dict())
        self.assertFalse(keys & {"weights", "primary_metric",
                                 "evaluation_split", "quality_evaluator",
                                 "gate_a", "gate_b"})


if __name__ == "__main__":
    unittest.main()
