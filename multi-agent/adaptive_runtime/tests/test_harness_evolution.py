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
from types import SimpleNamespace
from unittest import mock

from adaptive_runtime.evolution_spaces import CandidateType, is_frozen
from adaptive_runtime.harness_evolution import (
    EVALUATION_SUBSTRATE, HarnessCandidate, apply_harness_patch,
    degenerate_control_candidate, evaluation_substrate_snapshot,
    evaluate_candidate, generate_harness_candidates, promote_harness,
)
from adaptive_runtime.ledger import Ledger
from adaptive_runtime.research_failure_analyzer import (
    analyze_many, analyze_task_result,
)
from adaptive_runtime import harness_evolution as he
from adaptive_runtime.settings import Settings
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


# ---------------------------------------------------------------------------
# D1 + D2 (P0-1 / P0-2): negative-control integrity and missing-data handling.
# These exercise evaluate_candidate() with a *monkeypatched* _run_arm so the
# real paired-gate machinery is tested without running the research engine.
# ---------------------------------------------------------------------------

class _HarnessStub:
    """A minimal Harness stand-in that only needs .fingerprint() / .to_dict()."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def fingerprint(self) -> str:
        return "stub:" + self._tag

    def to_dict(self) -> dict:
        return {"tag": self._tag}


_BASE_STUB = _HarnessStub("base")
_CAND_STUB = _HarnessStub("cand")
_NEG_STUB = _HarnessStub("neg")


def _run_result(q: float | None) -> dict:
    quality = {"Q": q} if q is not None else {}
    return {
        "task_id": "t", "domain": "quant", "ok": True, "outcome": "completed",
        "quality": quality,
        "telemetry": {"missing_runs": 0, "timeout_runs": 0,
                      "total_trials": 10, "succeeded_trials": 10,
                      "critic_rejections": [], "proposer_skip_reasons": [],
                      "budget": {"max_total_seconds": 3600}},
        "provenance": {"result_path": "r.json"},
    }


def _make_fake_run(base_q, cand_q, neg_q, *,
                   missing_base=None, missing_holdout_id=None):
    missing_base = set(missing_base or [])

    def _fake(task, harness, *, run_root, experience_path):
        tid = getattr(task, "task_id", "")
        if harness is _NEG_STUB:
            return _run_result(neg_q)
        if harness is _CAND_STUB:
            if missing_holdout_id and tid == missing_holdout_id:
                return _run_result(None)
            return _run_result(cand_q)
        # base harness (train tasks + held-out base arm)
        if tid in missing_base:
            return _run_result(None)
        if missing_holdout_id and tid == missing_holdout_id:
            return _run_result(None)
        return _run_result(base_q)

    return _fake


def _candidate() -> HarnessCandidate:
    return HarnessCandidate(
        candidate_id="cand_d12", target="planner.strategy",
        old="fixed_first_knob", new="diagnosis_targeted",
        taxonomy="EXPERIMENT_DESIGN_FAILURE", patch={},
        hyp_fingerprint="research_harness::d12")


def _tasks(n: int):
    return [SimpleNamespace(task_id=f"task_{i}", domain="quant")
            for i in range(n)]


class TestNegativeControlAndMissingData(unittest.TestCase):
    """D1 + D2 behavioural guarantees (P0-1)."""

    def setUp(self):
        self.tasks = _tasks(5)
        self.heldout = SimpleNamespace(task_id="holdout", domain="quant")
        self.cfg = Settings()

    def _eval(self, base_q, cand_q, neg_q, *, with_neg=True,
              missing_base=None, missing_holdout=False, tmp=None):
        cand = _candidate()
        fake = _make_fake_run(
            base_q, cand_q, neg_q, missing_base=missing_base,
            missing_holdout_id=(self.heldout.task_id if missing_holdout else None))
        with mock.patch.object(he, "_run_arm", side_effect=fake):
            rec = evaluate_candidate(
                base_harness=_BASE_STUB, cand_harness=_CAND_STUB,
                candidate=cand, tasks=self.tasks,
                heldout_task=self.heldout,
                run_root=Path(tmp) / "run",
                experience_path=Path(tmp) / "exp.jsonl",
                repo_root=WORKSPACE, cfg=self.cfg,
                negative_harness=(_NEG_STUB if with_neg else None))
        return rec

    def test_missing_base_arm_is_not_imputed_as_zero(self):
        """P0-1 / D2: a crashed Base arm must stay None and fail CLOSED."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._eval(0.85, 0.90, 0.85,
                             missing_base={"task_0"}, tmp=tmp)
        # The missing arm is preserved as None, never coerced to 0.0.
        self.assertIn(None, rec.base_Q)
        self.assertNotIn(0.0, rec.base_Q)
        # Incomplete evidence cannot manufacture a fake improvement.
        self.assertEqual(rec.decision, "INCONCLUSIVE")
        self.assertGreaterEqual(rec.evidence["missing_runs"], 1)

    def test_inconclusive_negative_control_is_not_a_reject(self):
        """P0-1 / D1: INCONCLUSIVE must NEVER count as a successful rejection."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._eval(0.85, 0.90, 0.85, tmp=tmp)  # neg == base
        nc = rec.negative_control
        self.assertEqual(nc["rejected"], False)
        self.assertEqual(nc["accepted"], False)
        self.assertEqual(nc["not_accepted"], True)
        # The loop-integrity claim is simply left UNPROVEN for this candidate;
        # a genuinely improving candidate may still ACCEPT on the holdout.
        self.assertNotEqual(nc["rejected"], True)

    def test_genuine_negative_control_is_rejected(self):
        """P0-1 / D1: a real degradation must be recorded as a genuine REJECT."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._eval(0.85, 0.90, 0.70, tmp=tmp)  # neg degraded
        nc = rec.negative_control
        self.assertEqual(nc["rejected"], True)
        self.assertEqual(nc["accepted"], False)
        # The genuine improvement still ACCEPTs; the reject is of the control.
        self.assertEqual(rec.decision, "ACCEPT")

    def test_rigged_negative_control_fails_closed(self):
        """P0-1 / D1: a 'negative' control that improves proves a rigged loop."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._eval(0.85, 0.90, 0.98, tmp=tmp)  # neg improves
        nc = rec.negative_control
        self.assertEqual(nc["accepted"], True)
        # The candidate MUST NOT be accepted when its control was accepted.
        self.assertNotEqual(rec.decision, "ACCEPT")

    def test_missing_holdout_does_not_confirm_improvement(self):
        """P0-1 / D2: a missing holdout cannot confirm a candidate."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._eval(0.85, 0.90, 0.85, with_neg=False,
                             missing_holdout=True, tmp=tmp)
        self.assertIsNone(rec.holdout["delta"])
        self.assertEqual(rec.holdout["confirmed"], False)
        self.assertEqual(rec.decision, "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
