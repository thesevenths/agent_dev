"""ARR 0.2 / Package C: tests for the REAL candidate-runtime evaluation.

Design rules
------------
* The engine-dependent tests are OPT-IN (``RUN_REAL_SCINET=1``) AND gated on the
  engine actually being reachable. CI never runs a real training job by accident.
* The fast safety tests need no engine: they prove the real path can NEVER
  silently fall back to the fake profiles, that a no-op patch is a hard error,
  and that real failure counts flow into the non-compensable gates.
* A REAL negative control (deliberately poor learning rate) proves the measured
  deltas -- not the test author -- decide ACCEPT / REJECT / INCONCLUSIVE.

The deterministic fake gate-chain tests remain in ``test_research_ingest.py``;
this file adds the real path without changing any of them.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_runtime import gate, research_ingest as bridge
from adaptive_runtime.proposer import Candidate
from adaptive_runtime.settings import Settings
from research_runtime.candidate_config import CandidateConfigError

_REAL_OK = (os.getenv("RUN_REAL_SCINET") == "1" and bridge.scinet_available())


def _report() -> dict:
    return {
        "objective_id": "obj_pendulum_multistep",
        "outcome": "rejected",
        "comparisons": [{"decision": "rejected"}, {"decision": "rejected"}],
    }


def _candidate(new: str, *, cid: str = "rrsi_cand_test") -> Candidate:
    return Candidate(
        candidate_id=cid, target="train.lr", level=1,
        old="0.001", new=new, hypothesis=f"train.lr 0.001 -> {new}",
        patch={"kind": "research_config", "key": "train.lr",
               "old": "0.001", "new": new},
        hyp_fingerprint=f"train.lr|0.001->{new}|DECISION_UNCERTAINTY",
    )


class TestRealPathSafety(unittest.TestCase):
    """Fast, engine-free: structural invariants of the real path."""

    def test_engine_unreachable_never_falls_back_to_fake(self):
        """Package C forbids silently faking a 'real' evaluation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(bridge, "_scinet_engine",
                                   side_effect=RuntimeError("engine down")):
                with self.assertRaises(RuntimeError):
                    bridge.ingest(
                        _report(), ledger_path=root / "l.jsonl",
                        experience_path=root / "e.jsonl",
                        sandbox_root=root / "sb", real=True)
                with self.assertRaises(RuntimeError):
                    bridge.evaluate_research_runtime(
                        "base", real=True, candidate=_candidate("0.003"))
            # No fake verdict may have been written: the failure must be loud,
            # never a recorded 'accepted/inconclusive' decision.
            from adaptive_runtime.ledger import Ledger
            self.assertEqual(Ledger(root / "l.jsonl").rows, [])
        # The production wrapper states its refusal explicitly.
        with self.assertRaises(RuntimeError) as ctx:
            bridge._scinet_engine(
                scinet_project_dir=str(Path(tempfile.gettempdir())
                                       / "definitely_not_a_scinet_checkout"))
        self.assertIn("refusing to fake", str(ctx.exception))

    def test_real_search_space_is_disjoint_from_frozen_core(self):
        from research_runtime.candidate_config import DEFAULT_FORBIDDEN_ROOTS
        for key in bridge.RESEARCH_SEARCH_SPACE:
            self.assertNotIn(key, bridge.RESEARCH_FROZEN)
            self.assertNotIn(key.split(".")[0], DEFAULT_FORBIDDEN_ROOTS)
        # Evidence-integrity knobs are still frozen (no regression on R4 safety).
        self.assertIn("RESEARCH_EVIDENCE_STRICT", bridge.RESEARCH_FROZEN)
        for target in bridge.DOMINANT_TO_TARGET.values():
            self.assertIn(target, bridge.RESEARCH_SEARCH_SPACE)

    def test_missing_runs_force_incomplete_evidence(self):
        base = bridge.evaluate_research_runtime("base", "good")
        cand = bridge.evaluate_research_runtime("cand", "good")
        ev = bridge.build_research_evidence(base, cand, missing_runs=1)
        dec = gate.gate_b(ev, cfg=Settings())
        self.assertEqual(dec.reason, gate.Reason.INCOMPLETE_EVIDENCE.value)
        self.assertEqual(gate.ledger_status(dec, stopped=""), "inconclusive")

    def test_coerce_override_preserves_baseline_field_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cfg.yaml"
            p.write_text("train:\n  lr: 0.001\n  batch_size: 16\n"
                         "model:\n  latent_dim: 4\n", encoding="utf-8")
            self.assertEqual(
                bridge._coerce_override(Path(tmp), "cfg.yaml", "train.lr", "0.01"),
                0.01)
            self.assertEqual(
                bridge._coerce_override(Path(tmp), "cfg.yaml",
                                        "train.batch_size", "32"), 32)
            self.assertIsInstance(
                bridge._coerce_override(Path(tmp), "cfg.yaml",
                                        "model.latent_dim", "8"), int)
            with self.assertRaises(KeyError):
                bridge._coerce_override(Path(tmp), "cfg.yaml",
                                        "train.missing", "1")

    def test_fake_gate_chain_selftest_unchanged(self):
        out = bridge.selftest()
        self.assertEqual(out["good"]["ledger_status"], "accepted")
        self.assertEqual(out["zero"]["ledger_status"], "inconclusive")
        self.assertEqual(out["worse"]["ledger_status"], "rejected")


@unittest.skipUnless(_REAL_OK, "set RUN_REAL_SCINET=1 and make SciNet++ reachable")
class TestRealEngineEvaluation(unittest.TestCase):
    """REAL engine runs. Opt-in; never part of the default/CI suite."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_noop_patch_is_hard_error(self):
        """A patch that does not change the runnable config must NEVER be scored.

        If it were, cand == base -> ΔQ≈0 -> a real hypothesis would be recorded
        as 'no effect' and permanently pollute the ledger (probe.PatchError's
        exact failure mode).
        """
        with self.assertRaises(CandidateConfigError) as ctx:
            bridge._run_real_pair(_candidate("0.001", cid="rrsi_cand_noop"),
                                  paired_seeds=[42], holdout_split=None)
        self.assertIn("did NOT apply", str(ctx.exception))

    def test_real_candidate_changes_behavior_and_is_validated(self):
        """Full closed loop: real report -> diagnosis -> candidate -> real runs
        -> real deltas -> gates -> ACCEPT, with verified isolation."""
        res = bridge.ingest(
            _report(), ledger_path=self.root / "l.jsonl",
            experience_path=self.root / "e.jsonl",
            sandbox_root=self.root / "sb", real=True,
            paired_seeds=[42, 43], decision_split="validation",
            holdout_split="test", epochs=1)
        e = res.real_evidence
        self.assertEqual(res.evaluation_mode, "real")
        self.assertTrue(e["config_changed"])
        self.assertTrue(e["override_effective"])
        self.assertTrue(e["behavior_changed"])
        self.assertFalse(e["holdout_used_for_decision"])
        self.assertTrue(e["holdout_consistent"])
        self.assertEqual(e["missing_runs"], 0)
        self.assertEqual(len(res.evidence["task_deltas"]), 2)
        self.assertGreater(res.evidence["mean_delta"], 0)
        self.assertTrue(res.decision_a.startswith("ACCEPT"))
        self.assertEqual(res.decision_b, "ACCEPT(VALIDATED)")
        self.assertEqual(res.decision, "ACCEPT")
        self.assertTrue(res.live_runtime_untouched)
        self.assertEqual(len(res.artifact_sha256), 64)
        # real provenance, not a marker file
        self.assertTrue(e["dataset_version"])
        self.assertEqual(len(e["manifest_sha256"]), 64)
        self.assertNotEqual(e["baseline_config_sha256"],
                            e["candidate_config_sha256"])
        self.assertTrue((Path(res.sandbox_path) / "rrsi_real_evidence.json").is_file())

    def test_real_negative_control_is_not_accepted(self):
        """Anti-rigging: a genuinely worse config must NOT be validated.

        Deliberately shrink train.lr by 100x. With 1 epoch the model barely
        learns, so REAL paired deltas turn negative; the same gate chain that
        accepted the good candidate must refuse this one.
        """
        bad = _candidate("0.00001", cid="rrsi_cand_negctl")
        base_m, cand_m, pr = bridge._run_real_pair(
            bad, paired_seeds=[42, 43], decision_split="validation",
            holdout_split=None, cfg=Settings())
        ev = bridge.build_research_evidence(
            base_m, cand_m, missing_runs=pr["missing_runs"],
            regression_rate=pr["regression_rate"])
        dec_a = gate.gate_a(bad, cfg=Settings())
        dec_b = gate.gate_b(ev, cfg=Settings())
        print(f"[negctl] real task_deltas={ev.task_deltas} "
              f"mean_delta={ev.paired(Settings().alpha).mean:.6f} "
              f"gate_b={dec_b.decision}({dec_b.reason}) "
              f"base_Q={base_m.Q:.6f} cand_Q={cand_m.Q:.6f}")
        self.assertTrue(dec_a.accepted, "candidate must be admissible to reach scoring")
        self.assertLess(ev.paired(Settings().alpha).mean, 0,
                        "real metrics must show the negative control is worse")
        self.assertNotEqual(dec_b.decision, "ACCEPT",
                            "a real regression/no-gain must never be ACCEPTed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
