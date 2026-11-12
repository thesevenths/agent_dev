"""ARR 1.1 / G -- autonomous research-loop tests (engine-free, no FakeRunner).

These tests drive the REAL ``run_loop`` controller but inject a stub dual-layer
function and a stub harness, so they exercise the actual orchestration logic --
multi-round advancement, domain version inheritance (baking accepted overrides
into the next baseline), harness inheritance, the no-gain stop, and crash
resume -- without needing the heavy real engines / frozen datasets. The injected
stub returns a payload shaped exactly like ``run_dual_layer`` so the controller
code under test is the production code.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from research_runtime.loop import run_loop
from research_runtime.loop.loop_state import LoopState, save


def _payload(*, domain_accepted=(), harness_accepted=()):
    return {
        "schema_version": "arr-1.1-dual-layer-v1",
        "harness_layer": {"accepted": list(harness_accepted), "records": []},
        "domain_layer": {
            "domain": "embodied",
            "records": [{
                "candidate_type": "domain_model", "domain": "embodied",
                "target": t, "old": o, "new": n,
                "hyp_fingerprint": f"domain_model::{t}|{o}->{n}",
                "base_Q": [0.8], "cand_Q": [0.9], "mean_delta": 0.1,
                "decision": dec,
            } for (t, o, n, dec) in domain_accepted],
            "n_accepted": sum(1 for x in domain_accepted if x[3] == "accepted"),
            "n_rejected": 0, "n_inconclusive": 0,
        },
        "separation": {"candidate_types_distinct": True,
                       "ledger_namespaced": True, "frozen_protected": True},
        "version_inheritance": {"inheritable_next_round": bool(domain_accepted
                                                              or harness_accepted)},
    }


class TestMultiRound(unittest.TestCase):
    def _baseline(self, tmp: str) -> Path:
        cfg = Path(tmp) / "base.yaml"
        cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
        return cfg

    def test_runs_all_rounds_and_bakes_inherited_baseline(self):
        calls = []
        def stub(**kw):
            calls.append(kw.get("domain_base_config"))
            return _payload(domain_accepted=[("train.lr", 0.01, 0.005, "accepted")])

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._baseline(tmp)
            report = run_loop(
                state_path=Path(tmp) / "loop_state.json",
                domain="embodied", baseline_config_path=str(cfg),
                project_dir=tmp, max_rounds=3, dataset_version="test-v1",
                domain_seeds=(50, 51), domain_runner=None,
                domain_require_frozen_manifest=False,
                harness_evolve=None,
                run_dual_layer_fn=stub)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["rounds_completed"], 3)
            self.assertEqual(len(calls), 3)
            # Three versions inherited, one per accepted round.
            self.assertEqual(len(report["versions"]), 3)
            # Final baseline carries the accepted domain override.
            final = Path(report["final_baseline_config_path"])
            self.assertTrue(final.is_file())
            baked = yaml.safe_load(final.read_text(encoding="utf-8"))
            self.assertEqual(baked["train"]["lr"], 0.005)
            self.assertEqual(report["final_accepted_domain_overrides"],
                             {"train.lr": 0.005})


class TestNoGainStop(unittest.TestCase):
    def test_loop_stops_when_a_round_accepts_nothing(self):
        calls = []
        def stub(**kw):
            calls.append(1)
            return _payload()  # nothing accepted

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            report = run_loop(
                state_path=Path(tmp) / "loop_state.json",
                domain="embodied", baseline_config_path=str(cfg),
                project_dir=tmp, max_rounds=5, dataset_version="test-v1",
                domain_seeds=(50,), domain_runner=None,
                domain_require_frozen_manifest=False,
                harness_evolve=None,
                run_dual_layer_fn=stub)
            # No-gain stop: only the first round ran, then the loop ended.
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["rounds_completed"], 0)
            self.assertEqual(report["versions"], [])


class TestHarnessInheritance(unittest.TestCase):
    def test_accepted_harness_threads_into_next_round(self):
        seen_base = []
        def stub(**kw):
            seen_base.append(kw.get("base_harness"))
            rec = {"candidate": {
                "candidate_id": "h1", "target": "planner.strategy",
                "old": "fixed_first_knob", "new": "diagnosis_targeted",
                "taxonomy": "EXPERIMENT_DESIGN_FAILURE",
                "patch": {"kind": "research_harness", "target": "planner.strategy",
                          "old": "fixed_first_knob", "new": "diagnosis_targeted"},
                "hyp_fingerprint": "research_harness::a"}}
            return _payload(harness_accepted=[rec])

        counter = {"n": 0}
        def inherit(accepted, prev):
            counter["n"] += 1
            return f"harness_v{counter['n']}"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            report = run_loop(
                state_path=Path(tmp) / "loop_state.json",
                domain="embodied", baseline_config_path=str(cfg),
                project_dir=tmp, max_rounds=3, dataset_version="test-v1",
                domain_seeds=(50,), domain_runner=None,
                domain_require_frozen_manifest=False,
                harness_evolve=None, harness_inherit_fn=inherit,
                run_dual_layer_fn=stub)
            self.assertEqual(report["status"], "completed")
            # Round 1 had no inherited harness (base is None); rounds 2 and 3
            # received the threaded value from the previous round.
            self.assertEqual(seen_base[0], None)
            self.assertEqual(seen_base[1], "harness_v1")
            self.assertEqual(seen_base[2], "harness_v2")
            # Harness acceptance was recorded in every version.
            for v in report["versions"]:
                self.assertEqual(v["accepted_harness_targets"], ["planner.strategy"])


class TestCheckpointResume(unittest.TestCase):
    def test_resume_skips_already_completed_rounds(self):
        calls = []
        def stub(**kw):
            calls.append(1)
            return _payload(domain_accepted=[("train.lr", 0.01, 0.005, "accepted")])

        from research_runtime.loop.loop_state import LoopState, save

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            state_path = Path(tmp) / "loop_state.json"
            # Simulate an interrupted loop: round 1 finished and checkpointed,
            # but the process died before round 2. Resume must run ONLY rounds 2
            # and 3 (current_round == 2, status == running).
            pre = LoopState(
                loop_id="loop_resume", domain="embodied", max_rounds=3,
                current_round=2, status="running",
                baseline_config_path=str(cfg),
                dataset_version="test-v1", domain_seeds=[50])
            save(pre, state_path)

            report = run_loop(
                state_path=state_path,
                domain="embodied", baseline_config_path=str(cfg),
                project_dir=tmp, max_rounds=3, dataset_version="test-v1",
                domain_seeds=(50,), domain_runner=None,
                domain_require_frozen_manifest=False,
                harness_evolve=None, run_dual_layer_fn=stub)
            # Exactly the two remaining rounds (2 and 3) were executed; round 1
            # was NOT re-run.
            self.assertEqual(len(calls), 2)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["rounds_completed"], 2)


class TestRealHarnessInheritance(unittest.TestCase):
    """M2.5 P0: the loop threads the CONCRETE patched harness object (not a string
    tag, not the original base) into the next round, via the real
    ``default_harness_inherit`` primitive. Uses a stub dual-layer so the test is
    fast; the harness materialisation is the production code path.
    """

    def test_loop_threads_patched_harness_object_into_next_round(self):
        import os
        from research_runtime.arr_runtime import BASE_HARNESS
        from research_runtime.loop.version_inheritance import default_harness_inherit

        seen: list[Any] = []

        def stub(**kw):
            seen.append(kw.get("base_harness"))
            rec = {"candidate": {
                "candidate_id": "h1", "target": "planner.strategy",
                "old": "fixed_first_knob", "new": "diagnosis_targeted",
                "taxonomy": "EXPERIMENT_DESIGN_FAILURE",
                "patch": {"kind": "research_harness", "target": "planner.strategy",
                          "old": "fixed_first_knob", "new": "diagnosis_targeted"},
                "hyp_fingerprint": "research_harness::a"}}
            return _payload(harness_accepted=[rec])

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            cwd = os.getcwd()
            try:
                os.chdir(tmp)  # keep the harness sandbox inside temp dir
                report = run_loop(
                    state_path=Path(tmp) / "loop_state.json",
                    domain="embodied", baseline_config_path=str(cfg),
                    project_dir=tmp, max_rounds=3, dataset_version="test-v1",
                    domain_seeds=(50,), domain_runner=None,
                    domain_require_frozen_manifest=False,
                    harness_evolve=None,
                    base_harness=BASE_HARNESS,
                    harness_inherit_fn=default_harness_inherit,
                    run_dual_layer_fn=stub)
            finally:
                os.chdir(cwd)
            self.assertEqual(report["status"], "completed")
            # Round 1 starts from the real BASE_HARNESS object.
            self.assertIs(seen[0], BASE_HARNESS)
            # Round 2 receives the PATCHED harness object (real inheritance), not
            # the original base and not a string tag.
            self.assertIsNot(seen[1], BASE_HARNESS)
            self.assertNotEqual(seen[1].fingerprint(), BASE_HARNESS.fingerprint())
            # And round 3 carries an (again distinct) inherited object.
            self.assertIsNotNone(seen[2])
            self.assertNotEqual(seen[2].fingerprint(), BASE_HARNESS.fingerprint())


class TestRecoveryMismatch(unittest.TestCase):
    def test_resume_refuses_different_domain_checkpoint(self):
        from research_runtime.loop.recovery import LoopRecoveryError

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            state_path = Path(tmp) / "loop_state.json"
            # An OLD checkpoint from a DIFFERENT task (quant, not embodied).
            pre = LoopState(
                loop_id="old", domain="quant", max_rounds=3,
                current_round=1, status="running",
                baseline_config_path=str(cfg),
                dataset_version="robot_sim_v1", domain_seeds=[50])
            save(pre, state_path)
            with self.assertRaises(LoopRecoveryError):
                run_loop(
                    state_path=state_path,
                    domain="embodied", baseline_config_path=str(cfg),
                    project_dir=tmp, max_rounds=3, dataset_version="test-v1",
                    domain_seeds=(50,), domain_runner=None,
                    domain_require_frozen_manifest=False,
                    harness_evolve=None, run_dual_layer_fn=lambda **kw: {})


if __name__ == "__main__":
    unittest.main()
