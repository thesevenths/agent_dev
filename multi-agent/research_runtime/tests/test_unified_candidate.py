"""ARR 1.1 / F1+F2+F3 -- unified candidate protocol + dual-layer RRSI tests.

These are PURE / engine-free except where a real runner is injected. They prove:
* the shared lifecycle cannot be skipped or reversed;
* DOMAIN_MODEL candidates are name-spaced and scoped to a domain whitelist;
* the registry rejects frozen (evaluation-substrate) targets;
* config materialisation goes through the EXISTING CandidateConfigBuilder;
* the dual-layer controller keeps the two layers SEPARATE (distinct fingerprints,
  distinct records, frozen protected) and records an inheritable next-round.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_runtime.evolution.candidate import (
    CandidateStatus, DomainModelCandidate, CandidateType,
)
from research_runtime.evolution.candidate_registry import CandidateRegistry
from research_runtime.evolution.domain_evolver import (
    generate_domain_candidates, materialize_domain_candidate,
    evaluate_domain_candidate, DOMAIN_LAYER,
)
from research_runtime.evolution.evolution_controller import (
    run_dual_layer, DualLayerController,
)


class TestCandidateLifecycle(unittest.TestCase):
    def test_forward_transitions_allowed(self):
        s = CandidateStatus.PROPOSED
        for nxt in (CandidateStatus.CRITIC_APPROVED, CandidateStatus.SANDBOXED,
                    CandidateStatus.EVALUATED):
            self.assertTrue(s.can_transition_to(nxt))
            s = nxt

    def test_backward_transition_rejected(self):
        self.assertFalse(
            CandidateStatus.EVALUATED.can_transition_to(CandidateStatus.PROPOSED))

    def test_versioned_only_from_terminal(self):
        self.assertTrue(
            CandidateStatus.ACCEPTED.can_transition_to(CandidateStatus.VERSIONED))
        self.assertFalse(
            CandidateStatus.PROPOSED.can_transition_to(CandidateStatus.VERSIONED))

    def test_skip_stage_is_forbidden(self):
        # The whitelist forbids jumping over a mandatory stage.
        self.assertFalse(
            CandidateStatus.PROPOSED.can_transition_to(CandidateStatus.EVALUATED))
        self.assertFalse(
            CandidateStatus.CRITIC_APPROVED.can_transition_to(
                CandidateStatus.EVALUATED))

    def test_terminal_states_cannot_be_reversed(self):
        # A decision is final; terminal states must not convert into each other.
        self.assertFalse(
            CandidateStatus.ACCEPTED.can_transition_to(CandidateStatus.REJECTED))
        self.assertFalse(
            CandidateStatus.REJECTED.can_transition_to(CandidateStatus.ACCEPTED))
        self.assertFalse(
            CandidateStatus.INCONCLUSIVE.can_transition_to(
                CandidateStatus.ACCEPTED))

    def test_versioned_is_a_sink(self):
        # VERSIONED is reachable exactly once and has no outgoing edges.
        self.assertFalse(
            CandidateStatus.VERSIONED.can_transition_to(CandidateStatus.ACCEPTED))
        self.assertFalse(
            CandidateStatus.VERSIONED.can_transition_to(
                CandidateStatus.PROPOSED))

    def test_registry_rejects_illegal_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = CandidateRegistry(log_path=Path(tmp) / "reg.jsonl")
            c = DomainModelCandidate(
                candidate_id="x", domain="embodied",
                target="model.rollout_horizon", old=2.0, new=5.0,
                taxonomy="DECISION_UNCERTAINTY")
            reg.register(c)
            with self.assertRaises(ValueError):
                reg.transition(c.hyp_fingerprint, CandidateStatus.EVALUATED)


class TestDomainCandidate(unittest.TestCase):
    def test_fingerprint_is_namespaced(self):
        c = DomainModelCandidate(
            candidate_id="x", domain="embodied", target="model.rollout_horizon",
            old=2.0, new=5.0, taxonomy="DECISION_UNCERTAINTY")
        self.assertTrue(c.hyp_fingerprint.startswith("domain_model::"))
        # Distinct from any harness candidate (different namespace).
        self.assertNotEqual(CandidateType.RESEARCH_HARNESS.value, "domain_model")

    def test_generate_scoped_to_whitelist(self):
        # Embodied ALLOWED_KEYS include train.lr; dominant DECISION_UNCERTAINTY
        # maps to train.lr, so the generated candidate targets a real knob.
        cands = generate_domain_candidates(
            domain="embodied", dominant="DECISION_UNCERTAINTY", limit=3)
        self.assertTrue(cands, "expected at least one domain candidate")
        for c in cands:
            self.assertEqual(c.domain, "embodied")
            self.assertIn("train.lr", c.target)
            self.assertTrue(
                c.hyp_fingerprint.startswith("domain_model::"))

    def test_generate_empty_when_domain_has_no_whitelist(self):
        # SciNet++ exposes no ALLOWED_KEYS, so no DOMAIN_MODEL candidate can be
        # fabricated for it -- the loop must fall back to its harness layer.
        cands = generate_domain_candidates(
            domain="scinet", dominant="DECISION_UNCERTAINTY", limit=3)
        self.assertEqual(cands, [])


class TestRegistry(unittest.TestCase):
    def test_frozen_target_rejected(self):
        reg = CandidateRegistry()
        # 'evaluation_split' contains the frozen token 'evaluat' -> frozen.
        bad = DomainModelCandidate(
            candidate_id="bad", domain="embodied", target="evaluation_split",
            old=0, new=1, taxonomy="X")
        with self.assertRaises(ValueError):
            reg.register(bad)

    def test_register_and_transition_writes_audit_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "reg.jsonl"
            reg = CandidateRegistry(log_path=log)
            c = DomainModelCandidate(
                candidate_id="c1", domain="embodied",
                target="model.rollout_horizon", old=2.0, new=5.0,
                taxonomy="DECISION_UNCERTAINTY")
            reg.register(c)
            for st in (CandidateStatus.CRITIC_APPROVED, CandidateStatus.SANDBOXED,
                       CandidateStatus.EVALUATED, CandidateStatus.ACCEPTED,
                       CandidateStatus.VERSIONED):
                reg.transition(c.hyp_fingerprint, st)
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 6)  # register + 5 transitions
            self.assertTrue(all(line.startswith("{") for line in lines))


class TestMaterialize(unittest.TestCase):
    def _fixture(self, tmp: str) -> Path:
        cfg = Path(tmp) / "base.yaml"
        # Minimal baseline: a NESTED whitelisted key + seed + an (empty) data
        # section so the builder runs in require_frozen_manifest=False mode for
        # the test. CandidateConfigBuilder reads dotted keys into a nested mapping
        # (config["train"]["lr"]) -- the real embodied engine does the same -- so
        # the YAML must be nested, not a flat "train.lr" key.
        cfg.write_text(
            "seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
        return cfg

    def test_materialize_builds_isolated_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._fixture(tmp)
            c = DomainModelCandidate(
                candidate_id="m1", domain="embodied", target="train.lr",
                old=0.01, new=0.005, taxonomy="DECISION_UNCERTAINTY")
            arts = materialize_domain_candidate(
                domain="embodied", candidate=c, base_config=str(cfg),
                project_dir=tmp, campaign_id="t", seed=50,
                dataset_version="test-v1", require_frozen_manifest=False)
            self.assertIn("base", arts)
            self.assertIn("candidate", arts)
            # candidate config actually carries the override.
            import yaml
            cand_cfg = yaml.safe_load(
                Path(arts["candidate"]["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(cand_cfg["train"]["lr"], 0.005)
            self.assertEqual(cand_cfg["seed"], 50)


class TestEvaluateDomain(unittest.TestCase):
    def test_evaluate_requires_real_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text(
                "seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            c = DomainModelCandidate(
                candidate_id="e1", domain="embodied", target="train.lr",
                old=0.01, new=0.005, taxonomy="DECISION_UNCERTAINTY")
            with self.assertRaises(NotImplementedError):
                evaluate_domain_candidate(
                    domain="embodied", candidate=c, base_config=str(cfg),
                    project_dir=tmp, campaign_id="t", seeds=(50,),
                    dataset_version="test-v1", require_frozen_manifest=False)

    def test_evaluate_with_real_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text(
                "seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            c = DomainModelCandidate(
                candidate_id="e2", domain="embodied", target="train.lr",
                old=0.01, new=0.005, taxonomy="DECISION_UNCERTAINTY")
            # A real (injected) runner. Embodied's primary metric is state_mse@1
            # (minimize): the candidate arm (lr=0.005) yields a LOWER (=better)
            # value, the baseline arm (lr=0.01) a higher (=worse) value. The
            # improvement must be reported as POSITIVE despite the raw cand-base
            # delta being negative.
            def runner(art):
                import yaml
                cc = yaml.safe_load(
                    Path(art["config_path"]).read_text(encoding="utf-8"))
                return 0.7 if cc["train"]["lr"] == 0.005 else 0.9
            rec = evaluate_domain_candidate(
                domain="embodied", candidate=c, base_config=str(cfg),
                project_dir=tmp, campaign_id="t", seeds=(50, 51),
                dataset_version="test-v1", runner=runner,
                require_frozen_manifest=False)
            self.assertEqual(rec["base_Q"], [0.9, 0.9])
            self.assertEqual(rec["cand_Q"], [0.7, 0.7])
            self.assertEqual(rec["n_valid"], 2)
            self.assertEqual(rec["n_invalid"], 0)
            # Direction-normalized improvement > 0 for a lower (better) metric.
            self.assertGreater(rec["mean_improvement"], 0.0)
            # Raw delta (cand - base) is negative; kept for traceability only.
            self.assertLess(rec["deltas"][0], 0.0)


class TestDualLayerSeparation(unittest.TestCase):
    def _harness_stub(self, **kw):
        return {
            "schema_version": "stub",
            "dominant_failure": "DECISION_UNCERTAINTY",
            "records": [{"candidate": {"hyp_fingerprint": "research_harness::a",
                                       "target": "planner.strategy"},
                         "decision": "ACCEPT"}],
            "accepted": [{"candidate": {"hyp_fingerprint": "research_harness::a"}}],
            "n_accepted": 1,
        }

    def test_controller_keeps_layers_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text(
                "seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            payload = run_dual_layer(
                base_harness=None, tasks=[], heldout_task=None,
                run_root=tmp, repo_root=tmp, rounds=1, max_minutes=1.0,
                domain="embodied", domain_base_config=str(cfg),
                project_dir=tmp, dataset_version="test-v1",
                domain_seeds=(50, 51),
                domain_require_frozen_manifest=False,
                domain_runner=lambda art: 0.7 if "baseline" not in art["directory"] else 0.9,
                harness_evolve=self._harness_stub)
            self.assertIn("harness_layer", payload)
            self.assertIn("domain_layer", payload)
            self.assertTrue(payload["separation"]["candidate_types_distinct"])
            self.assertTrue(payload["separation"]["ledger_namespaced"])
            # Domain layer produced real, name-spaced candidate records.
            dom = payload["domain_layer"]
            self.assertTrue(dom["records"])
            for r in dom["records"]:
                self.assertTrue(
                    r["hyp_fingerprint"].startswith("domain_model::"))
                # The injected runner yields a strictly LOWER (better) minimize
                # metric for the candidate arm -> positive normalized improvement
                # -> ACCEPTED. This proves the decision is computed from the REAL
                # direction-aware metric, not a fabricated pass.
                self.assertEqual(r["decision"], "accepted")
                self.assertGreater(r["mean_improvement"], 0.0)
            # Harness layer's fingerprint lives in a different namespace.
            hfp = payload["harness_layer"]["records"][0]["candidate"]["hyp_fingerprint"]
            self.assertTrue(hfp.startswith("research_harness::"))
            # Both layers contributed accepted work -> next round inherits.
            self.assertTrue(payload["version_inheritance"]["inheritable_next_round"])
            self.assertEqual(payload["version_inheritance"]["domain_accepted"],
                             len(dom["records"]))


class TestDomainDecision(unittest.TestCase):
    """Black-box checks that the dual-layer DOMAIN decision is correct and honest.

    These exercise ``run_dual_layer`` (real evaluate + real decide) with injected
    engine runners -- no FakeRunner, no fabricated metric.
    """

    def _harness_stub(self, **kw):
        return {
            "schema_version": "stub",
            "dominant_failure": "DECISION_UNCERTAINTY",
            "records": [{"candidate": {"hyp_fingerprint": "research_harness::a",
                                       "target": "planner.strategy"},
                         "decision": "ACCEPT"}],
            "accepted": [{"candidate": {"hyp_fingerprint": "research_harness::a"}}],
            "n_accepted": 1,
        }

    def _run(self, runner):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text(
                "seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            payload = run_dual_layer(
                base_harness=None, tasks=[], heldout_task=None,
                run_root=tmp, repo_root=tmp, rounds=1, max_minutes=1.0,
                domain="embodied", domain_base_config=str(cfg),
                project_dir=tmp, dataset_version="test-v1",
                domain_seeds=(50, 51),
                domain_require_frozen_manifest=False,
                domain_runner=runner, harness_evolve=self._harness_stub)
            return payload["domain_layer"]["records"]

    def test_minimize_direction_accepts_real_improvement(self):
        # Embodied metric is minimized: candidate lower -> better -> ACCEPTED.
        recs = self._run(lambda art: 0.7 if "baseline" not in art["directory"] else 0.9)
        self.assertTrue(recs)
        for r in recs:
            self.assertEqual(r["decision"], "accepted")
            self.assertGreater(r["mean_improvement"], 0.0)

    def test_minimize_direction_rejects_real_degradation(self):
        # Candidate HIGHER (worse) for a minimize metric must be REJECTED, never
        # ACCEPTED -- this is the exact inversion the review flagged.
        recs = self._run(lambda art: 0.9 if "baseline" not in art["directory"] else 0.8)
        self.assertTrue(recs)
        for r in recs:
            self.assertEqual(r["decision"], "rejected")
            self.assertLess(r["mean_improvement"], 0.0)

    def test_failed_run_is_inconclusive_not_rejected(self):
        # NaN (engine failure / missing metric) must become INCONCLUSIVE, never a
        # fabricated REJECT that would pretend the candidate was genuinely worse.
        recs = self._run(lambda art: float("nan"))
        self.assertTrue(recs)
        for r in recs:
            self.assertEqual(r["decision"], "inconclusive")
            self.assertEqual(r["n_invalid"], 2)

    def test_single_seed_is_inconclusive(self):
        # One paired run cannot separate signal from noise -> INCONCLUSIVE.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text(
                "seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n", encoding="utf-8")
            payload = run_dual_layer(
                base_harness=None, tasks=[], heldout_task=None,
                run_root=tmp, repo_root=tmp, rounds=1, max_minutes=1.0,
                domain="embodied", domain_base_config=str(cfg),
                project_dir=tmp, dataset_version="test-v1",
                domain_seeds=(50,),
                domain_require_frozen_manifest=False,
                domain_runner=lambda art: 0.7 if "baseline" not in art["directory"] else 0.9,
                harness_evolve=self._harness_stub)
            recs = payload["domain_layer"]["records"]
            self.assertTrue(recs)
            for r in recs:
                self.assertEqual(r["decision"], "inconclusive")
                self.assertEqual(r["n_valid"], 1)


if __name__ == "__main__":
    unittest.main()
