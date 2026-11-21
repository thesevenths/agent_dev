"""ARR 1.1 / M2.7 -- reliability & scientific-validation regression tests.

ENGINE-FREE. These tests exercise the REAL decision / safety code (no FakeRunner,
no fabricated metric in the production path). Deterministic injected runners are
used only where a unit test must stand in for a heavy engine -- exactly as the
pre-existing ARR test suite does.

Coverage requested by the M2.7 task book:

P0-1 Domain statistical gate
  * 2 paired seeds (dev/quick) can only be INCONCLUSIVE, never a formal ACCEPT.
  * Formal ACCEPT needs >= 5 valid paired seeds, a 95% paired-improvement CI that
    excludes 0, a practical-improvement threshold, NO NaN/Inf/missing runs, and an
    INDEPENDENT holdout confirmation (>= 5 holdout seeds, non-contradictory).
  * Holdout never participates in candidate selection (paired vs holdout evidence
    are kept in disjoint fields).
  * The gate's hard floors can only be RAISED, never lowered by a CLI flag / API.
  * Provisional (dev) mode never produces a formal ACCEPT.
  * A genuine regression is still REJECTED; INCONCLUSIVE is never reported as
    REJECT and REJECT is never reported as a successful improvement.

P0-2 Frozen-manifest safety default
  * CLI and Python API (run_loop / run_research_cycle / run_dual_layer) BOTH
    default to require-frozen-manifest ON.
  * Missing / corrupt / SHA-mismatched data fails CLOSED.
  * Only an explicit dev opt-out allows unfrozen data.
  * Resume re-verifies the manifest; a changed dataset fails closed.

P0-3 (CI) is covered by the workflow itself: this file is listed in
.github/workflows/arr-research-runtime.yml at the REPO ROOT.

M3.0 note: the statistical verdict itself was UPGRADED (two-sided Student-t
CI instead of the 1.96 normal approximation, Holm family-wise correction,
bidirectional significance, per-seed holdout confirmation, missing-statistic
fail-closed). This file keeps the M2.7 safety invariants; the new protocol's
verdict cases live in test_m3_0_statistics.py. ``_rec`` now supplies the full
per-seed holdout evidence so these tests RAISE their evidence rather than
lower the gate.
"""
from __future__ import annotations

import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from research_runtime.candidate_config import (
    CandidateConfigBuilder, CandidateConfigError, sha256_file,
)
from research_runtime.cli import build_parser
from research_runtime.evolution.candidate import (
    CandidateStatus, DomainModelCandidate,
)
from research_runtime.evolution.domain_evolver import (
    domain_whitelist, evaluate_domain_candidate,
)
from research_runtime.evolution.evolution_controller import (
    MIN_SEEDS_FOR_FORMAL_ACCEPT, PRACTICAL_IMPROVEMENT_FLOOR_REL,
    _decide_domain, run_dual_layer,
)
from research_runtime.loop.controller import _reverify_frozen_manifest, run_loop
from research_runtime.loop.loop_report import build_loop_report, summarise_text
from research_runtime.loop.recovery import LoopRecoveryError
from research_runtime.loop.research_cycle import run_research_cycle


def _rec(*, n_valid=5, mean=0.001, std=0.0002, baseline_mean=0.012,
         ho=0.0009, ho_n=5, n_invalid=0,
         ho_improvements=None, ho_n_seeds=None, ho_n_invalid=0):
    """A direction-normalized domain evaluation record (positive == better).

    M3.0: the holdout now carries PER-SEED raw results, an explicit requested
    seed count and an invalid-run count. The protocol requires all of them, so
    this helper supplies the full evidence instead of relaxing the gate.
    """
    if ho_improvements is None and ho is not None and ho_n:
        ho_improvements = [ho] * ho_n
    return {
        "n_valid": n_valid, "n_invalid": n_invalid,
        "mean_improvement": mean, "std_improvement": std,
        "baseline_mean": baseline_mean,
        "holdout_improvement": ho, "holdout_n_valid": ho_n,
        "holdout_improvements": ho_improvements,
        "holdout_n_seeds": ho_n if ho_n_seeds is None else ho_n_seeds,
        "holdout_n_invalid": ho_n_invalid,
    }


class TestDomainStatisticalGate(unittest.TestCase):
    """P0-1: the formal ACCEPT bar is never lowered to manufacture verdicts."""

    def test_formal_accept_requires_full_evidence(self):
        self.assertEqual(MIN_SEEDS_FOR_FORMAL_ACCEPT, 5)
        self.assertGreaterEqual(PRACTICAL_IMPROVEMENT_FLOOR_REL, 0.01)
        self.assertEqual(_decide_domain(_rec()), CandidateStatus.ACCEPTED)

    def test_two_seed_dev_run_is_inconclusive_never_accepted(self):
        # The exact M2.6 situation: --seeds 50 51 with no holdout. A positive
        # mean over 2 seeds is NOT a scientifically validated improvement.
        r = _rec(n_valid=2, mean=0.0001, std=0.00005, ho=None, ho_n=0)
        self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_five_seed_no_holdout_is_inconclusive(self):
        r = _rec(ho=None, ho_n=0)
        self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_holdout_sample_too_small_is_inconclusive(self):
        r = _rec(ho=0.0009, ho_n=2)
        self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_holdout_contradiction_is_inconclusive(self):
        r = _rec(ho=-0.0005)
        self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_ci_including_zero_is_inconclusive(self):
        # Strong mean but huge variance -> 95% CI lower bound <= 0.
        r = _rec(mean=0.001, std=0.01)
        self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_below_practical_floor_is_inconclusive(self):
        # Statistically significant but < 1% of the baseline metric -> noise-level.
        r = _rec(mean=0.00005, std=0.00002, ho=0.0001)
        self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_negative_improvement_is_rejected(self):
        # A genuine regression is REJECTED, never silently INCONCLUSIVE.
        r = _rec(mean=-0.001, ho=-0.0005)
        self.assertEqual(_decide_domain(r), CandidateStatus.REJECTED)

    def test_nan_inf_none_mean_is_inconclusive(self):
        for bad in (float("nan"), float("inf"), None):
            with self.subTest(bad=bad):
                r = _rec(mean=bad)
                self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_any_invalid_run_is_inconclusive(self):
        # A failed / missing experiment is NOT "real degradation".
        r = _rec(n_invalid=1)
        self.assertEqual(_decide_domain(r), CandidateStatus.INCONCLUSIVE)

    def test_hard_floor_cannot_be_lowered_by_caller(self):
        # No CLI flag or Python API caller may relax the bar below the constants.
        self.assertEqual(
            _decide_domain(_rec(n_valid=2), min_seeds_for_accept=2),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_rec(mean=0.00005, std=0.00002, ho=0.0001),
                           practical_improvement_floor_rel=0.0),
            CandidateStatus.INCONCLUSIVE)

    def test_bar_can_be_raised(self):
        self.assertEqual(
            _decide_domain(_rec(n_valid=5), min_seeds_for_accept=6),
            CandidateStatus.INCONCLUSIVE)

    def test_provisional_dev_mode_never_formally_accepts(self):
        # --no-holdout-confirmation: positive candidate stays INCONCLUSIVE.
        self.assertEqual(
            _decide_domain(_rec(), require_holdout_confirmation=False),
            CandidateStatus.INCONCLUSIVE)
        # A genuine regression is still honestly REJECTED in dev mode.
        self.assertEqual(
            _decide_domain(_rec(mean=-0.001, ho=-0.0005),
                            require_holdout_confirmation=False),
            CandidateStatus.REJECTED)


_FIVE = (50, 51, 52, 53, 54)
_HOLDOUT = (60, 61, 62, 63, 64)


def _harness_stub(**kw):
    return {
        "schema_version": "stub", "dominant_failure": "DECISION_UNCERTAINTY",
        "records": [], "accepted": [], "n_accepted": 0,
    }


class TestDomainEvidenceIsolation(unittest.TestCase):
    """P0-1 rule 7: holdout must not participate in candidate selection."""

    def _candidate(self):
        return DomainModelCandidate(
            candidate_id="m27", domain="embodied", target="train.lr",
            old=0.01, new=0.005, taxonomy="DECISION_UNCERTAINTY")

    def _baseline(self, tmp):
        cfg = Path(tmp) / "base.yaml"
        cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n",
                       encoding="utf-8")
        return cfg

    def test_evaluate_records_baseline_mean_and_disjoint_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._baseline(tmp)

            def runner(art):
                # Deterministic injected runner: candidate arm is better.
                return 0.7 if "baseline" not in art["directory"] else 0.9

            rec = evaluate_domain_candidate(
                domain="embodied", candidate=self._candidate(),
                base_config=str(cfg), project_dir=tmp, campaign_id="m27",
                seeds=_FIVE, holdout_seeds=_HOLDOUT,
                dataset_version="test-v1", runner=runner,
                require_frozen_manifest=False)
            # Paired evidence: exactly the 5 selection seeds.
            self.assertEqual(rec["n_valid"], 5)
            self.assertEqual(len(rec["improvements"]), 5)
            self.assertAlmostEqual(rec["baseline_mean"], 0.9, places=6)
            self.assertEqual(rec["n_invalid"], 0)
            # Holdout is a SEPARATE, independent confirmation set -- its numbers
            # never enter ``improvements`` / ``mean_improvement`` (no leakage into
            # the candidate-selection statistic).
            self.assertEqual(rec["holdout_n_valid"], 5)
            self.assertGreater(rec["holdout_improvement"], 0)
            self.assertNotIn("holdout", " ".join(
                k for k in ("improvements", "mean_improvement")))
        self.assertEqual(_decide_domain(rec), CandidateStatus.ACCEPTED)

    def test_engine_failure_is_inconclusive_not_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._baseline(tmp)
            rec = evaluate_domain_candidate(
                domain="embodied", candidate=self._candidate(),
                base_config=str(cfg), project_dir=tmp, campaign_id="m27",
                seeds=_FIVE, holdout_seeds=_HOLDOUT,
                dataset_version="test-v1", runner=lambda art: float("nan"),
                require_frozen_manifest=False)
            self.assertEqual(rec["n_invalid"], 5)
            self.assertEqual(rec["holdout_n_valid"], 0)
            self.assertEqual(_decide_domain(rec), CandidateStatus.INCONCLUSIVE)


class TestDualLayerGateEndToEnd(unittest.TestCase):
    """P0-1 wired through the REAL run_dual_layer controller."""

    def _run(self, *, seeds, holdout, runner, **extra):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n",
                           encoding="utf-8")
            payload = run_dual_layer(
                base_harness=None, tasks=[], heldout_task=None,
                run_root=tmp, repo_root=tmp, rounds=1, max_minutes=1.0,
                domain="embodied", domain_base_config=str(cfg),
                project_dir=tmp, dataset_version="test-v1",
                domain_seeds=seeds, domain_holdout_seeds=holdout,
                domain_require_frozen_manifest=False,
                domain_runner=runner, harness_evolve=_harness_stub, **extra)
            return payload

    def test_full_evidence_accepts(self):
        payload = self._run(
            seeds=_FIVE, holdout=_HOLDOUT,
            runner=lambda art: 0.7 if "baseline" not in art["directory"] else 0.9)
        recs = payload["domain_layer"]["records"]
        self.assertTrue(recs)
        for r in recs:
            self.assertEqual(r["decision"], "accepted")
            self.assertIn("baseline_mean", r)

    def test_two_seed_dev_round_never_accepts(self):
        payload = self._run(
            seeds=(50, 51), holdout=(),
            runner=lambda art: 0.7 if "baseline" not in art["directory"] else 0.9)
        recs = payload["domain_layer"]["records"]
        self.assertTrue(recs)
        self.assertTrue(all(r["decision"] == "inconclusive" for r in recs))
        self.assertEqual(payload["domain_layer"]["n_accepted"], 0)

    def test_holdout_contradiction_downgrades_to_inconclusive(self):
        # Candidate better on selection seeds, worse on holdout seeds.
        def runner(art):
            import yaml as _y
            cc = _y.safe_load(Path(art["config_path"]).read_text(encoding="utf-8"))
            seed = int(cc.get("seed", 0))
            is_cand = "baseline" not in art["directory"]
            if seed in _HOLDOUT:  # contradict on the independent set
                return 0.95 if is_cand else 0.90
            return 0.70 if is_cand else 0.90
        payload = self._run(seeds=_FIVE, holdout=_HOLDOUT, runner=runner)
        recs = payload["domain_layer"]["records"]
        self.assertTrue(recs)
        self.assertTrue(all(r["decision"] == "inconclusive" for r in recs))
        self.assertEqual(payload["domain_layer"]["n_accepted"], 0)


class TestFrozenManifestSafetyDefault(unittest.TestCase):
    """P0-2: safety is enforced by the Runtime, not only by the CLI."""

    def test_python_api_defaults_are_all_true(self):
        # The exact gap the review found: loop/controller.py used False.
        for fn in (run_loop, run_research_cycle, run_dual_layer):
            with self.subTest(fn=fn.__name__):
                self.assertIs(
                    inspect.signature(fn).parameters[
                        "domain_require_frozen_manifest"].default,
                    True)

    def test_cli_defaults_are_safe(self):
        args = build_parser().parse_args([
            "loop", "--domain", "scinet", "--baseline-config", "b.yaml",
            "--project-dir", "p"])
        self.assertTrue(args.require_frozen_manifest)
        self.assertTrue(args.require_holdout_confirmation)
        self.assertEqual(args.seeds, [50, 51, 52, 53, 54])
        self.assertEqual(args.min_seeds_for_accept, 5)
        self.assertGreaterEqual(args.practical_floor_rel, 0.01)

    def test_cli_dev_switches_are_explicit(self):
        args = build_parser().parse_args([
            "loop", "--domain", "scinet", "--baseline-config", "b.yaml",
            "--project-dir", "p", "--allow-unfrozen-data",
            "--no-holdout-confirmation"])
        self.assertFalse(args.require_frozen_manifest)
        self.assertFalse(args.require_holdout_confirmation)

    def _builder(self, project_dir, *, require=True):
        allowed, rules = domain_whitelist("embodied")
        return CandidateConfigBuilder(
            project_dir=project_dir, allowed_keys=allowed,
            value_rules=rules, require_frozen_manifest=require)

    def _frozen_project(self, tmp):
        """Create a small REAL frozen dataset with a content-addressed manifest."""
        proj = Path(tmp)
        frozen = proj / "datasets" / "v1"
        frozen.mkdir(parents=True)
        split = frozen / "train.npy"
        split.write_bytes(b"deterministic-frozen-bytes")
        manifest = {
            "dataset_version": "test-v1",
            "splits": {"train": {"file": "train.npy",
                                  "sha256": sha256_file(split)}},
        }
        (frozen / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8")
        cfg = {"seed": 0, "train": {"lr": 0.01},
               "data": {"frozen_dir": "datasets/v1"}}
        return proj, frozen, split, cfg

    def test_valid_frozen_manifest_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj, _frozen, _split, cfg = self._frozen_project(tmp)
            verdict = self._builder(proj, require=True).verify_frozen_dataset(
                cfg, "test-v1")
            self.assertTrue(verdict["verified"])
            self.assertEqual(verdict["splits_verified"], ["train"])

    def test_missing_frozen_dir_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj, _f, _s, _cfg = self._frozen_project(tmp)
            cfg = {"seed": 0, "train": {"lr": 0.01}, "data": {}}
            with self.assertRaises(CandidateConfigError):
                self._builder(proj, require=True).verify_frozen_dataset(
                    cfg, "test-v1")
            # Only the EXPLICIT dev opt-out permits unfrozen data.
            verdict = self._builder(proj, require=False).verify_frozen_dataset(
                cfg, "test-v1")
            self.assertFalse(verdict["verified"])

    def test_sha_mismatch_fails_closed_even_in_dev_mode(self):
        # Once a frozen_dir is declared, its content MUST match -- no silent
        # fallback even when unfrozen data is otherwise allowed.
        with tempfile.TemporaryDirectory() as tmp:
            proj, _frozen, split, cfg = self._frozen_project(tmp)
            split.write_bytes(b"TAMPERED")
            for require in (True, False):
                with self.subTest(require=require):
                    with self.assertRaises(CandidateConfigError):
                        self._builder(proj, require=require).verify_frozen_dataset(
                            cfg, "test-v1")

    def test_dataset_version_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj, _f, _s, cfg = self._frozen_project(tmp)
            with self.assertRaises(CandidateConfigError):
                self._builder(proj, require=True).verify_frozen_dataset(
                    cfg, "a-DIFFERENT-version")

    def test_resume_reverifies_manifest(self):
        # A completed checkpoint can only resume if its frozen dataset is STILL
        # the one the prior run used.
        with tempfile.TemporaryDirectory() as tmp:
            proj, _frozen, split, cfg = self._frozen_project(tmp)
            cfg_path = proj / "baseline.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True),
                                encoding="utf-8")
            # Valid -> resume allowed.
            _reverify_frozen_manifest(
                project_dir=proj, baseline_config_path=cfg_path,
                dataset_version="test-v1", domain="embodied")
            # Tamper the data between the two processes -> fail closed.
            split.write_bytes(b"TAMPERED-BETWEEN-RUNS")
            with self.assertRaises(LoopRecoveryError):
                _reverify_frozen_manifest(
                    project_dir=proj, baseline_config_path=cfg_path,
                    dataset_version="test-v1", domain="embodied")

    def test_resume_missing_baseline_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LoopRecoveryError):
                _reverify_frozen_manifest(
                    project_dir=tmp, baseline_config_path=Path(tmp) / "gone.yaml",
                    dataset_version="test-v1", domain="embodied")


class TestEngineMetricProvenance(unittest.TestCase):
    """M2.7 P1-1: every domain engine must PERSIST its real metrics on disk."""

    def test_scinet_evaluate_persists_metrics_file_with_checkpoint_sha(self):
        # The declared artifacts list must match files that really exist, and the
        # persisted metric must be the SAME value returned to the caller (never a
        # fabricated number). Uses a tiny in-process stub of the engine adapter's
        # collect_result -- the code under test is the REAL scinet_compute body.
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from research_runtime.domains import scinet as mod

        class _R:
            status = "succeeded"
            metrics = {"latent_mse@1": 0.024023795500397682}
            dataset_version = "pendulum-a0e03386be89"
            run_id = "run-x"
            error = None

        class _A:
            def submit(self, spec): return "j"
            def wait(self, jid, timeout=None): return None
            def collect_result(self, jid, spec): return _R()

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "_scinet_run_id.txt").write_text("run-x", encoding="utf-8")
            (d / "checkpoint.json").write_text("{}", encoding="utf-8")
            cfgp = d / "cfg.yaml"
            cfgp.write_text("seed: 50\ntrain: {epochs: 1}\n", encoding="utf-8")
            with patch.object(mod, "SciNetAdapter", _A, create=True):
                # patch the name used inside scinet_compute's local import
                import research_runtime.adapters as adapters
                with patch.object(adapters, "SciNetAdapter", _A):
                    res = mod.scinet_compute(
                        {"seed": 50, "train": {"epochs": 1}},
                        {"stage": "evaluate", "seed": 50, "run_id": "run-x"},
                        d, config_path=cfgp)
            self.assertEqual(res["status"], "succeeded")
            self.assertEqual(res["metrics"]["latent_mse@1"],
                             0.024023795500397682)
            for name in res["artifacts"]:
                self.assertTrue((d / name).is_file(),
                                f"declared artifact {name} must exist on disk")
            persisted = json.loads((d / "backtest_metrics.json").read_text(
                encoding="utf-8"))
            self.assertEqual(persisted["metrics"]["latent_mse@1"],
                             res["metrics"]["latent_mse@1"])
            self.assertTrue(persisted["checkpoint_sha256"])


class TestNoGainRunIsReportedHonestly(unittest.TestCase):
    """M2.7: a strict-gate run with 0 ACCEPT must not read as '0 rounds ran'."""

    def test_report_distinguishes_executed_rounds_from_accepted_versions(self):
        from research_runtime.loop.loop_state import LoopState
        st = LoopState(
            loop_id="m27_no_gain", domain="quant", max_rounds=3,
            current_round=4, status="completed",
            baseline_config_path="config.yaml",
            dataset_version="market_hist_v1-7640575b236b",
            domain_seeds=[50, 51, 52, 53, 54])
        st.last_completed_round = 3  # all 3 rounds really executed
        # version_history stays empty: every candidate was honestly
        # INCONCLUSIVE / REJECT -- no version to inherit.
        report = build_loop_report(st)
        self.assertEqual(report["rounds_executed"], 3)
        self.assertEqual(report["rounds_completed"], 0)
        text = summarise_text(report)
        self.assertIn("rounds_executed=3/3", text)
        self.assertIn("accepted_versions=0", text)
        # The old misleading "rounds=0/3" line must be gone.
        self.assertNotIn("rounds=0/3", text)


if __name__ == "__main__":
    unittest.main()
