"""ARR 1.1 / M3.3 -- OOD (regime-shift) axis regression tests.

Two independent pieces are covered:

1. DATA HONESTY (engine-free file checks): the OOD dataset is a REAL regime
   shift, not an IID re-read. ARCHITECTURE_AUDIT.md forbids declaring IID
   same-generator re-splits as OOD; the genuine signal here is a changed physics
   knob (``t_max`` 8.0 -> 14.0) on the SAME generator, yielding a distinct
   content-addressed ``dataset_version``. We assert the manifest facts and SHA
   integrity without importing the torch engine.

2. GATE INDEPENDENCE (engine-free, injected runners): the OOD axis reuses the
   EXACT M3.0 statistical gate via the ``prefix`` parameter. Tests prove the
   namespaced ``ood_`` axis is decided on its own evidence and combined into the
   production verdict by ``_combine_dual_layer_decisions`` (ID ACCEPT + OOD
   ACCEPT -> ACCEPT; ID ACCEPT + OOD REJECT -> REJECT as ID-overfit; ID ACCEPT +
   OOD INCONCLUSIVE -> INCONCLUSIVE).

No FakeRunner in the production path: the integration test uses a deterministic
surrogate runner that writes BOTH the ID (return) and OOD (file) metrics exactly
like ``scinet_compute`` does, so the OOD number read off disk is genuine.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_runtime.evolution.candidate import CandidateStatus
from research_runtime.evolution.domain_evolver import evaluate_domain_candidate
from research_runtime.evolution.evolution_controller import (
    DEFAULT_ALPHA, HOLDOUT_DIRECTION_MIN_FRACTION, MIN_SEEDS_FOR_FORMAL_ACCEPT,
    _combine_dual_layer_decisions, _decide_domain, domain_family_significance,
    run_dual_layer,
)
from research_runtime.settings import DEFAULT_SCINET_DIR


_OOD_DATASET_DIR = (DEFAULT_SCINET_DIR / "datasets" / "pendulum_smoke_ood_v1")
_OOD_CONFIG = (DEFAULT_SCINET_DIR / "configs" / "pendulum_smoke_ood.yaml")
_ID_DATASET_DIR = (Path(__file__).resolve().parents[2] / "M2_7_runs" / "scinet"
                   / "datasets" / "pendulum_smoke_v1")

_FIVE = (50, 51, 52, 53, 54)
_HOLDOUT = (60, 61, 62, 63, 64)


class TestM3_3OodDatasetIsRegimeShiftNotIID(unittest.TestCase):
    """Engine-free verification that the OOD set is a genuine shift.

    Skipped (not failed) when the external SciNet++ checkout is absent, so the
    suite stays runnable without the engine; when present, it pins the artifact.
    """

    def test_ood_dataset_present_and_content_addressed(self):
        if not _OOD_DATASET_DIR.is_dir():
            self.skipTest(f"OOD dataset not present: {_OOD_DATASET_DIR}")
        manifest = json.loads(
            (_OOD_DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_version"], "pendulum-a884d327551d")
        # SHA integrity: no tampered / stale split file.
        for split, info in manifest["splits"].items():
            p = _OOD_DATASET_DIR / info["file"]
            self.assertTrue(p.is_file(), f"missing split {split}")
            import hashlib
            self.assertEqual(
                hashlib.sha256(p.read_bytes()).hexdigest(), info["sha256"],
                f"SHA mismatch for split {split}")

    def test_ood_is_real_regime_shift_not_iid(self):
        if not _OOD_DATASET_DIR.is_dir():
            self.skipTest(f"OOD dataset not present: {_OOD_DATASET_DIR}")
        ood = json.loads(
            (_OOD_DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
        ood_spec = ood["data_spec"]
        self.assertEqual(ood_spec["t_max"], 14.0,
                         "OOD must shift the physics knob t_max")
        # The ID set lives under the engine too; compare its t_max.
        id_dir = DEFAULT_SCINET_DIR / "datasets" / "pendulum_smoke_v1"
        if id_dir.is_dir():
            id_manifest = json.loads(
                (id_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(id_manifest["data_spec"]["t_max"], 8.0)
            # Distinct content-addressed version == NOT IID.
            self.assertNotEqual(
                id_manifest["dataset_version"], ood["dataset_version"])

    def test_ood_config_freezes_distinct_version(self):
        if not _OOD_CONFIG.is_file():
            self.skipTest(f"OOD config not present: {_OOD_CONFIG}")
        import yaml
        cfg = yaml.safe_load(_OOD_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(cfg["data"]["t_max"], 14.0)
        self.assertEqual(cfg["data"]["frozen_dir"],
                         "datasets/pendulum_smoke_ood_v1")
        # The data shape is unchanged -> validate_config still passes; only the
        # regime knob t_max moved -> a NEW dataset_version.
        self.assertEqual(cfg["data"]["sequence_length"], 80)
        self.assertEqual(cfg["data"]["block_size"], 20)


class TestM3_3OodAxisIndependentGate(unittest.TestCase):
    """The OOD axis reuses the SAME M3.0 gate via the ``prefix`` parameter."""

    def _full_rec_ood(self, *, n_valid=5, mean=0.001, std=0.0002,
                      baseline_mean=0.012, ho=0.0009, ho_n=5, n_invalid=0,
                      ho_improvements=None, ho_n_seeds=None, ho_n_invalid=0):
        if ho_improvements is None and ho is not None and ho_n:
            ho_improvements = [ho] * ho_n
        ho_pos = (sum(1 for x in ho_improvements if x > 0)
                  if ho_improvements else 0)
        return {
            "ood_n_valid": n_valid, "ood_n_invalid": n_invalid,
            "ood_mean_improvement": mean, "ood_std_improvement": std,
            "ood_baseline_mean": baseline_mean,
            "ood_holdout_improvement": ho, "ood_holdout_n_valid": ho_n,
            "ood_holdout_n_seeds": ho_n if ho_n_seeds is None else ho_n_seeds,
            "ood_holdout_n_invalid": ho_n_invalid,
            "ood_holdout_improvements": ho_improvements,
            "ood_holdout_n_positive": ho_pos,
            "ood_holdout_direction_fraction": (
                round(ho_pos / len(ho_improvements), 6)
                if ho_improvements else None),
        }

    @staticmethod
    def _drop(rec, *keys):
        out = dict(rec)
        for k in keys:
            out.pop(k, None)
        return out

    def test_ood_positive_is_accepted(self):
        self.assertEqual(_decide_domain(self._full_rec_ood(), prefix="ood_"),
                         CandidateStatus.ACCEPTED)

    def test_ood_negative_is_rejected(self):
        rec = self._full_rec_ood(mean=-0.001, ho=-0.0005)
        self.assertEqual(_decide_domain(rec, prefix="ood_"),
                         CandidateStatus.REJECTED)

    def test_ood_ci_straddles_is_inconclusive(self):
        # Same numbers as M3.0's CI-straddle case, on the ood_ axis.
        rec = self._full_rec_ood(mean=0.016844, std=0.014152,
                                 baseline_mean=0.023939, ho=0.029626)
        self.assertEqual(_decide_domain(rec, prefix="ood_"),
                         CandidateStatus.INCONCLUSIVE)

    def test_ood_missing_stat_fails_closed(self):
        self.assertEqual(
            _decide_domain(self._full_rec_ood(std=None), prefix="ood_"),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(self._drop(self._full_rec_ood(), "ood_baseline_mean"),
                          prefix="ood_"),
            CandidateStatus.INCONCLUSIVE)

    def test_ood_missing_holdout_fails_closed(self):
        self.assertEqual(
            _decide_domain(self._drop(self._full_rec_ood(),
                                     "ood_holdout_improvements"), prefix="ood_"),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(self._drop(self._full_rec_ood(), "ood_holdout_n_seeds"),
                          prefix="ood_"),
            CandidateStatus.INCONCLUSIVE)

    def test_ood_holdout_contradiction_is_inconclusive(self):
        rec = self._full_rec_ood(
            ho=0.0001,
            ho_improvements=[0.01, -0.001, -0.001, -0.001, -0.001])
        self.assertEqual(_decide_domain(rec, prefix="ood_"),
                         CandidateStatus.INCONCLUSIVE)

    def test_ood_family_holm_independent_of_id(self):
        # Strong, well-separated effect on the OOD axis (mean=0.020, std=0.003,
        # n=5 -> t ~ 14.9, p ~ 1e-9). The point of this test is that the OOD
        # family is corrected INDEPENDENTLY of the ID family, not that a
        # marginal effect sneaks past Holm; using a clear effect avoids the
        # M3.0 "lucky best winner" floor the gate is designed to catch.
        recs = [self._full_rec_ood(mean=0.020, std=0.003) for _ in range(3)]
        holm = domain_family_significance(recs, prefix="ood_")
        self.assertEqual([h["n_compared"] for h in holm], [3, 3, 3])
        for h in holm:
            self.assertTrue(h["significant"])

    def test_combine_id_accept_ood_accept(self):
        self.assertEqual(_combine_dual_layer_decisions(
            CandidateStatus.ACCEPTED, CandidateStatus.ACCEPTED),
            CandidateStatus.ACCEPTED)

    def test_combine_id_accept_ood_reject_is_rejected(self):
        # ID-overfit: improves the training regime but HARMS the deployment
        # regime -> rejected, never shipped.
        self.assertEqual(_combine_dual_layer_decisions(
            CandidateStatus.ACCEPTED, CandidateStatus.REJECTED),
            CandidateStatus.REJECTED)

    def test_combine_id_accept_ood_inconclusive_is_inconclusive(self):
        self.assertEqual(_combine_dual_layer_decisions(
            CandidateStatus.ACCEPTED, CandidateStatus.INCONCLUSIVE),
            CandidateStatus.INCONCLUSIVE)

    def test_combine_id_not_accepted_is_unchanged(self):
        self.assertEqual(_combine_dual_layer_decisions(
            CandidateStatus.REJECTED, CandidateStatus.ACCEPTED),
            CandidateStatus.REJECTED)
        self.assertEqual(_combine_dual_layer_decisions(
            CandidateStatus.INCONCLUSIVE, CandidateStatus.ACCEPTED),
            CandidateStatus.INCONCLUSIVE)


class TestM3_3RunDualLayerOodAxis(unittest.TestCase):
    """End-to-end OOD axis through run_dual_layer with an engine-free surrogate.

    The surrogate runner writes BOTH the ID (return value) and OOD (file) metric,
    exactly like scinet_compute, so evaluate_domain_candidate reads a genuine OOD
    number off disk. This exercises the real OOD plumbing (runner -> write
    ood_metrics.json -> _read_ood_q -> namespaced gate -> combine) without torch.
    """

    def _harness_stub(self, **kw):
        return {"schema_version": "stub", "dominant_failure": "DECISION_UNCERTAINTY",
                "records": [], "accepted": [], "n_accepted": 0}

    def _make_runner(self, ood_candidate_worse: bool):
        def runner(art):
            d = Path(art["directory"])
            d.mkdir(parents=True, exist_ok=True)
            is_base = "baseline" in art["directory"]
            id_q = 0.7 if not is_base else 0.9
            if ood_candidate_worse:
                # Candidate regresses on OOD -> ID-overfit exposed.
                ood_q = 0.95 if not is_base else 0.85
            else:
                ood_q = 0.6 if not is_base else 0.85
            (d / "ood_metrics.json").write_text(json.dumps(
                {"metrics": {"state_mse@1": ood_q}}), encoding="utf-8")
            return float(id_q)
        return runner

    def _run(self, ood_candidate_worse: bool):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n",
                           encoding="utf-8")
            payload = run_dual_layer(
                base_harness=None, tasks=[], heldout_task=None,
                run_root=tmp, repo_root=tmp, rounds=1, max_minutes=1.0,
                domain="embodied", domain_base_config=str(cfg),
                project_dir=tmp, dataset_version="test-v1",
                domain_seeds=_FIVE, domain_holdout_seeds=_HOLDOUT,
                domain_require_frozen_manifest=False,
                domain_runner=self._make_runner(ood_candidate_worse),
                harness_evolve=self._harness_stub,
                ood_config_path=str(Path(tmp) / "ood_fake.yaml"))
            return payload["domain_layer"]

    def test_ood_accept_combines_to_accepted(self):
        dom = self._run(ood_candidate_worse=False)
        self.assertIsNotNone(dom["ood_config_path"])
        self.assertEqual(dom["n_accepted"], len(dom["records"]))
        for r in dom["records"]:
            self.assertIn("ood_decision", r)
            self.assertEqual(r["decision"], "accepted")
            self.assertEqual(r["ood_decision"], "accepted")
            self.assertIsNotNone(r["ood_mean_improvement"])
            self.assertIsNotNone(r["ood_baseline_mean"])

    def test_ood_reject_combines_to_rejected(self):
        dom = self._run(ood_candidate_worse=True)
        for r in dom["records"]:
            # ID improved (deterministic runner) but OOD regressed -> rejected.
            self.assertEqual(r["decision"], "rejected")
            self.assertEqual(r["ood_decision"], "rejected")


class TestM3_3OodAxisIsOptIn(unittest.TestCase):
    """When ood_config_path is None the OOD axis is absent and behaviour is the
    original M3.0 behaviour (no ood_decision key, no combined downgrade)."""

    def _harness_stub(self, **kw):
        return {"schema_version": "stub", "dominant_failure": "DECISION_UNCERTAINTY",
                "records": [], "accepted": [], "n_accepted": 0}

    def test_no_ood_config_means_no_ood_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n",
                           encoding="utf-8")
            payload = run_dual_layer(
                base_harness=None, tasks=[], heldout_task=None,
                run_root=tmp, repo_root=tmp, rounds=1, max_minutes=1.0,
                domain="embodied", domain_base_config=str(cfg),
                project_dir=tmp, dataset_version="test-v1",
                domain_seeds=_FIVE, domain_holdout_seeds=_HOLDOUT,
                domain_require_frozen_manifest=False,
                domain_runner=lambda art: (0.7 if "baseline" not in art["directory"] else 0.9),
                harness_evolve=self._harness_stub)
            dom = payload["domain_layer"]
            self.assertIsNone(dom["ood_config_path"])
            self.assertIsNone(dom.get("ood_n_accepted"))
            for r in dom["records"]:
                self.assertNotIn("ood_decision", r)


if __name__ == "__main__":
    unittest.main()
