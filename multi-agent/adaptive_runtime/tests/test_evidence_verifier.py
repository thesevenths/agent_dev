"""ARR 1.0 / D3 + D5 -- EvidenceVerifier unit tests.

Pure, engine-free tests of the hash-recomputation verifier. The verifier must:

* recompute report / hypothesis / plan / config / dataset hashes from ACTUAL
  content and catch a tampered hash (the P0-2 "field exists != evidence real");
* NOT raise false failures on legacy hash-only reports that carry no raw content
  (so the existing valid artifacts stay green);
* verify a portable evidence bundle's internal self-consistency.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_runtime.evidence_verifier import (
    EvidenceVerifier, VerificationResult, sha256_of, sha256_file,
)


def _report(*, with_content: bool = False, tamper: bool = False) -> dict:
    """Build a minimal ResearchReport-shaped dict for verification tests."""
    if with_content:
        content = "E=mc^2" if not tamper else "E=mv^2"  # tamper changes content
        hyps = [{"hypothesis_id": "h1", "content": content,
                 "content_sha256": sha256_of("E=mc^2")}]
    else:
        hyps = [{"hypothesis_id": "h1", "content_sha256": "deadbeef"}]
    body = {
        "hypotheses": hyps,
        "reviews": [], "plans": [], "approvals": [],
        "campaigns": [{"status": "completed"}], "evidence_refs": [],
        "experience_refs": [],
    }
    report = dict(body, report_sha256=sha256_of(body))
    return report


class TestReportVerification(unittest.TestCase):
    def test_recomputed_report_sha256_matches_for_new_style(self):
        rep = _report(with_content=True)
        vr = EvidenceVerifier().verify_report(rep, only_verifiable=True)
        names = {c.name for c in vr.checks}
        self.assertIn("report_sha256_matches", names)
        self.assertTrue(vr.all_passed())

    def test_tampered_hypothesis_content_is_detected(self):
        rep = _report(with_content=True, tamper=True)
        vr = EvidenceVerifier().verify_report(rep, only_verifiable=True)
        failed = [c for c in vr.checks if not c.passed]
        self.assertTrue(failed, "tampered content must produce a failure")
        self.assertTrue(any("hypothesis_content_sha256" in c.name
                            for c in failed))

    def test_legacy_hash_only_report_does_not_false_fail(self):
        rep = _report(with_content=False)  # only hashes, no raw content
        vr = EvidenceVerifier().verify_report(rep, only_verifiable=True)
        # No verifiable content -> no recomputation check should be emitted that
        # would drop the Evidence score of a valid legacy artifact.
        verifiable = [c for c in vr.checks
                      if "content_sha256" in c.name
                      or "plans_sha256" in c.name
                      or "config_sha256" in c.name]
        self.assertEqual(verifiable, [])
        # report_sha256_matches is still emitted and should pass (computed right).
        self.assertTrue(any(c.name == "report_sha256_matches" and c.passed
                            for c in vr.checks))

    def test_config_hash_recomputed_and_tamper_caught(self):
        rep = _report(with_content=False)
        base_cfg = {"planner_strategy": "fixed_first_knob", "max_candidates": 1}
        cand_cfg = {"planner_strategy": "diagnosis_targeted", "max_candidates": 2}
        rep["comparisons"] = [{
            "baseline_config": base_cfg,
            "baseline_config_sha256": sha256_of(base_cfg),
            "candidate_config": cand_cfg,
            "candidate_config_sha256": sha256_of(cand_cfg),
        }]
        vr = EvidenceVerifier().verify_report(rep, only_verifiable=True)
        self.assertTrue(any(c.name == "config_sha256_matches" and c.passed
                            for c in vr.checks))

        # Tamper the declared candidate config hash.
        rep["comparisons"][0]["candidate_config_sha256"] = "0" * 64
        vr2 = EvidenceVerifier().verify_report(rep, only_verifiable=True)
        self.assertTrue(any(c.name == "config_sha256:candidate_config_sha256"
                            and not c.passed for c in vr2.checks))


class TestDatasetManifestVerification(unittest.TestCase):
    def test_manifest_and_split_hashes_recomputed(self):
        rows = [{"x": 1}, {"x": 2}]
        manifest = {
            "splits": {"train": {"rows": rows, "sha256": sha256_of(rows)}},
            "manifest_sha256": "",
        }
        manifest["manifest_sha256"] = sha256_of(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"})
        vr = EvidenceVerifier().verify_dataset_manifest(manifest)
        self.assertTrue(vr.all_passed())

    def test_split_tamper_caught(self):
        rows = [{"x": 1}, {"x": 2}]
        manifest = {"splits": {"train": {"rows": rows, "sha256": "0" * 64}}}
        vr = EvidenceVerifier().verify_dataset_manifest(manifest)
        self.assertFalse(vr.all_passed())


class TestBundleIntegrity(unittest.TestCase):
    def test_bundle_self_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arr_1_0_r1"
            root.mkdir()
            files = {
                "manifest.json": {"code_version": "94653a11"},
                "evolution_summary.json": {"decision": "ACCEPT"},
                "verification.json": {"passed": True},
            }
            for name, data in files.items():
                (root / name).write_text(
                    json.dumps(data, sort_keys=True), encoding="utf-8")
            hashes = {
                "bundle_files": {name: sha256_file(root / name)
                                 for name in files},
            }
            (root / "evidence_hashes.json").write_text(
                json.dumps(hashes, sort_keys=True), encoding="utf-8")
            vr = EvidenceVerifier().verify_bundle_integrity(root)
            self.assertTrue(vr.all_passed(),
                            f"bundle should be self-consistent: {vr.as_dict()}")

    def test_bundle_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arr_1_0_r1"
            root.mkdir()
            (root / "manifest.json").write_text(
                json.dumps({"code_version": "x"}, sort_keys=True),
                encoding="utf-8")
            (root / "evidence_hashes.json").write_text(
                json.dumps({"bundle_files": {
                    "manifest.json": "0" * 64}}), encoding="utf-8")
            vr = EvidenceVerifier().verify_bundle_integrity(root)
            self.assertFalse(vr.all_passed())

    def test_bundle_extra_undeclared_file_detected(self):
        """D-Fix / Problem C: an undeclared file on disk must fail the check."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arr_1_0_r1"
            root.mkdir()
            (root / "manifest.json").write_text(
                json.dumps({"a": 1}, sort_keys=True), encoding="utf-8")
            hashes = {"bundle_files": {
                "manifest.json": sha256_file(root / "manifest.json")}}
            (root / "evidence_hashes.json").write_text(
                json.dumps(hashes, sort_keys=True), encoding="utf-8")
            (root / "stowaway.txt").write_text("evil", encoding="utf-8")
            vr = EvidenceVerifier().verify_bundle_integrity(root)
            self.assertFalse(vr.all_passed())
            self.assertFalse(any(c.name == "bundle_no_extra_files" and c.passed
                                 for c in vr.checks))

    def test_bundle_path_traversal_rejected(self):
        """D-Fix / Problem C: a declared name with '..' must fail path safety."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arr_1_0_r1"
            root.mkdir()
            (root / "manifest.json").write_text(
                json.dumps({"a": 1}, sort_keys=True), encoding="utf-8")
            hashes = {"bundle_files": {
                "../escape.json": "0" * 64,
                "manifest.json": sha256_file(root / "manifest.json")}}
            (root / "evidence_hashes.json").write_text(
                json.dumps(hashes, sort_keys=True), encoding="utf-8")
            vr = EvidenceVerifier().verify_bundle_integrity(root)
            self.assertFalse(vr.all_passed())
            self.assertFalse(any(
                c.name == "bundle_path_safe:../escape.json" and c.passed
                for c in vr.checks))


class TestDatasetManifestWiredIntoQuality(unittest.TestCase):
    """D-Fix / Problem B: verify_dataset_manifest is wired into the Evidence (E)
    dimension of the quality evaluator, not just available as a standalone
    method. A tampered split hash must drop the Evidence dimension."""

    def _evaluator(self):
        from research_runtime.research_quality import (
            DefaultResearchQualityEvaluator, QualityEvidence)
        return DefaultResearchQualityEvaluator(), QualityEvidence

    def test_tampered_manifest_fails_evidence_integrity(self):
        ev_cls = self._evaluator()[1]
        rows = [{"x": 1}, {"x": 2}]
        manifest = {"splits": {"train": {"rows": rows, "sha256": "0" * 64}}}
        ev = ev_cls(report={"dataset_manifest": manifest})
        evaluator = self._evaluator()[0]
        _, checks = evaluator._evidence_integrity(ev)
        self.assertIn("V.split_sha256:train", checks)
        self.assertFalse(checks["V.split_sha256:train"])

    def test_valid_manifest_passes_when_wired(self):
        ev_cls = self._evaluator()[1]
        rows = [{"x": 1}, {"x": 2}]
        manifest = {"splits": {"train": {
            "rows": rows, "sha256": sha256_of(rows)}}}
        ev = ev_cls(report={"dataset_manifest": manifest})
        evaluator = self._evaluator()[0]
        _, checks = evaluator._evidence_integrity(ev)
        self.assertTrue(checks["V.split_sha256:train"])


if __name__ == "__main__":
    unittest.main()
