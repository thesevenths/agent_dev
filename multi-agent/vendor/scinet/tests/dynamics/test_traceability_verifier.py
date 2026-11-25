"""
tests.dynamics.test_traceability_verifier
=========================================
P1-A' review fix P2 -- NEGATIVE tests for the hardened number-traceability
verifier. A verifier that can never fail is not a verifier.

These tests inject deliberately corrupted report texts into
``verify_p1a_reports.verify`` and assert it returns FAIL, covering each v1
defect:
  1. un-compared scientific notation  -> change 7.628e-03 and require FAIL
  2. un-compared integers             -> change **14** and require FAIL
  3. hard-coded count                 -> corrupt the git count in the report
  4. concatenated-report flaw         -> move a value to the WRONG report
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_p1a_reports as V  # noqa: E402


class TestTraceabilityVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports, cls.evidence = V.load_inputs()

    def _run(self, reports=None, evidence=None):
        return V.verify(reports if reports is not None else self.reports,
                        evidence if evidence is not None else self.evidence)

    def test_baseline_passes(self):
        r = self._run()
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["n_pass"], r["n_total"])
        self.assertGreaterEqual(r["n_total"], 40)

    def test_all_cases_bind_a_specific_report_and_source(self):
        """No case may rely on a concatenated report string."""
        for c in V._build_cases():
            with self.subTest(label=c["label"]):
                self.assertIn(c["target_report"], V.REPORTS)
                self.assertIn(c["source"], V.EVIDENCE_FILES)
                self.assertIn("tolerance", c)
                self.assertIsInstance(c["tolerance"], (int, float))

    def test_no_case_can_skip_numeric_comparison(self):
        """Every case has an explicit numeric tolerance; none is default-True."""
        r = self._run()
        for row in r["rows"]:
            with self.subTest(label=row["label"]):
                self.assertIn("numeric_match", row)
                self.assertIsInstance(row["numeric_match"], bool)
                self.assertIsNotNone(row["source_value"])
                self.assertIsNotNone(row["parsed_report_value"])
                self.assertIsNotNone(row["numeric_delta"])

    # ---- negative tests: corrupted numbers MUST make the verifier FAIL ---- #
    def test_negative_scientific_notation(self):
        """Corrupt the value ON THE ANCHOR LINE of its case -> must FAIL."""
        reports = dict(self.reports)
        f = "P1_A_SCIENTIFIC_REVIEW.md"
        # the case binds the line containing "全局最大 7.628e-03"
        old = "全局最大 7.628e-03 rad"
        self.assertIn(old, reports[f])
        reports[f] = reports[f].replace(old, "全局最大 9.999e-03 rad", 1)
        r = self._run(reports)
        self.assertEqual(r["verdict"], "FAIL")
        failed = [x["label"] for x in r["rows"] if not x["passed"]]
        self.assertIn("scipy_dop853_max_state_error", failed)

    def test_anchor_binding_rejects_wrong_location(self):
        """A value that only exists on the WRONG line must NOT satisfy a case.

        This directly proves the v1 'concatenated report' loophole is closed.
        """
        case = next(c for c in V._build_cases()
                    if c["label"] == "scipy_dop853_max_state_error")
        reports = {n: "" for n in V.REPORTS}
        # value is present in the correct report, but NOT on the anchor line
        reports[case["target_report"]] = "unrelated line 7.628e-03\n"
        r = self._run(reports)
        self.assertEqual(r["verdict"], "FAIL")

    def test_negative_integer(self):
        reports = dict(self.reports)
        f = "P1_A_TRAJECTORY_DISTRIBUTION.md"
        reports[f] = reports[f].replace("**14**", "**15**", 1)
        r = self._run(reports)
        self.assertEqual(r["verdict"], "FAIL")
        failed = [x["label"] for x in r["rows"] if not x["passed"]]
        self.assertIn("split_train", failed)

    def test_negative_git_count_is_not_hardcoded(self):
        """Corrupt the evidence snapshot; the report's 78 must no longer match."""
        evidence = {k: dict(v) if isinstance(v, dict) else v
                    for k, v in self.evidence.items()}
        evidence["git_integrity.json"] = dict(evidence["git_integrity.json"])
        evidence["git_integrity.json"]["p2_02_evidence_file_count"] = 79
        r = self._run(evidence=evidence)
        self.assertEqual(r["verdict"], "FAIL")
        failed = [x["label"] for x in r["rows"] if not x["passed"]]
        self.assertIn("p2_02_evidence_file_count", failed)

    def test_negative_wrong_report_binding(self):
        """v1 concatenated all reports: a value in the WRONG report would pass.

        Here we REMOVE ALL occurrences from the target report and inject them
        into a DIFFERENT report. v2 must still FAIL.
        """
        reports = dict(self.reports)
        target = "P1_A_SCIENTIFIC_REVIEW.md"
        other = "P2_03A_READINESS_PLAN.md"
        reports[target] = reports[target].replace("7.628e-03", "REMOVED")
        reports[other] = reports[other] + "\n[negative test] 7.628e-03\n"
        r = self._run(reports)
        self.assertEqual(r["verdict"], "FAIL")
        failed = [x["label"] for x in r["rows"] if not x["passed"]]
        self.assertIn("scipy_dop853_max_state_error", failed)

    def test_negative_missing_jsonpath(self):
        evidence = dict(self.evidence)
        evidence["independent_verification.json"] = {
            k: v for k, v in evidence["independent_verification.json"].items()
            if k != "energy_balance_identity"
        }
        r = self._run(evidence=evidence)
        self.assertEqual(r["verdict"], "FAIL")
        failed = [x["label"] for x in r["rows"] if not x["passed"]]
        self.assertIn("energy_balance_max_rel_residual", failed)

    def test_percentage_and_bold_parsing(self):
        self.assertAlmostEqual(V.parse_literal("53.5%"), 53.5)
        self.assertAlmostEqual(V.parse_literal("**14**"), 14.0)
        self.assertAlmostEqual(V.parse_literal("7.628e-03"), 0.007628)
        self.assertAlmostEqual(V.parse_literal("-4.9863"), -4.9863)
        self.assertAlmostEqual(V.parse_literal("+4.9999"), 4.9999)

    def test_resolve_path_with_indices(self):
        v = V.resolve_path(
            self.evidence["trajectory_distribution.json"],
            "per_episode[5].theta_final_rad")
        self.assertAlmostEqual(v, 79.11, places=1)


if __name__ == "__main__":
    unittest.main()
