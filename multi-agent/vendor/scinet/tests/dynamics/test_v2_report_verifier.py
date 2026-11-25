"""
tests.dynamics.test_v2_report_verifier
=======================================
Negative tests for the v2-report number-traceability verifier.
A verifier that cannot fail is not a verifier.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_v2_reports as V  # noqa: E402


class TestV2ReportVerifier(unittest.TestCase):
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
        self.assertGreaterEqual(r["n_total"], 30)

    def test_all_cases_have_anchor_and_tolerance(self):
        for c in V._build_cases():
            with self.subTest(label=c["label"]):
                self.assertTrue(c["anchor"])
                self.assertIn(c["target_report"], V.REPORTS)
                self.assertIsInstance(c["tolerance"], (int, float))

    def test_all_cases_numerically_compared(self):
        for row in self._run()["rows"]:
            with self.subTest(label=row["label"]):
                self.assertIsInstance(row["numeric_match"], bool)
                self.assertIsNotNone(row["source_value"])
                self.assertIsNotNone(row["parsed_report_value"])
                self.assertIsNotNone(row["numeric_delta"])

    def test_negative_corrupt_scientific_notation(self):
        reports = dict(self.reports)
        f = "P1_B1_V2_SCIENTIFIC_REVIEW.md"
        old = "**1.6515e-05**"
        self.assertIn(old, reports[f])
        reports[f] = reports[f].replace(old, "**9.9999e-05**", 1)
        r = self._run(reports)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("v2_dop853_max_error",
                      [x["label"] for x in r["rows"] if not x["passed"]])

    def test_negative_corrupt_integer(self):
        reports = dict(self.reports)
        f = "P1_B1_V2_PILOT_REPORT.md"
        old = "[20, 1001, 2]"
        self.assertIn(old, reports[f])
        reports[f] = reports[f].replace(old, "[21, 1001, 2]", 1)
        r = self._run(reports)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("v2_n_episodes",
                      [x["label"] for x in r["rows"] if not x["passed"]])

    def test_negative_wrong_report_binding(self):
        reports = dict(self.reports)
        target = "P1_B1_V2_SCIENTIFIC_REVIEW.md"
        other = "P1_B1_V2_READINESS_REPORT.md"
        reports[target] = reports[target].replace("1.6515e-05", "REMOVED")
        reports[other] += "\n[negative] 1.6515e-05\n"
        r = self._run(reports)
        self.assertEqual(r["verdict"], "FAIL")

    def test_negative_wrong_line_binding(self):
        c = next(x for x in V._build_cases()
                 if x["label"] == "v2_dop853_max_error")
        reports = {n: "" for n in V.REPORTS}
        reports[c["target_report"]] = "unrelated line 1.6515e-05\n"
        self.assertEqual(self._run(reports)["verdict"], "FAIL")

    def test_negative_evidence_corruption(self):
        evidence = {k: dict(v) for k, v in self.evidence.items()}
        a = dict(evidence["audit_report.json"])
        a["checks"] = [dict(x) for x in a["checks"]]
        a["checks"][1] = dict(a["checks"][1])
        a["checks"][1]["detail"] = dict(a["checks"][1]["detail"])
        a["checks"][1]["detail"]["max_abs_state_error_over_all"] = 0.5
        evidence["audit_report.json"] = a
        r = self._run(evidence=evidence)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("v2_dop853_max_error",
                      [x["label"] for x in r["rows"] if not x["passed"]])

    def test_analytic_recomputation_values(self):
        """theta0=2 accelerations are recomputed from the equation, not copied."""
        import math
        base = -9.81 * math.sin(2.0)
        self.assertAlmostEqual(base + 2.5, -6.420208, places=5)
        self.assertAlmostEqual(base - 2.5, -11.420208, places=5)


if __name__ == "__main__":
    unittest.main()
