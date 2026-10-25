"""端到端集成测试：跑完整自检（含两个阴性对照）。

这是最慢的一组（会起子进程跑假跑器），但它是 P0 唯一的强制通过项，必须进 CI。
"""
import unittest
from pathlib import Path

from adaptive_runtime import selftest as st
from adaptive_runtime.settings import PATHS


class TestSelftest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 自检会起子进程，跑一次供全部用例复用
        cls.report = st.selftest(verbose=False)

    def test_all_checks_pass(self):
        failed = [c for c in self.report["checks"] if not c["pass"]]
        self.assertEqual(failed, [], f"自检失败项：{failed}")
        self.assertTrue(self.report["passed"])

    def test_cases_have_expected_reasons(self):
        cases = {c["case"]: c for c in self.report["cases"]}
        self.assertEqual(cases["zero"]["reason"], "within_noise")
        self.assertEqual(cases["base_vs_base"]["reason"], "within_noise")
        self.assertEqual(cases["good"]["decision"], "ACCEPT")
        # 「故意变差」必须被判负：REGRESSION / CONTRACT_FLOOR / WITHIN_NOISE 都算拒绝
        self.assertIn(cases["worse"]["reason"],
                      ("regression", "contract_floor", "within_noise"))
        self.assertEqual(cases["worse"]["decision"], "REJECT")

    def test_no_sandbox_leak(self):
        leftovers = list((PATHS.experiments / "selftest").rglob("sandbox-*"))
        self.assertEqual(leftovers, [], f"沙箱未清理：{leftovers[:3]}")


if __name__ == "__main__":
    unittest.main()
