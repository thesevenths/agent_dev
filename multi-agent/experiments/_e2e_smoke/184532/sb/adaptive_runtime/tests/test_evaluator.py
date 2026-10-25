import tempfile
import unittest
from pathlib import Path

from adaptive_runtime.evaluator import completeness, evidence_rate, score_trace
from adaptive_runtime.settings import CFG
from adaptive_runtime.trace_export import StepState, Trace


def _trace(**kw):
    tr = Trace(run_id="t1", log_path="test.log")
    tr.plan = [{"step_no": 1, "status": "pending", "title": "检索数据", "agent": "crawler_agent"},
               {"step_no": 2, "status": "pending", "title": "撰写报告并落盘", "agent": "code_agent"}]
    tr.total_steps = 2
    tr.steps = {1: StepState(1, passes=1, last="pass", agent="crawler_agent"),
                2: StepState(2, passes=1, last="pass", agent="code_agent")}
    tr.artifacts = [{"path": "documents/r.md", "bytes": 100, "ext": ".md",
                     "mtime": "2026-01-01T00:00:00", "deliverable": True}]
    for k, v in kw.items():
        setattr(tr, k, v)
    return tr


class TestEvidenceRate(unittest.TestCase):
    SRC = ("2025 年市场规模约 1200 亿美元（来源：https://example.com/a），同比增长 18%。\n\n"
           "毛利率 42%（数据来源：公司年报）。")
    NOSRC = "2025 年市场规模约 1200 亿美元，同比增长 18%。\n\n毛利率 42%。"

    def test_with_sources(self):
        rate, detail = evidence_rate(self.SRC)
        self.assertAlmostEqual(rate, 1.0)
        self.assertGreaterEqual(detail["claims"], 3)

    def test_without_sources(self):
        rate, detail = evidence_rate(self.NOSRC)
        self.assertEqual(rate, 0.0)
        self.assertGreater(detail["claims"], 0)

    def test_empty(self):
        rate, detail = evidence_rate("")
        self.assertEqual(rate, 0.0)
        self.assertEqual(detail["claims"], 0)

    def test_markdown_link_counts_as_source(self):
        rate, _ = evidence_rate("规模 100 亿（见 [报告](https://x.com/r)）。")
        self.assertEqual(rate, 1.0)

    def test_mixed(self):
        rate, _ = evidence_rate("规模 100 亿（来源：年报）。\n\n增速约 20%。")
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)


class TestCompleteness(unittest.TestCase):
    def test_full(self):
        c, d = completeness(_trace())
        self.assertEqual(c, 1.0)

    def test_missing_artifact(self):
        tr = _trace()
        tr.artifacts = []
        c, d = completeness(tr)
        self.assertLess(c, 1.0)
        self.assertEqual(d["missing"], 1)

    def test_truncated_and_salvaged(self):
        tr = _trace()
        tr.steps[2].fail_kind = "truncated"
        tr.result = {"salvaged": True}
        c, d = completeness(tr)
        self.assertLess(c, 0.7)

    def test_unavailable(self):
        c, d = completeness(_trace(), include=False)
        self.assertIsNone(c)
        self.assertFalse(d["available"])


class TestScoreTrace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "documents").mkdir()
        (self.root / "documents" / "r.md").write_text(
            "规模 100 亿（来源：https://a.com）。", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_full_weights(self):
        sc = score_trace(_trace(), root=self.root, mode="live")
        self.assertEqual(set(sc.weights_used), {"A", "E", "C"})
        self.assertAlmostEqual(sum(sc.weights_used.values()), 1.0, places=3)
        self.assertAlmostEqual(sc.Q, 100.0, places=1)

    def test_live_no_report_is_zero_not_renormalized(self):
        (self.root / "documents" / "r.md").unlink()
        tr = _trace()
        tr.artifacts = []
        sc = score_trace(tr, root=self.root, mode="live")
        self.assertIn("no_report", sc.flags)
        self.assertEqual(sc.E, 0.0)
        # 权重仍是 0.5/0.3/0.2，不得重分配
        self.assertAlmostEqual(sc.weights_used["A"], 0.5, places=3)

    def test_replay_gap_renormalizes(self):
        tr = _trace()
        tr.artifacts = []
        tr.result = {"artifacts_unavailable": True}
        sc = score_trace(tr, root=self.root, mode="replay")
        self.assertIsNone(sc.E)
        self.assertIsNone(sc.C)
        self.assertAlmostEqual(sc.weights_used["A"], 1.0, places=3)
        self.assertAlmostEqual(sc.Q, 100.0, places=1)

    def test_contract_rate_in_A(self):
        tr = _trace()
        tr.steps[2].last = "fail"
        tr.steps[2].fail_kind = "recursion"
        sc = score_trace(tr, root=self.root, mode="live")
        self.assertAlmostEqual(sc.A, 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
