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


#: 满足结构口径的报告（≥min_report_bytes、≥3 标题、有表格、有结论段）。
#: 反刷分口径下「一句话 md」不再计为报告，因此凡断言「满分 / Q=100」的用例
#: 都必须使用该样本，否则实际测的是「无报告情形下的分数」。
_GOOD_REPORT = """# 分析报告

## 一、背景

本报告梳理近三年规模变化与盈利结构，供决策参考。

## 二、概况

2025 年市场规模约 1200 亿美元（来源：https://example.com/a），同比增长 18%。

| 指标 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- |
| 规模（亿美元） | 860 | 1017 | 1200 |
| 增速 | 12% | 18% | 18% |

毛利率提升至 42%（数据来源：公司年报），主要来自产品结构变化，
服务收入占比由 31% 提升至 38%。期间费用率基本持平，
因此毛利率的提升大部分传导到了经营利润率。

行业集中度较上一年度提升，前三名厂商合计份额达到 46%（引自：https://example.com/s）。
新增产能主要来自头部企业的扩产项目，其中两家宣布的扩产计划合计约 80 亿美元。
这一变化与行业整体从产品销售向订阅服务转型的趋势一致，
也解释了服务收入占比在两年内提升 14 个百分点的原因。

## 三、结论

预计 2026 年增速回落至 12%（引自：https://example.com/f）。
主要风险为需求周期波动与价格竞争加剧。
综合判断，增长质量优于增长速度，建议关注服务收入占比的持续性与费用率的变化。
"""


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
    def test_no_report_text_halves_c(self):
        # 没有可读报告文本 → 结构分 0 → C 打折（不是 1.0）
        c, d = completeness(_trace())
        self.assertAlmostEqual(c, 1.0 * (1 - CFG.structure_share), places=4)
        self.assertEqual(d["structure_score"], 0.0)

    def test_full_with_structured_report(self):
        c, d = completeness(_trace(), report_text=_GOOD_REPORT)
        self.assertEqual(c, 1.0)
        self.assertEqual(d["structure_score"], 1.0)

    def test_missing_artifact(self):
        tr = _trace()
        tr.artifacts = []
        c, d = completeness(tr, report_text=_GOOD_REPORT)
        self.assertLess(c, 1.0)
        self.assertEqual(d["missing"], 1)

    def test_truncated_and_salvaged(self):
        tr = _trace()
        tr.steps[2].fail_kind = "truncated"
        tr.result = {"salvaged": True}
        c, d = completeness(tr, report_text=_GOOD_REPORT)
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
        (self.root / "documents" / "r.md").write_text(_GOOD_REPORT, encoding="utf-8")
        self.artifacts = [{"path": "documents/r.md",
                           "bytes": len(_GOOD_REPORT.encode("utf-8")), "ext": ".md",
                           "mtime": "2026-01-01T00:00:00", "deliverable": True}]

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_full_weights(self):
        sc = score_trace(_trace(artifacts=self.artifacts), root=self.root, mode="live")
        self.assertEqual(set(sc.weights_used), {"A", "E", "C"})
        self.assertAlmostEqual(sum(sc.weights_used.values()), 1.0, places=3)
        # 断言的是「Q 确实按 0.5/0.3/0.2 合成」，而不是「Q 恰好 100」——
        # 后者会把 E（证据率）的具体取值绑死，任何一个报告文本微调都会假失败。
        self.assertAlmostEqual(sc.A, 1.0, places=3)
        self.assertAlmostEqual(sc.C, 1.0, places=3)
        expected = 100.0 * (0.5 * sc.A + 0.3 * sc.E + 0.2 * sc.C)
        self.assertAlmostEqual(sc.Q, expected, places=1)
        self.assertGreater(sc.E, 0.3)

    def test_thin_report_is_not_a_report(self):
        """一句闲聊存成 .md 不再是「报告」——这是 Q 可被刷的主要通道。"""
        (self.root / "documents" / "r.md").write_text("1+1 等于 2。", encoding="utf-8")
        tr = _trace(artifacts=[{"path": "documents/r.md", "bytes": 14, "ext": ".md",
                                "mtime": "2026-01-01T00:00:00", "deliverable": True}])
        sc = score_trace(tr, root=self.root, mode="live")
        self.assertIn("no_report", sc.flags)
        self.assertIn("thin_deliverable", sc.flags)
        self.assertEqual(sc.E, 0.0)
        self.assertLess(sc.Q, 70.0)

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
