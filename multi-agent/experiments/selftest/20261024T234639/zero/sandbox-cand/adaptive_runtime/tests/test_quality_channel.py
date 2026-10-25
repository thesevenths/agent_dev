"""质量缺陷通道：诊断层能否发现「跑完了但产出不达标」。

覆盖的存在理由
--------------
诊断层原来的信号只有硬失败（``kind=recursion / truncated / exception``
以及 gate 反复 reject、replan 空转）。这条通道有个结构性盲区：
**一次每一步都 PASS、正常收尾、报告也落盘的 run，会被判成「无失败信号」。**

而真实数据里最大的失分项恰恰在这里 —— 一组实测 run 的 A/C 已接近满分，
27 分差距全部落在 E（268 个关键数字仅 27 个带来源）。只看硬失败，
闭环会认为这批 run 无改进空间，于是永远不会去提「给数字加来源」这类候选。

本文件锁住两件事：
1. 质量通道能从 Score 反推出 EVIDENCE_FAILURE / COMPLETENESS_FAILURE；
2. 主导模式为 EVIDENCE_FAILURE 时，提案器能路由到对应的提示补丁
   （否则通道打通了也没有候选可跑）。
"""
import tempfile
import unittest
from pathlib import Path

from adaptive_runtime import diagnosis, proposer
from adaptive_runtime.diagnosis import Taxonomy
from adaptive_runtime.evaluator import Score, score_trace
from adaptive_runtime.ledger import Ledger
from adaptive_runtime.settings import Settings
from adaptive_runtime.trace_export import StepState, Trace


def _clean_trace(run_id: str = "r1", agent: str = "CodeAgent") -> Trace:
    """构造一次「完全正常」的 run：三步全 PASS、有产物、正常收尾。

    这是质量通道的前提样本：它没有任何硬失败信号，
    因此任何被产出的 Finding 都只能来自 E/C。
    """
    tr = Trace(run_id=run_id, log_path="fake.log")
    tr.total_steps = 3
    for i in (1, 2, 3):
        tr.steps[i] = StepState(step=i, passes=1, last="pass", agent=agent)
    tr.artifacts = [{"path": "report.md", "ext": ".md",
                     "bytes": 5000, "deliverable": True}]
    tr.result = {"finalized": True}
    return tr


def _score(E: float | None = None, C: float | None = None,
           claims: int = 100, supported: int = 10,
           report: str | None = "report.md",
           **cdet: object) -> Score:
    """构造 Score。

    字段结构刻意与 ``evaluator.score_trace`` 的真实产出保持一致：
    ``detail["report"]`` 只在**确实读到成品文本**时才出现，
    ``detail["evidence"]`` 里没有 ``available`` 键。
    曾经因为测试 helper 自行塞了 ``available``，用例全绿但功能实际不生效。
    """
    detail = {
        "evidence": {
            "claims": claims,
            "supported": supported,
            "unsupported_samples": ["2025 年市场规模约 1200 亿美元"],
        },
        "completeness": {"available": C is not None, "truncated": False,
                         "salvaged": False, "missing": 0,
                         "structure": {"bytes": 4000}, **cdet},
    }
    if report is not None:
        detail["report"] = report
    return Score(run_id="r1", A=1.0, E=E, C=C, planned=3, detail=detail)


class TestEvidenceFailure(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()

    def test_clean_run_low_evidence_is_flagged(self):
        """无硬失败 + E 低于阈值 → 产出 EVIDENCE_FAILURE，不再是「无失败信号」。"""
        findings, dsum = diagnosis.from_trace(
            _clean_trace(), self.cfg, score=_score(E=0.10, claims=100, supported=10))
        taxes = [f.taxonomy for f in findings]
        self.assertIn(Taxonomy.EVIDENCE_FAILURE, taxes)
        self.assertEqual(dsum["dominant"], Taxonomy.EVIDENCE_FAILURE)
        # 关键回归：不能退化成 UNKNOWN / 无失败信号
        self.assertNotIn(Taxonomy.UNKNOWN, taxes)

    def test_high_evidence_not_flagged(self):
        """E 达标 → 干净 run 仍判 UNKNOWN（确实无改进信号）。"""
        findings, _ = diagnosis.from_trace(
            _clean_trace(), self.cfg, score=_score(E=0.90, claims=100, supported=90))
        self.assertEqual([f.taxonomy for f in findings], [Taxonomy.UNKNOWN])

    def test_few_claims_not_judged(self):
        """关键数字太少时 E 的方差不可信 → 不判证据缺陷。"""
        findings, _ = diagnosis.from_trace(
            _clean_trace(), self.cfg, score=_score(E=0.0, claims=3, supported=0))
        self.assertNotIn(Taxonomy.EVIDENCE_FAILURE, [f.taxonomy for f in findings])

    def test_unavailable_evidence_not_flagged(self):
        """产物不可读（E=None，detail 里无 report 键）是数据缺失，不是质量缺陷。"""
        findings, _ = diagnosis.from_trace(
            _clean_trace(), self.cfg, score=_score(E=None, C=0.9, report=None))
        self.assertNotIn(Taxonomy.EVIDENCE_FAILURE, [f.taxonomy for f in findings])

    def test_missing_artifact_not_mistaken_for_quality_defect(self):
        """回归：产物被外部清理时，E 按 live 语义记 0，但不得据此判质量缺陷。

        复盘历史日志必然遇到这种情况：日志还在、产物已被清理。
        若把「盘上文件没了」当成「报告质量差」，闭环会提出针对不存在问题的补丁。
        """
        sc = _score(E=0.0, C=0.50, claims=100, supported=0, report=None)
        findings, dsum = diagnosis.from_trace(_clean_trace(), self.cfg, score=sc)
        taxes = [f.taxonomy for f in findings]
        self.assertNotIn(Taxonomy.EVIDENCE_FAILURE, taxes)
        self.assertNotIn(Taxonomy.COMPLETENESS_FAILURE, taxes)

    def test_structure_unreadable_suppresses_completeness_finding(self):
        """C 的 structure 分量为 0 时，无法区分「结构差」与「没读到文本」。"""
        sc = _score(C=0.30, structure={"bytes": 0})
        findings, _ = diagnosis.from_trace(_clean_trace(), self.cfg, score=sc)
        self.assertNotIn(Taxonomy.COMPLETENESS_FAILURE,
                         [f.taxonomy for f in findings])

    def test_floor_is_configurable(self):
        cfg = Settings(evidence_floor=0.95)
        findings, _ = diagnosis.from_trace(
            _clean_trace(), cfg, score=_score(E=0.90, claims=100, supported=90))
        self.assertIn(Taxonomy.EVIDENCE_FAILURE, [f.taxonomy for f in findings])

    def test_finding_carries_unsupported_samples(self):
        """每条质量 Finding 必须能追溯到成品文本的具体数字，便于人工核验。"""
        findings, _ = diagnosis.from_trace(
            _clean_trace(), self.cfg,
            score=_score(E=0.05, claims=100, supported=5))
        f = next(x for x in findings if x.taxonomy == Taxonomy.EVIDENCE_FAILURE)
        self.assertEqual(f.evidence["claims"], 100)
        self.assertEqual(f.evidence["supported"], 5)
        self.assertTrue(f.evidence["unsupported_samples"])

    def test_agent_attribution_from_last_step(self):
        """质量缺陷要归到产出最终交付物的 agent；归不到则记 ``-``。"""
        findings, _ = diagnosis.from_trace(
            _clean_trace(agent="DataAgent"), self.cfg, score=_score(E=0.05))
        f = next(x for x in findings if x.taxonomy == Taxonomy.EVIDENCE_FAILURE)
        self.assertEqual(f.agent, "DataAgent")


class TestCompletenessFailure(unittest.TestCase):
    def test_low_completeness_is_flagged(self):
        findings, _ = diagnosis.from_trace(
            _clean_trace(), Settings(), score=_score(C=0.40))
        self.assertIn(Taxonomy.COMPLETENESS_FAILURE,
                      [f.taxonomy for f in findings])

    def test_completeness_at_floor_not_flagged(self):
        findings, _ = diagnosis.from_trace(
            _clean_trace(), Settings(), score=_score(C=0.70))
        self.assertNotIn(Taxonomy.COMPLETENESS_FAILURE,
                         [f.taxonomy for f in findings])

    def test_evidence_and_completeness_coexist(self):
        findings, _ = diagnosis.from_trace(
            _clean_trace(), Settings(), score=_score(E=0.05, C=0.40))
        taxes = [f.taxonomy for f in findings]
        self.assertIn(Taxonomy.EVIDENCE_FAILURE, taxes)
        self.assertIn(Taxonomy.COMPLETENESS_FAILURE, taxes)


class TestBackwardCompat(unittest.TestCase):
    def test_without_score_clean_run_is_unknown(self):
        """不传 score 时行为与首版一致，保证历史产物可复现。"""
        findings, dsum = diagnosis.from_trace(_clean_trace(), Settings())
        self.assertEqual([f.taxonomy for f in findings], [Taxonomy.UNKNOWN])
        self.assertEqual(dsum["dominant"], Taxonomy.UNKNOWN)

    def test_hard_failure_unaffected_by_score(self):
        """质量通道只做增补，不得吞掉硬失败信号。"""
        tr = _clean_trace()
        tr.steps[2] = StepState(step=2, failures=1, last="fail",
                                fail_kind="recursion", agent="CodeAgent")
        tr.events.append({"type": "step_failure", "step_no": 2,
                          "why": "recursion limit"})
        findings, _ = diagnosis.from_trace(tr, Settings(), score=_score(E=0.9, C=0.9))
        self.assertIn(Taxonomy.COMPLETION_FAILURE, [f.taxonomy for f in findings])


class TestProposerRouting(unittest.TestCase):
    def test_evidence_failure_routes_to_require_sources(self):
        """通道打通后必须有候选可跑 —— 否则诊断得出改进方向也无从验证。"""
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "h.jsonl")
            cands = proposer.propose(Taxonomy.EVIDENCE_FAILURE, ledger,
                                     cfg=Settings(max_edit_level=2))
        names = [c.target for c in cands]
        self.assertIn("prompt.py::coder.require-sources", names,
                      f"EVIDENCE_FAILURE 未路由到提示补丁，实际候选：{names}")

    def test_quality_taxonomies_are_classified(self):
        self.assertIn(Taxonomy.EVIDENCE_FAILURE, Taxonomy.QUALITY)
        self.assertIn(Taxonomy.COMPLETENESS_FAILURE, Taxonomy.QUALITY)
        self.assertTrue(set(Taxonomy.ALL) >= set(Taxonomy.HARD) | set(Taxonomy.QUALITY))

    def test_summarize_prioritizes_quality_over_planning_noise(self):
        """诊断细化（V3 §诊断）：只要出现质量缺陷，dominant 必为质量通道。

        否则闭环会被更高频的 PLANNING_FAILURE（replan≥2 的粗粒度噪声）主导，
        反复拧 replan 旋钮而忽略真正拉低 Q 的证据率缺陷。
        """
        findings = [
            diagnosis.Finding(Taxonomy.PLANNING_FAILURE, "supervisor", 0, reason="replan 3 次"),
            diagnosis.Finding(Taxonomy.PLANNING_FAILURE, "supervisor", 0, reason="replan 4 次"),
            diagnosis.Finding(Taxonomy.PLANNING_FAILURE, "supervisor", 0, reason="replan 5 次"),
            diagnosis.Finding(Taxonomy.EVIDENCE_FAILURE, "DataAgent", 0, reason="证据率 0.10"),
        ]
        dsum = diagnosis.summarize(findings, Settings())
        self.assertEqual(dsum["dominant"], Taxonomy.EVIDENCE_FAILURE,
                         "质量缺陷必须优先于更高频的 PLANNING_FAILURE")

    def test_summarize_falls_back_to_frequency_without_quality(self):
        """无质量缺陷时回落到硬失败频率主导（向后兼容）。"""
        findings = [
            diagnosis.Finding(Taxonomy.PLANNING_FAILURE, "supervisor", 0, reason="replan"),
            diagnosis.Finding(Taxonomy.PLANNING_FAILURE, "supervisor", 0, reason="replan"),
            diagnosis.Finding(Taxonomy.EXECUTION_FAILURE, "CodeAgent", 1, reason="x"),
        ]
        dsum = diagnosis.summarize(findings, Settings())
        self.assertEqual(dsum["dominant"], Taxonomy.PLANNING_FAILURE)


def _report_text(with_sources: bool) -> str:
    """生成一份体积达标的报告；``with_sources=True`` 时每段补上来源标记。"""
    para = ("2025 年市场规模约 1200 亿美元，同比增长 18%，头部三家合计份额 46%，"
            "服务收入占比由 31% 升至 38%，硬件收入下降 7 个百分点，"
            "订阅客户增加 52 万，客单价下降 12%，毛利率提升到 38%。")
    chunks = []
    for i in range(1, 7):
        body = f"第 {i} 部分，" + para
        if with_sources:
            body += "（来源：行业统计年鉴 2025）"
        chunks.append(f"## {i}. 章节\n\n" + body)
    return ("# 行业分析报告\n\n" + "\n\n".join(chunks)
            + "\n\n| 年份 | 规模（亿美元） |\n|---|---|\n| 2025 | 1200 |\n")


class TestEndToEndWithRealScorer(unittest.TestCase):
    """诊断行为必须经由**真实打分器**验证，不得手工构造 Score。

    存在理由：质量通道曾经「18 个单元测试全绿，但功能实际不生效」。
    根因是测试 helper 自行塞了 ``evidence.available`` 键，
    而 ``evaluator.evidence_rate`` 的真实产出里根本没有这个键，
    于是诊断里的 ``available is True`` 恒为假、通道从未被触发。

    教训固化：凡是断言「某类输入应产出某种 Finding」的用例，
    都要从 ``score_trace`` 的真实输出出发，这样实现与产出结构一旦脱节就会立刻变红。
    """

    def _run(self, with_sources: bool):
        td = tempfile.TemporaryDirectory()
        try:
            root = Path(td.name)
            rep = root / "report.md"
            rep.write_text(_report_text(with_sources), encoding="utf-8")
            assert rep.stat().st_size >= Settings().min_report_bytes, "样例报告体积未达门槛"

            tr = Trace(run_id="e2e", log_path="e2e.log")
            tr.total_steps = 3
            for i in (1, 2, 3):
                tr.steps[i] = StepState(step=i, passes=1, last="pass",
                                         agent="CodeAgent")
            tr.artifacts = [{"path": "report.md", "ext": ".md",
                             "bytes": rep.stat().st_size, "deliverable": True}]
            tr.result = {"finalized": True}
            return tr, score_trace(tr, root=root, mode="live")
        finally:
            td.cleanup()

    def test_unsourced_report_triggers_evidence_failure(self):
        """端到端：全 PASS 的 run，报告有数字无来源 → 自主定位到 EVIDENCE_FAILURE。

        这是「agent 怎么知道该往哪改」的核心证据：没有任何硬失败，
        闭环仍然指出了具体的质量维度。
        """
        tr, sc = self._run(with_sources=False)
        self.assertIsNotNone(sc.E, "应能读到报告文本")
        self.assertLess(sc.E, Settings().evidence_floor)

        findings, dsum = diagnosis.from_trace(tr, score=sc)
        self.assertEqual(dsum["dominant"], Taxonomy.EVIDENCE_FAILURE)
        f = findings[0]
        self.assertEqual(f.taxonomy, Taxonomy.EVIDENCE_FAILURE)
        self.assertGreater(f.evidence["claims"], 0)
        self.assertTrue(f.evidence["unsupported_samples"])

    def test_sourced_report_passes_clean(self):
        """同一份报告补上来源后 → 不再产出质量 Finding，且 E 显著提升。"""
        _tr_bad, sc_bad = self._run(with_sources=False)
        _tr_ok, sc_ok = self._run(with_sources=True)

        self.assertGreater(sc_ok.E, sc_bad.E)
        self.assertGreater(sc_ok.Q, sc_bad.Q)

        _, dsum_ok = diagnosis.from_trace(_tr_ok, score=sc_ok)
        self.assertNotIn(Taxonomy.EVIDENCE_FAILURE,
                         list(dsum_ok["counts"]))
        self.assertEqual(dsum_ok["dominant"], Taxonomy.UNKNOWN)

    def test_dominant_routes_to_prompt_patch(self):
        """端到端：由质量通道得到的主导模式必须能路由出可执行的提示补丁。"""
        tr, sc = self._run(with_sources=False)
        _findings, dsum = diagnosis.from_trace(tr, score=sc)

        with tempfile.TemporaryDirectory() as t2:
            cands = proposer.propose(dsum["dominant"], Ledger(Path(t2) / "h.jsonl"),
                                     cfg=Settings(max_edit_level=2))
        self.assertTrue(
            any(c.target == "prompt.py::coder.require-sources" for c in cands),
            f"主导模式 {dsum['dominant']} 未路由到提示补丁，实际：{[c.target for c in cands]}")


if __name__ == "__main__":
    unittest.main()
