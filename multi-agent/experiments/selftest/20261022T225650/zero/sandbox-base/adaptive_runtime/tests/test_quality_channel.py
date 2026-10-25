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
from adaptive_runtime.evaluator import Score
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
           **cdet: object) -> Score:
    return Score(
        run_id="r1", A=1.0, E=E, C=C, planned=3,
        detail={
            "evidence": {
                "claims": claims,
                "supported": supported,
                "unsupported_samples": ["2025 年市场规模约 1200 亿美元"],
            },
            "completeness": {"truncated": False, "salvaged": False,
                             "missing": 0, **cdet},
        },
    )


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
        """产物不可读（E=None）是数据缺失，不是质量缺陷。"""
        findings, _ = diagnosis.from_trace(
            _clean_trace(), self.cfg, score=_score(E=None, C=0.9))
        self.assertNotIn(Taxonomy.EVIDENCE_FAILURE, [f.taxonomy for f in findings])

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


if __name__ == "__main__":
    unittest.main()
