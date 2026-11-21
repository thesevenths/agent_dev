"""L2 提示补丁 + 反刷分（reward hacking）防线的测试。

本组用例针对两类「表面正常、实际未拦截」的高风险区域：
- L2：锚点写错时 patch 静默不生效，候选 ΔQ≈0 被记为「假设无效」，台账被无效证据污染。
- 反刷分：A 的分母为计划步数、C 的分母为计划产物数，两条路径均可通过「减少工作量」提分。
"""
import tempfile
import unittest
from pathlib import Path

from adaptive_runtime import proposer
from adaptive_runtime.comparator import PairRow, compare
from adaptive_runtime.evaluator import structure_score
from adaptive_runtime.gate import Evidence, Reason, gate_b
from adaptive_runtime.ledger import Ledger
from adaptive_runtime.probe import PatchError, apply_prompt_patch, create_sandbox
from adaptive_runtime.settings import CFG, Paths


class TestPromptPatches(unittest.TestCase):
    def test_all_patches_have_unique_anchor(self):
        """锚点必须恰好出现一次，否则 patch 会插错位置或失败。"""
        self.assertEqual(proposer.PROMPT_PATCH_PROBLEMS, [])

    def test_patch_line_budget(self):
        for name, spec in proposer.PROMPT_PATCHES.items():
            lines = [ln for ln in spec["insert"].splitlines() if ln.strip()]
            self.assertLessEqual(len(lines), proposer.MAX_PROMPT_DIFF_LINES,
                                 f"{name} 超出 {proposer.MAX_PROMPT_DIFF_LINES} 行")

    def test_level_2_candidates_appear_only_when_opened(self):
        old = CFG.max_edit_level
        try:
            CFG.max_edit_level = 1
            c1 = proposer.propose("COMPLETION_FAILURE", Ledger(), cfg=CFG, limit=20)
            self.assertTrue(all(c.level <= 1 for c in c1))

            CFG.max_edit_level = 2
            c2 = proposer.propose("COMPLETION_FAILURE", Ledger(), cfg=CFG, limit=20)
            self.assertTrue(any(c.level == 2 for c in c2), "打开 L2 后应出现提示层候选")
            self.assertTrue(all(c.level <= 2 for c in c2))
        finally:
            CFG.max_edit_level = old

    def test_prompt_candidate_carries_full_patch(self):
        old = CFG.max_edit_level
        try:
            CFG.max_edit_level = 2
            cands = [c for c in proposer.propose("COMPLETION_FAILURE", Ledger(),
                                                 cfg=CFG, limit=20)
                     if c.level == 2]
            self.assertTrue(cands)
            c = cands[0]
            self.assertTrue(c.is_prompt)
            self.assertEqual(c.env_overrides(), {}, "提示补丁不得走 env 通道")
            for key in ("file", "anchor", "insert"):
                self.assertTrue(c.patch.get(key), f"缺字段 {key}")
            self.assertIn(c.patch["anchor"], c.diff_text())
        finally:
            CFG.max_edit_level = old

    def test_apply_patch_only_touches_cand_sandbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = create_sandbox(Path(tmp) / "sb", "cand", keep=True)
            patch = proposer.PROMPT_PATCHES["prompt.py::coder.stop-on-repeat"]
            info = apply_prompt_patch(sb, {"file": "prompt.py", **patch})
            self.assertEqual(info["file"], "prompt.py")
            self.assertGreater(info["added"], 0)
            txt = (sb.path / "prompt.py").read_text(encoding="utf-8")
            self.assertIn(patch["insert"].splitlines()[0], txt)
            # 仍必须是合法 Python（否则整条 run 都会语法错误，而不是「效果不好」）
            compile(txt, str(sb.path / "prompt.py"), "exec")

    def test_apply_patch_fails_loudly_on_bad_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = create_sandbox(Path(tmp) / "sb", "cand", keep=True)
            with self.assertRaises(PatchError):
                apply_prompt_patch(sb, {"file": "prompt.py",
                                        "anchor": "<<<不存在的锚点>>>",
                                        "insert": "x\n"})

    def test_missing_fields_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = create_sandbox(Path(tmp) / "sb", "cand", keep=True)
            with self.assertRaises(PatchError):
                apply_prompt_patch(sb, {"file": "prompt.py", "anchor": "x"})


class TestStructureScore(unittest.TestCase):
    def test_one_line_is_not_a_report(self):
        s, d = structure_score("1+1 等于 2。", min_bytes=100)
        self.assertEqual(s, 0.0)
        self.assertFalse(d["size"])

    def test_full_structure(self):
        text = ("# 标题\n\n## 一\n\n正文\n\n| a | b |\n| - | - |\n\n## 二\n\n"
                "结论：增长质量良好。" + "补充说明。" * 60)
        s, d = structure_score(text, min_bytes=100)
        self.assertEqual(s, 1.0, d)


class TestPlanUndercut(unittest.TestCase):
    """A = 通过门数/计划步数 → 砍掉计划步数就能抬高 A，必须单独拦。"""

    def _cmp(self, pairs):
        return compare([{"task_id": t, "rep": 0, "Q": bq, "A": ba, "planned": bpl}
                        for t, bq, ba, bpl, _, _, _ in pairs],
                       [{"task_id": t, "rep": 0, "Q": cq, "A": ca, "planned": cpl}
                        for t, _, _, _, cq, ca, cpl in pairs])

    def test_undercut_detected(self):
        cmp = self._cmp([("T1", 50.0, 0.5, 5, 60.0, 1.0, 1),
                         ("T2", 50.0, 0.5, 5, 60.0, 1.0, 1)])
        ev = cmp.evidence()
        self.assertEqual(ev.plan_undercut_rate, 1.0)
        self.assertFalse(gate_b(ev).accepted)

    def test_shorter_plan_without_a_gain_is_fine(self):
        # 效率提升导致步数变少、但 A 没上升 → 不是刷分
        cmp = self._cmp([("T1", 50.0, 0.6, 5, 48.0, 0.6, 2),
                         ("T2", 50.0, 0.6, 5, 48.0, 0.6, 2)])
        self.assertEqual(cmp.evidence().plan_undercut_rate, 0.0)

    def test_no_gain_at_all(self):
        cmp = self._cmp([("T1", 50.0, 0.5, 5, 50.0, 0.5, 5)])
        self.assertEqual(cmp.evidence().plan_undercut_rate, 0.0)


class TestTaskLevelAggregation(unittest.TestCase):
    """同一任务的多个 rep 不是独立样本，不能用来抬高自由度。"""

    def test_reps_collapse_into_one_task(self):
        # 1 个任务 × 4 rep，delta 恒定 → 任务级 n=1，无法做 t 检验
        base = [{"task_id": "T1", "rep": r, "Q": 50.0, "A": 0.5, "planned": 4} for r in range(4)]
        cand = [{"task_id": "T1", "rep": r, "Q": 70.0, "A": 0.9, "planned": 4} for r in range(4)]
        ev = compare(base, cand).evidence()
        self.assertEqual(len(ev.deltas), 4)          # 配对仍是 4 条
        self.assertEqual(len(ev.task_deltas), 1)     # 但只有 1 个独立任务
        self.assertEqual(ev.paired().n, 1)
        self.assertFalse(ev.paired().positive)

    def test_single_task_rejected_by_min_tasks(self):
        base = [{"task_id": "T1", "rep": 0, "Q": 50.0, "A": 0.5, "planned": 4}]
        cand = [{"task_id": "T1", "rep": 0, "Q": 90.0, "A": 1.0, "planned": 4}]
        ev = compare(base, cand).evidence()
        dec = gate_b(ev)
        self.assertFalse(dec.accepted)
        self.assertEqual(dec.reason, Reason.INCOMPLETE_EVIDENCE.value)

    def test_two_tasks_can_pass_the_floor(self):
        base = [{"task_id": t, "rep": 0, "Q": 50.0, "A": 0.5, "planned": 4} for t in ("T1", "T2")]
        cand = [{"task_id": t, "rep": 0, "Q": 90.0, "A": 1.0, "planned": 4} for t in ("T1", "T2")]
        ev = compare(base, cand).evidence()
        dec = gate_b(ev)
        self.assertTrue(dec.accepted, dec.reason)

    def test_zero_variance_across_tasks_is_still_valid(self):
        """多个任务上 delta 完全一致 → sd=0 是真实的一致使然，允许通过。"""
        base = [{"task_id": t, "rep": 0, "Q": 50.0, "A": 0.5, "planned": 4} for t in ("T1", "T2", "T3")]
        cand = [{"task_id": t, "rep": 0, "Q": 90.0, "A": 1.0, "planned": 4} for t in ("T1", "T2", "T3")]
        ev = compare(base, cand).evidence()
        self.assertEqual(ev.paired().sd, 0.0)
        self.assertTrue(gate_b(ev).accepted)


class TestEvidenceDefaults(unittest.TestCase):
    def test_pairrow_defaults_keep_backward_compat(self):
        p = PairRow(task_id="T", rep=0, base_Q=1.0, cand_Q=2.0)
        self.assertEqual(p.base_planned, 0)
        self.assertEqual(p.cand_planned, 0)

    def test_evidence_without_task_deltas_falls_back(self):
        ev = Evidence(deltas=[1.0, 2.0, 3.0])
        self.assertEqual(ev.paired().n, 3)


if __name__ == "__main__":
    unittest.main()
