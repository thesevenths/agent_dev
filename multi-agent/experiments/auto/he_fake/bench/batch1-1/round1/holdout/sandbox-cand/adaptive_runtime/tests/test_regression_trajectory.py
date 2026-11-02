"""Regression 集（§13.4）与弱化版轨迹对比（§13.5）的单元测试。"""
import tempfile
import types
import unittest
from pathlib import Path

from adaptive_runtime import regression, trajectory
from adaptive_runtime.settings import CFG, Paths
from adaptive_runtime.trace_export import StepState, Trace


def _trace(run_id, steps, counters=None, total=None, truncated=False, salvaged=False):
    t = Trace(run_id=run_id, log_path="x.log")
    t.steps = steps
    t.total_steps = total if total is not None else len(steps)
    t.counters = counters or {}
    t.result = {"truncated": truncated, "salvaged": salvaged}
    return t


def _ok(run_id, tools=10):
    return _trace(run_id, {1: StepState(1, passes=1, last="pass", agent="CodeAgent"),
                           2: StepState(2, passes=1, last="pass", agent="CodeAgent")},
                  counters={"tool": tools})


def _bad(run_id, tools=40, kind="recursion"):
    return _trace(run_id, {1: StepState(1, passes=1, last="pass", agent="CodeAgent"),
                           2: StepState(2, failures=1, last="fail", fail_kind=kind,
                                        agent="CodeAgent", rejects=3)},
                  counters={"tool": tools})


class TestRegressionSet(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = regression.PATHS
        regression.PATHS = Paths(regression=Path(self._tmp.name))

    def tearDown(self):
        regression.PATHS = self._saved
        self._tmp.cleanup()

    def test_add_is_idempotent(self):
        tasks = [{"task_id": "T1", "query": "q1", "family": "crypto"}]
        self.assertEqual(regression.add_from_accepted(tasks, "c1"), 1)
        self.assertEqual(regression.add_from_accepted(tasks, "c2"), 0)
        self.assertEqual(len(regression.load()), 1)

    def test_coverage_reports_gap_honestly(self):
        regression.add_from_accepted([{"task_id": "T1", "query": "q", "family": "a"}])
        cov = regression.coverage()
        self.assertFalse(cov["meets_spec"])          # 只有 1 个任务，不该说满足
        self.assertEqual(cov["count"], 1)

    def test_check_not_covered_is_not_zero(self):
        cmp = types.SimpleNamespace(per_task={"OTHER-1": {"mean_delta": -30.0}})
        out = regression.check(cmp, {"T1"}, CFG)
        self.assertFalse(out["covered"])
        self.assertIsNone(out["regression_rate"])     # 没测 ≠ 0%
        self.assertIsNone(out["ok"])

    def test_check_detects_regression(self):
        cmp = types.SimpleNamespace(per_task={"T1": {"mean_delta": -30.0},
                                              "T2": {"mean_delta": 1.0},
                                              "T3": {"mean_delta": 0.5},
                                              "T4": {"mean_delta": 2.0},
                                              "T5": {"mean_delta": 1.5},
                                              "T6": {"mean_delta": 0.1},
                                              "T7": {"mean_delta": 0.2}})
        out = regression.check(cmp, {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}, CFG)
        self.assertTrue(out["covered"])
        self.assertEqual(out["regressed"], ["T1"])
        self.assertAlmostEqual(out["regression_rate"], 1 / 7, places=4)
        self.assertTrue(out["ok"])                    # 1/7 < R_MAX

    def test_select_rotates_families(self):
        tasks = [{"task_id": f"a{i}", "query": "q", "family": "a"} for i in range(3)]
        tasks += [{"task_id": f"b{i}", "query": "q", "family": "b"} for i in range(3)]
        picked = regression.select_for_probe(tasks, 4)
        self.assertEqual(len(picked), 4)
        self.assertEqual(len({p["family"] for p in picked}), 2)


class TestTrajectory(unittest.TestCase):
    def test_usable_requires_both_groups(self):
        rep = trajectory.contrast([_ok("a"), _ok("b")])
        self.assertFalse(rep["usable"])

    def test_contrast_computes_metrics(self):
        rep = trajectory.contrast([_ok("a", tools=10), _bad("b", tools=40)])
        self.assertTrue(rep["usable"])
        m = {x["metric"]: x for x in rep["metrics"]}
        self.assertIn("工具往返 / 步", m)
        self.assertGreater(m["工具往返 / 步"]["fail_mean"], m["工具往返 / 步"]["ok_mean"])

    def test_failure_clusters(self):
        rep = trajectory.contrast([_ok("a"), _bad("b"), _bad("c")])
        self.assertEqual(rep["failure_clusters"][0]["kind"], "recursion")
        self.assertEqual(rep["failure_clusters"][0]["count"], 2)

    def test_caveat_always_present(self):
        for traces in ([], [_ok("a")], [_ok("a"), _bad("b")]):
            self.assertTrue(trajectory.contrast(traces)["caveat"])

    def test_ok_definition_excludes_truncated(self):
        t = _trace("x", {1: StepState(1, passes=1, last="pass")}, truncated=True)
        ok, bad = trajectory.split([t])
        self.assertEqual((len(ok), len(bad)), (0, 1))

    def test_markdown_marks_insufficient_sample(self):
        md = trajectory.to_markdown(trajectory.contrast([_ok("a")]))
        self.assertIn("样本不足", md)
        self.assertIn(trajectory.CAVEAT[:10], md)


if __name__ == "__main__":
    unittest.main()
