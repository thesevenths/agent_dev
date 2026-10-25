"""benchmark：fake 模式绝不能污染台账/经验库。

为什么单独测这一条：fake 轨迹是伪造的（dQ 恒为 +38.1），一旦写进台账，
真实轮次的 `Ledger.is_duplicate` 会把这些假设当成"已试过"而跳过，
经验库也会把假结论检索进 Task Harness —— 系统会带着假记忆继续跑，且毫无提示。

诊断输入这里显式传 `trace=`：本测试只管「fake 守卫」这一件事，走哪条诊断来源
与它无关（默认那条 `round_traces` 路径由 `test_diag_batch` 覆盖）。之所以必须
给一个真实存在的日志 —— `run_benchmark` 会 `Path(log).exists()`，不存在就 SKIP，
那样「没写台账」是流程没走到，而不是守卫生效（假通过）。
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_runtime import benchmark
from adaptive_runtime.comparator import compare
from adaptive_runtime.experience import Store
from adaptive_runtime.ledger import Ledger
from adaptive_runtime.probe import ProbeOutcome
from adaptive_runtime.trace_export import StepState, Trace

TASKS = [{"task_id": f"T{i}", "query": f"任务 {i}"} for i in range(4)]


def _fake_outcome(cand):
    base = [{"task_id": t["task_id"], "rep": 0, "Q": 50.0, "A": 0.5, "planned": 4}
            for t in TASKS]
    cands = [{"task_id": t["task_id"], "rep": 0, "Q": 90.0, "A": 1.0, "planned": 4}
             for t in TASKS]
    return ProbeOutcome(base=base, cand=cands, comparison=compare(base, cands),
                        tokens=1234, n_planned=4, n_actual=4, out_dir="")


class TestFakeDoesNotPollute(unittest.TestCase):
    def _run(self, fake_profiles):
        ledger_calls, store_calls = [], []
        trace = Trace(run_id="r1", log_path="x.log")
        trace.plan = [{"step_no": 1, "status": "pending", "title": "撰写报告", "agent": "code_agent"}]
        trace.total_steps = 1
        trace.steps = {1: StepState(1, passes=1, last="pass", agent="code_agent")}

        # 必须给一个**真实存在**的日志路径：run_benchmark 会 Path(log).exists()，
        # 不存在就走「没有可诊断日志」分支直接 SKIP，于是 ledger 永远不会被写 ——
        # 那样这条测试会假通过（"不写"是因为流程没走到，不是因为守卫生效）。
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / "x.log"
        log.write_text("", encoding="utf-8")

        with mock.patch.object(benchmark, "export_from_log", return_value=trace), \
             mock.patch.object(benchmark, "probe_candidate",
                               side_effect=lambda cand, *a, **k: _fake_outcome(cand)), \
             mock.patch.object(benchmark, "cleanup", lambda *a: None), \
             mock.patch.object(benchmark, "_holdout_base_quality", return_value=(0.0, 0)), \
             mock.patch.object(Ledger, "update", lambda self, *a, **k: ledger_calls.append(a)), \
             mock.patch.object(Store, "create", lambda self, *a, **k: store_calls.append(k)):
            rep = benchmark.run_benchmark(
                TASKS, rounds=2, n=4, holdout_ratio=0.25,
                out_dir=Path(tmp.name) / "out", task_label="t", trace=log,
                fake_profiles=fake_profiles)
        return rep, ledger_calls, store_calls, trace

    def test_fake_leaves_ledger_and_experience_empty(self):
        rep, ledger_calls, store_calls, _ = self._run({"base": "base", "cand": "good"})
        self.assertEqual(ledger_calls, [], "fake 模式写了台账")
        self.assertEqual(store_calls, [], "fake 模式写了经验库")
        # 确认流程确实跑到了判定环节（而不是半路 SKIP 导致的假通过）
        self.assertTrue(rep["accepted"] + rep["rejected"] > 0,
                        f"流程没跑到判定环节：{rep['verdict']}")

    def test_real_mode_does_write(self):
        """反向验证：非 fake 时必须写，否则上面那条测试可能是「因为不写所以不写」。"""
        rep, ledger_calls, store_calls, _ = self._run(None)
        self.assertTrue(ledger_calls, f"真实模式应当写台账：{rep['verdict']}")
        self.assertTrue(store_calls, "真实模式应当写经验库")

    def test_manual_trace_is_flagged_as_non_self_contained(self):
        """人工指定日志必须被标成 manual_log —— 无人值守靠这个标记判断闭环是否自洽。"""
        rep, _, _, _ = self._run({"base": "base", "cand": "good"})
        self.assertEqual(rep["diag_sources"], ["manual_log"],
                         f"--trace 走的是人工日志，provenance 必须如实标注：{rep['diag_sources']}")


if __name__ == "__main__":
    unittest.main()
