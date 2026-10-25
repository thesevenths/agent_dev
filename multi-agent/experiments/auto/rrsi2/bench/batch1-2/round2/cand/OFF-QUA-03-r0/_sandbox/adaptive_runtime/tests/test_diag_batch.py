"""``diag_batch``：诊断输入必须是「本轮自己跑出来的产物」，且反序列化不能失真。

这个模块存在的理由（``benchmark.cmd_round`` 的老 bug）：
原实现用 ``_newest_log()`` 取 ``log/`` 里 mtime 最新的日志当诊断输入 —— 人工在 Studio
里随手问一次，闭环下一轮的「症状」就被换成了一份与本轮 probe 毫无因果关系的历史记录。
无人值守时这不闭合，所以诊断源改成 run 目录里的 ``trace.json`` + ``score.json``。

被锁住的行为分三类：

1. **反序列化保真**：``Trace``/``Score`` 只有 ``to_dict()``，没有反向构造。``steps`` 的
   int 键被 JSON 变成 str，``to_dict()`` 还附加了一个 ``derived`` 块。若还原不成立，
   ``classify`` 拿到的就是 dict 而不是 ``StepState``，失败信号读不出来 —— 于是
   「全部跑崩」会被诊断成「无失败信号」，闭环对着一个不存在的问题开药。
2. **泄漏与假跑防线**：holdout 任务混进诊断池 ⇒ 泛化证据作废；假轨迹（dQ 恒为 +38.1）
   混进真实轮次 ⇒ 闭环给自己编的病历开药。两者都必须**机器排除**，不靠调用方自觉。
3. **来源可证**：``summary["source"]`` 与 ``is_usable`` 让报告能自证「吃的是本轮产物」，
   并在没有产物 / UNKNOWN 爆表时拒绝继续。
"""
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

from adaptive_runtime import diag_batch, settings
from adaptive_runtime.evaluator import Score
from adaptive_runtime.io_util import write_json
from adaptive_runtime.trace_export import StepState, Trace


# ---------------------------------------------------------------- 造 run 目录

def _trace(*, failures: int = 0, kind: str = "exception", last: str = "fail",
           rejects: int = 0, steps: int = 2, deliverable: bool = False) -> Trace:
    tr = Trace(run_id="run-1", log_path="log/x.log", goal="g", query="q",
               total_steps=steps,
               plan=[{"step": i + 1} for i in range(steps)],
               events=[{"type": "step_failure", "step_no": 1, "why": "tool crashed"}]
               if failures else [])
    for i in range(steps):
        tr.steps[i + 1] = StepState(step=i + 1, passes=0 if failures else 1,
                                    rejects=rejects, failures=failures,
                                    last=last, fail_kind=kind if failures else "",
                                    agent="CodeAgent")
    if deliverable:
        tr.artifacts = [{"path": "tmp/report.md", "deliverable": True, "bytes": 5000}]
    return tr


def _score(task_id: str, run_name: str, **kw) -> Score:
    return Score(run_id=run_name, task_id=task_id, arm="base",
                 A=kw.pop("A", 0.5), Q=kw.pop("Q", 50.0), **kw)


def _make_run(probe: Path, task_id: str, rep: int, *, arm: str = "base",
              trace: Trace | None = None, score: Score | None = None,
              fake: bool | None = None) -> Path:
    rd = probe / arm / f"{task_id}-r{rep}"
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "trace.json", (trace or _trace()).to_dict())
    if score is not None:
        write_json(rd / "score.json", score.to_dict())
    if fake is not None:
        write_json(rd / "result.json", {"fake": fake, "ok": True})
    return rd


# ---------------------------------------------------------------- 反序列化

class TestLoadTrace(unittest.TestCase):
    def test_roundtrip_restores_int_keys_and_stepstate(self):
        tr = _trace(failures=2, kind="recursion")
        back = diag_batch.load_trace(json.loads(json.dumps(tr.to_dict())))
        self.assertEqual(set(back.steps), {1, 2})
        self.assertIsInstance(back.steps[1], StepState)
        self.assertEqual(back.steps[1].failures, 2)
        self.assertEqual(back.steps[1].fail_kind, "recursion")
        self.assertFalse(back.steps[1].achieved)

    def test_derived_block_is_dropped_not_passed_to_constructor(self):
        """``to_dict()`` 附加的 ``derived`` 是 property 的结果，当构造参数会 TypeError。"""
        tr = _trace(failures=0, last="pass")
        data = tr.to_dict()
        self.assertIn("derived", data)
        back = diag_batch.load_trace(data)
        self.assertEqual(back.passed_steps, tr.passed_steps)   # property 重算，不靠 derived
        self.assertEqual(back.contract_rate, tr.contract_rate)

    def test_non_integer_step_keys_are_skipped(self):
        data = _trace().to_dict()
        data["steps"]["oops"] = {"step": 99}
        back = diag_batch.load_trace(data)
        self.assertNotIn("oops", back.steps)

    def test_all_crashed_run_is_not_diagnosed_as_no_signal(self):
        """反序列化失真的后果：崩掉的 run 被读成「无失败信号」。"""
        tr = _trace(failures=3, kind="exception")
        restored = diag_batch.load_trace(json.loads(json.dumps(tr.to_dict())))
        taxonomies = [f.taxonomy for f in diag_batch.diagnosis.classify(restored)]
        self.assertNotIn("UNKNOWN", taxonomies)
        self.assertTrue(any("EXECUTION" in t or "TOOL" in t for t in taxonomies), taxonomies)


class TestLoadScore(unittest.TestCase):
    def test_empty_or_missing_is_none(self):
        self.assertIsNone(diag_batch.load_score({}))
        self.assertIsNone(diag_batch.load_score(None))

    def test_missing_both_Q_and_A_is_none(self):
        self.assertIsNone(diag_batch.load_score({"run_id": "r", "flags": ["no_report"]}))

    def test_full_score_roundtrips(self):
        sc = _score("T1", "T1-r0", A=0.9, Q=72.5, flags=["ok"])
        back = diag_batch.load_score(sc.to_dict())
        self.assertEqual((back.task_id, back.A, back.Q, back.flags), ("T1", 0.9, 72.5, ["ok"]))


# ---------------------------------------------------------------- 扫描与排除

class TestFromRunDirs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.probe = Path(self._tmp.name) / "PROBE-1"

    def tearDown(self):
        self._tmp.cleanup()

    def test_iter_run_dirs_sorted_and_requires_trace(self):
        _make_run(self.probe, "T2", 0)
        _make_run(self.probe, "T1", 0)
        (self.probe / "base" / "ZZ-no-trace").mkdir(parents=True, exist_ok=True)
        got = [p.name for p in diag_batch.iter_run_dirs(self.probe)]
        self.assertEqual(got, ["T1-r0", "T2-r0"])
        self.assertEqual(diag_batch.iter_run_dirs(self.probe / "nope"), [])

    def test_holdout_is_mechanically_excluded(self):
        _make_run(self.probe, "T1", 0, score=_score("T1", "T1-r0"))
        _make_run(self.probe, "H1", 0, score=_score("H1", "H1-r0"))
        findings, summary = diag_batch.from_run_dirs(
            self.probe, exclude_task_ids=["H1"])
        self.assertEqual(summary["holdout_runs_excluded"], 1)
        self.assertEqual(summary["n_runs"], 1)
        self.assertEqual(summary["task_ids"], ["T1"])
        self.assertNotIn("H1", json.dumps(summary))

    def test_fake_runs_are_excluded_unless_explicitly_allowed(self):
        _make_run(self.probe, "T1", 0, fake=True, trace=_trace(failures=2))
        _make_run(self.probe, "T2", 0, fake=False)
        _, blocked = diag_batch.from_run_dirs(self.probe)
        self.assertEqual(blocked["fake_runs_excluded"], 1)
        self.assertEqual(blocked["task_ids"], ["T2"], "假跑不得进入真实轮次的症状池")
        _, allowed = diag_batch.from_run_dirs(self.probe, allow_fake=True)
        self.assertEqual(allowed["fake_runs_excluded"], 0)
        self.assertEqual(set(allowed["task_ids"]), {"T1", "T2"})

    def test_missing_score_falls_back_to_run_dir_name_as_task_id(self):
        rd = _make_run(self.probe, "OFF-STAT-01", 0, score=None)
        self.assertEqual(rd.name, "OFF-STAT-01-r0")
        _, summary = diag_batch.from_run_dirs(self.probe)
        self.assertEqual(summary["task_ids"], ["OFF-STAT-01"])
        self.assertEqual(summary["n_scored"], 0, "没有 score.json 就不该谎称已评分")

    def test_source_marks_provenance(self):
        _make_run(self.probe, "T1", 0)
        _, summary = diag_batch.from_run_dirs(self.probe)
        self.assertEqual(summary["source"], "round_traces")
        self.assertEqual(summary["arm"], "base")

    def test_use_score_false_skips_quality_channel(self):
        """低分但每一步都成功：只有带上 score 才会出现「质量缺陷」这一路。"""
        _make_run(self.probe, "T1", 0, trace=_trace(failures=0, last="pass"),
                  score=_score("T1", "T1-r0", A=0.0, Q=5.0))
        _, with_score = diag_batch.from_run_dirs(self.probe, use_score=True)
        _, no_score = diag_batch.from_run_dirs(self.probe, use_score=False)
        self.assertEqual(with_score["n_scored"], 1)
        self.assertEqual(no_score["n_scored"], 0)


# ---------------------------------------------------------------- 可用性与落盘

class TestIsUsableAndWrite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.probe = Path(self._tmp.name) / "PROBE-1"

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_runs_is_unusable(self):
        _, summary = diag_batch.from_run_dirs(self.probe)
        ok, why = diag_batch.is_usable(summary)
        self.assertFalse(ok)
        self.assertIn("trace.json", why)

    def test_unknown_ratio_over_threshold_blocks(self):
        """每一步都成功 ⇒ 硬通道无信号 ⇒ 全是 UNKNOWN ⇒ UNKNOWN 爆表必须停下。"""
        _make_run(self.probe, "T1", 0, trace=_trace(failures=0, last="pass"))
        _, summary = diag_batch.from_run_dirs(self.probe)
        self.assertGreater(summary["unknown_ratio"], settings.CFG.unknown_max)
        ok, why = diag_batch.is_usable(summary)
        self.assertFalse(ok, "UNKNOWN 爆表时闭环不该继续提案")
        self.assertTrue(why)

    def test_clean_failure_evidence_is_usable(self):
        _make_run(self.probe, "T1", 0, trace=_trace(failures=2, kind="recursion",
                                                    deliverable=True))
        findings, summary = diag_batch.from_run_dirs(self.probe)
        ok, why = diag_batch.is_usable(summary)
        self.assertTrue(ok, why)
        out = diag_batch.write_diagnosis(self._tmp.name, findings, summary)
        payload = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["source"], "round_traces")
        self.assertEqual(payload["unknown_max"], settings.CFG.unknown_max)
        self.assertTrue(payload["findings"])
        for v in payload["by_taxonomy"].values():
            self.assertIsInstance(v["agents"], list, "set 不能 JSON 序列化")


# ---------------------------------------------------------------- 最近轮次发现

class TestRecentTraceDirs(unittest.TestCase):
    """闭环只能从「有真实产物的目录」里续跑，且顺序必须是新→旧。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runs = root / "runs"
        self.mods = dataclasses.replace(settings.PATHS,
                                        experiments=root, runs=self.runs)
        self._orig = settings.PATHS
        settings.PATHS = self.mods

    def tearDown(self):
        settings.PATHS = self._orig
        self._tmp.cleanup()

    def _probe(self, name: str, *, fake: bool, age: int) -> Path:
        probe = self.runs / name
        rd = _make_run(probe, "T1", 0, fake=fake)
        write_json(rd / "trace.json", _trace(failures=1).to_dict())
        st = 1_700_000_000 - age
        os.utime(probe, (st, st))
        return probe

    def test_all_fake_probe_is_skipped(self):
        self._probe("PROBE-FAKE", fake=True, age=0)
        self.assertEqual(diag_batch.recent_trace_dirs(), [])

    def test_orders_by_mtime_desc_and_respects_limit(self):
        old = self._probe("PROBE-OLD", fake=False, age=500)
        new = self._probe("PROBE-NEW", fake=False, age=0)
        self.assertEqual(diag_batch.recent_trace_dirs(limit=2), [new, old],
                         "最新一轮必须是第一个")
        self.assertEqual(diag_batch.recent_trace_dirs(), [new],
                         "默认只取最近一轮（limit=1）")

    def test_calibration_and_auto_roots_are_also_scanned(self):
        probe = self.mods.experiments / "calibration" / "CALB-1"
        _make_run(probe, "OFF-STAT-01", 0, fake=False)
        got = diag_batch.recent_trace_dirs()
        self.assertEqual([p.name for p in got], ["CALB-1"])


if __name__ == "__main__":
    unittest.main()
