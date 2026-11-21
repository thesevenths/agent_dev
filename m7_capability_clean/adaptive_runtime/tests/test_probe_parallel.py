"""并发调度与每 run 隔离：墙钟时间是离线闭环唯一的硬约束，但并发会把证据搅浑。

``ADAPTIVE_MAX_PARALLEL`` 打开后，两个 run 会同时在跑。危险的的不是 CPU 争抢，而是**产物归属**：
``runner_once`` 靠 ``AGENT_CHECKPOINT_DB`` / ``log`` / ``tmp`` / ``documents`` 隔离自己，
而 ``trace_export`` 是**按 mtime 扫描 ``documents/``/``tmp/``** 来认定「这些报告是本次 run 产出的」。
共用一份臂目录并发跑，A 的报告就会落进 B 的证据里 —— 得到的是一个很大但完全假的 ΔQ。
所以并发必须每 run 一份物理副本，本文件锁的就是这两件事：

1. ``run_jobs``：关掉开关（默认 ``workers=1``）时必须与串行旧代码逐字同序，不能悄悄改语义；
2. ``fork_sandbox`` / ``_env_for_run``：副本之间、副本与模板之间不共享产物目录，
   且离线开关由 harness 注入、**候选不能用 overrides 关掉它**。
"""
import dataclasses
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from adaptive_runtime import probe
from adaptive_runtime.probe import Sandbox, fork_sandbox, run_jobs
from adaptive_runtime.settings import CFG


class TestRunJobsSerialPath(unittest.TestCase):
    """workers<=1 是默认路径：求值顺序必须与 jobs 列表逐字一致。"""

    def test_serial_preserves_submission_order(self):
        order: list[str] = []
        jobs = [(k, (lambda k=k: order.append(k))) for k in ("a", "b", "c", "d")]
        res = run_jobs(jobs, workers=1)
        self.assertEqual(order, ["a", "b", "c", "d"])
        self.assertEqual(set(res), {"a", "b", "c", "d"})

    def test_single_job_is_serial_even_with_many_workers(self):
        order: list[str] = []
        run_jobs([("only", lambda: order.append("only"))], workers=8)
        self.assertEqual(order, ["only"])

    def test_should_stop_prevents_remaining_jobs_in_serial_mode(self):
        ran: list[str] = []
        jobs = [(k, (lambda k=k: ran.append(k))) for k in ("a", "b", "c")]
        res = run_jobs(jobs, workers=1, should_stop=lambda: len(ran) >= 1)
        self.assertEqual(ran, ["a"], "止损必须在 run 之间生效，否则预算形同虚设")
        self.assertEqual(list(res), ["a"])

    def test_serial_exception_propagates(self):
        def boom():
            raise ValueError("run failed")
        with self.assertRaises(ValueError):
            run_jobs([("a", boom)], workers=1)


class TestRunJobsPool(unittest.TestCase):
    def test_jobs_really_overlap(self):
        """不用 barrier 断言并发是无效的：串行执行时这里会超时抛错。"""
        n = 3
        barrier = threading.Barrier(n, timeout=10)

        def job():
            barrier.wait()
            return time.time()

        res = run_jobs([(f"r{i}", job) for i in range(n)], workers=n)
        self.assertEqual(len(res), n)
        self.assertTrue(all(isinstance(v, float) for v in res.values()))

    def test_results_are_keyed_not_positional(self):
        """乱序完成时结果必须仍挂在正确的 key 上。"""
        jobs = [("slow", lambda: time.sleep(0.25) or "S"),
                ("fast", lambda: "F")]
        res = run_jobs(jobs, workers=2)
        self.assertEqual(res, {"slow": "S", "fast": "F"})

    def test_should_stop_clears_pending_in_pool_mode(self):
        ran: list[str] = []

        def job(k: str):
            ran.append(k)
            return k

        res = run_jobs([(k, (lambda k=k: job(k))) for k in "abcdefgh"],
                       workers=2, should_stop=lambda: True)
        self.assertEqual(res, {}, "止损信号为真时不该再提交任何 run")
        self.assertEqual(ran, [])

    def test_pool_exception_is_visible(self):
        def boom():
            raise RuntimeError("cadence")
        with self.assertRaises(RuntimeError):
            run_jobs([("a", lambda: None), ("b", boom)], workers=2)


class TestEnvForRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.sb = Sandbox(path=Path(self._tmp.name) / "sb", arm="base", keep=True)
        self.sb.path.mkdir(parents=True, exist_ok=True)
        self.out = Path(self._tmp.name) / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def test_offline_is_injected_and_cannot_be_overridden_by_candidate(self):
        cfg = dataclasses.replace(CFG, offline=True)
        env = probe._env_for_run(self.sb, {"ADAPTIVE_OFFLINE": "0"}, self.out,
                                 fake=False, cfg=cfg)
        self.assertEqual(env["ADAPTIVE_OFFLINE"], "1",
                         "「base 联网 / cand 离线」会造出一个与 harness 无关的 ΔQ")

    def test_online_mode_does_not_inject(self):
        """非离线不主动注入；用干净环境是为了不拿宿主的残留变量当结论。"""
        import os
        from unittest import mock
        cfg = dataclasses.replace(CFG, offline=False)
        with mock.patch.dict(os.environ, {}, clear=True):
            env = probe._env_for_run(self.sb, {}, self.out, fake=False, cfg=cfg)
        self.assertNotEqual(env.get("ADAPTIVE_OFFLINE"), "1")

    def test_checkpoint_db_and_token_sink_are_per_run(self):
        sb2 = Sandbox(path=Path(self._tmp.name) / "sb2", arm="cand", keep=True)
        sb2.path.mkdir(parents=True, exist_ok=True)
        out2 = Path(self._tmp.name) / "out2"
        a = probe._env_for_run(self.sb, {}, self.out, fake=False, cfg=CFG)
        b = probe._env_for_run(sb2, {}, out2, fake=False, cfg=CFG)
        self.assertNotEqual(a["AGENT_CHECKPOINT_DB"], b["AGENT_CHECKPOINT_DB"])
        self.assertNotEqual(a["ADAPTIVE_TOKEN_SINK"], b["ADAPTIVE_TOKEN_SINK"])
        self.assertEqual(a["ADAPTIVE_TOKEN_SINK"], str(self.out / "tokens.jsonl"))
        self.assertIn(str(sb2.path), b["PYTHONPATH"])

    def test_parent_sink_is_purged_before_being_set(self):
        """继承来的 SINK 会把本次 run 的 token 记到别人的账本上。"""
        import os
        os.environ["ADAPTIVE_TOKEN_SINK"] = str(Path(self._tmp.name) / "stale.jsonl")
        try:
            env = probe._env_for_run(self.sb, {}, self.out, fake=False, cfg=CFG)
            self.assertEqual(env["ADAPTIVE_TOKEN_SINK"], str(self.out / "tokens.jsonl"))
        finally:
            os.environ.pop("ADAPTIVE_TOKEN_SINK", None)

    def test_fake_flag_is_separate_from_offline(self):
        cfg = dataclasses.replace(CFG, offline=True)
        env = probe._env_for_run(self.sb, {}, self.out, fake=True, cfg=cfg)
        self.assertEqual(env["ADAPTIVE_FAKE"], "1")
        self.assertEqual(env["ADAPTIVE_OFFLINE"], "1")


class TestForkSandboxIsolation(unittest.TestCase):
    """并发时每个 run 用自己那份副本：产物目录必须是空的。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tpl = probe.create_sandbox(Path(self._tmp.name) / "tpl", "base", keep=True)
        (self.tpl.path / "documents").mkdir(parents=True, exist_ok=True)
        (self.tpl.path / "documents" / "leftover.md").write_text("上一个 run 的残渣",
                                                                 encoding="utf-8")
        (self.tpl.path / "seed_marker.txt").write_text("template", encoding="utf-8")

    def tearDown(self):
        self.tpl.cleanup()
        self._tmp.cleanup()

    def test_artifact_dirs_are_empty_in_the_fork(self):
        fork = fork_sandbox(self.tpl, Path(self._tmp.name) / "forkA")
        self.addCleanup(fork.cleanup)
        self.assertTrue((fork.path / "documents").is_dir())
        self.assertEqual(list((fork.path / "documents").iterdir()), [],
                         "残留报告会污染 mtime 归属")

    def test_siblings_do_not_share_writes(self):
        a = fork_sandbox(self.tpl, Path(self._tmp.name) / "forkA")
        b = fork_sandbox(self.tpl, Path(self._tmp.name) / "forkB")
        self.addCleanup(a.cleanup)
        self.addCleanup(b.cleanup)
        (a.path / "tmp" / "only_a.md").write_text("A", encoding="utf-8")
        self.assertFalse((b.path / "tmp" / "only_a.md").exists())
        self.assertFalse((self.tpl.path / "tmp" / "only_a.md").exists())

    def test_fork_carries_source_files_but_keeps_template_untouched(self):
        fork = fork_sandbox(self.tpl, Path(self._tmp.name) / "forkC")
        self.addCleanup(fork.cleanup)
        self.assertTrue((fork.path / "seed_marker.txt").exists())
        (fork.path / "seed_marker.txt").write_text("modified by candidate", encoding="utf-8")
        self.assertEqual((self.tpl.path / "seed_marker.txt").read_text(encoding="utf-8"),
                         "template", "候选只能改自己的副本")

    def test_fork_replacing_existing_dest_is_clean(self):
        dest = Path(self._tmp.name) / "forkD"
        dest.mkdir(parents=True)
        (dest / "stale.txt").write_text("旧副本", encoding="utf-8")
        fork = fork_sandbox(self.tpl, dest)
        self.addCleanup(fork.cleanup)
        self.assertFalse((dest / "stale.txt").exists(),
                         "复用目录必须整体重建，否则碎片会顺着副本链传下去")
        self.assertEqual(fork.arm, "base")


class TestRunRetry(unittest.TestCase):
    """失败 run 重跑：缺任何一个 run 都会让整轮 INCOMPLETE_EVIDENCE。

    离线实测单 run 失败率 ≈ 3%，一轮 78 个 run 全干净的概率只有 ~9%。
    不重试 = 闭环九成轮次白烧墙钟（并且旧台账写法会把假设永久烧掉）。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.sb = Sandbox(path=Path(self._tmp.name) / "sb", arm="base", keep=True)
        self.sb.path.mkdir(parents=True, exist_ok=True)
        self.task = {"task_id": "T-1", "query": "q"}
        self.calls: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def _stub(self, fail_times: int, bucket: list[str] | None = None):
        """前 ``fail_times`` 次返回失败，之后成功（只统计调用次数，不真跑子进程）。"""
        def fake_run_one(sandbox, task, rep, arm, out_dir, overrides=None, cfg=None,
                         fake_profile=None):
            acc = self.calls if bucket is None else bucket
            ok = len(acc) >= fail_times        # 前 fail_times 次皆失败，之后成功
            acc.append(str(sandbox.path))
            return probe.RunOutcome(ok=ok, task_id="T-1", rep=rep, arm=arm,
                                    Q=80.0 if ok else 0.0, out_dir=str(out_dir))
        return fake_run_one

    def test_serial_retries_until_success(self):
        cfg = dataclasses.replace(CFG, max_parallel=1, probe_retry=2)
        with mock.patch.object(probe, "run_one", self._stub(2)):
            ro = probe.run_one_isolated(self.sb, self.task, 0, "base",
                                        Path(self._tmp.name) / "o", cfg=cfg)
        self.assertTrue(ro.ok, "前两次只是测量事故，第三次真的跑通了就该收下")
        self.assertEqual(ro.attempts, 3)
        self.assertEqual(len(self.calls), 3)

    def test_gives_up_after_configured_attempts(self):
        cfg = dataclasses.replace(CFG, max_parallel=1, probe_retry=1)
        with mock.patch.object(probe, "run_one", self._stub(99)):
            ro = probe.run_one_isolated(self.sb, self.task, 0, "base",
                                        Path(self._tmp.name) / "o", cfg=cfg)
        self.assertFalse(ro.ok)
        self.assertEqual(ro.attempts, 2)
        self.assertEqual(len(self.calls), 2, "重试次数必须有限，否则卡死的任务能跑到天亮")

    def test_probe_retry_zero_is_exactly_one_attempt(self):
        """``ADAPTIVE_PROBE_RETRY=0`` 必须回到旧行为：只跑一次。"""
        cfg = dataclasses.replace(CFG, max_parallel=1, probe_retry=0)
        with mock.patch.object(probe, "run_one", self._stub(99)):
            ro = probe.run_one_isolated(self.sb, self.task, 0, "base",
                                        Path(self._tmp.name) / "o", cfg=cfg)
        self.assertFalse(ro.ok)
        self.assertEqual(ro.attempts, 1)
        self.assertEqual(len(self.calls), 1)

    def test_concurrent_path_retries_with_a_fork_per_attempt(self):
        """并发分支：每次尝试都重新 fork，不能复用上一次尝试写脏的副本。"""
        cfg = dataclasses.replace(CFG, max_parallel=4, probe_retry=1)
        seen: list[str] = []
        with mock.patch.object(probe, "run_one", self._stub(1, seen)):
            ro = probe.run_one_isolated(self.sb, self.task, 0, "base",
                                        Path(self._tmp.name) / "o", cfg=cfg)
        self.assertTrue(ro.ok)
        self.assertEqual(ro.attempts, 2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1], "同一 run 的两次尝试共用同一个副本路径（重打而不是新建）")

    def test_probe_outcome_counts_retries(self):
        """重跑次数要写进 probe.json：零失败轮与满盘超时轮的证据等级不同。"""
        self.assertIn("retries", probe.ProbeOutcome().to_dict())


class TestDropStale(unittest.TestCase):
    def test_scoring_artifacts_are_removed_but_logs_survive(self):
        out = Path(tempfile.mkdtemp())
        (out / "score.json").write_text('{"Q": 99.0}', encoding="utf-8")
        (out / "result.json").write_text('{"ok": true}', encoding="utf-8")
        (out / "trace.json").write_text("{}", encoding="utf-8")
        (out / "usage.json").write_text("{}", encoding="utf-8")
        (out / "tokens.jsonl").write_text("{}\n", encoding="utf-8")
        (out / "stdout.log").write_text("可查的旧输出", encoding="utf-8")
        probe._drop_stale(out)
        for name in ("score.json", "result.json", "trace.json", "usage.json", "tokens.jsonl"):
            self.assertFalse((out / name).exists(),
                             f"{name} 若留着，失败的尝试会「继承」上一次的 Q —— 那是插补")
        self.assertTrue((out / "stdout.log").exists())


class TestStopLossUnderFreeTokens(unittest.TestCase):
    """不计费时止损要「幅度且显著」：误触一次就把本轮降到 4 个任务的配对。"""

    SMALL = [-5.0, -4.0, -6.0, -3.0, -5.0, -4.0, -6.0, -3.0]      # mean=-4.5 se≈0.42
    HUGE = [-30.0, -28.0, -32.0, -30.0, -29.0, -31.0, -30.0, -30.0]

    def test_governed_keeps_the_cheap_1_5_se_rule(self):
        cfg = dataclasses.replace(CFG, cost_governed=True)
        self.assertTrue(probe._stop_loss(self.SMALL, cfg))

    def test_free_tokens_does_not_stop_on_noise(self):
        """同样的数据在免费模式下不该止损：df=3 时本轮证据永远过不了门。"""
        cfg = dataclasses.replace(CFG, cost_governed=False, stop_loss_points=10.0,
                                  alpha=0.05)
        self.assertFalse(probe._stop_loss(self.SMALL, cfg))

    def test_free_tokens_stops_on_real_regression(self):
        cfg = dataclasses.replace(CFG, cost_governed=False, stop_loss_points=10.0)
        self.assertTrue(probe._stop_loss(self.HUGE, cfg))

    def test_single_delta_never_stops(self):
        cfg = dataclasses.replace(CFG, cost_governed=False)
        self.assertFalse(probe._stop_loss([-50.0], cfg))


if __name__ == "__main__":
    unittest.main()
