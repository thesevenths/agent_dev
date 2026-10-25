"""``auto.assess``：「是否实现 RRSI」必须是能从磁盘产物重放的判据表。

用户的要求是两条：闭环不许人工干预，以及**凭什么说它做到了**。第二条比第一条难，
因为它必须能被反驳。所以判据读的是 ``diagnosis.json`` / ``policy.json`` /
``base_history.jsonl`` / ``learning_curve.json`` 这些落盘工件，而不是运行时内存里的印象。

这里要锁住的是「判据不许自我拔高」：

- 只有 ``BLOCKING_CRITERIA`` 里那九条决定「达成」，观察项（环境健康、改进空间）
  再难看也不能把结论打成未达成 —— 反过来，环境塌了也必须看得见，否则「未达成」
  会被读成「agent 不行」，而实际问题在链路；
- 「上一次 auto 的 promote」不能为本次作证：判据只算 ``started_at`` 之后的事件；
- 只升版本号不改内容 = 自欺，所以 ``behavior_actually_changed`` 比的是内容指纹。
"""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from adaptive_runtime import auto, autopromote
from adaptive_runtime.settings import Paths

STAMP_LIVE = "stamp-live-production"
STARTED = "2026-10-23T09:00:00"


def _promote(when: str = "2026-10-23T10:00:00", who: str = "auto:adaptive",
             tag: str = "base/v4") -> dict:
    return {"event": "promote", "created_by": who, "created_at": when, "tag": tag,
            "evidence": [{"holdout": {}}, {"candidate": {"target": "t"}}]}


def _demote(when: str = "2026-10-23T11:00:00") -> dict:
    return {"event": "demote", "created_by": "auto:adaptive", "created_at": when,
            "tag": "base/v3"}


def _round(**kw) -> dict:
    r = {"round_no": 1, "diag_source": "round_traces", "decision": "ACCEPT",
         "holdout_pairs": 2, "holdout_mean_delta": 3.0, "reason": ""}
    r.update(kw)
    return r


class AssessBase(unittest.TestCase):
    """把 assess 的四个外部依赖（标定文件、版本台账、生产指纹、产物目录）指向临时目录。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = self.root / "auto" / "RUN-1"
        self.out.mkdir(parents=True)
        self.cal = self.root / "calibration.json"

        mods = dataclasses.replace(Paths(), calibration=self.cal, experiments=self.root)
        self._patch = [
            mock.patch.object(auto, "PATHS", mods),
            mock.patch.object(autopromote, "PATHS", mods),
            mock.patch.object(auto, "landing",
                              SimpleNamespace(content_stamp=lambda: STAMP_LIVE)),
        ]
        for p in self._patch:
            p.start()
        self._write_cal()
        self._write_diagnosis(["T1", "T2"])

    def tearDown(self):
        for p in self._patch:
            p.stop()
        self._tmp.cleanup()

    # ---- 工件构造 ----
    def _write_cal(self, payload: dict | None = None):
        if payload is None:
            payload = {"passed": True, "sd_diff_null": 8.4, "k": 4,
                       "tasks": 8, "n_pairs": 48, "target": 15.0}
        if payload:
            self.cal.write_text(json.dumps(payload), encoding="utf-8")
        elif self.cal.exists():
            self.cal.unlink()

    def _write_diagnosis(self, task_ids: list[str], *, excluded: int = 2,
                         source: str = "round_traces") -> None:
        d = self.out / "bench" / "PROBE-1" / "round1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "diagnosis.json").write_text(json.dumps(
            {"summary": {"source": source, "task_ids": task_ids,
                         "holdout_runs_excluded": excluded}}), encoding="utf-8")

    def _assess(self, hist: list[dict], state: dict | None = None):
        stub = SimpleNamespace(history=lambda: hist,
                               current=lambda: SimpleNamespace(version=4))
        with mock.patch.object(auto, "version", stub):
            return auto.assess(self.out, state if state is not None else self._state())

    def _state(self, **kw) -> dict:
        st = {"started_at": STARTED, "curve": [_round()],
              "holdout_ids": ["H1", "H2"],
              "content_stamp_before": "stamp-before",
              "content_stamp_after": STAMP_LIVE,
              "verifies": [{"ok": True, "tag": "base/v4", "drift": 0.5,
                            "tol": -5.0, "n": 2}]}
        st.update(kw)
        return st

    def _failed(self, verdict: dict) -> set[str]:
        return {c["name"] for c in verdict["criteria"] if not c["passed"]}


class TestGreenPath(AssessBase):
    def test_all_nine_blocking_criteria_pass(self):
        v = self._assess([_promote()])
        self.assertTrue(v["achieved"], [c for c in v["criteria"] if not c["passed"]])
        self.assertEqual(v["n_blocking"], len(auto.BLOCKING_CRITERIA))
        self.assertEqual(v["passed_blocking"], v["n_blocking"])
        self.assertEqual(self._failed(v) & set(auto.BLOCKING_CRITERIA), set())

    def test_observations_present_but_non_blocking(self):
        v = self._assess([_promote()])
        obs = {c["name"] for c in v["criteria"] if not c["blocking"]}
        self.assertEqual(obs, {"environment_healthy", "improvement_capacity"})

    def test_environment_healthy_passes_when_no_usage_artifacts(self):
        v = self._assess([_promote()])
        c = [x for x in v["criteria"] if x["name"] == "environment_healthy"][0]
        self.assertTrue(c["passed"])
        self.assertEqual(c["detail"]["infra_runs"], 0)


class TestEachCriterionCanFail(AssessBase):
    """每一条硬条件都必须能被单独拦下：否则「达成」是偶然得来的。"""

    def _one(self, name: str, hist, state=None):
        v = self._assess(hist, state)
        self.assertFalse(v["achieved"], f"{name} 应当是硬条件")
        self.assertIn(name, self._failed(v))
        return v

    def test_missing_calibration_blocks_noise_calibrated(self):
        self._write_cal({})
        self._one("noise_calibrated", [_promote()])

    def test_degenerate_calibration_blocks_even_when_passed_true(self):
        """今天加上的守卫：sd=0 / validity 非空的报表不能为 δ 背书。"""
        self._write_cal({"passed": True, "sd_diff_null": 0.0, "k": 2,
                         "validity": ["重复任务数 2 < 最少 4"]})
        v = self._one("noise_calibrated", [_promote()])
        c = [x for x in v["criteria"] if x["name"] == "noise_calibrated"][0]
        self.assertTrue(c["detail"]["validity"])

    def test_manual_log_diagnosis_source_blocks_self_contained(self):
        self._one("loop_self_contained", [_promote()],
                  self._state(curve=[_round(diag_source="manual_log")]))

    def test_no_candidate_round_is_exempt_from_provenance(self):
        """没提案的轮次没有诊断输入，不该因为 diag_source 被判泄漏。"""
        v = self._assess([_promote()],
                         self._state(curve=[_round(diag_source="manual_log",
                                                   reason="no_candidate")]))
        self.assertTrue(v["achieved"], self._failed(v))

    def test_holdout_leak_blocks_quarantine(self):
        self._write_diagnosis(["T1", "H1"])
        self._one("holdout_quarantined", [_promote()])

    def test_previous_run_promote_does_not_vouch_for_this_one(self):
        self._one("auto_promote_happened",
                  [_promote(when="2026-10-22T10:00:00")],
                  self._state(curve=[_round()], verifies=[{"ok": True, "drift": 0.0,
                                                           "tol": -5.0, "n": 2}]))

    def test_human_signoff_in_window_blocks(self):
        self._one("no_human_promote_in_run",
                  [_promote(), _promote(who="user", tag="base/v5")])

    def test_version_bump_without_content_change_blocks(self):
        self._one("behavior_actually_changed", [_promote()],
                  self._state(content_stamp_before=STAMP_LIVE,
                              content_stamp_after=STAMP_LIVE))

    def test_negative_holdout_delta_blocks_paired_gain(self):
        self._one("holdout_paired_gain", [_promote()],
                  self._state(curve=[_round(holdout_mean_delta=-4.0)]))

    def test_missing_post_promote_verify_blocks(self):
        self._one("post_promote_stable", [_promote()], self._state(verifies=[]))

    def test_bad_drift_blocks(self):
        self._one("post_promote_stable", [_promote()],
                  self._state(verifies=[{"ok": False, "tag": "base/v4", "drift": -9.0,
                                         "tol": -5.0, "n": 2}]))

    def test_demote_after_promote_blocks(self):
        self._one("no_unresolved_demote", [_promote(), _demote()])

    def test_repromote_after_demote_resolves(self):
        hist = [_promote(tag="base/v4"), _demote(),
                _promote(when="2026-10-23T12:00:00", tag="base/v5")]
        self.assertTrue(self._assess(hist)["achieved"], self._failed(self._assess(hist)))


class TestObservationDoesNotBlock(AssessBase):
    def test_infrastructure_failures_are_visible_but_not_blocking(self):
        for i, (calls, errs) in enumerate([(4, 4), (20, 3), (6, 6)]):
            rd = self.out / "bench" / "PROBE-1" / "base" / f"T{i}-r0"
            rd.mkdir(parents=True)
            (rd / "usage.json").write_text(
                json.dumps({"calls": calls, "errors": errs, "total": 1000}),
                encoding="utf-8")
        v = self._assess([_promote()])
        c = [x for x in v["criteria"] if x["name"] == "environment_healthy"][0]
        self.assertEqual(c["detail"]["infra_runs"], 2)
        self.assertEqual(c["detail"]["runs_with_usage"], 3)
        self.assertFalse(c["passed"])
        self.assertFalse(c["blocking"])
        self.assertTrue(v["achieved"], "链路故障不该把「已达成」判成未达成，但必须看得见")

    def test_legacy_usage_without_calls_field_is_not_counted(self):
        """旧产物里没有 calls：不能把历史抹成故障。"""
        rd = self.out / "bench" / "PROBE-1" / "base" / "OLD-r0"
        rd.mkdir(parents=True)
        (rd / "usage.json").write_text(json.dumps({"total": 123}), encoding="utf-8")
        v = self._assess([_promote()])
        c = [x for x in v["criteria"] if x["name"] == "environment_healthy"][0]
        self.assertEqual((c["detail"]["infra_runs"], c["detail"]["runs_with_usage"]), (0, 0))

    def test_every_round_exhausted_is_observed_not_blocking(self):
        v = self._assess([_promote()],
                         self._state(curve=[_round(decision="REJECT", reason="no_candidate",
                                                   holdout_pairs=1, holdout_mean_delta=1.0)]))
        cap = [x for x in v["criteria"] if x["name"] == "improvement_capacity"][0]
        self.assertFalse(cap["passed"])
        self.assertFalse(cap["blocking"])


class TestVerdictArtifact(AssessBase):
    def test_write_verdict_lands_json_and_markdown(self):
        state = self._state()
        with mock.patch.object(auto, "version",
                               SimpleNamespace(history=lambda: [_promote()],
                                               current=lambda: SimpleNamespace(version=4))):
            v = auto.write_verdict(self.out, state)
        self.assertTrue((self.out / "verdict.json").exists())
        md = (self.out / auto.VERDICT_NAME).read_text(encoding="utf-8")
        self.assertIn("**已达成**", md)
        self.assertIn("noise_calibrated", md)
        self.assertIn("这份判据为什么不是自证", md)
        self.assertEqual(v["out_dir"], str(self.out))

    def test_unachieved_verdict_lists_what_is_missing(self):
        self._write_cal({})
        state = self._state()
        with mock.patch.object(auto, "version",
                               SimpleNamespace(history=lambda: [_promote()],
                                               current=lambda: SimpleNamespace(version=4))):
            auto.write_verdict(self.out, state)
        md = (self.out / auto.VERDICT_NAME).read_text(encoding="utf-8")
        self.assertIn("**未达成**", md, "结论行必须是未达成")
        self.assertIn("## 未达成时缺什么", md)
        self.assertIn("noise_calibrated", md.split("## 未达成时缺什么")[1])


class TestResumeAccounting(AssessBase):
    """续跑记账必须**累计**：判据是跨批读的，不能被下一批的空 tally 抹掉。

    真实跑法就是一批一轮地续跑（单批 6 小时，中途断供就会把已跑轮次全丢），而
    ``state.json`` 只在整批跑完后写。于是「一轮一批」是唯一能存活的跑法，而它要求
    ``run_auto`` 把上一批的 verifies/promotes 带回来 —— 旧实现只带了 promotes。
    """

    def _batch2(self, prior: dict, *, hist: list[dict], new_round: dict) -> dict:
        tasks = [{"task_id": "T1"}, {"task_id": "T2"}]
        # 上一批跑完 1 轮：state 里的 rounds_done 是续跑编号的根，缺它就会被当成从头开始
        prior = dict({"rounds_done": 1}, **prior, task_hash=auto._task_hash(tasks),
                     label="RUN-1")
        auto.save_state(self.out, prior)
        bench = SimpleNamespace(
            run_benchmark=lambda *a, **kw: {
                "curve": [new_round], "holdout_ids": ["H1"], "train_ids": ["T1", "T2"],
                "base_version": 4,
                "delta_series": [], "holdout_delta_series": [], "holdout_drift": 0.0,
                "diag_sources": ["round_traces"], "accepted": 0, "rejected": 1,
                "skipped": 0},
            base_sweep=lambda *a, **kw: None)
        verse = SimpleNamespace(history=lambda: hist,
                               current=lambda: SimpleNamespace(version=4))
        with mock.patch.object(auto, "benchmark", bench), \
                mock.patch.object(auto, "version", verse):
            rep = auto.run_auto(tasks, rounds=2, label="RUN-1",
                                out_dir=self.out.parent)
        self.assertNotIn("error", rep)
        return rep["state"]

    def _no_gain_round(self) -> dict:
        return _round(round_no=1, decision="REJECT", reason="within_noise",
                      holdout_pairs=0, holdout_mean_delta=0.0)

    def test_verifies_survive_a_batch_without_promote(self):
        st = self._batch2(self._state(), hist=[_promote()], new_round=self._no_gain_round())
        self.assertEqual(len(st["verifies"]), 1,
                         "第 2 批没 promote，不能把第 1 批的落地后复查记成没发生过")
        self.assertTrue(st["verifies"][0]["ok"])
        v = self._assess([_promote()], st)
        self.assertTrue([c for c in v["criteria"]
                         if c["name"] == "post_promote_stable"][0]["passed"],
                        "判据 8 不得因续跑记账而由绿转红")

    def test_promote_count_is_cumulative_across_batches(self):
        """限次保险看的是累计值：每批重置就等于允许无限次换基线。"""
        st = self._batch2(dict(self._state(), promotes=1, demotes=0),
                          hist=[_promote()], new_round=self._no_gain_round())
        self.assertEqual(st["promotes"], 1)
        self.assertEqual(st["rounds_done"], 2)
        self.assertEqual([r["round_no"] for r in st["curve"]], [1, 2],
                         "学习曲线得接着往后排，不能两个 round 1")


if __name__ == "__main__":
    unittest.main()
