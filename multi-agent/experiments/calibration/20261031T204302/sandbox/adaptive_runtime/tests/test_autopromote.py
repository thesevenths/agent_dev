"""``autopromote``：无人工门的换基线必须比人工评审更难通过。

RRSI 的要求是「不需要人工干预」，但 ``version.promote`` 原本禁止 system 操作者 ——
原因不是流程洁癖，而是**无锚漂移**：拿上一轮候选当新基线，正向噪声会被逐轮累积成
「看起来一直在进步」。所以自动 promote 的做法是换一组机器可判定、且更硬的证据，
而不是取消门槛。这个测试锁的就是「硬」的部分：

- 每一条 blocking 条件都必须能单独把 promote 拦下来（否则「全部通过」是可被偶然达成的）；
- 未覆盖的回归集只能告警不能阻断（否则首轮任何机制都上不了线）；
- 标定这一条必须拒掉「退化样本给出的 sd=0」——那是假通过，δ 归零后任何噪声都算显著改进。
"""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive_runtime import autopromote
from adaptive_runtime.settings import CFG, Paths


def _cand(target: str = "降低 CodeAgent 重试次数") -> SimpleNamespace:
    return SimpleNamespace(target=target, level=1, patch_hash="abc123",
                           to_dict=lambda: {"target": target, "level": 1})


def _cmp(deltas: list[float]) -> SimpleNamespace:
    """假的 holdout 配对比较结果（``comparator`` 的返回形状）。"""
    return SimpleNamespace(
        pairs=[SimpleNamespace(delta=d) for d in deltas],
        per_task={f"H{i}": d for i, d in enumerate(deltas)},
    )


class CalibrationFile(unittest.TestCase):
    """``calibration_ok`` 的基类：把 ``PATHS.calibration`` 指到临时文件。

    注意必须 patch ``autopromote.PATHS`` 而不是 ``settings.PATHS`` —— 本模块是
    ``from .settings import PATHS``，导入时已绑定，改 settings 里的名字不影响它。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cal_path = Path(self._tmp.name) / "calibration.json"
        self._orig = autopromote.PATHS
        autopromote.PATHS = dataclasses.replace(Paths(), calibration=self.cal_path)

    def tearDown(self):
        autopromote.PATHS = self._orig
        self._tmp.cleanup()

    def _write(self, payload: dict) -> None:
        self.cal_path.write_text(json.dumps(payload), encoding="utf-8")


class TestCalibrationOk(CalibrationFile):
    def test_absent_file_is_not_ok(self):
        ok, detail = autopromote.calibration_ok()
        self.assertFalse(ok)
        self.assertFalse(detail["exists"])

    def test_measured_pass_is_ok(self):
        self._write({"passed": True, "sd_diff_null": 8.4, "k": 4})
        ok, detail = autopromote.calibration_ok()
        self.assertTrue(ok, detail)

    def test_failed_calibration_is_not_ok(self):
        self._write({"passed": False, "sd_diff_null": 40.0, "k": 4})
        self.assertFalse(autopromote.calibration_ok()[0])

    def test_zero_sd_is_not_ok(self):
        self._write({"passed": True, "sd_diff_null": 0.0, "k": 2})
        self.assertFalse(autopromote.calibration_ok()[0],
                         "sd=0 会让 δ=0，之后任何正数 ΔQ 都判显著")

    def test_validity_warnings_are_blocking(self):
        """今天加上的守卫：报表自己承认样本退化时，passed=true 也不能背书。"""
        self._write({"passed": True, "sd_diff_null": 4.17, "k": 2,
                     "validity": ["重复任务数 2 < 最少 4"]})
        ok, detail = autopromote.calibration_ok()
        self.assertFalse(ok)
        self.assertTrue(detail["validity"])

    def test_legacy_report_without_validity_key_still_counts(self):
        self._write({"passed": True, "sd_diff_null": 6.0, "k": 3})
        self.assertTrue(autopromote.calibration_ok()[0])


class TestPolicy(CalibrationFile):
    def setUp(self):
        super().setUp()
        self._write({"passed": True, "sd_diff_null": 8.4, "k": 4})

    def _eval(self, **kw):
        base = dict(cand=_cand(), dec_b_accepted=True, stopped="",
                    holdout_cmp=_cmp([2.0, 3.0]), regression=None,
                    frozen_violations=[], promotes_done=0, cfg=CFG)
        base.update(kw)
        return autopromote.evaluate(**base)

    def test_all_evidence_present_allows(self):
        pol = self._eval(regression={"covered": True, "ok": True,
                                     "regression_rate": 0.0, "regressed": 0,
                                     "r_max": 0.1})
        self.assertTrue(pol.allowed, pol.blocking)
        self.assertEqual(pol.blocking, [])
        self.assertEqual(pol.warnings, [])

    def test_gate_reject_blocks(self):
        pol = self._eval(dec_b_accepted=False)
        self.assertFalse(pol.allowed)
        self.assertIn("evolve_gate", pol.blocking)

    def test_stop_loss_blocks_even_if_gate_accepted(self):
        pol = self._eval(stopped="stop_loss")
        self.assertIn("evolve_gate", pol.blocking)

    def test_single_holdout_task_blocks_as_insufficient_evidence(self):
        pol = self._eval(holdout_cmp=_cmp([9.0]))
        self.assertIn("holdout_evidence", pol.blocking,
                      "一个任务的 holdout 是「一次噪声」，不是泛化证据")

    def test_holdout_regression_blocks(self):
        pol = self._eval(holdout_cmp=_cmp([-CFG.holdout_tol - 1.0,
                                           -CFG.holdout_tol - 3.0]))
        self.assertIn("holdout_no_regression", pol.blocking)

    def test_holdout_flat_is_allowed(self):
        """门槛是「不退步」而非「必须涨」：0 必须放过，否则机制永远上不了线。"""
        self.assertTrue(self._eval(holdout_cmp=_cmp([0.0, 0.0])).allowed)

    def test_frozen_violation_blocks(self):
        pol = self._eval(frozen_violations=["agents.py:_rule_gate 被修改"])
        self.assertIn("frozen_intact", pol.blocking)

    def test_promote_budget_blocks(self):
        pol = self._eval(promotes_done=CFG.auto_promote_max)
        self.assertIn("promote_budget", pol.blocking)

    def test_covered_regression_failure_blocks(self):
        pol = self._eval(regression={"covered": True, "ok": False,
                                     "regression_rate": 0.3, "regressed": 2,
                                     "r_max": 0.1})
        self.assertIn("regression_gate", pol.blocking)

    def test_uncovered_regression_warns_but_does_not_block(self):
        """首轮 probe 只覆盖 train 任务；要求覆盖回归集会让任何机制都上不了线。"""
        pol = self._eval(regression={"covered": False})
        self.assertTrue(pol.allowed)
        self.assertIn("regression_gate", pol.warnings)

    def test_missing_calibration_blocks_promote(self):
        self.cal_path.unlink()
        pol = self._eval()
        self.assertIn("calibration", pol.blocking)

    def test_policy_to_dict_is_audit_shape(self):
        d = self._eval(regression={"covered": False}).to_dict()
        self.assertEqual(set(d), {"allowed", "blocking", "warnings", "checks"})
        names = {c["name"] for c in d["checks"]}
        self.assertEqual(names, {"calibration", "evolve_gate", "holdout_evidence",
                                 "holdout_no_regression", "regression_gate",
                                 "frozen_intact", "promote_budget"})
        for c in d["checks"]:
            self.assertIn(c["status"], ("ok", "BLOCK", "warn"))


class TestCheckStatus(unittest.TestCase):
    def test_status_vocabulary(self):
        C = autopromote.Check
        self.assertEqual(C("a", True).status, "ok")
        self.assertEqual(C("a", False, blocking=True).status, "BLOCK")
        self.assertEqual(C("a", False, blocking=False).status, "warn")


class TestVerifyAndRollbackEvidence(unittest.TestCase):
    def test_within_tol_is_ok(self):
        r = autopromote.verify_after_promote(holdout_prev_mean=70.0,
                                             holdout_new_mean=70.0 - CFG.holdout_tol / 2,
                                             tol=CFG.holdout_tol, prev_tag="base/v3")
        self.assertTrue(r["ok"])
        self.assertEqual(r["drift"], -CFG.holdout_tol / 2)

    def test_dropping_below_tol_is_not_ok(self):
        r = autopromote.verify_after_promote(holdout_prev_mean=70.0,
                                             holdout_new_mean=70.0 - CFG.holdout_tol * 3,
                                             tol=CFG.holdout_tol, prev_tag="base/v3")
        self.assertFalse(r["ok"], "换基线后 holdout 明显更差 = 该回滚")

    def test_improvement_is_ok(self):
        self.assertTrue(autopromote.verify_after_promote(
            holdout_prev_mean=60.0, holdout_new_mean=75.0,
            tol=CFG.holdout_tol, prev_tag="base/v3")["ok"])


class TestVerifyAfterPromotePaired(unittest.TestCase):
    """给了任务级配对输入后，回滚必须是「幅度超容忍 **且** 配对检验显著」。"""

    PREV = {"T1": 80.0, "T2": 60.0, "T3": 75.0, "T4": 65.0}          # 均值 70

    def _v(self, new):
        return autopromote.verify_after_promote(
            holdout_prev_mean=70.0, holdout_new_mean=sum(new.values()) / len(new),
            tol=CFG.holdout_tol, prev_tag="base/v3",
            prev_by_task=dict(self.PREV), new_by_task=new)

    def test_noisy_drift_does_not_rollback(self):
        # 漂移 -6.25 超容忍，但逐任务差值 [-15, +1, -1, -10] 方向不一致 → 噪声
        r = self._v({"T1": 65.0, "T2": 61.0, "T3": 74.0, "T4": 55.0})
        self.assertEqual(r["mode"], "paired")
        self.assertFalse(r["significant_regression"])
        self.assertTrue(r["ok"], "误回滚会同时打掉 auto_promote_happened 与 no_unresolved_demote")

    def test_consistent_regression_rolls_back(self):
        r = self._v({"T1": 68.0, "T2": 48.0, "T3": 63.0, "T4": 53.0})   # 每任务都 -12
        self.assertEqual(r["mode"], "paired")
        self.assertTrue(r["significant_regression"])
        self.assertFalse(r["ok"])

    def test_small_common_set_falls_back_to_means(self):
        r = autopromote.verify_after_promote(
            holdout_prev_mean=70.0, holdout_new_mean=50.0, tol=CFG.holdout_tol,
            prev_tag="base/v3", prev_by_task={"T1": 70.0}, new_by_task={"T1": 50.0})
        self.assertEqual(r["mode"], "means_only")
        self.assertFalse(r["ok"])

    def test_legacy_signature_unchanged(self):
        """不传配对输入（旧调用方）：行为与改动前一致。"""
        r = autopromote.verify_after_promote(holdout_prev_mean=70.0,
                                             holdout_new_mean=69.0, tol=5.0,
                                             prev_tag="base/v3")
        self.assertEqual(r["mode"], "means_only")
        self.assertTrue(r["ok"])
        self.assertNotIn("significant_regression", r)


if __name__ == "__main__":
    unittest.main()
