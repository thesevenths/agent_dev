"""基础设施故障不得被当成「质量信号」：run 级报废 + 标定级不可信。

为什么这组用例存在（一次真实事故的复盘）
------------------------------------------
自托管 vLLM 的 TLS 链路被系统代理掐断后，闭环照样「跑成功」了 4 个 run：
每个 run 的 ``result.json`` 都是 ``ok=true``，分数全是 **Q=5.0**（地板分），
因为 agent 一次 LLM 调用都没成功，什么也没产出，判分器忠实地给了最低分。

后果不是「分数低」，而是**判分口径被环境故障接管**：

1. ``calibrate`` 看到「同一任务两次都 5.0」→ sd(Δ)=0 → 「噪声远低于目标 15，标定通过」；
   δ 随之归零，之后任何正数的 ΔQ 都会被判「显著改进」。
2. 配对比较把两次网络故障读成「两臂持平」，真改进被记成无效。
3. 诊断层把这批 run 当成本轮症状，闭环开始给自己没跑过的失败开药。

一条「sd=0 的标定」比没有标定危险得多，因为它是**假通过**：机器判据会当成尺子可信。
所以这里同时锁两层：执行器必须把这种 run 标成失败（不进样本），
标定必须把退化样本标成不可信（不发布）。
"""
import unittest
from types import SimpleNamespace

from adaptive_runtime import calibrate
from adaptive_runtime.runner_once import _infra_error
from adaptive_runtime.settings import CFG


def _score(A: float):
    return SimpleNamespace(A=A)


class TestRunLevelInfraGuard(unittest.TestCase):
    """``_infra_error``：测的是端点可达性，就不是 harness。"""

    def test_all_llm_calls_failed_is_infra(self):
        got = _infra_error({"calls": 4, "errors": 4}, _score(0.0))
        self.assertIsNotNone(got, "全部调用失败必须报废，否则地板分 5.0 会进样本")
        self.assertIn("4", got)

    def test_zero_llm_calls_is_infra(self):
        """一次调用都没记录到：探针没收到请求或请求在发出前就崩了，同样不是质量信号。"""
        self.assertIsNotNone(_infra_error({"calls": 0, "errors": 0}, _score(0.0)))

    def test_missing_usage_fields_default_to_no_calls(self):
        self.assertIsNotNone(_infra_error({}, _score(0.0)))

    def test_majority_errors_without_any_achievement_is_infra(self):
        got = _infra_error({"calls": 10, "errors": 6}, _score(0.0))
        self.assertIsNotNone(got, "过半调用失败且没有任何步骤达成 = 链路故障")

    def test_healthy_run_is_not_flagged(self):
        """误报的代价是白扔一次真跑（十几分钟），所以只在证据明确时报废。"""
        self.assertIsNone(_infra_error({"calls": 20, "errors": 3}, _score(0.9)))

    def test_minority_errors_with_zero_achievement_is_real_quality(self):
        """少量调用失败但确实达成了步骤：这是 harness 的表现，必须保留记账。"""
        self.assertIsNone(_infra_error({"calls": 20, "errors": 2}, _score(0.5)))

    def test_all_failed_but_a_achieved_still_infra(self):
        """A>0 也救不回来：所有 LLM 调用都失败时，A 只可能来自上一个 run 的残留状态。"""
        self.assertIsNotNone(_infra_error({"calls": 5, "errors": 5}, _score(1.0)))


class TestCalibrationValidity(unittest.TestCase):
    """标定报表的「能不能信」必须与「噪声大小」分开判定。"""

    def test_zero_variance_is_not_a_pass(self):
        groups = {f"T{i}": [5.0, 5.0] for i in range(CFG.min_calib_tasks)}
        rep = calibrate.summarize(groups)
        self.assertEqual(rep["sd_diff_null"], 0.0)
        self.assertFalse(rep["passed"], "sd=0 是样本退化，不是「噪声极低」")
        self.assertTrue(rep["validity"])
        self.assertFalse(calibrate.passes(rep))

    def test_too_few_repeated_tasks_is_not_a_pass(self):
        groups = {"T1": [50.0, 52.0], "T2": [60.0], "T3": [40.0]}
        rep = calibrate.summarize(groups)
        self.assertFalse(rep["passed"])
        self.assertTrue(any("任务" in x for x in rep["validity"]),
                        f"应说明任务数不足：{rep['validity']}")

    def test_healthy_sample_passes(self):
        """真实但有限的抖动：sd(Δ) 落在目标内、样本够格 ⇒ 必须通过（别把守卫写成死门）。"""
        groups = {f"T{i}": [60.0 + (j % 2) * (1.0 + i * 0.1) for j in range(4)]
                  for i in range(CFG.min_calib_tasks)}
        rep = calibrate.summarize(groups)
        self.assertEqual(rep["validity"], [], f"合格样本不该被拦：{rep['validity']}")
        self.assertTrue(rep["passed"], rep)
        self.assertTrue(calibrate.passes(rep))

    def test_n_pairs_counts_only_repeat_tasks(self):
        self.assertEqual(calibrate.n_pairs({"A": [1, 2, 3], "B": [4.0]}), 3)

    def test_passes_falls_back_when_validity_absent(self):
        """旧的 calibration.json 没有 validity 字段：不能因此把已有实测标定判死。"""
        legacy = {"passed": True, "sd_diff_null": 8.0}
        self.assertTrue(calibrate.passes(legacy))


if __name__ == "__main__":
    unittest.main()
