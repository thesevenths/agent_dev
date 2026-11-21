"""归因三元组（base_commit / patch_hash / judge_version）与 quarantine 的回归测试。

为什么这三项要有测试：它们是 V3 §16 B 组的验收项，也是「同一 Q 是否可比」的前提。
一旦 patch_hash 用 candidate_id 冒充（每次运行都变）或 judge_version 固定不变，
历史经验会被静默错配到新尺子上，而结果看起来完全正常 —— 这种错最难发现。
"""
import tempfile
import types
import unittest
from pathlib import Path

from adaptive_runtime import evaluator, proposer, quarantine
from adaptive_runtime.settings import CFG


def _cand():
    return proposer.Candidate(
        candidate_id="cand_test", target="AGENT_PARSE_TRIES", level=1,
        old="3", new="5", hypothesis="h",
        patch={"kind": "env", "key": "AGENT_PARSE_TRIES", "old": "3", "new": "5"})


class TestAttributionTriple(unittest.TestCase):
    def test_patch_hash_is_content_based(self):
        a, b = _cand(), _cand()
        b.candidate_id = "cand_other"
        self.assertEqual(a.patch_hash, b.patch_hash)
        b.new = "9"
        self.assertNotEqual(a.patch_hash, b.patch_hash)

    def test_patch_hash_in_to_dict(self):
        d = _cand().to_dict()
        self.assertEqual(d["patch_hash"], _cand().patch_hash)
        self.assertEqual(len(d["patch_hash"]), 16)

    def test_judge_version_is_stable_and_sensitive(self):
        v1 = evaluator.judge_version()
        self.assertEqual(v1, evaluator.judge_version())

        class C:
            pass

        c = C()
        for k, v in CFG.to_dict().items():
            setattr(c, k, v)
        c.w_a, c.w_e, c.w_c = 0.9, 0.05, 0.05
        c.regression_min_points = CFG.regression_min_points
        c.w_a, c.w_e, c.w_c = 0.9, 0.05, 0.05
        c.normalized_weights = lambda: (0.9, 0.05, 0.05)
        self.assertNotEqual(v1, evaluator.judge_version(c))


class TestQuarantine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = quarantine.PATHS
        quarantine.PATHS = types.SimpleNamespace(quarantine=Path(self._tmp.name))

    def tearDown(self):
        quarantine.PATHS = self._saved
        self._tmp.cleanup()

    def test_record_then_find(self):
        cand = _cand().to_dict()
        cand["base_commit"] = "abc1234"
        payload = quarantine.record(cand, "within_noise", "gate_b", round_dir="/tmp/r")
        self.assertTrue(payload["file"].endswith(".json"))
        self.assertEqual(payload["patch_hash"], cand["patch_hash"])
        found = quarantine.find_by_patch(cand["patch_hash"])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["reason"], "within_noise")

    def test_index_is_append_only(self):
        cand = _cand().to_dict()
        quarantine.record(cand, "regression", "gate_b")
        quarantine.record(cand, "within_noise", "gate_b")
        self.assertEqual(len(quarantine.entries()), 2)
        self.assertEqual(quarantine.summary()["total"], 2)
        self.assertEqual(set(quarantine.summary()["by_reason"]),
                         {"regression", "within_noise"})

    def test_files_created(self):
        cand = _cand().to_dict()
        quarantine.record(cand, "cost_rule", "gate_b")
        files = list(Path(self._tmp.name).glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertTrue(quarantine.index_path().exists())


if __name__ == "__main__":
    unittest.main()
