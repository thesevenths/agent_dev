"""演化正则化与版本模型的单元测试（exploration / pruning / base promote）。"""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from adaptive_runtime import evolution, version
from adaptive_runtime.io_util import read_json
from adaptive_runtime.proposer import SEARCH_SPACE


def _row(target, status, reason="", taxonomy="completion_failure", ts=None):
    return {"fingerprint": f"{target}|1->2|{taxonomy}", "status": status,
            "evidence": {"reason": reason},
            "created_at": ts or datetime.now().isoformat(timespec="seconds")}


class TestOutcome(unittest.TestCase):
    def test_accepted(self):
        self.assertEqual(evolution.outcome_of(_row("X", "accepted")), "accepted")

    def test_within_noise_is_not_plain_inconclusive(self):
        # 台账把 within_noise 记成 inconclusive，但 pruning 必须能把它单独识别出来
        r = _row("X", "inconclusive", reason="within_noise")
        self.assertEqual(evolution.outcome_of(r), "within_noise")

    def test_inconclusive_without_reason(self):
        r = _row("X", "inconclusive", reason="incomplete_evidence")
        self.assertEqual(evolution.outcome_of(r), "inconclusive")


class TestExploration(unittest.TestCase):
    def test_single_level_never_explores(self):
        # P0 只开放 L1，层间不存在冷落，不应误触发
        hit, why = evolution.should_explore([], evolution.CFG)
        self.assertFalse(hit)
        self.assertEqual(why["reason"], "single_level")

    def test_cold_shoulder_triggers(self):
        ss = {"A": {"level": 1, "taxonomies": ("completion_failure",)},
              "B": {"level": 2, "taxonomies": ("completion_failure",)}}

        class C:
            max_edit_level = 2

        rows = [_row("A", "accepted") for _ in range(6)]
        hit, why = evolution.should_explore(rows, C(), ss)
        self.assertTrue(hit)
        self.assertEqual(why["reason"], "cold_shoulder")

    def test_balanced_does_not_trigger(self):
        ss = {"A": {"level": 1, "taxonomies": ("completion_failure",)},
              "B": {"level": 2, "taxonomies": ("completion_failure",)}}

        class C:
            max_edit_level = 2

        rows = [_row("A", "accepted")] * 3 + [_row("B", "accepted")] * 3
        hit, _ = evolution.should_explore(rows, C(), ss)
        self.assertFalse(hit)

    def test_neglected_pairs_cover_search_space(self):
        pairs = evolution.neglected_pairs([], SEARCH_SPACE)
        self.assertTrue(pairs)
        self.assertTrue(all(lv <= 3 for _, lv in pairs))


class TestPruning(unittest.TestCase):
    def test_streak_stops_at_accepted(self):
        rows = [_row("AGENT_PARSE_TRIES", "inconclusive", "within_noise")] * 3
        rows.append(_row("AGENT_PARSE_TRIES", "accepted"))
        rows.extend([_row("AGENT_PARSE_TRIES", "inconclusive", "within_noise")] * 2)
        self.assertEqual(evolution.within_noise_streak(rows, "AGENT_PARSE_TRIES"), 2)

    def test_prune_after_threshold(self):
        rows = [_row("AGENT_PARSE_TRIES", "inconclusive", "within_noise")
                for _ in range(evolution.PRUNE_STREAK)]
        out = evolution.prune_candidates(rows)
        self.assertEqual([p["target"] for p in out], ["AGENT_PARSE_TRIES"])

    def test_not_pruned_below_threshold(self):
        rows = [_row("AGENT_PARSE_TRIES", "inconclusive", "within_noise")
                for _ in range(evolution.PRUNE_STREAK - 1)]
        self.assertEqual(evolution.prune_candidates(rows), [])

    def test_dormant_expires(self):
        old = (datetime.now() - timedelta(days=evolution.DORMANT_DAYS + 2)).isoformat(timespec="seconds")
        rows = [_row("AGENT_PARSE_TRIES", "inconclusive", "within_noise", ts=old)
                for _ in range(evolution.PRUNE_STREAK)]
        self.assertEqual(evolution.dormant_targets(rows), set())

    def test_dormant_active(self):
        rows = [_row("AGENT_PARSE_TRIES", "inconclusive", "within_noise")
                for _ in range(evolution.PRUNE_STREAK)]
        self.assertIn("AGENT_PARSE_TRIES", evolution.dormant_targets(rows))

    def test_evolve_plan_excludes_dormant(self):
        rows = [_row("AGENT_PARSE_TRIES", "inconclusive", "within_noise")
                for _ in range(evolution.PRUNE_STREAK)]
        plan = evolution.evolve_plan(rows, "completion_failure")
        self.assertIn("AGENT_PARSE_TRIES", plan["exclude_targets"])


class TestVersionModel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (version._state_path, version._history_path,
                       version.env_snapshot, version.git_head)
        root = Path(self._tmp.name)
        version._state_path = lambda: root / "base_state.json"
        version._history_path = lambda: root / "base_history.jsonl"
        version.env_snapshot = lambda: {"AGENT_N": "15"}
        version.git_head = lambda root=None: "abc1234"
        self.root = root

    def tearDown(self):
        (version._state_path, version._history_path,
         version.env_snapshot, version.git_head) = self._saved
        self._tmp.cleanup()

    def test_ensure_initial_is_idempotent(self):
        a = version.ensure_initial()
        b = version.ensure_initial()
        self.assertEqual(a.version, 1)
        self.assertEqual(a.tag, "base/v1")
        self.assertEqual(b.version, 1)
        self.assertEqual(len(version.history()), 1)

    def test_promote_requires_human(self):
        version.ensure_initial()
        for bad in ("system", "auto", "robot", "  "):
            with self.assertRaises(ValueError):
                version.promote([], operator=bad)

    def test_promote_bumps_version(self):
        version.ensure_initial()
        new = version.promote([{"experience_id": "e1"}], operator="alice", note="ok")
        self.assertEqual(new.version, 2)
        self.assertEqual(version.current().version, 2)
        self.assertEqual(version.current().created_by, "alice")
        events = [r.get("event") for r in version.history()]
        self.assertEqual(events, ["init", "promote"])

    def test_state_file_written(self):
        version.ensure_initial()
        data = read_json(version._state_path())
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["env"]["AGENT_N"], "15")
        self.assertIn("tag", data)

    def test_incarnate_shape(self):
        rows = version.incarnate([{"candidate_id": "c1", "target": "T",
                                   "old": "1", "new": "2",
                                   "decision": "ACCEPT", "reason": "VALIDATED"}])
        self.assertEqual(rows[0]["candidate_id"], "c1")
        self.assertEqual(rows[0]["decision"], "ACCEPT")


if __name__ == "__main__":
    unittest.main()
