"""落地层的可撤销性：被接受的候选必须真的写进生产，并且每条都能逐条拆掉。

缺口原本是致命的：probe 的补丁只打在**沙箱副本**上，即使候选被判 ACCEPT，
生产的 ``.env`` / ``prompt.py`` 一个字都没变 —— 用户下次提问，行为与改进前完全一致。
那不是自我改进，是一次「只写报告的实验」。

这里钉住四条不显然的性质：

1. **托管块必须在文件末尾**：python-dotenv 对同文件重复键取后者。用户手工区里
   ``AGENT_MAX_ITERATIONS=15`` 不能被覆盖（那是人的行），所以机器改动追加在末尾 ——
   生效靠的是「后者胜出」这条 dotenv 语义，因此必须由测试钉住，不能靠记忆。
2. **回滚不依赖 git**：git 回滚会连带抹掉用户其它未提交的改动。
3. **落地进程环境的刷新**只在落地到真实生产目录时做，测试用临时 root 不得污染本进程。
4. Frozen Core 覆盖的 env **一律拒绝落地**：自我修改判分口径是最危险的失效模式。
"""
import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dotenv import dotenv_values

from adaptive_runtime import landing
from adaptive_runtime.settings import Paths

USER_LINE = "AGENT_MAX_ITERATIONS=15"
ANCHOR = "SUPERVISOR_SYSTEM_PROMPT = \"\"\""


def _cand_env(key: str, value: str, *, tag: str = "base/v1", ph: str = "h1") -> SimpleNamespace:
    return SimpleNamespace(is_env=True, is_prompt=False, target=key, new=value,
                           level=1, patch_hash=ph, candidate_id=f"c-{ph}",
                           patch={"kind": "env"})


def _cand_prompt(insert: str, *, tag: str = "base/v1", ph: str = "p1",
                 file: str = "prompt.py") -> SimpleNamespace:
    return SimpleNamespace(is_env=False, is_prompt=True, target=file, new="",
                           level=2, patch_hash=ph, candidate_id=f"c-{ph}",
                           patch={"kind": "prompt", "file": file,
                                  "anchor": ANCHOR, "insert": insert})


class LandingBase(unittest.TestCase):
    """所有落地都打到临时 root：不许碰用户的生产 .env / prompt.py。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.env = self.root / ".env"
        self.env.write_text(f"# 用户手工维护\n{USER_LINE}\nLLM_MODEL=qwen\n", encoding="utf-8")
        self.mods = dataclasses.replace(Paths(),
                                        landing=self.root / "landing",
                                        landing_history=self.root / "landing" / "history.jsonl")
        self._patch = mock.patch.object(landing, "PATHS", self.mods)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()


class TestManagedBlock(LandingBase):
    def test_managed_value_wins_over_user_line_via_dotenv(self):
        """核心性质：块在末尾 + dotenv 后者胜出 ⇒ 托管值真的生效。"""
        landing.upsert_entry("AGENT_MAX_ITERATIONS", "25", version_tag="base/v1",
                             patch_hash="aa", root=self.root)
        self.assertEqual(dotenv_values(self.env)["AGENT_MAX_ITERATIONS"], "25")

    def test_user_line_is_never_rewritten(self):
        before = self.env.read_text(encoding="utf-8")
        landing.upsert_entry("AGENT_MAX_ITERATIONS", "25", root=self.root)
        after = self.env.read_text(encoding="utf-8")
        self.assertIn(f"\n{USER_LINE}\n", after, "机器不许改人的行")
        self.assertTrue(after.startswith(before.rstrip("\n")),
                        "托管块只能追加在用户区之后，不能重排用户区")

    def test_upsert_is_idempotent_per_key(self):
        landing.upsert_entry("X", "1", version_tag="base/v1", root=self.root)
        landing.upsert_entry("X", "2", version_tag="base/v2", root=self.root)
        entries = landing.read_block(self.root)
        self.assertEqual([e.key for e in entries], ["X"])
        self.assertEqual(entries[0].value, "2")
        self.assertEqual(entries[0].version_tag, "base/v2")

    def test_block_carries_attribution_metadata(self):
        landing.upsert_entry("X", "3", version_tag="base/v7", patch_hash="deadbeef",
                             root=self.root)
        text = self.env.read_text(encoding="utf-8")
        self.assertIn("landed by base/v7", text)
        self.assertIn("patch=deadbeef", text)
        e = landing.read_block(self.root)[0]
        self.assertEqual((e.version_tag, e.patch_hash), ("base/v7", "deadbeef"))

    def test_content_after_block_is_preserved(self):
        """重写块时只动块内区域：块前的用户行、块后的任意行都得逐字保留。"""
        self.env.write_text(f"A=1\n{landing.BLOCK_START}\n旧条目\n{landing.BLOCK_END}\nTAIL=9\n",
                            encoding="utf-8")
        landing.upsert_entry("B", "2", root=self.root)
        text = self.env.read_text(encoding="utf-8")
        self.assertIn("A=1", text)
        self.assertIn("TAIL=9", text)
        self.assertEqual(text.count(landing.BLOCK_START), 1)
        self.assertEqual([e.key for e in landing.read_block(self.root)], ["B"],
                         "块内不是 key=value 的行应该被整体重写掉")

    def test_drop_entries_restores_user_value_in_file(self):
        landing.upsert_entry("AGENT_MAX_ITERATIONS", "25", root=self.root)
        removed = landing.drop_entries(["AGENT_MAX_ITERATIONS"], self.root)
        self.assertEqual(removed, ["AGENT_MAX_ITERATIONS"])
        self.assertEqual(landing.read_block(self.root), [])
        self.assertEqual(dotenv_values(self.env)["AGENT_MAX_ITERATIONS"], "15",
                         "撤销后必须退回用户区那个值")

    def test_lines_without_block_markers_are_not_parsed(self):
        self.env.write_text("A=1\n# 没有块\nB=2\n", encoding="utf-8")
        self.assertEqual(landing.read_block(self.root), [])


class TestLiveValue(LandingBase):
    def test_precedence_process_env_then_block_then_user(self):
        landing.upsert_entry("K", "block", root=self.root)
        self.assertEqual(landing.live_value("K", self.root), "block")
        with mock.patch.dict(os.environ, {"K": "process"}):
            self.assertEqual(landing.live_value("K", self.root), "process")

    def test_falls_back_to_user_area_when_absent(self):
        self.assertEqual(landing.live_value("LLM_MODEL", self.root), "qwen")
        self.assertIsNone(landing.live_value("NOT_THERE_AT_ALL", self.root))

    def test_blank_process_env_does_not_shadow(self):
        """进程里有个空串不算「设过了」：否则 proposer 会拿空值当现值去搜索。"""
        landing.upsert_entry("K", "block", root=self.root)
        with mock.patch.dict(os.environ, {"K": "   "}):
            self.assertEqual(landing.live_value("K", self.root), "block")


class TestProcessEnvSync(LandingBase):
    def test_temp_root_does_not_pollute_control_plane(self):
        landing.upsert_entry("AGENT_MAX_ITERATIONS", "25", root=self.root)
        self.assertNotEqual(os.environ.get("AGENT_MAX_ITERATIONS"), "25")

    def test_production_root_refreshes_process_env(self):
        """否则刚落地的值被旧环境变量遮住：代码层生效、行为层没生效。"""
        with mock.patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "15"}):
            landing._sync_process_env("AGENT_MAX_ITERATIONS", "25", None)
            self.assertEqual(os.environ["AGENT_MAX_ITERATIONS"], "25")
            landing._sync_process_env("AGENT_MAX_ITERATIONS", None, None)
            self.assertNotIn("AGENT_MAX_ITERATIONS", os.environ)


class TestLandCandidate(LandingBase):
    def test_l1_env_candidate_lands_into_block(self):
        res = landing.land(_cand_env("AGENT_MAX_ITERATIONS", "25"),
                           version_tag="base/v1", root=self.root)
        self.assertEqual(res["action"], "upsert")
        self.assertEqual(landing.read_block(self.root)[0].value, "25")
        self.assertEqual(dotenv_values(self.env)["AGENT_MAX_ITERATIONS"], "25")

    def test_frozen_env_key_is_refused(self):
        with self.assertRaises(RuntimeError) as ctx:
            landing.land(_cand_env("AGENT_CRITIC_MIN_LEN", "1"), root=self.root)
        self.assertIn("冻结", str(ctx.exception))

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(RuntimeError):
            landing.land(SimpleNamespace(is_env=False, is_prompt=False,
                                         patch={"kind": "mystery"}, target="t",
                                         patch_hash="h"), root=self.root)

    def test_l2_prompt_candidate_lands_at_anchor(self):
        (self.root / "prompt.py").write_text(
            f'{ANCHOR}\n你是 supervisor\n"""\n', encoding="utf-8")
        res = landing.land(_cand_prompt("# 候选提示\n"), version_tag="base/v2",
                           root=self.root)
        self.assertEqual(res["action"], "land_prompt")
        text = (self.root / "prompt.py").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# 候选提示\n" + ANCHOR))

    def test_ambiguous_anchor_is_refused(self):
        (self.root / "prompt.py").write_text(f"{ANCHOR}x\n{ANCHOR}y\n", encoding="utf-8")
        with self.assertRaises(RuntimeError) as ctx:
            landing.land(_cand_prompt("# ins\n"), root=self.root)
        self.assertIn("恰好 1 次", str(ctx.exception))

    def test_missing_target_file_is_refused(self):
        with self.assertRaises(RuntimeError):
            landing.land(_cand_prompt("# ins\n", file="nope.py"), root=self.root)


class TestRollback(LandingBase):
    def setUp(self):
        super().setUp()
        (self.root / "prompt.py").write_text(f"{ANCHOR}\nbody\n\"\"\"\n", encoding="utf-8")

    def test_rollback_drops_only_newer_env_entries(self):
        landing.land(_cand_env("K1", "1", ph="h1"), version_tag="base/v1", root=self.root)
        landing.land(_cand_env("K2", "2", ph="h2"), version_tag="base/v2", root=self.root)
        out = landing.rollback_to("base/v1", self.root)
        self.assertEqual(out["removed_env"], ["K2"])
        self.assertEqual(out["kept_env"], ["K1"])
        self.assertEqual([e.key for e in landing.read_block(self.root)], ["K1"])

    def test_rollback_is_recorded_in_history(self):
        landing.land(_cand_env("K1", "1"), version_tag="base/v1", root=self.root)
        landing.rollback_to("base/v0", self.root)
        kinds = [h.get("kind") for h in
                 landing.read_jsonl(self.mods.landing_history)]
        self.assertIn("rollback", kinds, "回滚必须留痕：不然没人能证明它发生过")

    def test_revert_prompt_removes_exactly_the_inserted_text(self):
        insert = "# 候选提示\n"
        original = (self.root / "prompt.py").read_text(encoding="utf-8")
        landing.land(_cand_prompt(insert, ph="p9"), version_tag="base/v3", root=self.root)
        self.assertTrue(landing.revert_prompt("prompt.py", insert, self.root))
        self.assertEqual((self.root / "prompt.py").read_text(encoding="utf-8"), original)

    def test_revert_prompt_is_honest_when_text_absent(self):
        (self.root / "prompt.py").write_text(f"{ANCHOR}\nbody\n\"\"\"\n", encoding="utf-8")
        self.assertFalse(landing.revert_prompt("prompt.py", "# 从没落过\n", self.root))
        self.assertFalse(landing.revert_prompt("missing.py", "# x\n", self.root))


class TestVersionOrdering(LandingBase):
    def test_le_compares_by_trailing_number(self):
        self.assertTrue(landing._le("base/v1", "base/v3"))
        self.assertFalse(landing._le("base/v4", "base/v3"))
        self.assertTrue(landing._le("base/v3", "base/v3"))

    def test_unparseable_tag_is_treated_as_newest(self):
        """解不出版本号 ⇒ 当作比目标新：宁可多撤销，不可漏撤销。"""
        self.assertFalse(landing._le("", "base/v1"))
        self.assertFalse(landing._le("nonsense", "base/v0"))
        self.assertFalse(landing._le("base/v2", ""))

    def test_rollback_uses_le_semantics_for_empty_tag(self):
        """没写 version_tag 的条目（人工 upsert）在回滚时会被撤掉。"""
        landing.upsert_entry("MANUAL", "9", root=self.root)
        landing.land(_cand_env("K1", "1"), version_tag="base/v9", root=self.root)
        out = landing.rollback_to("base/v5", self.root)
        self.assertEqual(sorted(out["removed_env"]), ["K1", "MANUAL"])
        self.assertEqual(landing.read_block(self.root), [])


class TestContentStamp(LandingBase):
    def test_stamp_changes_when_a_value_lands(self):
        """只升版本号不改内容 = 自欺；判据比的是这个指纹。"""
        before = landing.content_stamp(self.root)
        landing.upsert_entry("AGENT_MAX_ITERATIONS", "25", root=self.root)
        self.assertNotEqual(landing.content_stamp(self.root), before)

    def test_stamp_is_stable_for_identical_content(self):
        landing.upsert_entry("K", "1", root=self.root)
        self.assertEqual(landing.content_stamp(self.root), landing.content_stamp(self.root))


if __name__ == "__main__":
    unittest.main()
