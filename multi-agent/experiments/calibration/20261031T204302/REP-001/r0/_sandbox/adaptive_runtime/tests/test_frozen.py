import os
import tempfile
import unittest
from pathlib import Path

from adaptive_runtime import frozen


class TestSymbolSource(unittest.TestCase):
    def test_function(self):
        seg = frozen.symbol_source(Path(frozen.ROOT / "agents.py").read_text(encoding="utf-8"),
                                   "_critic_gate")
        self.assertIsNotNone(seg)
        self.assertIn("def _critic_gate", seg)

    def test_assignment(self):
        seg = frozen.symbol_source(Path(frozen.ROOT / "agents.py").read_text(encoding="utf-8"),
                                   "_CRITIC_MIN_LEN")
        self.assertIsNotNone(seg)
        self.assertIn("_CRITIC_MIN_LEN", seg)

    def test_missing(self):
        self.assertIsNone(frozen.symbol_source("x = 1\n", "nope"))

    def test_syntax_error_is_safe(self):
        self.assertIsNone(frozen.symbol_source("def broken(:\n", "broken"))


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "manifest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_freeze_and_verify_clean(self):
        m = frozen.freeze(self.path)
        self.assertGreater(len(m["symbols"]), 0)
        self.assertEqual(frozen.verify(self.path), [])

    def test_missing_manifest(self):
        v = frozen.verify(Path(self.tmp.name) / "nope.json")
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].kind, "manifest_missing")

    def test_env_override_detected(self):
        frozen.freeze(self.path)
        key = "AGENT_CRITIC_MIN_LEN"
        old = os.environ.get(key)
        try:
            os.environ[key] = "999"
            v = frozen.verify(self.path)
            self.assertTrue(any(x.kind == "env_changed" and x.target == key for x in v))
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        self.assertEqual(frozen.verify(self.path), [])

    def test_manifest_summary(self):
        frozen.freeze(self.path)
        s = frozen.manifest_summary(self.path)
        self.assertGreaterEqual(s["symbols"], 15)
        self.assertEqual(s["missing"], 0)
        self.assertEqual(s["env_keys"], len(frozen.FROZEN_ENV))

    def test_env_file_parsing(self):
        p = Path(self.tmp.name) / ".env"
        p.write_text("A=1\nB = 2  # note\nC='3'\n# comment\nD=\n", encoding="utf-8")
        env = frozen.parse_env_file(p)
        self.assertEqual(env["A"], "1")
        self.assertEqual(env["B"], "2")
        self.assertEqual(env["C"], "3")
        self.assertEqual(env["D"], "")


if __name__ == "__main__":
    unittest.main()
