import os
import tempfile
import time
import unittest
from pathlib import Path

from datetime import datetime, timedelta

from adaptive_runtime.trace_export import export_from_log, scan_artifacts


def _log_text(base: datetime) -> str:
    def ts(sec: int) -> str:
        return (base + timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M:%S") + ",000"
    return f"""{ts(0)} INFO === run log started: x ===
{ts(1)} INFO [supervisor] created execution plan (goal='测试目标'):
1.    [pending] 检索数据: 抓取数据 → crawler_agent
2.    [pending] 计算指标: 计算 → code_agent
3.    [pending] 撰写报告: 写报告并落盘 → code_agent
{ts(2)} INFO [critic+summary] step 1/3 crawler_agent: PASS via FUSED judge+summary (1 LLM call)
{ts(3)} INFO [tool][crawler_agent] ✔ web_search (1.20s, 300 chars) -> ok
{ts(4)} INFO [critic+summary] step 2/3 code_agent: PASS via FUSED judge+summary (1 LLM call)
{ts(5)} INFO [step-failure] step 3 attempt #1 kind=truncated agent=code_agent
  WHY: 截断，已按完成处理
{ts(6)} INFO [supervisor] re-planning step 3/3:
{ts(7)} INFO [tool][code_agent] ✖ create_file (0.01s, 10 chars) -> err
{ts(8)} INFO [supervisor] FINISH
"""


class TestTraceExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "documents").mkdir()
        self.log = self.root / "log" / "default_20260101T100000_abcd1234.log"
        self.log.parent.mkdir()
        self.base = datetime.now() - timedelta(seconds=60)
        self.log.write_text(_log_text(self.base), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_and_steps(self):
        tr = export_from_log(self.log, root=self.root)
        self.assertEqual(len(tr.plan), 3)
        self.assertEqual(tr.total_steps, 3)
        self.assertEqual(tr.steps[1].last, "pass")
        self.assertEqual(tr.steps[3].last, "fail")
        self.assertEqual(tr.steps[3].fail_kind, "truncated")
        self.assertAlmostEqual(tr.contract_rate, 2 / 3, places=3)

    def test_goal(self):
        tr = export_from_log(self.log, root=self.root)
        self.assertEqual(tr.goal, "测试目标")

    def test_result_flags(self):
        tr = export_from_log(self.log, root=self.root)
        self.assertTrue(tr.truncated)
        self.assertTrue(tr.salvaged)
        self.assertTrue(tr.result["finalized"])

    def test_tool_events(self):
        tr = export_from_log(self.log, root=self.root)
        tools = [e for e in tr.events if e["type"] == "tool"]
        self.assertEqual(len(tools), 2)
        self.assertTrue(tools[0]["ok"])
        self.assertFalse(tools[1]["ok"])

    def test_artifact_window_excludes_old_files(self):
        old = self.root / "documents" / "old.md"
        old.write_text("old", encoding="utf-8")
        os.utime(old, (time.time() - 86400 * 30,) * 2)
        new = self.root / "documents" / "new.md"
        new.write_text("new content", encoding="utf-8")
        arts = scan_artifacts(self.root, since_ts=time.time() - 60)
        paths = [a["path"] for a in arts]
        self.assertIn("documents/new.md", paths)
        self.assertNotIn("documents/old.md", paths)


if __name__ == "__main__":
    unittest.main()
