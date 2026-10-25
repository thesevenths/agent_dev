"""结构性（代码级）补丁的落地 / 回滚 / 分发。

证明「结构性编辑空间」真的可用：kind=code 的候选能落进运行时文件、能逐字回滚、
且 land() 分发与 rollback_to 按 hash 反查都正确。全部打到临时 root，不碰生产。

这是 RRSI 自我改进的关键缺口补全：之前只能落 env 旋钮与 prompt 补丁，结构性候选
（改控制流 / 加校验）无处落地。本测试钉住 kind=code 这条通路。
"""
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from adaptive_runtime import landing
from adaptive_runtime.landing import wrap_code_patch, CODE_PATCHES
from adaptive_runtime.io_util import append_jsonl
from adaptive_runtime.settings import PATHS


SUP_ANCHOR = "def supervisor(state: AgentState) -> Dict[str, Any]:"
SUP_INSERT = (
    "def _rrsi_test_guard():  # 测试 fixture\n"
    "    return None"
)


def _ph(wrapped: str) -> str:
    return hashlib.sha256(wrapped.encode("utf-8")).hexdigest()[:16]


class StructuralCodePatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rrsi_code_"))
        sup_src = PATHS.root / "supervisor.py"
        self.sup = self.tmp / "supervisor.py"
        if sup_src.exists():
            self.sup.write_text(sup_src.read_text(encoding="utf-8", errors="replace"),
                                encoding="utf-8")
        self.name = "selftest_fixture_guard"
        self.wrapped = wrap_code_patch(self.name, SUP_INSERT)
        CODE_PATCHES[self.name] = {"file": "supervisor.py", "anchor": SUP_ANCHOR,
                                    "insert": SUP_INSERT, "wrapped": self.wrapped}

    def tearDown(self):
        CODE_PATCHES.pop(self.name, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cand_code(self):
        return type("C", (), {
            "is_env": False, "is_prompt": False, "target": self.name, "new": self.wrapped,
            "level": 3, "patch_hash": _ph(self.wrapped), "candidate_id": "c-code",
            "patch": {"kind": "code", "name": self.name, "file": "supervisor.py",
                      "anchor": SUP_ANCHOR, "insert": self.wrapped},
        })()

    def test_land_code_writes_block(self):
        rec = landing.land_code(self.name, "supervisor.py", SUP_ANCHOR, SUP_INSERT,
                                version_tag="base/v2", patch_hash="x", root=self.tmp)
        self.assertEqual(rec["action"], "land_code")
        self.assertIn(self.wrapped, self.sup.read_text(encoding="utf-8"))

    def test_revert_code_removes_block(self):
        landing.land_code(self.name, "supervisor.py", SUP_ANCHOR, SUP_INSERT,
                          version_tag="base/v2", patch_hash="x", root=self.tmp)
        ok = landing.revert_code("supervisor.py", self.wrapped, root=self.tmp)
        self.assertTrue(ok)
        self.assertNotIn(self.wrapped, self.sup.read_text(encoding="utf-8"))

    def test_land_dispatch_code(self):
        cand = self._cand_code()
        res = landing.land(cand, version_tag="base/v2", root=self.tmp)
        self.assertIn("land_code", res.get("action", ""))
        self.assertIn(self.wrapped, self.sup.read_text(encoding="utf-8"))

    def test_rollback_to_by_hash(self):
        # 写一条 code 历史，验证 rollback_to 能按 patch_hash 反查并拆块
        landing.land_code(self.name, "supervisor.py", SUP_ANCHOR, SUP_INSERT,
                          version_tag="base/v2", patch_hash="x", root=self.tmp)
        append_jsonl(PATHS.landing_history, [{"ts": "t", "kind": "code",
                                              "file": "supervisor.py", "version_tag": "base/v2",
                                              "patch_hash": _ph(self.wrapped)}])
        out = landing.rollback_to("base/v1", root=self.tmp)
        self.assertIn("supervisor.py", out.get("reverted_code_files", []))
        self.assertNotIn(self.wrapped, self.sup.read_text(encoding="utf-8"))

    def test_propose_can_emit_code_candidate(self):
        # proposer 的 code 分支处于休眠态（SEARCH_SPACE 暂无 code 项），但通路必须可用：
        # 直接登记一个 code 项并放开 edit level，propose 应当能产出 kind=code 的候选。
        import copy
        from adaptive_runtime import proposer as P
        P.SEARCH_SPACE[self.name] = {"level": 3, "kind": "code", "values": ["on"],
                                     "file": "supervisor.py", "anchor": SUP_ANCHOR,
                                     "insert": SUP_INSERT, "rationale": "结构性自检 fixture",
                                     "taxonomies": ("PLANNING_FAILURE",), "expectation": "structure"}
        cfg = copy.copy(P.CFG)
        cfg.max_edit_level = 3
        try:
            cands = P.propose("PLANNING_FAILURE", P.Ledger(), cfg=cfg, limit=12)
            code_cands = [c for c in cands if c.patch.get("kind") == "code"]
            self.assertTrue(code_cands, "propose 应能从 code 项产出候选")
            self.assertEqual(code_cands[0].patch["name"], self.name)
            self.assertIn(self.wrapped, code_cands[0].diff_text())
        finally:
            P.SEARCH_SPACE.pop(self.name, None)


if __name__ == "__main__":
    unittest.main()
