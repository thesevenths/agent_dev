"""结构性（代码级）补丁的落地 / 回滚 / 分发。

证明「结构性编辑空间」真的可用：kind=code 的候选能落进运行时文件、能逐字回滚、
且 land() 分发与 rollback_to 按 hash 反查都正确。全部打到临时 root，不碰生产。

这是 RRSI 自我改进的关键缺口补全：之前只能落 env 旋钮与 prompt 补丁，结构性候选
（改控制流 / 加校验）无处落地。本测试钉住 kind=code 这条通路。
"""
import ast
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
            # 真实生产 code 项（B1-B4）也在 SEARCH_SPACE 里，会排在 fixture 之前，
            # 所以只断言 fixture 这一条确实被产出，而不是断言它排第一。
            names = [c.patch.get("name") for c in code_cands]
            self.assertIn(self.name, names, "propose 应能从动态登记的 code 项产出候选")
            fixture = next(c for c in code_cands if c.patch.get("name") == self.name)
            self.assertIn(self.wrapped, fixture.diff_text())
        finally:
            P.SEARCH_SPACE.pop(self.name, None)


class ProductionCodeCandidates(unittest.TestCase):
    """钉住 4 个生产代码级候选（B1-B4）能落地 / 回滚 / 且插入后 supervisor.py 仍可解析。

    这些候选锚点带前导缩进（位于缩进块内）。land_code 必须用「含前导缩进的锚点行」做替换，
    否则锚点会落到列 0、其后的块体缩进全乱，整个文件无法 parse。本用例在临时副本上验证，
    不碰生产文件。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rrsi_prod_"))
        self.sup_src = PATHS.root / "supervisor.py"
        self.sup = self.tmp / "supervisor.py"
        self.sup.write_text(self.sup_src.read_text(encoding="utf-8", errors="replace"),
                            encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_four_land_parse_and_revert(self):
        from adaptive_runtime import proposer as P
        for name, spec in P._code_entries().items():
            anchor, insert, file = spec["anchor"], spec["insert"], spec["file"]
            # 锚点（含前导缩进）在 supervisor.py 中必须恰好出现一次
            self.assertEqual(self.sup.read_text(encoding="utf-8").count(anchor), 1,
                             f"{name} 锚点必须唯一")
            rec = landing.land_code(name, file, anchor, insert,
                                    version_tag="v", patch_hash="h", root=self.tmp)
            self.assertEqual(rec["action"], "land_code", name)
            txt = self.sup.read_text(encoding="utf-8")
            try:
                ast.parse(txt)
            except SyntaxError as e:
                self.fail(f"{name} 落地后 supervisor.py 无法解析: {e}")
            ok = landing.revert_code(file, landing.wrap_code_patch(name, insert), root=self.tmp)
            self.assertTrue(ok, f"{name} 回滚失败")
            self.assertEqual(self.sup.read_text(encoding="utf-8"),
                             self.sup_src.read_text(encoding="utf-8", errors="replace"),
                             f"{name} 回滚后未完全还原")

    def test_all_four_landed_together_parse(self):
        """四个候选同时落地（最坏情况）也不得破坏 supervisor.py 的可解析性。"""
        from adaptive_runtime import proposer as P
        for name, spec in P._code_entries().items():
            landing.land_code(name, spec["file"], spec["anchor"], spec["insert"],
                              version_tag="v", patch_hash="h", root=self.tmp)
        try:
            ast.parse(self.sup.read_text(encoding="utf-8"))
        except SyntaxError as e:
            self.fail(f"四候选同时落地后 supervisor.py 无法解析: {e}")

    def test_propose_emits_all_four(self):
        """放开 edit level 后，propose 必须能从 4 个生产 code 项产出候选。"""
        import copy
        from adaptive_runtime import proposer as P
        with tempfile.TemporaryDirectory() as td:
            ledger = P.Ledger(Path(td) / "h.jsonl")
            cfg = copy.copy(P.CFG)
            cfg.max_edit_level = 3
            cands = P.propose("PLANNING_FAILURE", ledger, cfg=cfg, limit=20)
        code_cands = [c for c in cands if c.patch.get("kind") == "code"]
        names = {c.patch.get("name") for c in code_cands}
        self.assertEqual(names, set(P._code_entries().keys()),
                         f"propose 应产出全部 4 个生产代码候选，实际：{names}")


if __name__ == "__main__":
    unittest.main()
