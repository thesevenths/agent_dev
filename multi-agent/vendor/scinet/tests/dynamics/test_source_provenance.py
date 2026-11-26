"""
tests.dynamics.test_source_provenance
=====================================
P1-B1.1 / P6: source-provenance hardening for data collection.

Positive + negative coverage for ``data.source_provenance`` and the collector's
enforcement path. These tests must NOT mutate the real v1/v2 datasets.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data.source_provenance as sp  # noqa: E402
from data.source_provenance import (  # noqa: E402
    ProvenanceError,
    assert_provenance_ok,
    capture_source_identity,
    file_sha256,
    git_head_sha,
    git_tracked_clean,
    verify_identity_stable,
)


class TestGitHelpers(unittest.TestCase):
    def test_git_head_sha_is_valid(self):
        sha = git_head_sha(REPO_ROOT)
        self.assertIsNotNone(sha)
        self.assertEqual(len(sha), 40)
        int(sha, 16)  # parseable hex

    def test_git_tracked_clean_is_bool(self):
        self.assertIsInstance(git_tracked_clean(REPO_ROOT), bool)

    def test_file_sha256_stable_and_different(self):
        f = REPO_ROOT / "data" / "source_provenance.py"
        a = file_sha256(f)
        self.assertEqual(len(a), 64)
        # A different (non-existent) file must raise.
        with self.assertRaises(ProvenanceError):
            file_sha256(REPO_ROOT / "does_not_exist_xyz.py")


class TestCaptureIdentity(unittest.TestCase):
    def _identity(self):
        return capture_source_identity(
            collector_path=REPO_ROOT / "scripts" / "collect_controlled_pendulum_v2.py",
            simulator_path=REPO_ROOT / "data" / "controlled_pendulum_v2.py",
            config_path=REPO_ROOT / "configs" / "dynamics"
            / "controlled_pendulum_v2_pilot.yaml",
            cwd=REPO_ROOT,
        )

    def test_captures_source_hashes_and_deps(self):
        ident = self._identity()
        for k in ("git_commit", "collector_sha256",
                  "simulator_sha256", "config_sha256"):
            self.assertIn(k, ident)
            self.assertIsNotNone(ident[k])
        deps = ident["dependency_versions"]
        self.assertIn("scipy", deps)
        self.assertIn("numpy", deps)
        self.assertIn("torch", deps)
        self.assertNotEqual(deps["scipy"], "unavailable")

    def test_missing_source_file_raises(self):
        with self.assertRaises(ProvenanceError):
            capture_source_identity(
                collector_path=REPO_ROOT / "scripts" / "collect_controlled_pendulum_v2.py",
                simulator_path=REPO_ROOT / "data" / "nope_v2.py",
                config_path=REPO_ROOT / "configs" / "dynamics"
                / "controlled_pendulum_v2_pilot.yaml",
                cwd=REPO_ROOT,
            )


class TestProvenanceEnforcement(unittest.TestCase):
    def test_required_commit_missing_raises(self):
        ident = {"git_commit": None, "_tracked_clean": True}
        with self.assertRaises(ProvenanceError):
            assert_provenance_ok(ident, git_commit_required=True)
        # Without the requirement, missing commit is tolerated.
        assert_provenance_ok(ident, git_commit_required=False)

    def test_dirty_tree_raises(self):
        ident = {"git_commit": "a" * 40, "_tracked_clean": False}
        with self.assertRaises(ProvenanceError):
            assert_provenance_ok(ident, git_commit_required=True)
        # allow_dirty explicitly bypasses the gate but is recorded as a risk.
        assert_provenance_ok(ident, git_commit_required=True, allow_dirty=True)

    def test_clean_tree_ok(self):
        ident = {"git_commit": "a" * 40, "_tracked_clean": True}
        assert_provenance_ok(ident, git_commit_required=True)


class TestIdentityStability(unittest.TestCase):
    def _base(self):
        return {
            "git_commit": "a" * 40,
            "collector_sha256": "c" * 64,
            "simulator_sha256": "s" * 64,
            "config_sha256": "k" * 64,
            "dependency_versions": {"python": "3.13.0", "numpy": "2.2.6",
                                    "scipy": "1.16.3", "torch": "2.14.1"},
        }

    def test_stable_identical_passes(self):
        b = self._base()
        self.assertTrue(verify_identity_stable(b, dict(b)))

    def test_changed_git_commit_fails(self):
        b, a = self._base(), self._base()
        a["git_commit"] = "b" * 40
        self.assertFalse(verify_identity_stable(b, a))

    def test_changed_source_hash_fails(self):
        b, a = self._base(), self._base()
        a["simulator_sha256"] = "z" * 64
        self.assertFalse(verify_identity_stable(b, a))

    def test_changed_dependency_fails(self):
        b, a = self._base(), self._base()
        a["dependency_versions"]["scipy"] = "1.99.0"
        self.assertFalse(verify_identity_stable(b, a))


class TestCollectorEnforcesDirtyTree(unittest.TestCase):
    """The collector must refuse to run on a dirty tracked tree (P6)."""

    def test_main_returns_2_on_dirty_tree(self):
        import scripts.collect_controlled_pendulum_v2 as coll
        orig = coll.git_tracked_clean
        coll.git_tracked_clean = lambda cwd: False  # simulate dirty tree
        try:
            out = Path(tempfile.mkdtemp(prefix="scinet-p6-neg-"))
            rc = coll.main(["--output-dir", str(out)])
            # Simulated dirty tree -> assert_provenance_ok raises -> return 2.
            self.assertEqual(rc, 2)
        finally:
            coll.git_tracked_clean = orig
            import shutil
            shutil.rmtree(str(out), ignore_errors=True)


class TestWorktreeIsolationHarness(unittest.TestCase):
    """The isolation harness must create AND tear down a worktree cleanly."""

    def test_worktree_created_and_removed(self):
        sentinel = {}

        def runner(wt_root: Path):
            sentinel["wt"] = Path(wt_root)
            # The worktree is created at the git *repo* root (E:/LLM), and
            # SciNet++ is a subdirectory of it, so the committed source lives at
            # wt/<relative-to-repo>/data/controlled_pendulum.py.
            repo_root = sp.git_repo_root(REPO_ROOT)
            repo_rel = REPO_ROOT.relative_to(repo_root) if repo_root else Path(".")
            committed = Path(wt_root) / repo_rel / "data" / "controlled_pendulum.py"
            self.assertTrue(committed.is_file(),
                            f"committed source not found at {committed}")
            return 0

        rc = sp.git_worktree_isolation(REPO_ROOT, runner)
        self.assertEqual(rc, 0)
        self.assertIn("wt", sentinel)
        # After isolation the throwaway worktree must be gone.
        self.assertFalse(sentinel["wt"].exists())


if __name__ == "__main__":
    unittest.main()
