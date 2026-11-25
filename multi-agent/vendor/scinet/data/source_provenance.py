"""
data.source_provenance
=======================

SciNet++ P1-B1.1 / P6: machine-verifiable source provenance for data collection.

The historical V2 collector recorded ``git_commit`` via a best-effort
``git rev-parse HEAD`` that returned the string ``"unavailable"`` when git was
not reachable, stored it silently, and never checked whether the working tree
was clean or whether the source files themselves could be re-identified. It also
never re-verified source identity *after* a (potentially long) collection run.

This module closes those gaps with pure, testable functions:

  * ``git_head_sha``            -- current HEAD SHA or None if unavailable
  * ``git_tracked_clean``       -- True iff no tracked files are modified/staged
  * ``file_sha256``             -- SHA-256 of a source file's bytes
  * ``capture_source_identity`` -- snapshot of HEAD, source-file hashes, deps
  * ``assert_provenance_ok``    -- enforce git_commit_required + clean tree
  * ``verify_identity_stable``  -- fail-closed comparison of before/after run
  * ``dependency_versions``     -- python/numpy/scipy/torch versions

Design note on "dirty tree":
  The precondition rejects only *tracked* modifications (``git diff --quiet HEAD``
  and ``git diff --cached --quiet HEAD``). Untracked outputs produced by a
  previous run are expected and must NOT block a re-run; only uncommitted
  changes to the *source* (collector / simulator / config) matter for
  provenance. This is documented so the gate is unambiguous and verifiable.
"""

from __future__ import annotations

import datetime
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:  # SciPy is a hard dependency for the scientific audits; record its version.
    import scipy
    _SCIPY_VERSION = scipy.__version__
except Exception:  # pragma: no cover - environment dependent
    _SCIPY_VERSION = "unavailable"


class ProvenanceError(RuntimeError):
    """Raised when a provenance precondition is violated or identity drifts."""


def git_head_sha(cwd: Path) -> Optional[str]:
    """Return the current HEAD commit SHA, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True,
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def git_repo_root(cwd: Path) -> Optional[Path]:
    """Return the top-level working tree of the git repo containing ``cwd``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd), capture_output=True, text=True)
        if out.returncode != 0:
            return None
        root = out.stdout.strip()
        return Path(root) if root else None
    except Exception:
        return None


def git_tracked_clean(cwd: Path) -> bool:
    """True iff no tracked file is modified or staged.

    Untracked files (e.g. dataset outputs) are intentionally ignored, because
    they do not affect the provenance of the *source* used to generate data.
    """
    try:
        for args in (["git", "diff", "--quiet", "HEAD"],
                     ["git", "diff", "--cached", "--quiet", "HEAD"]):
            r = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
            if r.returncode != 0:
                return False
        return True
    except Exception:
        return False


def file_sha256(path: Path) -> str:
    """SHA-256 of a source file's bytes. Raises if the file is missing."""
    p = Path(path)
    if not p.is_file():
        raise ProvenanceError(f"required source file not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _SCIPY_VERSION,
        "torch": torch.__version__,
    }


def capture_source_identity(
    collector_path: Path,
    simulator_path: Path,
    config_path: Path,
    cwd: Path,
) -> Dict[str, Any]:
    """Snapshot the source identity used for a collection run.

    Records:
      * git HEAD SHA (or None if unavailable)
      * SHA-256 of the collector, simulator and config source files
      * python / numpy / scipy / torch versions

    Raises ``ProvenanceError`` if any required source file is missing.
    """
    return {
        "git_commit": git_head_sha(cwd),
        "collector_sha256": file_sha256(collector_path),
        "simulator_sha256": file_sha256(simulator_path),
        "config_sha256": file_sha256(config_path),
        "dependency_versions": dependency_versions(),
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def assert_provenance_ok(
    identity: Dict[str, Any],
    git_commit_required: bool,
    allow_dirty: bool = False,
) -> None:
    """Enforce the collection-time provenance preconditions.

    Raises ``ProvenanceError`` when:
      * ``git_commit_required`` is true but no valid HEAD SHA is available, or
      * the tracked working tree is dirty (unless ``allow_dirty`` is set).
    """
    sha = identity.get("git_commit")
    if git_commit_required and (sha is None or sha == "unavailable" or sha == ""):
        raise ProvenanceError(
            "git_commit_required is true but no valid git commit SHA is "
            "available; aborting collection (cannot prove source identity)."
        )
    if not identity.get("_tracked_clean", True):
        if allow_dirty:
            # Still recorded as a risk by the caller; proceed only explicitly.
            pass
        else:
            raise ProvenanceError(
                "git working tree has tracked modifications; collection "
                "rejected to preserve source provenance. Commit or stash first."
            )


def verify_identity_stable(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> bool:
    """Fail-closed comparison of source identity before vs after a run.

    Compares the git HEAD SHA, the SHA-256 of every source file, and the
    dependency versions. Returns False (i.e. identity changed) if ANY differ.
    The git *cleanliness* of the tree is intentionally NOT compared here,
    because a successful run legitimately produces new untracked output files.
    """
    for key in ("git_commit", "collector_sha256",
                "simulator_sha256", "config_sha256"):
        if before.get(key) != after.get(key):
            return False
    b_deps = before.get("dependency_versions", {})
    a_deps = after.get("dependency_versions", {})
    for k in ("python", "numpy", "scipy", "torch"):
        if b_deps.get(k) != a_deps.get(k):
            return False
    return True


def git_worktree_isolation(cwd: Path, runner, worktree_dir: Optional[Path] = None):
    """Run ``runner`` inside a throwaway git worktree at the current HEAD.

    Creates ``<worktree_dir>`` (default: a temp dir under the system temp dir),
    checks out the current HEAD commit, invokes ``runner(worktree_root)``, then
    removes the worktree. Provides true source isolation: the generation runs
    against a snapshot of the exact committed source, so even mid-run edits to
    the working tree cannot affect it.

    Returns whatever ``runner`` returns. Raises on worktree setup/teardown error.
    """
    import tempfile
    import shutil

    wt = Path(worktree_dir) if worktree_dir else Path(
        tempfile.mkdtemp(prefix="scinet-provenance-"))
    try:
        add = subprocess.run(
            ["git", "worktree", "add", "--force", "--detach", str(wt), "HEAD"],
            cwd=str(cwd), capture_output=True, text=True,
        )
        if add.returncode != 0:
            raise ProvenanceError(
                f"git worktree add failed: {add.stderr.strip()}")
        try:
            return runner(wt)
        finally:
            rm = subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt)],
                cwd=str(cwd), capture_output=True, text=True)
            if rm.returncode != 0:  # pragma: no cover - best effort cleanup
                shutil.rmtree(str(wt), ignore_errors=True)
    finally:
        pass
