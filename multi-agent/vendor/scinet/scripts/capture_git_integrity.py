"""
scripts/capture_git_integrity.py
================================
Capture a machine-readable Git integrity snapshot so downstream verifiers
never have to hard-code counts such as "78 P2-02 evidence files".

All values are obtained from real git commands at capture time. The output
records the exact commands, the full HEAD SHA and a UTC timestamp, so the
snapshot itself is auditable.

Usage:
  python scripts/capture_git_integrity.py
  -> datasets/controlled_pendulum_pilot/analysis/git_integrity.json
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "controlled_pendulum_pilot" / "analysis" / "git_integrity.json"


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> int:
    head = sh(["git", "rev-parse", "HEAD"])
    main_head = sh(["git", "rev-parse", "origin/main"])
    merge_base = sh(["git", "merge-base", "origin/main", "HEAD"])
    left_right = sh(["git", "rev-list", "--left-right", "--count",
                     "origin/main...HEAD"]).split()
    behind, ahead = int(left_right[0]), int(left_right[1])

    name_status = sh(["git", "diff", "--name-status", "origin/main...HEAD"])
    counts = {"A": 0, "M": 0, "D": 0, "OTHER": 0}
    for line in name_status.splitlines():
        kind = line.split("\t")[0][0]
        counts[kind] = counts.get(kind, 0) + 1
        if kind not in ("A", "M", "D"):
            counts["OTHER"] += 1
    changed_files = len([l for l in name_status.splitlines() if l])

    tree = sh(["git", "ls-tree", "-r", "HEAD", "--name-only"]).splitlines()
    p2_02_files = [p for p in tree if "outputs/dynamics/pusher_v1/" in p]

    commits_since_a5b8396 = sh(
        ["git", "log", "--oneline", "a5b8396..HEAD"]).splitlines()
    clean = sh(["git", "status", "--porcelain"]) == ""

    snap = {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "branch": sh(["git", "branch", "--show-current"]),
        "head_full_sha": head,
        "origin_main_full_sha": main_head,
        "merge_base_full_sha": merge_base,
        "origin_main_is_ancestor_of_head": merge_base == main_head,
        "ahead": ahead,
        "behind": behind,
        "changed_files_vs_main": changed_files,
        "file_status_counts": {k: v for k, v in counts.items() if k != "OTHER"},
        "commits_since_a5b8396": commits_since_a5b8396,
        "working_tree_clean": clean,
        "p2_02_evidence_file_count": len(p2_02_files),
        "commands": [
            "git rev-parse HEAD",
            "git rev-parse origin/main",
            "git merge-base origin/main HEAD",
            "git rev-list --left-right --count origin/main...HEAD",
            "git diff --name-status origin/main...HEAD",
            "git ls-tree -r HEAD --name-only",
            "git log --oneline a5b8396..HEAD",
            "git status --porcelain",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"captured: {OUT}")
    print(f"  HEAD={head[:12]} ahead={ahead} behind={behind} "
          f"changed={changed_files} A/M/D={counts['A']}/{counts['M']}/{counts['D']}")
    print(f"  p2_02_evidence_file_count={len(p2_02_files)} clean={clean}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
