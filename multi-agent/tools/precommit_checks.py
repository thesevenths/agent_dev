"""Read-only pre-commit checks for the ARR 1.1 workspace (P0-4).

These hooks NEVER modify files. ``pre-commit run --all-files`` is executed on
the WHOLE repository (which contains several unrelated legacy projects), so the
checks are deliberately restricted to the ARR subtree by .pre-commit-config.yaml
and use only the Python standard library -- no network, no third-party hook
repository, no formatter that would rewrite historical evidence files.

Usage:
    python multi-agent/tools/precommit_checks.py <mode> <file> [<file> ...]

Modes: ast (Python syntax), json (JSON validity), conflict (merge markers).
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

# Line-ANCHORED and length-exact. A bare "=======" substring also appears in
# legitimate docstring/section separators ("========================"), so a
# naive substring search produces false positives on the existing codebase.
_CONFLICT_LINE = re.compile(r"^(?:<{7}(?:\s.*)?$|>{7}(?:\s.*)?$|={7}$)")


def check_ast(paths: list[Path]) -> list[str]:
    errors = []
    for p in paths:
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            errors.append(f"{p}: {exc}")
    return errors


def check_json(paths: list[Path]) -> list[str]:
    errors = []
    for p in paths:
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            errors.append(f"{p}: invalid JSON: {exc}")
    return errors


def check_conflict(paths: list[Path]) -> list[str]:
    errors = []
    for p in paths:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError) as exc:
            errors.append(f"{p}: unreadable: {exc}")
            continue
        for lineno, line in enumerate(lines, 1):
            if _CONFLICT_LINE.match(line):
                errors.append(f"{p}:{lineno}: merge-conflict marker {line.strip()!r}")
                break
    return errors


MODES = {"ast": check_ast, "json": check_json, "conflict": check_conflict}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in MODES:
        print(f"usage: {argv[0]} <{'|'.join(MODES)}> <files...>", file=sys.stderr)
        return 2
    mode, files = argv[1], [Path(x) for x in argv[2:]]
    if not files:
        return 0
    errors = MODES[mode](files)
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        print(f"{mode}: {len(errors)} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
