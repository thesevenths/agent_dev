"""Command builder for the evaluation stage (pure function, unit-testable)."""
from __future__ import annotations

import sys


def build_evaluate_command(config: str, run_id: str, *,
                           python_exe: str | None = None,
                           extra_args: list[str] | None = None,
                           split: str | None = None) -> list[str]:
    # evaluate.py reads <run_dir>/checkpoint.pt, so it MUST receive the same
    # run-id that training used. No --new, no implicit "latest run" fallback.
    cmd = [
        python_exe or sys.executable,
        "evaluate.py",
        "--config", config,
        "--run-id", run_id,
    ]
    # ``split=None`` deliberately emits no flag: evaluate.py's own default
    # (test) preserves the historical command shape and behaviour.
    if split:
        cmd += ["--split", str(split)]
    cmd += list(extra_args or [])
    return cmd
