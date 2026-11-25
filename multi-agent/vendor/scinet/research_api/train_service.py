"""Command builders for the training stage.

These are pure functions so they can be unit-tested without launching a GPU
process. The stage scripts themselves are untouched.

Important: ``--run-id`` is ALWAYS supplied. ``utils.config.ensure_output_dir``
falls back to *reusing the most recent run directory* when no run-id is given
(config.py:98), which would make concurrent experiments silently share a
checkpoint directory.
"""
from __future__ import annotations

import sys


def build_train_command(config: str, run_id: str, *, epochs: int | None = None,
                        python_exe: str | None = None,
                        extra_args: list[str] | None = None) -> list[str]:
    cmd = [
        python_exe or sys.executable,
        "train.py",
        "--config", config,
        "--run-id", run_id,
    ]
    if epochs is not None:
        cmd += ["--epochs", str(int(epochs))]
    cmd += list(extra_args or [])
    return cmd


def build_analyze_command(config: str, run_id: str, *,
                          python_exe: str | None = None) -> list[str]:
    return [python_exe or sys.executable, "analyze.py",
            "--config", config, "--run-id", run_id]


def build_symbolic_command(config: str, run_id: str, *,
                           python_exe: str | None = None) -> list[str]:
    return [python_exe or sys.executable, "symbolic.py",
            "--config", config, "--run-id", run_id]


def build_tta_command(config: str, run_id: str, *,
                      python_exe: str | None = None) -> list[str]:
    return [python_exe or sys.executable, "tta.py",
            "--config", config, "--run-id", run_id]
