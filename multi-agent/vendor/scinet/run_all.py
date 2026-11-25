"""
run_all.py
==========
SciNet++ (2026) -- one-command reproduction of the full pipeline.

Runs the four connected threads in order:

    train.py     -> V-JEPA representation learning      (V-JEPA)
    analyze.py   -> physics probe / concept discovery   (SciNet)
    symbolic.py  -> AI-Feynman formula discovery (PySR)  (AI Feynman)
    evaluate.py  -> multi-step latent evaluation         (V-JEPA)
    tta.py       -> online test-time adaptation         (AdaJEPA)

Each stage is executed in its own subprocess so a failure is isolated and the
config stays the single source of truth.

Usage:
    python run_all.py --config configs/pendulum.yaml
    python run_all.py --config configs/pendulum.yaml --epochs 3 --skip-symbolic
    python run_all.py --config configs/double_pendulum.yaml
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys

from utils.config import make_run_id

STAGES = [
    ("train.py", "V-JEPA representation learning"),
    ("analyze.py", "Physics probe (SciNet concept discovery)"),
    ("symbolic.py", "AI-Feynman symbolic regression (PySR)"),
    ("evaluate.py", "V-JEPA multi-step latent evaluation"),
    ("tta.py", "AdaJEPA online test-time adaptation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SciNet++ pipeline")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override train epochs")
    parser.add_argument(
        "--skip-symbolic",
        action="store_true",
        help="skip symbolic.py (useful when PySR/Julia is not installed)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="reuse a specific run sub-directory (timestamp); a fresh one is made if omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # One fresh timestamp shared by every stage so all artefacts of a single
    # run land in the same time-stamped sub-directory: outputs/<exp>/<run_id>/.
    run_id = args.run_id or make_run_id()
    print(f">>> run-id: {run_id}  (outputs/<experiment>/{run_id}/)")

    for script, description in STAGES:
        if args.skip_symbolic and script == "symbolic.py":
            print(f"\n>>> Skipping {script} (--skip-symbolic)")
            continue

        cmd = [sys.executable, script, "--config", args.config, "--run-id", run_id]
        if script == "train.py" and args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        print("\n" + "=" * 72)
        print(f">>> {script}: {description}")
        print("=" * 72, flush=True)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStage '{script}' failed with exit code {result.returncode}.")
            sys.exit(result.returncode)

    print("\nAll stages completed successfully.")


if __name__ == "__main__":
    main()
