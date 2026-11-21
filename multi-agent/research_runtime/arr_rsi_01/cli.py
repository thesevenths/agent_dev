"""CLI for ARR-RSI-01 P0.1 (strict concurrency=1).

    python -m research_runtime.arr_rsi_01.cli --tasks 1  --stage pilot \
        --out ARR_RSI_01_P0_1_PILOT
    python -m research_runtime.arr_rsi_01.cli --tasks 5  --stage reliability \
        --out ARR_RSI_01_P0_1_RELIABILITY
    python -m research_runtime.arr_rsi_01.cli --tasks 20 --stage full \
        --out ARR_RSI_01_P0_1_FULL
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import finalize_run, run_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARR-RSI-01 P0.1 runner")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--out", default="ARR_RSI_01_P0_1_RUN")
    parser.add_argument("--scinet-dir", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stage", default="p0_1_run",
                        choices=("pilot", "reliability", "full", "p0_1_run"))
    parser.add_argument("--resume", action="store_true",
                        help="reuse recorded arms instead of re-running them")
    parser.add_argument("--finalize", action="store_true",
                        help="recompute evaluation/artifacts from recorded arms only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.finalize:
        print(json.dumps(finalize_run(run_root=Path(args.out)),
                         ensure_ascii=False, indent=2))
        return
    result = run_pilot(
        n_tasks=args.tasks, run_root=Path(args.out),
        scinet_dir=args.scinet_dir or None, verbose=args.verbose,
        concurrency=1, resume=args.resume, stage=args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
