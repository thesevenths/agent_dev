"""ARR_EMB_RSI_01 -- CLI. Pilot only; full experiment is not implemented."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prereg import freeze_prereg
from .runner import finalize_pilot, run_pilot


def main() -> None:
    p = argparse.ArgumentParser(description="ARR_EMB_RSI_01 pilot runner")
    p.add_argument("--stage", choices=("prereg", "pilot", "finalize"), default="pilot")
    p.add_argument("--out", default="ARR_EMB_RSI_01_PILOT")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.stage == "finalize":
        print(json.dumps(finalize_pilot(run_root=args.out),
                         indent=2, ensure_ascii=False, default=str))
    elif args.stage == "prereg":
        print(json.dumps(freeze_prereg(Path(args.out) / "preregistration.json"),
                         indent=2, ensure_ascii=False))
    else:
        print(json.dumps(run_pilot(run_root=args.out, seeds=(args.seed,)),
                         indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
