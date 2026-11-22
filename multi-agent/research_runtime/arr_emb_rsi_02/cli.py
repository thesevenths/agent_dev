"""ARR_EMB_RSI_02 CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import run_campaign
from .report import write_report


def main() -> None:
    p = argparse.ArgumentParser(description="ARR_EMB_RSI_02 blind validation")
    p.add_argument("--stage", choices=("campaign",), default="campaign")
    p.add_argument("--out", default="ARR_EMB_RSI_02_FULL")
    p.add_argument("--concurrency", type=int, default=3)
    args = p.parse_args()
    canonical = run_campaign(run_root=args.out,
                             concurrency=args.concurrency)
    report = write_report(canonical=canonical, root=Path(args.out))
    print(json.dumps({"verdicts": canonical["verdicts"],
                      "scientific_claim": canonical["scientific_claim"],
                      "report": str(report),
                      "canonical": str(Path(args.out) / "canonical_results.json")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
