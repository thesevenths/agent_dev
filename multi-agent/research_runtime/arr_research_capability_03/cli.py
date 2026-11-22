"""ARR-03 CLI: prereg / preflight / campaign / finalize / report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prereg import freeze_prereg


def main() -> None:
    p = argparse.ArgumentParser(description="ARR_RESEARCH_CAPABILITY_03")
    p.add_argument("--stage", choices=("prereg", "campaign", "finalize", "report"),
                   default="campaign")
    p.add_argument("--out", default="ARR_RESEARCH_CAPABILITY_03_FULL")
    p.add_argument("--seal-dir", default="ARR_03_SEALED")
    p.add_argument("--concurrency-m2", type=int, default=3)
    p.add_argument("--concurrency-m3", type=int, default=1)
    args = p.parse_args()

    if args.stage == "prereg":
        pr = freeze_prereg(Path(args.out) / "preregistration.json")
        print(json.dumps({"preregistration_sha256": pr["preregistration_sha256"]},
                         indent=2))
        return
    if args.stage == "finalize":
        from .finalize import finalize
        canonical = finalize(run_root=args.out)
        print(json.dumps({"verdicts": canonical["verdicts"],
                          "canonical_sha256": canonical["canonical_sha256"]},
                         indent=2, ensure_ascii=False))
        return
    if args.stage == "report":
        from .report import write_report
        canonical = json.loads((Path(args.out) / "canonical_results.json").read_text(
            encoding="utf-8"))
        rp = write_report(canonical=canonical, root=Path(args.out))
        print(json.dumps({"report": str(rp)}, indent=2))
        return
    from .campaign import run_campaign
    canonical = run_campaign(
        run_root=args.out, seal_dir=args.seal_dir,
        concurrency_m2=args.concurrency_m2, concurrency_m3=args.concurrency_m3)
    print(json.dumps({"verdicts": canonical["verdicts"],
                      "canonical": str(Path(args.out) / "canonical_results.json")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
