"""ARR-04 CLI: single end-to-end campaign entry point with resume support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import execute
from .report import generate_final_report


def main() -> None:
    p = argparse.ArgumentParser(description="ARR-04 unified research capability campaign")
    p.add_argument("--root", default="experiments/campaigns/ARR-04-2026-001")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.report_only:
        canonical = json.loads((root / "final" / "ARR04_FINAL_REPORT.json").read_text(
            encoding="utf-8"))
        rp = generate_final_report(root, canonical)
        print(json.dumps({"report": str(rp)}, indent=2))
        return
    canonical = execute(campaign_root=root, resume=not args.no_resume)
    rp = generate_final_report(root, canonical)
    print(json.dumps({"terminal": canonical.get("terminal_state"),
                      "verdicts": canonical["verdicts"], "report": str(rp)},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
