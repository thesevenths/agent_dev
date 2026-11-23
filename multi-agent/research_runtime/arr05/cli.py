"""ARR-05 CLI: single end-to-end campaign entry point with resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import execute
from .report import generate


def main() -> None:
    p = argparse.ArgumentParser(description="ARR-05 unified campaign")
    p.add_argument("--root", default="experiments/campaigns/ARR-05-2026-001")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.report_only:
        c = json.loads((root / "final" / "ARR05_FINAL_REPORT.json").read_text(encoding="utf-8"))
        print(json.dumps({"files": [str(x) for x in generate(root, c).values()]}, indent=2))
        return
    c = execute(root)
    files = generate(root, c)
    print(json.dumps({"terminal": c["terminal_state"], "verdicts": c["verdicts"],
                      "files": {k: str(v) for k, v in files.items()}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
