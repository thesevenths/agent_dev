"""ARR-06 CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import execute
from .report import generate


def main() -> None:
    p = argparse.ArgumentParser(description="ARR-06 campaign")
    p.add_argument("--root", default="experiments/campaigns/ARR-06-2026-001")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    root = Path(a.root).resolve()
    if a.report_only:
        c = json.loads((root / "final" / "ARR06_FINAL_REPORT.json").read_text(encoding="utf-8"))
        print(json.dumps({k: str(v) for k, v in generate(root, c).items()}, indent=2))
        return
    c = execute(root)
    print(json.dumps({"terminal": c["terminal_state"], "verdicts": c["verdicts"],
                      "files": {k: str(v) for k, v in generate(root, c).items()}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
