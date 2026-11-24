"""ARR-07 CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import execute


def main() -> None:
    p = argparse.ArgumentParser(description="ARR-07 autonomous loop validation")
    p.add_argument("--root", default="experiments/campaigns/ARR-07-2026-001")
    a = p.parse_args()
    c = execute(Path(a.root).resolve())
    print(json.dumps({"overall": c["overall"], "verdicts": c["verdicts"],
                      "V1": {"converged": c["validations"]["V1"]["n_converged"],
                             "spurious": c["validations"]["V1"]["n_spurious_convergence"]},
                      "V2": {"blind": c["validations"]["V2"]["blind_converged"],
                             "leaky": c["validations"]["V2"]["leaky_converged"]},
                      "V3": {"pass": c["validations"]["V3"]["pass"]}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
