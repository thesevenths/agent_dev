"""CI helper: rebuild an equivalent research.sqlite from the committed,
READ-ONLY export of the REAL P1-06 rows.

``research.sqlite`` is git-ignored (runtime state), so a GitHub Actions
checkout has no DB. Tests must not depend on a developer's private DB. The
seed is a byte-faithful export of the historical rows (NOT fabricated data);
it is used ONLY by tests / the offline backfill, never by production code.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research_runtime.research_store import ResearchStores

SEED_PATH = Path(__file__).resolve().parents[1] / \
    "M4_0_runs" / "evidence" / "p1_06_db_seed.json"


def real_db_path(workspace: Path) -> Path:
    return workspace / "research_runtime" / "_data" / "research.sqlite"


def build_equivalent_db(dest: Path, *, workspace: Path,
                        seed_path: Path | None = None) -> dict:
    """Prefer a COPY of the real DB; otherwise rebuild from the committed seed."""
    real = real_db_path(workspace)
    if real.is_file():
        import shutil
        shutil.copy2(real, dest)
        return {"source": "real_db_copy", "path": str(dest)}
    seed_path = seed_path or SEED_PATH
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    assert seed.get("schema_version") == "m4.0-p1-06-seed-v1"
    ResearchStores(dest)                      # creates ALL tables idempotently
    conn = sqlite3.connect(str(dest))
    try:
        for table, payload in seed["tables"].items():
            cols = payload["columns"]
            for row in payload["rows"]:
                values = [row.get(c) for c in cols]
                placeholders = ",".join("?" for _ in cols)
                conn.execute(
                    f'INSERT OR IGNORE INTO "{table}" '
                    f'({",".join(chr(34)+c+chr(34) for c in cols)}) '
                    f"VALUES ({placeholders})", tuple(values))
        conn.commit()
    finally:
        conn.close()
    return {"source": "committed_p1_06_seed", "path": str(dest),
            "seed": seed_path.name,
            "tables": list(seed["tables"])}
