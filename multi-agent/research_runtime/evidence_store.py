"""Append-only evidence ledger for research runs.

Separate from ``longterm.py`` (conversation memory) on purpose: this is a
*machine-readable audit trail* of what was submitted, what came back and when.
No dedup, no LLM-written prose -- every record is a structured event with an
ISO timestamp, so a result can always be traced back to its job and spec.
"""
from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Any

from .settings import ensure_data_dir


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class EvidenceStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else ensure_data_dir() / "evidence.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event_type: str, payload: dict[str, Any]) -> dict:
        event = {
            "at": _now(),
            "event_type": event_type,
            "payload": payload,
        }
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return event

    def query(self, *, experiment_id: str | None = None,
              event_type: str | None = None,
              limit: int | None = None) -> list[dict]:
        if not self.path.is_file():
            return []
        found: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event_type and event.get("event_type") != event_type:
                    continue
                payload = event.get("payload") or {}
                if experiment_id and (
                    payload.get("experiment_id") != experiment_id
                    and (payload.get("result") or {}).get("experiment_id") != experiment_id
                ):
                    continue
                found.append(event)
        if limit is not None:
            found = found[-limit:]
        return found

    def clear(self) -> None:
        """Only for tests / explicit maintenance."""
        with self._lock:
            self.path.write_text("", encoding="utf-8")
