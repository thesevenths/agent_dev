"""ARR-05 orchestrator: state machine, checkpoints, resume, budget controller."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .core import CAMPAIGN_ID, sha256_json, now_iso

PHASES = ("p0_source", "preflight", "protocol_freeze", "m1", "m2", "m3",
          "rq", "rsi", "independent_audit", "final_report")


class BudgetController:
    def __init__(self, *, max_formal: int = 1500, max_llm: int = 2500,
                 max_preflight: int = 60, max_replay: int = 200):
        self.max_formal, self.max_llm = max_formal, max_llm
        self.max_preflight, self.max_replay = max_preflight, max_replay
        self.formal = self.llm = self.preflight = self.replay = 0

    def consume(self, kind: str) -> None:
        limits = {"formal": (self.formal, self.max_formal),
                  "llm": (self.llm, self.max_llm),
                  "preflight": (self.preflight, self.max_preflight),
                  "replay": (self.replay, self.max_replay)}
        if kind not in limits:
            raise ValueError(f"unknown budget kind {kind}")
        used, cap = limits[kind]
        if used >= cap:
            raise RuntimeError(f"{kind} budget exhausted")
        setattr(self, kind, used + 1)

    def to_dict(self) -> dict[str, Any]:
        return {"formal": self.formal, "llm": self.llm,
                "preflight": self.preflight, "replay": self.replay,
                "max_formal": self.max_formal, "max_llm": self.max_llm,
                "max_preflight": self.max_preflight, "max_replay": self.max_replay}


class CampaignState:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "campaign_state.json"
        self.ckpt = self.root / "checkpoints"
        self.ckpt.mkdir(exist_ok=True)
        self.log_path = self.root / "logs" / "campaign.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"campaign_id": CAMPAIGN_ID, "created_at": now_iso(),
                "phase": "init", "phase_status": {}, "checkpoints": {},
                "terminal_state": None, "budget": {}}

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        self.state_path.write_text(json.dumps(self.data, indent=2, sort_keys=True,
                                              ensure_ascii=False, default=str),
                                   encoding="utf-8")

    def log(self, msg: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{now_iso()} {msg}\n")

    def phase_done(self, p: str) -> bool:
        return self.data["phase_status"].get(p) == "completed"

    def run_phase(self, p: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        if self.phase_done(p):
            self.log(f"SKIP {p}")
            return self.data["phase_status"].get(p + "_summary", {})
        self.data["phase"] = p
        self.data["phase_status"][p] = "in_progress"
        self.log(f"START {p}")
        self.save()
        try:
            s = fn()
            self.data["phase_status"][p] = "completed"
            self.data["phase_status"][p + "_summary"] = s
            self.log(f"DONE {p}")
            self.save()
            return s
        except Exception as exc:
            self.data["phase_status"][p] = f"error: {type(exc).__name__}: {exc}"
            self.log(f"ERROR {p}: {exc}")
            self.save()
            raise

    def checkpoint(self, name: str, payload: Any) -> str:
        d = sha256_json(payload)
        p = self.ckpt / f"{name}.json"
        p.write_text(json.dumps({"name": name, "sha256": d, "payload": payload},
                                indent=2, sort_keys=True, ensure_ascii=False,
                                default=str), encoding="utf-8")
        self.data["checkpoints"][name] = {"sha256": d, "path": str(p)}
        self.save()
        return d

    def set_terminal(self, state: str) -> None:
        self.data["terminal_state"] = state
        self.log(f"TERMINAL {state}")
        self.save()


__all__ = ["PHASES", "BudgetController", "CampaignState"]
