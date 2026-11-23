"""ARR-04 campaign orchestrator: state machine, checkpoints, resume, budget."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .core import CAMPAIGN_ID, sha256_json, now_iso

PHASES = (
    "phase0_audit", "preflight", "protocol_freeze", "m1_active_diagnosis",
    "m2_research_selection", "m3_knowledge_transfer", "recursive_improvement",
    "independent_audit", "final_report",
)


class BudgetController:
    def __init__(self, *, max_formal_runs: int, max_preflight_runs: int,
                 max_llm_calls: int = 400):
        self.max_formal_runs = max_formal_runs
        self.max_preflight_runs = max_preflight_runs
        self.max_llm_calls = max_llm_calls
        self.formal_runs = 0
        self.preflight_runs = 0
        self.llm_calls = 0

    def consume(self, *, kind: str) -> None:
        if kind == "formal":
            self.formal_runs += 1
            if self.formal_runs > self.max_formal_runs:
                raise RuntimeError("formal run budget exhausted")
        elif kind == "preflight":
            self.preflight_runs += 1
            if self.preflight_runs > self.max_preflight_runs:
                raise RuntimeError("preflight run budget exhausted")
        elif kind == "llm":
            self.llm_calls += 1
            if self.llm_calls > self.max_llm_calls:
                raise RuntimeError("LLM call budget exhausted")

    def stopping_condition(self) -> dict[str, Any]:
        return {"formal_exhausted": self.formal_runs >= self.max_formal_runs,
                "preflight_exhausted": self.preflight_runs >= self.max_preflight_runs,
                "llm_exhausted": self.llm_calls >= self.max_llm_calls}

    def to_dict(self) -> dict[str, Any]:
        return {"max_formal_runs": self.max_formal_runs,
                "max_preflight_runs": self.max_preflight_runs,
                "max_llm_calls": self.max_llm_calls,
                "formal_runs": self.formal_runs,
                "preflight_runs": self.preflight_runs,
                "llm_calls": self.llm_calls}


class CampaignState:
    """Persisted state machine with per-phase checkpoints and safe resume."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "campaign_state.json"
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "logs" / "campaign.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"campaign_id": CAMPAIGN_ID, "created_at": now_iso(),
                "phase": "init", "phase_status": {}, "checkpoints": {},
                "budget": {}, "terminal_state": None}

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        self.state_path.write_text(json.dumps(
            self.data, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")

    def log(self, message: str, level: str = "INFO") -> None:
        line = f"{now_iso()} [{level}] {message}\n"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def phase_done(self, phase: str) -> bool:
        return self.data["phase_status"].get(phase) == "completed"

    def start_phase(self, phase: str) -> None:
        self.data["phase"] = phase
        self.data["phase_status"][phase] = "in_progress"
        self.log(f"PHASE START {phase}")
        self.save()

    def complete_phase(self, phase: str, summary: dict[str, Any]) -> None:
        self.data["phase_status"][phase] = "completed"
        self.data["phase_status"][phase + "_summary"] = summary
        self.log(f"PHASE DONE {phase}")
        self.save()

    def block_phase(self, phase: str, reason: str) -> None:
        self.data["phase_status"][phase] = "blocked"
        self.data["phase_status"][phase + "_reason"] = reason
        self.log(f"PHASE BLOCKED {phase}: {reason}", "WARN")
        self.save()

    def checkpoint(self, name: str, payload: Any) -> str:
        digest = sha256_json(payload)
        path = self.checkpoint_dir / f"{name}.json"
        path.write_text(json.dumps(
            {"name": name, "sha256": digest, "timestamp": now_iso(),
             "payload": payload}, indent=2, sort_keys=True,
            ensure_ascii=False, default=str), encoding="utf-8")
        self.data["checkpoints"][name] = {"sha256": digest, "path": str(path)}
        self.save()
        return digest

    def load_checkpoint(self, name: str) -> Any | None:
        entry = self.data["checkpoints"].get(name)
        if not entry:
            return None
        payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        if sha256_json(payload["payload"]) != payload["sha256"]:
            raise RuntimeError(f"checkpoint {name} hash mismatch")
        return payload["payload"]

    def set_terminal(self, state: str) -> None:
        self.data["terminal_state"] = state
        self.log(f"TERMINAL {state}")
        self.save()

    def run_phase(self, phase: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Execute a phase once; skip if a valid completed checkpoint exists."""
        if self.phase_done(phase):
            self.log(f"PHASE SKIP (already completed) {phase}")
            return self.data["phase_status"].get(phase + "_summary", {})
        self.start_phase(phase)
        try:
            summary = fn()
            self.complete_phase(phase, summary)
            return summary
        except Exception as exc:
            self.log(f"PHASE ERROR {phase}: {type(exc).__name__}: {exc}", "ERROR")
            raise


__all__ = ["PHASES", "BudgetController", "CampaignState"]
