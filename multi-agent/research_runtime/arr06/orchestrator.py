"""ARR-06 orchestrator: state machine, checkpoints, resume, budget, R0-R3 contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .core import CAMPAIGN_ID, sha256_json, now_iso

PHASES = ("phase0_source", "phase1_hardening", "preflight", "protocol_freeze",
          "rq1_belief", "rq2_selection", "rq3_transfer", "rq4_recursive",
          "rq5_discovery", "independent_audit", "final_report")


class BudgetController:
    def __init__(self, *, max_formal=700, max_llm=600,
                 max_preflight=80, max_replay=120):
        self.caps = {"formal": max_formal, "llm": max_llm,
                     "preflight": max_preflight, "replay": max_replay}
        self.used = {k: 0 for k in self.caps}

    def consume(self, kind: str) -> None:
        if kind not in self.caps:
            raise ValueError(kind)
        if self.used[kind] >= self.caps[kind]:
            raise RuntimeError(f"{kind} budget exhausted")
        self.used[kind] += 1

    def to_dict(self) -> dict[str, Any]:
        return {**self.used, **{f"max_{k}": v for k, v in self.caps.items()}}


class CampaignState:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "campaign_state.json"
        self.ckpt = self.root / "checkpoints"
        self.ckpt.mkdir(exist_ok=True)
        self.log_path = self.root / "logs" / "campaign.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.is_file() else {
                "campaign_id": CAMPAIGN_ID, "created_at": now_iso(),
                "phase": "init", "phase_status": {}, "checkpoints": {},
                "terminal_state": None}

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        self.state_path.write_text(json.dumps(self.data, indent=2, sort_keys=True,
                                              ensure_ascii=False, default=str),
                                   encoding="utf-8")

    def log(self, msg: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{now_iso()} {msg}\n")

    def done(self, phase: str) -> bool:
        return self.data["phase_status"].get(phase) == "completed"

    def run_phase(self, phase: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        if self.done(phase):
            self.log(f"SKIP {phase}")
            return self.data["phase_status"].get(phase + "_summary", {})
        self.data["phase"] = phase
        self.data["phase_status"][phase] = "in_progress"
        self.log(f"START {phase}")
        self.save()
        try:
            s = fn()
            self.data["phase_status"][phase] = "completed"
            self.data["phase_status"][phase + "_summary"] = s
            self.log(f"DONE {phase}")
            self.save()
            return s
        except Exception as exc:
            self.data["phase_status"][phase] = f"error: {type(exc).__name__}: {exc}"
            self.log(f"ERROR {phase}: {exc}")
            self.save()
            raise

    def checkpoint(self, name: str, payload: Any) -> str:
        d = sha256_json(payload)
        p = self.ckpt / f"{name}.json"
        p.write_text(json.dumps({"sha256": d, "payload": payload}, indent=2,
                                sort_keys=True, ensure_ascii=False, default=str),
                     encoding="utf-8")
        self.data["checkpoints"][name] = d
        self.save()
        return d

    def set_terminal(self, state: str) -> None:
        self.data["terminal_state"] = state
        self.log(f"TERMINAL {state}")
        self.save()


def reproducibility_contract(root: Path) -> dict[str, Any]:
    """Report R0-R3 levels from actual artifacts, never a single boolean."""
    raw = list((root / "raw_data" / "immutable").glob("*.json"))
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in raw]
    r0 = bool(raw) and all(r.get("config_sha256") and r.get("result_hash")
                           and r.get("source_commit") for r in recs)
    stats = root / "independent_audit" / "statistics_audit.json"
    r1 = stats.is_file() and bool(json.loads(stats.read_text(encoding="utf-8")).get("pass"))
    replay = root / "independent_audit" / "replay_audit.json"
    r2 = replay.is_file() and bool(json.loads(replay.read_text(encoding="utf-8")).get("pass"))
    r3 = bool((root / "final" / "ARR06_FINAL_REPORT.json").is_file()
              and (root / "protocol" / "protocol.sha256").is_file() and r0 and r1)
    return {"R0_artifact_integrity": r0, "R1_statistical_recalculation": r1,
            "R2_experiment_replay": r2, "R3_full_campaign_reproduction": r3,
            "n_records": len(raw)}


__all__ = ["PHASES", "BudgetController", "CampaignState", "reproducibility_contract"]
