"""ARR_EMB_RSI_01 -- benchmark integrity primitives.

Ground truth is sealed OUTSIDE the agent-visible run directory. Evidence is a
bundle of artifact claims with path + hash + claim hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEAK_KEYS = frozenset({
    "ground_truth", "correct_intervention", "oracle_answer",
    "hidden_test_result", "failure_id", "failure_mode",
    "ground_truth_root_cause",
})


class IntegrityError(RuntimeError):
    pass


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactClaim:
    claim_id: str
    artifact_path: str
    artifact_sha256: str
    claim_text: str
    claim_sha256: str = ""
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["claim_sha256"]:
            d["claim_sha256"] = claim_hash(
                self.claim_id, self.artifact_sha256, self.claim_text, self.run_id)
        return d


def claim_hash(claim_id: str, artifact_sha256: str,
               claim_text: str, run_id: str) -> str:
    return sha256_json({"claim_id": claim_id,
                        "artifact_sha256": artifact_sha256,
                        "claim_text": claim_text, "run_id": run_id})


class ArtifactResolver:
    """Path -> SHA-256 -> claim hash -> run id. Fail closed."""

    def verify(self, claims: list[ArtifactClaim]) -> dict[str, Any]:
        results = []
        for claim in claims:
            d = claim.to_dict()
            path = Path(d["artifact_path"])
            reasons = []
            if not path.is_file():
                reasons.append("artifact_missing")
            actual_hash = sha256_file(path) if path.is_file() else ""
            if actual_hash != d["artifact_sha256"]:
                reasons.append("artifact_hash_mismatch")
            if d["claim_sha256"] != claim_hash(
                    d["claim_id"], d["artifact_sha256"],
                    d["claim_text"], d["run_id"]):
                reasons.append("claim_hash_mismatch")
            results.append({**d, "ok": not reasons, "reasons": reasons,
                            "actual_sha256": actual_hash})
        return {"n_claims": len(claims),
                "n_verified": sum(r["ok"] for r in results),
                "results": results,
                "pass": bool(claims) and all(r["ok"] for r in results)}


@dataclass(frozen=True)
class EvidencePackage:
    """Only this object is given to diagnosis controllers."""

    evidence_id: str
    independent_unit: str
    metrics: dict[str, float]
    config_snapshot: dict[str, Any]
    budget_usage: dict[str, Any]
    claims: tuple[ArtifactClaim, ...]
    task_label: str = "unlabelled_dynamics_case"

    def to_agent_dict(self) -> dict[str, Any]:
        d = {"evidence_id": self.evidence_id,
             "independent_unit": self.independent_unit,
             "metrics": dict(self.metrics),
             "config_snapshot": dict(self.config_snapshot),
             "budget_usage": dict(self.budget_usage),
             "claims": [c.to_dict() for c in self.claims],
             "task_label": self.task_label}
        assert_no_ground_truth(d)
        return d


class SealedGroundTruthStore:
    """Seal location is derived from run_root's PARENT, never inside run_root."""

    def __init__(self, *, run_root: str | Path,
                 seal_dir: str | Path | None = None):
        self.run_root = Path(run_root).resolve()
        self.seal_dir = Path(seal_dir or self.run_root.parent / "GT_SEALED")
        self.seal_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.seal_dir / "ground_truth.sealed.json"

    def seal(self, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
        obj = {"experiment_id": "ARR_EMB_RSI_01",
               "records": [records[k] for k in sorted(records)]}
        digest = sha256_json(obj)
        self.path.write_text(json.dumps({**obj, "sha256": digest}, indent=2,
                                        sort_keys=True, ensure_ascii=False),
                             encoding="utf-8")
        return {"path": str(self.path), "sha256": digest,
                "ground_truth_sha256": digest,
                "outside_run_root": not str(self.path).startswith(
                    str(self.run_root))}

    def load(self, expected_sha256: str) -> dict[str, dict[str, Any]]:
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        records = obj.get("records", [])
        if sha256_json({"experiment_id": obj.get("experiment_id"),
                        "records": records}) != expected_sha256:
            raise IntegrityError("ground truth SHA mismatch")
        return {str(r["case_id"]): r for r in records}


def assert_no_ground_truth(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in LEAK_KEYS:
                raise IntegrityError(f"ground truth leak at {path}.{key}")
            assert_no_ground_truth(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            assert_no_ground_truth(value, f"{path}[{index}]")


__all__ = ["IntegrityError", "ArtifactClaim", "ArtifactResolver",
           "EvidencePackage", "SealedGroundTruthStore", "claim_hash",
           "assert_no_ground_truth", "sha256_json", "sha256_file"]
