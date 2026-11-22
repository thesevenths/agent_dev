"""ARR_EMB_RSI_02 -- artifact/evidence evaluation with claim verification."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .integrity import IntegrityError, assert_no_ground_truth, sha256_json


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def claim_hash(claim_id: str, artifact_sha256: str,
               claim_text: str, run_id: str) -> str:
    return sha256_json({"claim_id": claim_id,
                        "artifact_sha256": artifact_sha256,
                        "claim_text": claim_text, "run_id": run_id})


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
                self.claim_id, self.artifact_sha256,
                self.claim_text, self.run_id)
        return d


class ArtifactResolver:
    def verify(self, claims: list[ArtifactClaim]) -> dict[str, Any]:
        rows = []
        for claim in claims:
            d = claim.to_dict()
            p = Path(d["artifact_path"])
            reasons = []
            actual = ""
            if not p.is_file():
                reasons.append("artifact_missing")
            else:
                actual = sha256_file(p)
                if actual != d["artifact_sha256"]:
                    reasons.append("artifact_hash_mismatch")
            expected_claim = claim_hash(
                d["claim_id"], d["artifact_sha256"],
                d["claim_text"], d["run_id"])
            if d["claim_sha256"] != expected_claim:
                reasons.append("claim_hash_mismatch")
            rows.append({**d, "actual_sha256": actual, "ok": not reasons,
                         "reasons": reasons})
        return {"n_claims": len(claims),
                "n_verified": sum(r["ok"] for r in rows),
                "integrity_pass": bool(claims) and all(r["ok"] for r in rows),
                "claims": rows}


def evidence_audit(*, resolver_result: dict[str, Any],
                   agent_citations: list[str],
                   relevant_prefixes: tuple[str, ...] = (
                       "artifact:", "metric:", "result:")) -> dict[str, Any]:
    verified = {r["claim_id"] for r in resolver_result["claims"] if r["ok"]}
    cited = set(agent_citations or [])
    valid = cited & verified
    # Relevance is judged against recorded artifact ids, not GT.
    relevant = {c for c in cited if any(c.startswith(p) for p in relevant_prefixes)}
    sufficient = len(valid) >= 2 and len(relevant) >= 2
    return {
        "artifact_integrity": resolver_result["integrity_pass"],
        "citation_validity": (len(valid) / len(cited)) if cited else 0.0,
        "evidence_relevance": (len(relevant) / len(cited)) if cited else 0.0,
        "evidence_sufficiency": sufficient,
        "n_verified_artifacts": len(verified),
        "n_cited": len(cited),
    }


def build_evidence(*, case_id: str, unit: str,
                   metrics: dict[str, float], run: dict[str, Any]) -> dict[str, Any]:
    claims = []
    for name, path in (run.get("artifact_paths") or {}).items():
        claims.append(ArtifactClaim(
            claim_id=f"artifact:{case_id}:{name}",
            artifact_path=path,
            artifact_sha256=(run.get("artifact_hashes") or {}).get(name, ""),
            claim_text=f"engine artifact {name} from run {run.get('run_id')}",
            run_id=str(run.get("run_id") or "")))
    claims.append(ArtifactClaim(
        claim_id=f"artifact:{case_id}:config",
        artifact_path=run["config_path"],
        artifact_sha256=run["config_sha256"],
        claim_text="engine config actually submitted",
        run_id=str(run.get("run_id") or "")))
    curves = {k: v for k, v in metrics.items()
              if "@" in k or "cosine" in k or "mse" in k}
    payload = {"evidence_id": "ev_" + sha256_json(
                   {"case": case_id, "metrics": metrics})[:12],
               "regime_id": "regime_" + sha256_json(unit)[:12],
               "metrics": dict(metrics),
               "curves": curves,
               "claims": [c.to_dict() for c in claims],
               "budget": {"max_jobs": 4, "max_seconds": 900}}
    assert_no_ground_truth(payload)
    return {"payload": payload, "claims": claims,
            "resolver": ArtifactResolver().verify(claims)}


__all__ = ["ArtifactClaim", "ArtifactResolver", "evidence_audit",
           "build_evidence", "sha256_file", "claim_hash"]
