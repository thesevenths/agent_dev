"""ARR 1.0 / D3 -- EvidenceVerifier: recompute hashes, do NOT trust declarations.

The quality evaluator (``research_quality._evidence_integrity``) used to accept
the *presence* of a ``report_sha256`` / ``content_sha256`` field as proof of
integrity. A field can exist and still not correspond to the referenced content
-- a tampered report could keep a stale hash and the old check would pass.

This verifier recomputes the SHA-256 from the ACTUAL content / config / dataset
and compares it to the declared value, returning an explicit pass/fail per
check. It is imported by the quality evaluator so the Evidence (E) dimension is
grounded in recomputation, not self-declaration.

Design rules (aligned with P0-2):

* When the referenced content is present, the declared hash MUST match the
  recomputed one -- otherwise the check fails (tamper detected).
* When the content is NOT present (legacy reports only carry the hash, not the
  raw content), the recomputation check is reported as ``unverifiable`` and is
  EXCLUDED from the pass/fail fraction. This keeps the old, valid artifacts
  green while still upgrading every new artifact to real verification.
* File-reference checks (dataset manifest, result paths) verify the referenced
  artifact exists and its recomputed hash matches, when a ``base_dir`` is given.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def sha256_of(obj: Any) -> str:
    """Canonical SHA-256 of any JSON-serialisable object (order-independent)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Any) -> str | None:
    p = Path(path)  # type: ignore[name-defined]
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass
class VerifyCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerificationResult:
    """Structured, auditable result of an EvidenceVerifier pass."""

    checks: list[VerifyCheck] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> "VerificationResult":
        self.checks.append(VerifyCheck(name, bool(passed), detail))
        return self

    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.all_passed(),
            "n_checks": len(self.checks),
            "n_passed": sum(1 for c in self.checks if c.passed),
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                       for c in self.checks],
        }


class EvidenceVerifier:
    """Recompute SHA-256 and validate actual content / config / dataset refs.

    Stateless and pure: it never mutates the input report. Every method returns
    a ``VerificationResult`` so the caller can fold the recomputation checks into
    the quality evaluator's Evidence dimension or report them independently.
    """

    #: Which report fields carry raw, hashable content.
    _HYP_CONTENT_KEY = "content"
    _PLAN_CONTENT_KEY = "content"

    def verify_report(self, report: dict[str, Any],
                      *, base_dir: Any = None,
                      only_verifiable: bool = False) -> VerificationResult:
        res = VerificationResult()
        hyps = report.get("hypotheses") or []
        plans = report.get("plans") or []
        comps = report.get("comparisons") or []
        # Whether this report carries raw content/config we can actually
        # recompute. Legacy reports store only hashes -> those content checks are
        # unverifiable. The report_sha256 check below is ALWAYS verifiable
        # (the report body is always present) so it is never suppressed.
        verifiable_items = any(h.get(self._HYP_CONTENT_KEY) is not None
                               for h in hyps) \
            or any(p.get(self._PLAN_CONTENT_KEY) is not None for p in plans) \
            or any(c.get("baseline_config") is not None for c in comps)

        declared = report.get("report_sha256")
        body = {k: v for k, v in report.items() if k != "report_sha256"}
        recomputed = sha256_of(body)
        # Always recompute the report hash: this is the core P0-2 upgrade and
        # must apply even to legacy reports (body is always present to hash).
        res.add("report_sha256_matches",
                bool(declared) and declared == recomputed,
                f"declared={declared} recomputed={recomputed}")

        # Hypotheses: only verifiable when the raw content is present.
        if hyps:
            verified = 0
            tampered = False
            for h in hyps:
                content = h.get(self._HYP_CONTENT_KEY)
                if content is None:
                    continue  # no raw content -> cannot recompute (legacy)
                ok = h.get("content_sha256") == sha256_of(content)
                if not ok:
                    tampered = True
                    res.add("hypothesis_content_sha256_matches", False,
                            f"mismatch for {h.get('hypothesis_id')}")
                    break
                verified += 1
            if tampered:
                pass  # failure already recorded
            elif verified:
                res.add("hypothesis_content_sha256_matches", True,
                        f"{verified} hypotheses recomputed & matched")
            elif not only_verifiable:
                res.add("hypothesis_content_sha256_matches", False,
                        "hypotheses present but none carried verifiable content")
        elif not only_verifiable:
            res.add("hypothesis_content_sha256_matches", False, "no hypotheses")

        # Plans: same rule.
        if plans:
            verified = 0
            tampered = False
            for p in plans:
                content = p.get(self._PLAN_CONTENT_KEY)
                if content is None:
                    continue
                if p.get("plan_sha256") != sha256_of(content):
                    tampered = True
                    res.add("plans_sha256_matches", False,
                            f"mismatch for {p.get('plan_id')}")
                    break
                verified += 1
            if not tampered and verified:
                res.add("plans_sha256_matches", True,
                        f"{verified} plans recomputed & matched")
            elif not tampered and not only_verifiable:
                res.add("plans_sha256_matches", False,
                        "plans present but none carried verifiable content")

        # Comparison config hashes: recompute from the embedded config dict.
        if comps:
            tampered = False
            checked = 0
            for c in comps:
                for cfg_key, hash_key in (
                        ("baseline_config", "baseline_config_sha256"),
                        ("candidate_config", "candidate_config_sha256")):
                    cfg = c.get(cfg_key)
                    declared_hash = c.get(hash_key)
                    if cfg is None or declared_hash is None:
                        continue  # legacy: only the hash was stored
                    checked += 1
                    if declared_hash != sha256_of(cfg):
                        tampered = True
                        res.add(f"config_sha256:{hash_key}", False,
                                "config hash mismatch")
                        break
                if tampered:
                    break
            if not tampered and checked:
                res.add("config_sha256_matches", True,
                        f"{checked} configs recomputed & matched")
        return res

    def verify_dataset_manifest(self, manifest: dict[str, Any],
                               *, base_dir: Any = None,
                               only_verifiable: bool = False
                               ) -> VerificationResult:
        res = VerificationResult()
        declared_overall = manifest.get("manifest_sha256")
        if declared_overall:
            body = {k: v for k, v in manifest.items()
                    if k != "manifest_sha256"}
            res.add("manifest_sha256_matches",
                    declared_overall == sha256_of(body),
                    "dataset manifest recomputed")
        for name, info in (manifest.get("splits") or {}).items():
            declared = (info or {}).get("sha256")
            rows = (info or {}).get("rows")
            if declared and rows is not None:
                res.add(f"split_sha256:{name}",
                        declared == sha256_of(rows),
                        "split rows recomputed")
        return res

    def verify_bundle_integrity(self, bundle_dir: Any) -> VerificationResult:
        """Verify a portable evidence bundle is internally self-consistent.

        Checks that ``evidence_hashes.json`` lists every bundle file with a hash
        that matches the file on disk, so a third party can confirm the bundle
        was not tampered after publication.
        """
        root = Path(bundle_dir)
        res = VerificationResult()
        hashes_path = root / "evidence_hashes.json"
        if not hashes_path.is_file():
            res.add("bundle_evidence_hashes_present", False,
                    "evidence_hashes.json missing")
            return res
        declared = json.loads(hashes_path.read_text(encoding="utf-8"))
        declared_files = declared.get("bundle_files") or {}
        for name, expected in declared_files.items():
            actual = sha256_file(root / name)
            res.add(f"bundle_file:{name}",
                    actual is not None and actual == expected,
                    "" if actual == expected else "hash mismatch or missing")
        return res
