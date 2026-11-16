"""M4.3-02 -- Confirmatory protocol builder + freeze gate.

The confirmatory experiment may only start after a COMPLETE, FROZEN protocol
exists and a human has approved THAT exact protocol SHA. This module builds the
protocol from the pilot evidence, freezes it, and refuses to freeze when the
pilot cannot support the design.

Freeze preconditions (spec section 7)
-------------------------------------
1. the pilot completed and a pilot analysis exists;
2. the smallest effect of interest is pre-specified (not read off the results);
3. the sample size is ESTIMATED from the pilot variance, not invented -- if the
   pilot data are insufficient the builder returns a BLOCKED draft and freezing
   is refused;
4. every snapshot id, seed, budget, metric, model/prompt/ranker/policy version
   and exclusion rule is explicit;
5. the protocol carries a single SHA-256 over all of the above.

There is no path from "pilot finished" to "confirmatory executing" that skips
either the freeze or the human approval file.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .experiment_registry import FrozenProtocol
from .metrics import PRIMARY_METRICS, frozen_metric_contract
from .protocol_schema import (
    CONFIRMATORY_REQUIRED_FIELDS, missing_confirmatory_fields,
)

PROTOCOL_FILENAME = "confirmatory_protocol.json"
APPROVAL_FILENAME = "confirmatory_approval.json"


class ProtocolBlocked(RuntimeError):
    """The pilot cannot support a confirmatory design (fail-closed)."""


class ProtocolNotApproved(RuntimeError):
    """No valid human approval is bound to this exact protocol SHA."""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_confirmatory_protocol(*, pilot_analysis: dict[str, Any],
                                snapshot_ids: list[str],
                                paired_seeds: list[int],
                                sample_size: int,
                                campaign_budget: dict[str, Any],
                                model_version: str,
                                prompt_version: str,
                                ranker_version: str,
                                domain_policy_version: str,
                                smallest_effect_of_interest: float,
                                alpha: float = 0.05,
                                target_power: float = 0.8,
                                correction: str = "holm",
                                exclusion_rules: tuple[str, ...] = (),
                                stopping_rule: str = "",
                                analysis_plan_version: str = "m4_3_confirmatory_v1",
                                seed: int = 42) -> FrozenProtocol:
    """Assemble a COMPLETE confirmatory protocol (all 20 fields).

    Raises ``ProtocolBlocked`` when the pilot cannot justify the design.
    """
    sample = pilot_analysis.get("sample_size") or {}
    if not pilot_analysis:
        raise ProtocolBlocked("no pilot analysis: run analyze-pilot first")
    if not snapshot_ids:
        raise ProtocolBlocked("the confirmatory snapshot set must be frozen")
    if not paired_seeds:
        raise ProtocolBlocked("paired seeds must be frozen")
    if sample_size <= 0:
        raise ProtocolBlocked("sample_size must be a positive block count")
    if smallest_effect_of_interest <= 0:
        raise ProtocolBlocked(
            "smallest_effect_of_interest must be pre-specified and positive")
    if sample.get("status") != "estimated":
        raise ProtocolBlocked(
            "the pilot does not support a sample-size estimate "
            f"(status={sample.get('status')!r}, "
            f"contrasts_without_variance={sample.get('contrasts_without_variance')}); "
            "complete the human blind review and re-run analyze-pilot")
    recommended = sample.get("n_blocks_recommended")
    if recommended and sample_size < recommended:
        raise ProtocolBlocked(
            f"sample_size={sample_size} is below the pilot recommendation "
            f"({recommended} blocks); a smaller confirmatory run would be "
            "underpowered by construction")

    protocol = FrozenProtocol(
        confirmatory=True,
        snapshot_ids=tuple(snapshot_ids),
        sample_size=int(sample_size),
        seed=seed,
        model_version=model_version,
        prompt_version=prompt_version,
        ranker_version=ranker_version,
        domain_policy_version=domain_policy_version,
        analysis_plan_version=analysis_plan_version,
        campaign_budget=dict(campaign_budget),
        paired_seeds=tuple(paired_seeds),
        budget=dict(campaign_budget),
        primary_metrics=tuple(PRIMARY_METRICS),
        metric_contract_sha256=frozen_metric_contract()["contract_sha256"],
        exclusion_rules=tuple(exclusion_rules or
                              ("no post-hoc exclusion; failed/interrupted/"
                               "invalid runs are retained",)),
        statistical_method="paired-block t-test on block differences",
        multiple_comparison_correction=correction,
        stopping_rule=(stopping_rule or
                       "all frozen blocks complete; no interim analysis"),
        smallest_effect_of_interest=float(smallest_effect_of_interest),
        alpha=float(alpha),
        target_power=float(target_power))
    missing = missing_confirmatory_fields(protocol)
    if missing:
        raise ProtocolBlocked(
            f"protocol is incomplete, missing {missing}; refusing to freeze")
    return protocol


def freeze_confirmatory_protocol(protocol: FrozenProtocol,
                                 *, out_dir: str | Path) -> dict[str, Any]:
    """Persist the frozen protocol (append-only version) and return its SHA."""
    if not protocol.confirmatory:
        raise ProtocolBlocked("only a confirmatory protocol can be frozen")
    missing = missing_confirmatory_fields(protocol)
    if missing:
        raise ProtocolBlocked(f"protocol is incomplete: missing {missing}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha = protocol.freeze_sha256()
    path = out / PROTOCOL_FILENAME
    versions: list[dict[str, Any]] = []
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        versions = list(previous.get("versions") or [])
        if any(v.get("protocol_sha256") == sha for v in versions):
            return previous          # idempotent: identical content
    entry = {"version": len(versions) + 1, "protocol_sha256": sha,
             "frozen_at": _now(), "frozen_dict": protocol.frozen_dict()}
    versions.append(entry)
    payload = {"schema": "m4_3_confirmatory_protocol_v1",
               "protocol_sha256": sha,
               "current_version": entry["version"],
               "versions": versions,
               "required_fields": list(CONFIRMATORY_REQUIRED_FIELDS)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return payload


def request_confirmatory_approval(*, protocol_sha256: str,
                                  out_dir: str | Path,
                                  requested_by: str = "m4_3_cli",
                                  notes: str = "") -> dict[str, Any]:
    """Write a PENDING approval request bound to the protocol SHA."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "m4_3_confirmatory_approval_v1",
               "protocol_sha256": protocol_sha256,
               "status": "pending",
               "requested_by": requested_by,
               "requested_at": _now(),
               "approved_by": "",
               "approved_at": None,
               "notes": notes}
    (out / APPROVAL_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def approve_confirmatory_protocol(*, protocol_sha256: str, approved_by: str,
                                  out_dir: str | Path, notes: str = ""
                                  ) -> dict[str, Any]:
    """Record a HUMAN approval bound to the exact protocol SHA.

    Mirrors the out-of-band approval console: the approver identity is required,
    the SHA must match the frozen protocol, and an earlier approval for a
    DIFFERENT protocol is never silently reused.
    """
    if not approved_by or not approved_by.strip():
        raise ProtocolNotApproved("approved_by is required (a human identity)")
    out = Path(out_dir)
    protocol_path = out / PROTOCOL_FILENAME
    if not protocol_path.is_file():
        raise ProtocolNotApproved("no frozen protocol to approve")
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if frozen.get("protocol_sha256") != protocol_sha256:
        raise ProtocolNotApproved(
            "the approval SHA does not match the frozen protocol; a human must "
            "approve the version they actually reviewed")
    payload = {"schema": "m4_3_confirmatory_approval_v1",
               "protocol_sha256": protocol_sha256,
               "status": "approved",
               "requested_by": "m4_3_cli",
               "requested_at": _now(),
               "approved_by": approved_by.strip(),
               "approved_at": _now(),
               "notes": notes}
    (out / APPROVAL_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_approval(*, protocol_sha256: str, out_dir: str | Path
                  ) -> dict[str, Any]:
    """Verify a human approval exists AND is bound to this exact SHA."""
    path = Path(out_dir) / APPROVAL_FILENAME
    if not path.is_file():
        raise ProtocolNotApproved(
            "no confirmatory approval file; a human must approve the frozen "
            "protocol before the confirmatory run can start")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "approved":
        raise ProtocolNotApproved(
            f"confirmatory approval status is {payload.get('status')!r}")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ProtocolNotApproved(
            "the approval is bound to a DIFFERENT protocol SHA")
    if not str(payload.get("approved_by") or "").strip():
        raise ProtocolNotApproved("the approval has no human identity")
    return payload


def load_frozen_protocol(out_dir: str | Path) -> FrozenProtocol:
    """Reconstruct the CURRENT frozen protocol from disk."""
    path = Path(out_dir) / PROTOCOL_FILENAME
    if not path.is_file():
        raise ProtocolBlocked("no frozen confirmatory protocol")
    payload = json.loads(path.read_text(encoding="utf-8"))
    current = payload["current_version"]
    entry = next(v for v in payload["versions"] if v["version"] == current)
    fields = entry["frozen_dict"]
    known = {f for f in FrozenProtocol.__dataclass_fields__}
    proto = FrozenProtocol(**{k: v for k, v in fields.items() if k in known})
    if proto.freeze_sha256() != entry["protocol_sha256"]:
        raise ProtocolBlocked(
            "the stored protocol does not reproduce its recorded SHA")
    return proto


__all__ = [
    "PROTOCOL_FILENAME", "APPROVAL_FILENAME", "ProtocolBlocked",
    "ProtocolNotApproved", "build_confirmatory_protocol",
    "freeze_confirmatory_protocol", "request_confirmatory_approval",
    "approve_confirmatory_protocol", "load_approval", "load_frozen_protocol",
]
