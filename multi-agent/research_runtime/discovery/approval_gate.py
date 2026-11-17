"""M7.6 P0-A: fail-closed EXTERNAL approval verifier.

This module is the M7 approval boundary. It never grants an approval and never
accepts an in-process callable/dict as evidence. The ONLY trusted source is the
M4.3 ``ApprovalStore``: a human operator records a decision out-of-band through
``approval_cli.py`` (a separate process), and this gate re-reads it from durable
storage and re-verifies the SHA it is bound to.

Trust model (unchanged from M4.3 / APPROVAL_BOUNDARY_M4_3.md)
-------------------------------------------------------------
``approved_by`` is an UNAUTHENTICATED operator-supplied string. There is no
signature, certificate or ACL lookup. ``source="human"`` is a caller-supplied
label. What this gate DOES enforce: a persisted, immutable, SHA-bound decision,
created by the out-of-band console, from an authoritative source, naming a
non-forbidden approver, with the required scope, re-read on every check.
"""
from __future__ import annotations

from typing import Any

from ..harness_rsi.identity import sha256_json

# Identifiers that must never count as a human operator. Includes the legacy
# M7 fake approver and demo/automation identities that could otherwise slip
# through a blacklist that only covered ["agent", "llm", ...].
FORBIDDEN_APPROVERS = frozenset({
    "", "agent", "llm", "research_agent", "auto", "automation", "system",
    "system_policy", "human-operator", "human_operator", "demo",
    "demo-script", "demo_script", "test", "ci", "workbuddy", "assistant",
})

AUTHORITATIVE_SOURCES = frozenset({"human"})


def verify_external_approval(*, approval_store: Any, approval_id: str,
                             protocol_sha: str,
                             allowed_sources: frozenset[str] | tuple[str, ...]
                             = AUTHORITATIVE_SOURCES,
                             scope: str = "run_campaign") -> dict[str, Any]:
    """Re-read and verify an out-of-band approval. Fail closed on every doubt.

    Returns a structured verdict; ``ok`` is True only when ALL of the following
    hold against the record currently in trusted storage:

      * a record with ``approval_id`` exists;
      * ``status == "approved"`` (pending/rejected/revoked all fail);
      * ``resource_sha256 == protocol_sha`` (SHA-bound; a changed preregistration
        invalidates the decision);
      * ``source`` is authoritative (``human`` by default; ``system_policy``
        can NEVER authorise a production campaign);
      * ``approved_by`` is non-empty and not a known automation identity;
      * ``scope`` is granted.

    No callable, no in-memory dict, no env var and no Git HEAD is consulted.
    """
    verdict: dict[str, Any] = {
        "ok": False, "reason": "", "approval_id": str(approval_id or ""),
        "fixture": False, "verifier": "m7_6_external_approval_v1",
    }
    if approval_store is None:
        verdict["reason"] = "no_approval_store"
        return verdict
    if not str(approval_id or "").strip():
        verdict["reason"] = "approval_id_missing"
        return verdict
    try:
        record = approval_store.get(str(approval_id))
    except Exception as exc:  # noqa: BLE001 - storage failure fails closed
        verdict["reason"] = f"approval_store_error:{type(exc).__name__}"
        return verdict
    if record is None:
        verdict["reason"] = "approval_not_found"
        return verdict

    status = str(getattr(record, "status", "") or "").lower()
    source = str(getattr(record, "source", "") or "").lower()
    approved_by = str(getattr(record, "approved_by", "") or "").strip()
    resource_sha = str(getattr(record, "resource_sha256", "") or "").strip()
    scopes = list(getattr(record, "scope", []) or [])
    events = []
    try:
        events = approval_store.events(str(approval_id))
    except Exception:  # noqa: BLE001 - events are supporting evidence only
        pass

    verdict.update({
        "status": status, "source": source, "resource_sha256": resource_sha,
        "approved_by_present": bool(approved_by), "scope": scopes,
        "evidence_sha": sha256_json({
            "approval_id": str(approval_id), "status": status,
            "source": source, "resource_sha256": resource_sha,
            "approved_by": approved_by, "scope": scopes,
            "events": events,
        }),
    })

    if status != "approved":
        verdict["reason"] = f"approval_status_{status or 'missing'}"
        return verdict
    if resource_sha != str(protocol_sha or "").strip():
        verdict["reason"] = "approval_not_bound_to_protocol_sha"
        return verdict
    if source not in set(allowed_sources):
        verdict["reason"] = f"approval_source_rejected:{source}"
        return verdict
    if approved_by.lower() in FORBIDDEN_APPROVERS:
        verdict["reason"] = "approval_approver_forbidden_or_missing"
        return verdict
    if scope not in scopes:
        verdict["reason"] = f"approval_scope_missing:{scope}"
        return verdict

    verdict["ok"] = True
    verdict["approved_by"] = approved_by
    verdict["reason"] = "verified_external_human_approval"
    return verdict


__all__ = ["FORBIDDEN_APPROVERS", "AUTHORITATIVE_SOURCES",
           "verify_external_approval"]
