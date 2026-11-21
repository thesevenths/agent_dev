"""M4.0.1 P1-04 -- Discovery Evidence Adapter (READ-ONLY contract facade).

This module codifies the Evidence-Reader usage contract that M4.1 Discovery
MUST depend on, and makes the boundary with the legacy 12-step fail-closed
gate evidence source explicit.

Two distinct readers exist in ARR 1.1 and must NOT be conflated:

  1. LEGACY gate provider -- ``research_service.default_evidence_provider``.
     Reads ONLY the append-only ``evidence.jsonl`` ledger (event_type="result")
     and is consumed by the 12-step fail-closed statistical gate. It is
     intentionally NOT replaced by M4.0 (change-scope control) and this
     adapter never routes through it.

  2. UNIFIED read model -- ``evidence_read_model.query_run_evidence`` /
     ``UnifiedEvidenceReader`` (M4.0-04). Enumerates ALL six existing evidence
     carriers for a run, dedupes by SHA-256 of real bytes, preserves raw
     verdict strings verbatim, never recomputes a statistical verdict, never
     writes. This is the canonical source for M4.1 Discovery.

``DiscoveryEvidenceAdapter`` is a STABLE, thin facade over reader #2. M4.1
Discovery depends ONLY on this facade and never on the internal carrier
reading details of ``UnifiedEvidenceReader``.

INVARIANTS (enforced by research_runtime/tests/test_m4_0_discovery_adapter.py):
  * READ-ONLY: never writes, never calls ``default_evidence_provider``, never
    participates in a statistical gate.
  * NO VERDICT GENERATION: returns the EvidenceView verbatim, including the
    raw-verdict census; it never recomputes or manufactures a verdict.
  * NO FABRICATION: every returned evidence entity has >=1 real carrier; the
    adapter never synthesises an entity (e.g. an Experience) that was not
    actually recorded. Missing Experience therefore stays UNRESOLVED.
  * STABLE CONTRACT: M4.1 must depend only on this facade's public methods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .evidence_read_model import query_run_evidence


# The M4.1 Discovery facade is explicitly forbidden from routing through the
# legacy 12-step gate provider. If that name ever appears in this module's
# source, the contract test fails (see test_m4_0_discovery_adapter.py).
_LEGACY_GATE_PROVIDER = "default_evidence_provider"


class DiscoveryEvidenceAdapter:
    """Read-only Evidence Reader contract for M4.1 Discovery.

    Wraps the unified ``query_run_evidence`` entry point (the same one
    ResearchService and UnifiedResearchScheduler use) so Discovery and the rest
    of ARR see an IDENTICAL evidence set. It adds discovery-oriented accessors
    but NO new evidence semantics.
    """

    def __init__(self, workspace: str | Path | None = None):
        # None -> the default workspace (multi-agent/) is resolved by
        # query_run_evidence's module-level default reader.
        self.workspace = Path(workspace) if workspace is not None else None

    # -- canonical read ------------------------------------------------
    def query_run(self, run_id: str) -> dict[str, Any]:
        """Return the full, deterministic EvidenceView for a run.

        Delegates to the shared ``query_run_evidence`` entry point. READ-ONLY:
        no writes, no verdict recomputation, no gate participation.
        """
        return query_run_evidence(run_id, workspace=self.workspace)

    # -- discovery-oriented accessors (thin views over the view) --------
    def discover_entities(self, run_id: str) -> list[dict[str, Any]]:
        """The deduplicated evidence entities (identity = SHA-256 of real bytes)."""
        return self.query_run(run_id).get("evidence", [])

    def carriers(self, run_id: str) -> list[dict[str, Any]]:
        """Which carriers were read and how many records each contributed."""
        return self.query_run(run_id).get("carriers", [])

    def raw_verdict_census(self, run_id: str) -> dict[str, int]:
        """A REPORT over the stored raw verdict strings -- NOT a recomputed gate.

        M4.1 Discovery may read this to understand what verdict strings exist,
        but it must NOT treat it as an accept/reject decision.
        """
        return self.query_run(run_id).get("raw_verdict_census", {})

    def unresolved(self, run_id: str) -> list[dict[str, Any]]:
        """Honest, explicit gaps (e.g. engine-metric collisions) -- never hidden."""
        return self.query_run(run_id).get("unresolved", [])


def _self_source_uses_legacy_gate_provider() -> bool:
    """Static contract check: this module must never route through the legacy
    gate provider. Used by the contract test; also documents intent."""
    import inspect
    return _LEGACY_GATE_PROVIDER in inspect.getsource(DiscoveryEvidenceAdapter)
