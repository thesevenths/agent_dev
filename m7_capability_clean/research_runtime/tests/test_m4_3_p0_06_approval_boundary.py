"""M4.3-00 / P0-06 -- approval boundary is documented ACCURATELY (engine-free).

These tests are executable fact-checks for APPROVAL_BOUNDARY_M4_3.md. They
assert both what the boundary DOES enforce and -- crucially -- what it does
NOT, so the documentation cannot silently overclaim a strong identity model:

  * Director has no self-grant path;
  * Agent-facing tool surface has NO grant tool (only pending requests);
  * approval_cli is a separate console entry point;
  * ApprovalStore.grant is STILL a public Python Store API (convention, not a
    sandbox) and ``approved_by`` is only checked for non-emptiness.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from research_runtime.research_director import ResearchDirector
from research_runtime.research_store.approval_store import ApprovalStore
from research_runtime import tools as tool_module

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------- #1 Director
def test_director_has_no_self_grant_capability():
    assert not hasattr(ResearchDirector, "approve_approval")
    assert not hasattr(ResearchDirector, "grant")
    assert "approval_granter" not in inspect.signature(
        ResearchDirector.__init__).parameters
    src = Path(inspect.getfile(ResearchDirector)).read_text(encoding="utf-8")
    assert "approvals.grant(" not in src, "Director must never write a grant"


# --------------------------------------------------------------------- #2 tool surface
def test_agent_tool_surface_exposes_no_grant_tool():
    for attr in ("CAMPAIGN_RESEARCH_TOOLS", "RESEARCH_TOOLS"):
        names = [getattr(t, "name", None) or getattr(t, "__name__", None)
                 for t in getattr(tool_module, attr)]
        assert names, attr
        assert not [n for n in names
                    if "grant" in str(n).lower() or "approve" in str(n).lower()], \
            (attr, names)
    # The strongest Agent capability is a PENDING request, not a decision.
    assert "request_plan_approval" in [
        getattr(t, "name", None) or getattr(t, "__name__", None)
        for t in tool_module.CAMPAIGN_RESEARCH_TOOLS]
    all_public = [n for n in dir(tool_module) if not n.startswith("_")]
    assert not [n for n in all_public
                if "grant" in n.lower() or "approve" in n.lower()]


# --------------------------------------------------------------------- #3 separate console
def test_approval_cli_is_a_standalone_entry_point():
    """The console runs as its own process with a grant subcommand."""
    result = subprocess.run(
        [sys.executable, "-m",
         "research_runtime.research_store.approval_cli", "--help"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "grant" in result.stdout, "console exposes a grant subcommand"
    # It is NOT registered as an Agent tool.
    tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None)
                  for t in tool_module.CAMPAIGN_RESEARCH_TOOLS]
    assert "approval_cli" not in tool_names and "grant" not in tool_names


# --------------------------------------------------------------------- #4 honest trust model
def test_grant_is_a_public_store_api_not_a_sandbox():
    """grant() is importable/callable by any in-process Python caller."""
    assert callable(ApprovalStore.grant)
    params = inspect.signature(ApprovalStore.grant).parameters
    assert "approved_by" in params and "source" in params
    assert "grant" in (ApprovalStore.grant.__doc__ or "").lower() or \
        "human" in (ApprovalStore.grant.__doc__ or "").lower()


def test_approved_by_is_only_checked_for_non_emptiness():
    """ACCURACY: approved_by is an unauthenticated STRING, not an identity.

    The store only rejects an empty/blank value; it performs NO authentication,
    signature check, ACL lookup or token validation. This test pins that fact
    so no future doc can claim strong identity without changing this behavior.
    """
    from research_runtime.research_store import ResearchStores
    from research_runtime.research_store.db import ResearchStoreError
    import tempfile

    stores = ResearchStores(Path(tempfile.mkdtemp()) / "b.sqlite")
    req = stores.approvals.request(
        resource_type="experiment_plan", resource_id="plan_x",
        resource_sha256="a" * 64, requested_by="research_agent")
    # A fabricated identity string is ACCEPTED (only non-emptiness is checked):
    # this is exactly the weak trust model the document must describe.
    granted = stores.approvals.grant(
        req.approval_id, approved_by="totally_fabricated_identity",
        source="human")
    assert granted.approved_by == "totally_fabricated_identity"
    assert granted.source == "human"
    # Blank identity is rejected, but that is the ONLY check on the string.
    req2 = stores.approvals.request(
        resource_type="experiment_plan", resource_id="plan_y",
        resource_sha256="b" * 64, requested_by="research_agent")
    try:
        stores.approvals.grant(req2.approval_id, approved_by="   ",
                               source="human")
        raise AssertionError("blank approved_by must be rejected")
    except ResearchStoreError:
        pass


def test_boundary_document_exists_and_states_weak_identity():
    doc = Path(__file__).resolve().parents[1] / "APPROVAL_BOUNDARY_M4_3.md"
    text = doc.read_text(encoding="utf-8")
    assert "unauthenticated string" in text
    assert "does NOT enforce" in text
    assert "ApprovalStore.grant()" in text
    assert "no enterprise" in text.lower() or "no enterprise" in text
