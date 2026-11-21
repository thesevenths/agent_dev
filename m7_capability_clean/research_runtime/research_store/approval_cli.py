"""Out-of-band HUMAN approval console. This is deliberately NOT an Agent tool.

Run in a terminal (separate process from the Agent)::

    D:/python313/python.exe -m research_runtime.research_store.approval_cli \
        grant approval_ab12cd34 --approved-by "alice" --yes

Other commands: ``list``, ``show``, ``reject``, ``revoke``.

Why a console and not a tool
-----------------------------
An LLM can only reach ``research_runtime.tools``. There is no
``approve_research_plan`` tool: the strongest record a tool can create is
``status="pending"``. Only this console (or a test acting as the human
operator) can call ``ApprovalStore.grant()``, which writes ``source="human"``.
The console additionally re-reads the CURRENT plan hash and refuses to grant if
the plan changed while the request was waiting -- a human must approve the
version they actually reviewed.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import ResearchStores, default_db_path


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research-approval",
        description="Human-side approval console for autonomous research plans",
    )
    parser.add_argument("--db", default=str(default_db_path()),
                        help="research state SQLite path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list approvals for a plan")
    p_list.add_argument("plan_id", nargs="?")

    p_show = sub.add_parser("show", help="show one approval")
    p_show.add_argument("approval_id")

    p_grant = sub.add_parser("grant", help="approve a pending request as a human")
    p_grant.add_argument("approval_id")
    p_grant.add_argument("--approved-by", required=True,
                         help="the human operator's identity")
    p_grant.add_argument("--source", default="human",
                         choices=["human", "system_policy"])
    p_grant.add_argument("--notes", default="")
    p_grant.add_argument("--yes", action="store_true",
                         help="skip the interactive confirmation prompt")

    p_reject = sub.add_parser("reject", help="reject a pending request")
    p_reject.add_argument("approval_id")
    p_reject.add_argument("--decided-by", required=True)
    p_reject.add_argument("--notes", default="")

    p_revoke = sub.add_parser("revoke", help="invalidate an approval")
    p_revoke.add_argument("approval_id")
    p_revoke.add_argument("--decided-by", required=True)
    p_revoke.add_argument("--notes", default="")

    args = parser.parse_args(argv)
    stores = ResearchStores(args.db)

    if args.command == "list":
        if args.plan_id:
            _print([r.model_dump(mode="json")
                    for r in stores.approvals.for_resource(args.plan_id)])
        else:
            rows = stores.db.rows(
                "SELECT * FROM approvals ORDER BY created_at")
            _print([dict(r) for r in rows])
        return 0

    if args.command == "show":
        record = stores.approvals.get(args.approval_id)
        if record is None:
            print("unknown approval_id", file=sys.stderr)
            return 1
        _print({"approval": record.model_dump(mode="json"),
                "events": stores.approvals.events(args.approval_id)})
        return 0

    record = stores.approvals.get(args.approval_id)
    if record is None:
        print("unknown approval_id", file=sys.stderr)
        return 1

    if args.command in ("grant", "reject", "revoke"):
        if not args.yes and args.command == "grant":
            print(f"Approve {record.resource_type} {record.resource_id} "
                  f"@ {record.resource_sha256[:12]}? [y/N] ", end="")
            if input().strip().lower() not in ("y", "yes"):
                print("not granted")
                return 1
        try:
            if args.command == "grant":
                current = None
                if record.resource_type == "experiment_plan":
                    current = stores.plans.current_sha256(record.resource_id)
                    if current is None:
                        raise ValueError("plan no longer exists")
                result = stores.approvals.grant(
                    args.approval_id, approved_by=args.approved_by,
                    source=args.source, notes=args.notes,
                    current_resource_sha256=current)
            elif args.command == "reject":
                result = stores.approvals.reject(
                    args.approval_id, decided_by=args.decided_by, notes=args.notes)
            else:
                result = stores.approvals.revoke(
                    args.approval_id, decided_by=args.decided_by, notes=args.notes)
        except Exception as exc:  # console: a clear message, never a traceback
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        _print(result.model_dump(mode="json"))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
