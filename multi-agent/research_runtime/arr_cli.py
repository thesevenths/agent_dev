"""ARR v1.0 unified operations CLI (WP-10).

Real commands only. Each subcommand calls an implemented module; no decorative
placeholders. Reuses existing research_runtime.cli for legacy run/loop/verify.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _cmd_program_create(args) -> int:
    from .program import ResearchProgramState
    st = ResearchProgramState(
        root=Path(args.root) / args.program_id, program_id=args.program_id,
        objective={"question": args.objective, "domain": args.domain},
        domain_scope=args.domain,
        resource_policy=json.loads(args.resource_policy or "{}"),
        approval_policy=json.loads(args.approval_policy or "{}"))
    st.transition("OBJECTIVE_VALIDATED", reason="cli_created")
    st.transition("RESEARCH_QUEUED", reason="cli_created")
    _print({"program_id": args.program_id, "status": st.status,
            "manifest_sha": st.obj.get("program_manifest_sha", "")})
    return 0


def _cmd_program_inspect(args) -> int:
    from .program import ResearchProgramState
    st = ResearchProgramState(root=Path(args.root) / args.program_id,
                              program_id=args.program_id,
                              objective={}, domain_scope="",
                              resource_policy={}, approval_policy={})
    _print({"status": st.status, "rounds": st.obj["rounds"],
            "budget_used": st.budget_used(),
            "integrity": st.verify_integrity()})
    return 0


def _cmd_method_list(args) -> int:
    from .harness_rsi.method_registry import MethodRegistry
    from .research_store.db import ResearchDatabase
    reg = MethodRegistry(db=ResearchDatabase(Path(args.db)),
                         root=Path(args.root) / "methods")
    _print({"methods": reg.list()})
    return 0


def _cmd_memory_search(args) -> int:
    from .discovery.memory import VerifiedResearchMemory
    from .discovery.knowledge import ClaimEvidenceGraph
    mem = VerifiedResearchMemory(
        Path(args.root) / "memory",
        graph=ClaimEvidenceGraph(Path(args.root) / "claims"),
        db=None, evidence_resolver=lambda digest: None)
    out = mem.retrieve(domain=args.domain or "", problem=args.query, top=args.top)
    _print({"memories": out})
    return 0


def _cmd_benchmark_coverage(args) -> int:
    from .capability_bench.external_benchmarks import coverage_matrix
    _print(coverage_matrix(base_dir=Path(args.root) / "external_benchmarks"))
    return 0


def _cmd_recovery_inspect(args) -> int:
    from .research_store.db import ResearchDatabase
    from .runtime_state import RuntimeState
    rt = RuntimeState(db=ResearchDatabase(Path(args.db)))
    _print({"classification": rt.classify(args.run_id),
            "reservations": rt.reservations(args.run_id),
            "journal": rt.verify_journal(args.run_id)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arr", description="ARR v1.0 operations")
    p.add_argument("--root", default="ARR_RUNTIME",
                   help="runtime root for program/memory/benchmark state")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("program", help="research program operations")
    ps = pr.add_subparsers(dest="action", required=True)
    pc = ps.add_parser("create")
    pc.add_argument("--program-id", required=True)
    pc.add_argument("--objective", required=True)
    pc.add_argument("--domain", default="embodied_world_model")
    pc.add_argument("--resource-policy", default="")
    pc.add_argument("--approval-policy", default="")
    pc.set_defaults(func=_cmd_program_create)
    pi = ps.add_parser("inspect")
    pi.add_argument("--program-id", required=True)
    pi.set_defaults(func=_cmd_program_inspect)

    me = sub.add_parser("memory", help="verified memory operations")
    ms = me.add_subparsers(dest="action", required=True)
    mse = ms.add_parser("search")
    mse.add_argument("--query", required=True)
    mse.add_argument("--domain", default="")
    mse.add_argument("--top", type=int, default=5)
    mse.set_defaults(func=_cmd_memory_search)

    me2 = sub.add_parser("method", help="research method registry")
    m2s = me2.add_subparsers(dest="action", required=True)
    ml = m2s.add_parser("list")
    ml.add_argument("--db", default="ARR_RUNTIME/methods.sqlite")
    ml.set_defaults(func=_cmd_method_list)

    bm = sub.add_parser("benchmark", help="benchmark operations")
    bs = bm.add_subparsers(dest="action", required=True)
    bc = bs.add_parser("coverage")
    bc.set_defaults(func=_cmd_benchmark_coverage)

    rc = sub.add_parser("recovery", help="recovery inspection")
    rcs = rc.add_subparsers(dest="action", required=True)
    ri = rcs.add_parser("inspect")
    ri.add_argument("--run-id", required=True)
    ri.add_argument("--db", default="ARR_RUNTIME/runtime.sqlite")
    ri.set_defaults(func=_cmd_recovery_inspect)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
