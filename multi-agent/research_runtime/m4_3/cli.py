"""M4.3-02 -- unified CLI for the ARR A/B/C value validation runtime.

    python -m research_runtime.m4_3.cli preflight
    python -m research_runtime.m4_3.cli run --stage pilot --blocks 4
    python -m research_runtime.m4_3.cli analyze-pilot
    python -m research_runtime.m4_3.cli prepare-confirmatory --blocks 8 --seoi 0.5
    python -m research_runtime.m4_3.cli freeze-confirmatory
    python -m research_runtime.m4_3.cli approve-confirmatory --approved-by alice
    python -m research_runtime.m4_3.cli run --stage confirmatory
    python -m research_runtime.m4_3.cli report --stage pilot|final
    python -m research_runtime.m4_3.cli status | recover | export | verify-integrity
    python -m research_runtime.m4_3.cli review list|show|submit|status|unblind

The reviewer-facing commands print ONLY the blinded payload; the arm mapping
stays sealed until ``review unblind`` is allowed to run.

Nothing here fabricates an engine result, a review or an approval:
``run --stage pilot`` refuses without a real SciNet++ checkout and real
evidence, and ``run --stage confirmatory`` refuses without a frozen protocol
AND a human approval bound to its SHA.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "M4_3_runs"


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _root(args) -> Path:
    return Path(args.run_root).resolve()


def _paths(args, stage: str) -> dict[str, Path]:
    root = _root(args)
    base = root / stage
    base.mkdir(parents=True, exist_ok=True)
    return {"base": base, "db": base / "research.sqlite"}


def _db(args, stage: str):
    """Plain ARR SQLite handle (no engine) for the bookkeeping commands."""
    from ..research_store.db import ResearchDatabase
    return ResearchDatabase(_paths(args, stage)["db"])


def _service(args, stage: str):
    """Build the REAL SciNet++ service with the experiment's paired seeds.

    The domain policy is what decides a plan's paired seeds, so the seeds from
    the command line (the frozen protocol) MUST be injected here -- otherwise
    the campaign would silently run the domain default and the recorded paired
    seeds would not describe what actually ran.
    """
    from ..domains.scinet import scinet_research_policy
    from ..research_environment import build_default_research_service
    paths = _paths(args, stage)
    return build_default_research_service(
        "scinet", db_path=paths["db"],
        domain_policy=scinet_research_policy(
            paired_seeds=tuple(getattr(args, "seeds", (42, 43)))))


def _config(args, *, stage: str, experiment_id: str | None = None):
    from .pilot_runner import PilotConfig
    return PilotConfig(
        experiment_id=experiment_id or args.experiment_id,
        run_id=args.run_id, n_blocks=args.blocks,
        paired_seeds=tuple(args.seeds),
        candidate_budget=args.candidate_budget,
        human_time_budget_s=args.human_time_seconds,
        # The domain policy caps a plan at 2 candidates; the recorded budget
        # must describe what actually runs (no optimistic under-reporting).
        campaign_budget={"max_jobs": args.max_jobs, "max_seeds": len(args.seeds),
                         "max_candidates": 2},
        approver_id=args.approver,
        max_campaigns=args.max_campaigns,
        max_wall_time_s=args.max_seconds,
        require_review_complete=(stage == "confirmatory"
                                 or args.review == "complete"),
        source_experiment_id=args.source_experiment)


# --------------------------------------------------------------- commands
def cmd_preflight(args) -> int:
    from .pilot_runner import preflight
    service = None
    try:
        service = _service(args, "pilot")
    except Exception as exc:  # noqa: BLE001 - report, never crash
        _print({"service_error": f"{type(exc).__name__}: {exc}"})
    cfg = _config(args, stage="pilot")
    rep = preflight(cfg, service=service)
    _print(rep.to_dict())
    return 0 if rep.ready else 2


def cmd_run(args) -> int:
    from .experiment_controller import ExperimentController, BudgetGuard
    from .pilot_runner import PilotRunner, PreflightError, ReviewIncomplete
    from .review_queue import ReviewQueue

    stage = args.stage
    paths = _paths(args, stage)
    service = _service(args, stage)
    cfg = _config(args, stage=stage)
    queue = ReviewQueue(db=service.stores.db)  # same SQLite as the service

    if stage == "confirmatory":
        from .confirmatory_protocol_builder import (
            ProtocolBlocked, ProtocolNotApproved, load_frozen_protocol,
        )
        from .confirmatory_runner import ConfirmatoryRunner
        protocol_dir = _root(args) / "confirmatory" / "protocol"
        # ONE lifecycle record (pilot db) carries the human gates; the
        # confirmatory REGISTRY (and its per-block bookkeeping) is separate.
        from .confirmatory_protocol_builder import load_approval
        lifecycle = ExperimentController.load(cfg.experiment_id,
                                              db=_db(args, "pilot"))
        protocol = load_frozen_protocol(protocol_dir)
        approval = load_approval(protocol_sha256=protocol.freeze_sha256(),
                                 out_dir=protocol_dir)
        if lifecycle.state == "CONFIRMATORY_AWAITING_APPROVAL":
            lifecycle.start_confirmatory(
                approval_reference=(f"{approval['approved_by']}@"
                                    f"{approval['approved_at']}"),
                detail={"protocol_sha256": protocol.freeze_sha256()})
        controller_id = f"{cfg.experiment_id}::confirmatory"
        controller = ExperimentController.load(controller_id, db=_db(args, stage)) \
            if _experiment_exists(_db(args, stage), controller_id) else None
        if controller is None:
            controller = ExperimentController.create(
                experiment_id=controller_id, db=_db(args, stage),
                config=cfg.to_dict(),
                budget=BudgetGuard(max_campaigns=cfg.max_campaigns,
                                   max_wall_time_s=cfg.max_wall_time_s))
            controller.preflight(detail={"stage": stage})
            controller.freeze_snapshots(list(protocol.snapshot_ids))
        try:
            runner = ConfirmatoryRunner.from_frozen(
                service=service, config=cfg, out_dir=paths["base"],
                controller=controller, protocol_dir=protocol_dir,
                review_queue=queue, lifecycle=lifecycle)
        except (ProtocolBlocked, ProtocolNotApproved) as exc:
            _print({"ok": False, "refused": type(exc).__name__, "error": str(exc)})
            return 3
    else:
        existing = ExperimentController.load(cfg.experiment_id,
                                             db=service.stores.db) \
            if _experiment_exists(service.stores.db, cfg.experiment_id) else None
        if existing is not None and existing.is_terminal:
            _print({"ok": False, "state": existing.state,
                    "error": "experiment is terminal; nothing to resume"})
            return 3
        controller = existing or ExperimentController.create(
            experiment_id=cfg.experiment_id, db=service.stores.db,
            config=cfg.to_dict(),
            budget=BudgetGuard(max_campaigns=cfg.max_campaigns,
                               max_wall_time_s=cfg.max_wall_time_s))
        if existing is None:
            controller.preflight(detail={"stage": stage})
            ids = _snapshot_ids(service, cfg)
            controller.freeze_snapshots(ids)
        runner = PilotRunner(service=service, config=cfg,
                             out_dir=paths["base"], controller=controller,
                             review_queue=queue)
    try:
        result = runner.run()
    except (PreflightError, ReviewIncomplete) as exc:
        _print({"ok": False, "refused": type(exc).__name__, "error": str(exc),
                "controller_state": controller.state})
        return 3
    _print({"ok": True, "stage": stage, "state": controller.state,
            "n_runs": result.get("n_runs"),
            "elapsed_s": result.get("elapsed_s"),
            "blocks": [b["block_id"] for b in result.get("blocks", [])],
            "outcomes": {b["block_id"]: b["outcomes"]
                         for b in result.get("blocks", [])}})
    return 0


def _experiment_exists(db: Any, experiment_id: str) -> bool:
    """Does this experiment already exist? (creates the additive schema)."""
    from .experiment_state import ExperimentEventStore
    store = ExperimentEventStore(db=db)      # idempotent CREATE TABLE IF NOT EXISTS
    return store.load(experiment_id) is not None


def _advance_to_pilot_completed(controller: Any, pilot: dict[str, Any]) -> None:
    """Move the lifecycle to PILOT_COMPLETED exactly once (idempotent)."""
    state = controller.state
    if state == "BLOCK_COMPLETED":
        controller.analyze_pilot(detail={
            "n_blocks": pilot.get("n_blocks"),
            "conclusion": (pilot.get("conclusion") or {}).get("verdict")})
        state = controller.state
    if state == "PILOT_ANALYZING":
        controller.pilot_completed(detail={
            "sample_size_status": (pilot.get("sample_size") or {}).get("status")})


def _snapshot_ids(service, cfg) -> list[str]:
    return [f"{cfg.run_id}::block_{i}" for i in range(1, cfg.n_blocks + 1)]


def cmd_analyze_pilot(args) -> int:
    from .pilot_analysis import analyze_pilot
    from .value_report import generate_value_report

    paths = _paths(args, "pilot")
    from ..research_store.db import ResearchDatabase
    db = ResearchDatabase(paths["db"])
    analysis = analyze_pilot(db=db, out_dir=_root(args) / "pilot",
                             experiment_id=args.experiment_id,
                             alpha=args.alpha,
                             seoi_quality=args.seoi,
                             write=True)
    generate_value_report(db=db, out_dir=_root(args) / "pilot",
                          experiment_id=args.experiment_id, stage="pilot",
                          pilot_analysis=analysis, write=True)
    _print({"ok": True,
            "n_blocks": analysis["n_blocks"],
            "n_runs": analysis["n_runs"],
            "conclusion": analysis["conclusion"]["verdict"],
            "sample_size_status": analysis["sample_size"]["status"],
            "n_blocks_recommended":
                analysis["sample_size"]["n_blocks_recommended"],
            "outputs": ["pilot_analysis.json", "pilot_report.md",
                        "sample_size_recommendation.json",
                        "pilot_value_report.json",
                        "M4_3_PILOT_VALUE_REPORT.md"]})
    return 0


def cmd_prepare_confirmatory(args) -> int:
    from .confirmatory_protocol_builder import (
        ProtocolBlocked, build_confirmatory_protocol,
    )
    from .experiment_controller import ExperimentController

    pilot = json.loads((_root(args) / "pilot" / "pilot_analysis.json")
                       .read_text(encoding="utf-8"))
    pilot_db = _db(args, "pilot")
    if not _experiment_exists(pilot_db, args.experiment_id):
        _print({"ok": False,
                "error": ("no pilot experiment record for "
                          f"{args.experiment_id!r}; run the pilot first")})
        return 3
    controller = ExperimentController.load(args.experiment_id, db=pilot_db)
    _advance_to_pilot_completed(controller, pilot)
    try:
        protocol = build_confirmatory_protocol(
            pilot_analysis=pilot,
            snapshot_ids=pilot["data_integrity"].get("snapshot_ids")
            or [b["snapshot_id"] for b in pilot["blocks"]],
            paired_seeds=list(args.seeds),
            sample_size=args.blocks,
            campaign_budget={"max_jobs": args.max_jobs,
                             "max_seeds": len(args.seeds),
                             "max_candidates": 1},
            model_version=pilot["metric_contract"].get("contract_sha256", "")[:0]
            or "scinet-pendulum-smoke-v1",
            prompt_version="objective-candidates-v1",
            ranker_version=pilot.get("ranker_version", "deterministic-v1"),
            domain_policy_version="scinet-domain-v1",
            smallest_effect_of_interest=args.seoi,
            alpha=args.alpha, target_power=args.power,
            correction=args.correction)
    except ProtocolBlocked as exc:
        draft = {"schema": "m4_3_confirmatory_draft_v1",
                 "status": "blocked",
                 "blocked_reason": str(exc),
                 "pilot_sample_size": pilot.get("sample_size", {}),
                 "requested_blocks": args.blocks,
                 "requested_seoi": args.seoi}
        out = _root(args) / "confirmatory" / "protocol"
        out.mkdir(parents=True, exist_ok=True)
        (out / "confirmatory_draft.json").write_text(
            json.dumps(draft, indent=2, sort_keys=True), encoding="utf-8")
        _print({"ok": False, "status": "blocked", "error": str(exc),
                "draft": str(out / "confirmatory_draft.json")})
        return 3
    controller.prepare_confirmatory(protocol_draft=protocol.frozen_dict())
    _print({"ok": True, "protocol_sha256": protocol.freeze_sha256(),
            "sample_size": protocol.sample_size,
            "note": "draft prepared; freeze it explicitly next"})
    return 0


def cmd_freeze_confirmatory(args) -> int:
    from .confirmatory_protocol_builder import (
        ProtocolBlocked, build_confirmatory_protocol,
        freeze_confirmatory_protocol, request_confirmatory_approval,
    )
    from .experiment_controller import ExperimentController

    pilot_path = _root(args) / "pilot" / "pilot_analysis.json"
    if not pilot_path.is_file():
        _print({"ok": False,
                "error": "no pilot analysis; run analyze-pilot first"})
        return 3
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    controller = ExperimentController.load(args.experiment_id,
                                           db=_db(args, "pilot"))
    try:
        protocol = build_confirmatory_protocol(
            pilot_analysis=pilot,
            snapshot_ids=list(controller.record.config.get("snapshot_ids")
                              or []),
            paired_seeds=list(args.seeds), sample_size=args.blocks,
            campaign_budget={"max_jobs": args.max_jobs,
                             "max_seeds": len(args.seeds),
                             "max_candidates": 1},
            model_version="scinet-pendulum-smoke-v1",
            prompt_version="objective-candidates-v1",
            ranker_version=pilot.get("ranker_version", "deterministic-v1"),
            domain_policy_version="scinet-domain-v1",
            smallest_effect_of_interest=args.seoi,
            alpha=args.alpha, target_power=args.power,
            correction=args.correction)
    except ProtocolBlocked as exc:
        _print({"ok": False, "refused": "ProtocolBlocked", "error": str(exc)})
        return 3
    out = _root(args) / "confirmatory" / "protocol"
    frozen = freeze_confirmatory_protocol(protocol, out_dir=out)
    controller.freeze_confirmatory(protocol_sha256=frozen["protocol_sha256"])
    controller.request_confirmatory_approval()
    request_confirmatory_approval(protocol_sha256=frozen["protocol_sha256"],
                                  out_dir=out)
    _print({"ok": True, "protocol_sha256": frozen["protocol_sha256"],
            "version": frozen["current_version"],
            "next": "approve-confirmatory --approved-by <human>"})
    return 0


def cmd_approve_confirmatory(args) -> int:
    from .confirmatory_protocol_builder import (
        ProtocolNotApproved, approve_confirmatory_protocol,
        load_frozen_protocol,
    )
    from .experiment_controller import ExperimentController

    out = _root(args) / "confirmatory" / "protocol"
    try:
        protocol = load_frozen_protocol(out)
        approval = approve_confirmatory_protocol(
            protocol_sha256=protocol.freeze_sha256(),
            approved_by=args.approved_by, out_dir=out, notes=args.notes)
    except ProtocolNotApproved as exc:
        _print({"ok": False, "error": str(exc)})
        return 3
    controller = ExperimentController.load(args.experiment_id,
                                           db=_db(args, "pilot"))
    _print({"ok": True, "approval": approval,
            "controller_state": controller.state})
    return 0


def cmd_report(args) -> int:
    from ..research_store.db import ResearchDatabase
    from .value_report import generate_value_report

    stage = args.stage
    if stage == "pilot":
        db = ResearchDatabase(_paths(args, "pilot")["db"])
        out = _root(args) / "pilot"
        pilot = None
        pilot_path = out / "pilot_analysis.json"
        if pilot_path.is_file():
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        report = generate_value_report(db=db, out_dir=out,
                                       experiment_id=args.experiment_id,
                                       stage="pilot", pilot_analysis=pilot)
    else:
        db = ResearchDatabase(_paths(args, "confirmatory")["db"])
        out = _root(args) / "final"
        protocol = None
        proto_path = (_root(args) / "confirmatory" / "protocol"
                      / "confirmatory_protocol.json")
        if proto_path.is_file():
            protocol = json.loads(proto_path.read_text(encoding="utf-8"))
        report = generate_value_report(
            db=db, out_dir=out, experiment_id=args.experiment_id,
            stage="final",
            protocol={"protocol_sha256": (protocol or {}).get(
                "protocol_sha256", "")})
    _print({"ok": True, "stage": stage,
            "conclusion": report["conclusion"]["outcome"],
            "significance_claimed": report["conclusion"]["significance_claimed"],
            "out_dir": str(out)})
    return 0


def cmd_status(args) -> int:
    from .experiment_controller import ExperimentController
    from .review_queue import ReviewQueue

    db = _db(args, args.stage)
    out: dict[str, Any] = {"stage": args.stage, "db": str(db.db_path)}
    try:
        controller = ExperimentController.load(args.experiment_id, db=db)
        out["experiment"] = controller.status()
    except Exception as exc:  # noqa: BLE001
        out["experiment_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .registry_store import RegistryStore
        store = RegistryStore(db=db)
        out["registry"] = {"n_blocks": len(store.load_blocks()),
                           "n_runs": len(store.load_runs()),
                           "n_reviews": len(store.load_reviews()),
                           "integrity": store.verify_integrity()}
    except Exception as exc:  # noqa: BLE001
        out["registry_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["reviews"] = ReviewQueue(db=db).queue_status(
            args.experiment_id)
    except Exception as exc:  # noqa: BLE001
        out["reviews_error"] = f"{type(exc).__name__}: {exc}"
    _print(out)
    return 0


def cmd_recover(args) -> int:
    from .experiment_controller import ExperimentController
    from .experiment_recovery import recover_experiment

    controller = ExperimentController.load(args.experiment_id,
                                           db=_db(args, args.stage))
    report = recover_experiment(controller, apply=not args.dry_run)
    _print({"ok": True, "dry_run": bool(args.dry_run),
            "report": report.to_dict()})
    return 0


def cmd_export(args) -> int:
    from .registry_store import RegistryStore

    store = RegistryStore(db=_db(args, args.stage))
    bundle = store.export()
    if getattr(args, "with_evidence", False):
        from ..settings import DEFAULT_SCINET_DIR
        from .evidence_bundle import build_evidence_bundle
        evidence_dir = (_root(args) / args.stage / "evidence")
        ev = build_evidence_bundle(
            db=_db(args, args.stage), out_dir=evidence_dir,
            experiment_id=args.experiment_id,
            campaign_root=Path(DEFAULT_SCINET_DIR)
            / "outputs" / "research_campaigns",
            job_log_dir=Path(DEFAULT_SCINET_DIR) / "outputs" / "agent_jobs")
        _print({"ok": True, "evidence_dir": str(evidence_dir),
                "registry_integrity": ev["registry_integrity"],
                "n_runs": ev["n_runs"],
                "copied_engine_logs": len(ev["copied_engine_logs"]),
                "missing_artefacts": ev["missing_artefacts"]})
        return 0
    if args.out:
        Path(args.out).write_text(
            json.dumps(bundle, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        _print({"ok": True, "out": str(args.out),
                "export_sha256": bundle.get("export_sha256")})
    else:
        _print({"export_sha256": bundle.get("export_sha256"),
                "n_blocks": len(bundle.get("blocks", [])),
                "n_runs": len(bundle.get("runs", [])),
                "n_events": len(bundle.get("events", []))})
    return 0


def cmd_verify_integrity(args) -> int:
    from .registry_store import RegistryStore

    store = RegistryStore(db=_db(args, args.stage))
    ok = store.verify_integrity()
    export = store.export()
    _print({"integrity": ok,
            "export_sha256": export.get("export_sha256"),
            "n_events": len(export.get("events", []))})
    return 0 if ok else 4


# ------------------------------------------------------------ review cmds
def cmd_review(args) -> int:
    from .review_queue import ReviewQueue, ReviewQueueViolation

    queue = ReviewQueue(db=_db(args, args.stage),
                        required_human_reviews=args.required_reviews)
    try:
        if args.review_cmd == "list":
            _print({"pending": queue.list_pending(args.experiment_id)})
        elif args.review_cmd == "show":
            _print(queue.show(args.review_id))
        elif args.review_cmd == "submit":
            scores = json.loads(args.scores)
            review = queue.submit(review_id=args.review_id,
                                 reviewer_id=args.reviewer,
                                 scores=scores, reason=args.reason,
                                 review_kind=args.kind)
            _print({"ok": True, "review": review,
                    "completeness": queue.completeness(args.review_id)})
        elif args.review_cmd == "status":
            _print(queue.queue_status(args.experiment_id))
        elif args.review_cmd == "unblind":
            _print(queue.unblind(args.experiment_id))
        else:  # pragma: no cover - argparse enforces the choices
            raise SystemExit(2)
    except ReviewQueueViolation as exc:
        _print({"ok": False, "error": str(exc)})
        return 3
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="research_runtime.m4_3.cli",
        description="ARR A/B/C value-validation runtime (M4.3-02)")
    p.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    p.add_argument("--experiment-id", default="m4_3_pilot")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, *, stage_default: str | None = None) -> None:
        sp.add_argument("--run-id", default="arr_m34_001")
        sp.add_argument("--blocks", type=int, default=4)
        sp.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
        sp.add_argument("--candidate-budget", type=int, default=3)
        sp.add_argument("--human-seconds", dest="human_time_seconds",
                        type=float, default=900.0)
        sp.add_argument("--max-jobs", type=int, default=8)
        sp.add_argument("--approver", default="operator")
        sp.add_argument("--max-campaigns", type=int, default=0)
        sp.add_argument("--max-seconds", type=float, default=0.0)
        sp.add_argument("--review", choices=("deferred", "complete"),
                        default="deferred")
        sp.add_argument("--source-experiment", default="")
        if stage_default:
            sp.add_argument("--stage", choices=("pilot", "confirmatory"),
                            default=stage_default)

    sp = sub.add_parser("preflight", help="check the real environment")
    common(sp, stage_default="pilot")
    sp.set_defaults(func=cmd_preflight)

    sp = sub.add_parser("run", help="execute a stage")
    common(sp, stage_default="pilot")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("analyze-pilot", help="pilot analysis + sample size")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--seoi", type=float, default=0.5)
    sp.set_defaults(func=cmd_analyze_pilot)

    for name, func in (("prepare-confirmatory", cmd_prepare_confirmatory),
                       ("freeze-confirmatory", cmd_freeze_confirmatory)):
        sp = sub.add_parser(name, help=name.replace("-", " "))
        sp.add_argument("--blocks", type=int, required=True)
        sp.add_argument("--seoi", type=float, required=True)
        sp.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
        sp.add_argument("--max-jobs", type=int, default=8)
        sp.add_argument("--alpha", type=float, default=0.05)
        sp.add_argument("--power", type=float, default=0.8)
        sp.add_argument("--correction", default="holm")
        sp.set_defaults(func=func)

    sp = sub.add_parser("approve-confirmatory",
                        help="record a HUMAN approval of the frozen protocol")
    sp.add_argument("--approved-by", required=True)
    sp.add_argument("--notes", default="")
    sp.set_defaults(func=cmd_approve_confirmatory)

    sp = sub.add_parser("report", help="generate the value report")
    sp.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("status", help="experiment + registry + review status")
    sp.add_argument("--stage", choices=("pilot", "confirmatory"),
                    default="pilot")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("recover", help="reconcile and resume after a crash")
    sp.add_argument("--stage", choices=("pilot", "confirmatory"),
                    default="pilot")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_recover)

    sp = sub.add_parser("export", help="export the registry (with SHA)")
    sp.add_argument("--stage", choices=("pilot", "confirmatory"),
                    default="pilot")
    sp.add_argument("--out", default="")
    sp.add_argument("--with-evidence", action="store_true",
                    help="also copy real engine logs + campaign artefacts")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("verify-integrity", help="verify the event hash chain")
    sp.add_argument("--stage", choices=("pilot", "confirmatory"),
                    default="pilot")
    sp.set_defaults(func=cmd_verify_integrity)

    sp = sub.add_parser("review", help="human blind-review console")
    rsub = sp.add_subparsers(dest="review_cmd", required=True)
    sp.add_argument("--stage", choices=("pilot", "confirmatory"),
                    default="pilot")
    sp.add_argument("--required-reviews", type=int, default=2)
    rl = rsub.add_parser("list", help="list reviews still pending")
    rl.set_defaults(func=cmd_review)
    rs = rsub.add_parser("show", help="show the blinded objective")
    rs.add_argument("review_id")
    rs.set_defaults(func=cmd_review)
    rsubp = rsub.add_parser("submit", help="submit one review")
    rsubp.add_argument("review_id")
    rsubp.add_argument("--reviewer", required=True)
    rsubp.add_argument("--scores", required=True,
                       help='JSON, e.g. {"evidence_grounding": 3, ...}')
    rsubp.add_argument("--reason", required=True)
    rsubp.add_argument("--kind", choices=("human_blind", "automated"),
                       default="human_blind")
    rsubp.set_defaults(func=cmd_review)
    rst = rsub.add_parser("status", help="completeness per queued review")
    rst.set_defaults(func=cmd_review)
    ru = rsub.add_parser("unblind", help="release the sealed arm mapping")
    ru.set_defaults(func=cmd_review)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run one command. Domain refusals become exit code 3, never a traceback."""
    from .confirmatory_protocol_builder import ProtocolBlocked, ProtocolNotApproved
    from .experiment_state import IllegalTransition
    from .pilot_runner import PreflightError, ReviewIncomplete
    from .review_queue import ReviewQueueViolation

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ProtocolBlocked, ProtocolNotApproved, IllegalTransition,
            ReviewQueueViolation, PreflightError, ReviewIncomplete) as exc:
        _print({"ok": False, "refused": type(exc).__name__, "error": str(exc)})
        return 3
    except ValueError as exc:
        _print({"ok": False, "error": str(exc)})
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main", "DEFAULT_RUN_ROOT"]
