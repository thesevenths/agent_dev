"""JSON-only command line interface for the external Research Runtime.

Every subcommand prints a single JSON document to stdout. Errors are also JSON
(with a non-zero exit code), so the caller never has to scrape human prose.

    python -m research_api.cli submit --experiment-id exp1 --stage train \
        --config configs/pendulum.yaml --epochs 3
    python -m research_api.cli status --job-id <id>
    python -m research_api.cli wait   --job-id <id> --timeout 600
    python -m research_api.cli result --job-id <id>
    python -m research_api.cli list   --experiment-id exp1
    python -m research_api.cli freeze --config configs/pendulum.yaml \
        --out datasets/pendulum_v1 --splits train validation adapt test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import JobRunner, SUPPORTED_STAGES
from .settings import DATA_SPLITS, resolve_config_path


def _emit(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _parse_spec_json(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--spec-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--spec-json must be a JSON object")
    return value


def _parse_extra_args(text: str | None) -> list[str]:
    if not text:
        return []
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError("--extra-args must be a JSON array of strings")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SciNet++ research_api JSON CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="submit an async stage job")
    p_submit.add_argument("--experiment-id", required=True)
    p_submit.add_argument("--stage", required=True, choices=list(SUPPORTED_STAGES))
    p_submit.add_argument("--config", required=True)
    p_submit.add_argument("--run-id", default=None)
    p_submit.add_argument("--job-id", default=None)
    p_submit.add_argument("--epochs", type=int, default=None)
    p_submit.add_argument("--spec-json", default=None)
    p_submit.add_argument("--extra-args", default=None)
    p_submit.add_argument(
        "--split", default=None, choices=list(DATA_SPLITS),
        help="evaluation split (only forwarded to the evaluate stage). "
             "Omit to use the stage script's default (test).",
    )
    p_submit.add_argument("--db", default=None)

    p_status = sub.add_parser("status", help="query job status")
    p_status.add_argument("--job-id", required=True)
    p_status.add_argument("--db", default=None)

    p_result = sub.add_parser("result", help="read structured result.json")
    p_result.add_argument("--job-id", required=True)
    p_result.add_argument("--db", default=None)

    p_wait = sub.add_parser("wait", help="block until the job reaches a terminal state")
    p_wait.add_argument("--job-id", required=True)
    p_wait.add_argument("--timeout", type=float, default=3600.0)
    p_wait.add_argument("--poll", type=float, default=1.0)
    p_wait.add_argument("--db", default=None)

    p_list = sub.add_parser("list", help="list jobs")
    p_list.add_argument("--experiment-id", default=None)
    p_list.add_argument("--status", default=None)
    p_list.add_argument("--db", default=None)

    p_freeze = sub.add_parser("freeze", help="materialise an immutable dataset")
    p_freeze.add_argument("--config", required=True)
    p_freeze.add_argument("--out", required=True)
    p_freeze.add_argument("--splits", nargs="+",
                          default=["train", "validation", "adapt", "test"])
    p_freeze.add_argument("--db", default=None)

    return parser


def dispatch(args: argparse.Namespace) -> dict:
    if args.command == "submit":
        runner = JobRunner(db_path=args.db)
        spec = _parse_spec_json(args.spec_json)
        spec.setdefault("experiment_id", args.experiment_id)
        spec.setdefault("stage", args.stage)
        return runner.submit(
            experiment_id=args.experiment_id, stage=args.stage,
            config=args.config, run_id=args.run_id, epochs=args.epochs,
            job_id=args.job_id, spec=spec,
            extra_args=_parse_extra_args(args.extra_args),
            split=args.split,
        )

    if args.command == "freeze":
        # Imported lazily: this path needs the full SciNet++ env (torch/scipy),
        # whereas status/result/submit work in a lightweight interpreter.
        import yaml
        from data.frozen import build_frozen_dataset
        path = resolve_config_path(args.config)
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        manifest = build_frozen_dataset(
            cfg, args.out, splits=tuple(args.splits), config_path=str(path)
        )
        return {
            "dataset_version": manifest["dataset_version"],
            "out_dir": str(Path(args.out).resolve()),
            "config": str(path),
            "splits": {
                k: {"file": v["file"], "sha256": v["sha256"],
                    "samples": v["samples"]}
                for k, v in manifest["splits"].items()
            },
        }

    runner = JobRunner(db_path=args.db)
    if args.command == "status":
        job = runner.get_status(args.job_id)
        if job is None:
            return {"error": f"unknown job_id: {args.job_id}", "job_id": args.job_id}
        return job

    if args.command == "result":
        return runner.collect_result(args.job_id)

    if args.command == "wait":
        return runner.wait(args.job_id, timeout=args.timeout,
                           poll_interval=args.poll)

    if args.command == "list":
        return {"jobs": runner.list_jobs(experiment_id=args.experiment_id,
                                         status=args.status)}
    raise ValueError(f"unknown command: {args.command}")


def main() -> None:
    args = build_parser().parse_args()
    try:
        _emit(dispatch(args))
    except Exception as exc:  # noqa: BLE001 - CLI contract: errors are JSON too
        _emit({"error": str(exc), "error_type": type(exc).__name__,
               "command": args.command})
        sys.exit(1)


if __name__ == "__main__":
    main()
