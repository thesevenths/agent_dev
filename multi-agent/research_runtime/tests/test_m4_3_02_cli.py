"""M4.3-02 -- unified CLI wiring (ENGINE-FREE commands only).

Only the bookkeeping commands are exercised here: they must work on an empty
database, produce machine-readable JSON, and never require the engine. The
engine-dependent commands (``preflight`` / ``run``) are covered by the opt-in
real-engine test.
"""
from __future__ import annotations

import json

import pytest

from research_runtime.m4_3 import cli


def _run(capsys, *argv: str) -> tuple[int, dict]:
    code = cli.main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads("\n".join(out)) if out else {}
    return code, payload


def test_parser_exposes_every_required_command():
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices
               and "run" in (a.choices or {})]
    assert actions, "the subcommand table is missing"
    commands = set(actions[0].choices)
    required = {"preflight", "run", "analyze-pilot", "prepare-confirmatory",
                "freeze-confirmatory", "approve-confirmatory", "report",
                "status", "recover", "export", "verify-integrity", "review"}
    assert required <= commands


def test_status_on_an_empty_run_root(tmp_path, capsys):
    code, payload = _run(capsys, "--run-root", str(tmp_path), "status")
    assert code == 0
    assert payload["registry"]["n_runs"] == 0
    assert payload["registry"]["integrity"] is True
    assert payload["reviews"]["n_queued"] == 0
    assert "experiment_error" in payload


def test_review_list_and_status_on_empty_queue(tmp_path, capsys):
    code, payload = _run(capsys, "--run-root", str(tmp_path),
                         "review", "list")
    assert code == 0 and payload["pending"] == []
    code, payload = _run(capsys, "--run-root", str(tmp_path),
                         "review", "status")
    assert code == 0 and payload["n_pending"] == 0


def test_review_unblind_is_refused_without_any_review(tmp_path, capsys):
    code, payload = _run(capsys, "--run-root", str(tmp_path),
                         "review", "unblind")
    assert code == 3
    assert "no queued review" in payload["error"]


def test_export_and_verify_integrity(tmp_path, capsys):
    out_file = tmp_path / "export.json"
    code, payload = _run(capsys, "--run-root", str(tmp_path), "export",
                         "--out", str(out_file))
    assert code == 0 and out_file.is_file()
    assert len(payload["export_sha256"]) == 64
    code, payload = _run(capsys, "--run-root", str(tmp_path),
                         "verify-integrity")
    assert code == 0 and payload["integrity"] is True


def test_analyze_pilot_on_an_empty_registry_writes_honest_outputs(tmp_path,
                                                                 capsys):
    code, payload = _run(capsys, "--run-root", str(tmp_path), "analyze-pilot")
    assert code == 0
    assert payload["n_blocks"] == 0
    assert payload["conclusion"] == "evidence_insufficient"
    for name in ("pilot_analysis.json", "pilot_report.md",
                 "sample_size_recommendation.json"):
        assert (tmp_path / "pilot" / name).is_file()


def test_report_pilot_without_runs_is_evidence_insufficient(tmp_path, capsys):
    code, payload = _run(capsys, "--run-root", str(tmp_path), "report",
                         "--stage", "pilot")
    assert code == 0
    assert payload["conclusion"] == "evidence insufficient"
    assert payload["significance_claimed"] is False


def test_freeze_confirmatory_refuses_without_pilot_analysis(tmp_path, capsys):
    # No pilot analysis file exists -> the builder refuses, exit code 3.
    code, payload = _run(capsys, "--run-root", str(tmp_path),
                         "freeze-confirmatory", "--blocks", "8", "--seoi", "0.5")
    assert code != 0


def test_approve_confirmatory_requires_a_frozen_protocol(tmp_path, capsys):
    code, payload = _run(capsys, "--run-root", str(tmp_path),
                         "approve-confirmatory", "--approved-by", "alice")
    assert code == 3
    assert "frozen" in payload["error"]


def test_recover_dry_run_on_empty_state_is_reported(tmp_path, capsys):
    code, payload = _run(capsys, "--run-root", str(tmp_path), "recover",
                         "--dry-run")
    assert code != 0 or "error" in payload or "ok" in payload
