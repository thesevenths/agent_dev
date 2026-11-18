"""ARR v1.0 WP-09/10 tests: operations CLI + reproducibility + security."""
from __future__ import annotations

import json

import pytest

from research_runtime.arr_cli import build_parser
from research_runtime.reproducibility import export


def test_cli_parses_all_real_commands():
    p = build_parser()
    for argv in (["program", "create", "--program-id", "P", "--objective", "q"],
                 ["program", "inspect", "--program-id", "P"],
                 ["memory", "search", "--query", "q"],
                 ["method", "list"],
                 ["benchmark", "coverage"],
                 ["recovery", "inspect", "--run-id", "r"]):
        args = p.parse_args(argv)
        assert hasattr(args, "func") and callable(args.func)


def test_cli_has_no_decorative_placeholder():
    import inspect
    import research_runtime.arr_cli as cli
    src = inspect.getsource(cli)
    assert "NOT_IMPLEMENTED" not in src
    # Every registered subcommand handler must be a real function in this module,
    # not a stub that returns a constant "not implemented" value.
    p = build_parser()
    import argparse
    handlers = []
    for action in p._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sp in action.choices.values():
                for a2 in sp._actions:
                    if isinstance(a2, argparse._SubParsersAction):
                        handlers.extend(
                            s2.get_default("func") for s2 in a2.choices.values())
    handlers = [h for h in handlers if h]
    assert handlers
    for h in handlers:
        body = inspect.getsource(h)
        assert "NOT_IMPLEMENTED" not in body
        assert "return 0" in body or "int(" in body  # real command returns


def test_reproducibility_export_and_verify(tmp_path):
    prog = tmp_path / "prog"
    prog.mkdir()
    (prog / "program_state.json").write_text(json.dumps({
        "program_id": "P", "status": "PROGRAM_COMPLETED",
        "program_manifest_sha": "a" * 64, "rounds": []}), encoding="utf-8")
    (prog / "program_report.json").write_text(json.dumps({
        "program_id": "P", "status": "PROGRAM_COMPLETED",
        "program_manifest_sha": "a" * 64, "verdicts": {}}), encoding="utf-8")
    out = export(program_dir=prog, out_dir=tmp_path / "bundle")
    assert out["ok"] and out["n_files"] == 2
    man = json.loads((tmp_path / "bundle" /
                      "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["verification"]["all_files_have_sha"] is True
    assert all(f["sha256"] for f in man["files"])


def test_reproducibility_fails_closed_on_missing(tmp_path):
    out = export(program_dir=tmp_path / "none", out_dir=tmp_path / "b")
    assert out["ok"] is False


def test_no_arbitrary_shell_from_cli_source():
    import inspect
    import research_runtime.arr_cli as cli
    src = inspect.getsource(cli)
    # No os.system / shell=True / subprocess in the operations CLI.
    assert "os.system" not in src
    assert "shell=True" not in src
    assert "subprocess" not in src
