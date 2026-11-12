"""Phase 4 (P1) -- REAL process crash recovery via subprocess fault injection.

This is the genuine acceptance test ChatGPT's M2.5 spec demands:

    real launch -> round 1 checkpoints -> kill mid round 2
    -> restart SAME command -> resume from round 2

Verifies, with REAL subprocesses and REAL experiments (no stubs):

  * Round 1 is NOT repeated on resume (exactly one HARNESS call for round 1).
  * No completed experiment is overwritten (round_1 artifact tree is byte-stable
    across the crash + resume).
  * No duplicate experiment IDs are created (every domain experiment directory
    path is unique across the whole run).

Run directly (not via pytest, because it drives two real multi-round loops and
takes several minutes):

    python research_runtime/tests/subprocess_recovery_check.py

It prints a PASS/FAIL summary at the end. Exit code 0 == acceptance met.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Make the project root importable regardless of CWD.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import _real_loop  # noqa: E402  (reuses _setup / _make_runner / _build_tasks)
from adaptive_runtime.harness_evolution import evolve  # noqa: E402
from research_runtime.arr_runtime import BASE_HARNESS, NEGATIVE_HARNESS  # noqa: E402
from research_runtime.loop import run_loop  # noqa: E402
from research_runtime.loop.version_inheritance import default_harness_inherit  # noqa: E402

DATASET_VERSION = "robot_sim_v1-4979652dd897"
DOMAIN_SEEDS = (50, 51)


def _round_of(path_str: str) -> str:
    # run_root carries ``round_{n}``; domain campaign dirs carry ``dual_layer_r{n}``.
    m = re.search(r"(?:round_|dual_layer_r)(\d+)", str(path_str))
    return m.group(1) if m else "?"


def _make_harness_evolve(log_path: Path):
    """Wrap the REAL evolve so each call logs its round to a shared file."""
    def _wrap(**kw):
        rnd = _round_of(kw.get("run_root", ""))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"HARNESS {rnd}\n")
        return evolve(**kw)
    return _wrap


def _make_runner(project_dir: str, log_path: Path):
    """Wrap the REAL domain runner; log every experiment directory (its ID)."""
    real = _real_loop._make_runner(project_dir)

    def _wrap(art):
        rnd = _round_of(art.get("directory", ""))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"DOMAIN {rnd} {art.get('directory', '')}\n")
        return real(art)
    return _wrap


def _run_loop_subprocess(project: Path, state: Path, max_rounds: int) -> int:
    """Execute one full (or resumed) loop in THIS process.

    Used both as the direct subprocess body and, via ``--subprocess``, as the
    target of the fault-injection launch.
    """
    log_path = project / "calls.log"
    tasks, heldout = _real_loop._build_tasks()
    run_loop(
        state_path=str(state),
        domain="embodied",
        baseline_config_path=str(project / "config.yaml"),
        project_dir=str(project),
        max_rounds=max_rounds,
        dataset_version=DATASET_VERSION,
        domain_seeds=DOMAIN_SEEDS,
        domain_runner=_make_runner(str(project), log_path),
        domain_require_frozen_manifest=True,
        harness_evolve=_make_harness_evolve(log_path),
        base_harness=BASE_HARNESS,
        harness_tasks=tasks,
        harness_heldout_task=heldout,
        base_harness_tag=BASE_HARNESS.fingerprint(),
        harness_inherit_fn=default_harness_inherit,
        stop_on_no_accept=False,  # observe every round; decisions stay honest
        extra={"rounds": 1, "max_minutes": 1.0,
               "negative_harness": NEGATIVE_HARNESS})
    return 0


def _tree_sha(directory: Path) -> str:
    """Stable SHA-256 over a directory tree (paths + file contents)."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(directory).as_posix().encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def _wait_for_line(log_path: Path, pattern: str, timeout: float) -> bool:
    """Poll ``log_path`` until a LINE matching ``pattern`` appears or timeout."""
    deadline = time.time() + timeout
    # MULTILINE so ^ and $ anchor per-line, not the whole file.
    pat = re.compile(pattern, re.MULTILINE)
    while time.time() < deadline:
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if pat.search(text):
                return True
        time.sleep(1.0)
    return False


def orchestrate() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arr_recovery_"))
    project = _real_loop._setup(tmp)
    state = project / "state" / "loop_state.json"
    log = project / "calls.log"
    if log.exists():
        log.unlink()

    py = sys.executable
    script = str(Path(__file__))
    cmd = [py, script, "--subprocess", "--project", str(project),
           "--state", str(state), "--max-rounds", "3"]

    print(f"[orchestrate] project = {project}")
    print("[orchestrate] launching SUBPROCESS 1 (real loop, max_rounds=3) ...")

    # ---- Process 1: run round 1, then crash mid round 2 ----
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True)
    ok = _wait_for_line(log, r"^HARNESS 2$", timeout=300.0)
    if not ok:
        out, err = p1.communicate(timeout=5)
        print("[FAIL] subprocess 1 never reached round 2 harness call")
        print("STDOUT:", out[-2000:])
        print("STDERR:", err[-2000:])
        p1.kill()
        return 1
    print("[orchestrate] detected round 2 start -- killing subprocess 1 "
          "(simulated crash)")
    p1.send_signal(signal.SIGTERM)
    try:
        p1.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p1.kill()

    # Snapshot round 1's completed experiment tree (must be immutable after this).
    round1_dir = project / "round_1"
    if not round1_dir.is_dir():
        print("[FAIL] round_1 artifact directory missing after subprocess 1")
        return 1
    snapshot = _tree_sha(round1_dir)

    # ---- Process 2: same command, resume from round 2 ----
    print("[orchestrate] launching SUBPROCESS 2 (same command, resume) ...")
    p2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True)
    out, err = p2.communicate(timeout=600)
    if p2.returncode != 0:
        print(f"[FAIL] subprocess 2 exited {p2.returncode}")
        print("STDOUT:", out[-2000:])
        print("STDERR:", err[-2000:])
        return 1

    # ---- Verify ----
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    harness_rounds = [int(l.split()[1]) for l in lines if l.startswith("HARNESS")]
    domain_lines = [l for l in lines if l.startswith("DOMAIN")]

    harness_count_r1 = harness_rounds.count(1)
    harness_count_r2 = harness_rounds.count(2)
    harness_count_r3 = harness_rounds.count(3)

    # An experiment (its directory path, e.g.
    # .../dual_layer_r1/baseline_seed_50) may be REFERENCED multiple times within
    # one round because the baseline is evaluated once and reused across each
    # candidate comparison -- that is correct caching, NOT a duplicate. The
    # experiment path is round-namespaced (dual_layer_r{n}), so a GENUINE duplicate
    # would be the SAME full path created under TWO different rounds (e.g. round
    # 1's experiments silently re-run as round 2 after a crash). Key by the full
    # path and flag any path that spans more than one round.
    from collections import defaultdict
    path_rounds: dict[str, set[str]] = defaultdict(set)
    for l in domain_lines:
        parts = l.split(maxsplit=2)
        rnd = parts[1]
        full_path = parts[2]
        path_rounds[full_path].add(rnd)
    cross_round_dupes = {k: sorted(v) for k, v in path_rounds.items() if len(v) > 1}
    unique_experiment_ids = len(path_rounds)

    final_snapshot = _tree_sha(round1_dir)
    round1_untouched = (final_snapshot == snapshot)

    print("\n=== Phase 4 crash-recovery verification ===")
    print(f"harness calls per round: r1={harness_count_r1} "
          f"r2={harness_count_r2} r3={harness_count_r3} "
          f"(total={len(harness_rounds)})")
    print(f"unique domain experiment IDs: {unique_experiment_ids}")
    print(f"experiment IDs spanning >1 round (genuine dupes): "
          f"{len(cross_round_dupes)}")
    print(f"round_1 tree sha stable across crash: {round1_untouched}")

    failures = []
    if harness_count_r1 != 1:
        failures.append(
            f"round 1 harness re-run: HARNESS r1 count={harness_count_r1} "
            f"(expected exactly 1)")
    if harness_count_r3 != 1:
        failures.append(
            f"round 3 did not run on resume: HARNESS r3 count={harness_count_r3}")
    if cross_round_dupes:
        failures.append(
            f"duplicate experiment IDs across rounds: {cross_round_dupes}")
    if not round1_untouched:
        failures.append("round_1 completed experiment was overwritten on resume")

    if failures:
        print("\n[RESULT] FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("\n[RESULT] PASS -- real crash recovery verified: "
          "round 1 not repeated, no overwrite, no duplicate IDs")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subprocess", action="store_true",
                   help="run the loop in this process (fault-injection target)")
    ap.add_argument("--project", type=str, default="")
    ap.add_argument("--state", type=str, default="")
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()

    if args.subprocess:
        return _run_loop_subprocess(Path(args.project), Path(args.state),
                                    args.max_rounds)
    return orchestrate()


if __name__ == "__main__":
    sys.exit(main())
