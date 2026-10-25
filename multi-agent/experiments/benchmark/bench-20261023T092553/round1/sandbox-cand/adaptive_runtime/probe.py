"""Probe：沙箱隔离 + 子进程执行 + 两级 Racing + 预算记账。

为什么是「沙箱复制」而不是 git worktree
---------------------------------------
本仓库在 ``E:/agent_dev`` 这个 git 仓库的**子目录**里，``git worktree add`` 会把整个
agent_dev 物化出来（含其它项目）。P0/P1 的候选只改 env（不改代码），需要的是
「独立的 checkpoint DB / log / tmp / documents」，复制一个几百 KB 的干净副本更合适。
需要严格分支语义时设 ``ADAPTIVE_USE_GIT_WORKTREE=1``。

两级 Racing
-----------
Stage 1：4 个任务 × 2 次 = 8 对，只做**止损**（n=2 没有判定力，不录取）；
Stage 2：对没被止损的候选补齐到 n。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import Budget, BudgetExceeded
from .comparator import Comparison, compare
from .io_util import now_iso, read_json, write_json
from .settings import CFG, PATHS, Settings

#: 不复制进沙箱的目录（要么是运行产物，要么是别的任务的记忆）
IGNORE_DIRS = {
    "log", "tmp", "documents", "contexts", "snapshots", "memory", "docs",
    "experiments", "__pycache__", ".git", ".workbuddy", ".venv", "venv",
    ".langgraph_api", ".pytest_cache", ".mypy_cache",
}
#: 沙箱里需要存在但必须为空的目录
EMPTY_DIRS = ("log", "tmp", "documents", "contexts", "snapshots", "memory")
_MAX_FILE_BYTES = 5 * 1024 * 1024


@dataclass
class Sandbox:
    path: Path
    arm: str
    keep: bool = True

    @property
    def created(self) -> bool:
        return self.path.exists()

    def cleanup(self) -> None:
        if self.keep:
            return
        _rmtree(self.path)

    def to_dict(self) -> dict:
        return {"path": str(self.path), "arm": self.arm}


def _rmtree(path: Path) -> None:
    """Windows 上 rmtree 遇到只读文件/被占用目录会失败，必须去只读位并重试。"""
    import stat as _stat
    import time as _time

    def _onexc(func, p, exc):
        try:
            os.chmod(p, _stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for attempt in range(4):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_onexc)     # Python >= 3.12
        except TypeError:
            try:
                shutil.rmtree(path, onerror=lambda f, p, e: _onexc(f, p, e))
            except Exception:
                pass
        except Exception:
            pass
        if not path.exists():
            return
        _time.sleep(0.15 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)


def create_sandbox(dest: str | Path, arm: str, root: str | Path | None = None,
                   keep: bool = True) -> Sandbox:
    """复制一份干净的运行副本（不含历史产物与记忆）。"""
    root = Path(root) if root else PATHS.root
    dest = Path(dest)
    if dest.exists():
        _rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    def _ignore(dirpath, names):
        return [n for n in names if n in IGNORE_DIRS or n.endswith(".pyc")]

    for entry in root.iterdir():
        if entry.name in IGNORE_DIRS or entry.name.endswith(".pyc"):
            continue
        target = dest / entry.name
        try:
            if entry.is_dir():
                shutil.copytree(entry, target, ignore=_ignore, dirs_exist_ok=True)
            elif entry.is_file() and entry.stat().st_size <= _MAX_FILE_BYTES:
                shutil.copy2(entry, target)
        except OSError:
            continue

    for d in EMPTY_DIRS:
        (dest / d).mkdir(parents=True, exist_ok=True)
    for d in ("adaptive_runtime",):
        (dest / d).mkdir(parents=True, exist_ok=True)
    return Sandbox(path=dest, arm=arm, keep=keep)


# ---------------------------------------------------------------- 单次 run

@dataclass
class RunOutcome:
    ok: bool
    task_id: str
    rep: int
    arm: str
    Q: float = 0.0
    A: float = 0.0
    planned: int = 0
    tokens: int = 0
    estimated_ratio: float = 0.0
    error: str = ""
    timeout: bool = False
    duration_s: float = 0.0
    out_dir: str = ""

    def to_score(self) -> dict:
        return {"task_id": self.task_id, "rep": self.rep, "arm": self.arm,
                "Q": self.Q, "A": self.A, "planned": self.planned,
                "tokens": self.tokens}

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _env_for_run(sandbox: Sandbox, overrides: dict[str, str], out_dir: Path,
                 fake: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ADAPTIVE_TOKEN_SINK", None)
    env.update({str(k): str(v) for k, v in (overrides or {}).items()})
    env.update({
        "AGENT_CHECKPOINT_DB": str(sandbox.path / "memory" / "checkpoints.sqlite"),
        "ADAPTIVE_TOKEN_SINK": str(out_dir / "tokens.jsonl"),
        "PYTHONPATH": str(sandbox.path) + os.pathsep + env.get("PYTHONPATH", ""),
        "PYTHONIOENCODING": "utf-8",
    })
    if fake:
        env["ADAPTIVE_FAKE"] = "1"
    return env


def run_one(sandbox: Sandbox, task: dict, rep: int, arm: str, out_dir: str | Path,
            overrides: dict[str, str] | None = None, cfg: Settings | None = None,
            fake_profile: str | None = None) -> RunOutcome:
    cfg = cfg or CFG
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_id = task["task_id"]
    cmd = [sys.executable, "-m", "adaptive_runtime.runner_once",
           "--query", task["query"], "--task-id", task_id, "--rep", str(rep),
           "--arm", arm, "--out", str(out)]
    if fake_profile:
        cmd += ["--fake-profile", fake_profile]
    if task.get("thread_id"):
        cmd += ["--thread-id", task["thread_id"]]

    env = _env_for_run(sandbox, overrides or {}, out, fake=bool(fake_profile))
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=str(sandbox.path), env=env,
                              capture_output=True, text=True, timeout=cfg.run_timeout,
                              encoding="utf-8", errors="replace")
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout, stderr, rc = "", f"timeout after {cfg.run_timeout}s", -1
    except OSError as e:
        stdout, stderr, rc = "", f"{type(e).__name__}: {e}", -1
    dur = time.time() - t0

    (out / "stdout.log").write_text(stdout, encoding="utf-8")
    (out / "stderr.log").write_text(stderr, encoding="utf-8")

    result = read_json(out / "result.json", default={}) or {}
    score = read_json(out / "score.json", default={}) or {}
    usage = read_json(out / "usage.json", default={}) or {}
    tokens = int(usage.get("total") or 0)
    if not tokens:
        tokens = _sum_tokens(out / "tokens.jsonl")

    ok = bool(result.get("ok")) and rc == 0
    return RunOutcome(
        ok=ok, task_id=task_id, rep=rep, arm=arm,
        Q=float(score.get("Q") or 0.0), A=float(score.get("A") or 0.0),
        planned=int(score.get("planned") or 0),
        tokens=tokens, estimated_ratio=float(usage.get("estimated_ratio") or 0.0),
        error="" if ok else (result.get("error") or stderr.strip()[-400:] or f"rc={rc}"),
        timeout=timed_out, duration_s=round(dur, 2), out_dir=str(out))


def _sum_tokens(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            total += int(json.loads(line).get("total") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return total


# ---------------------------------------------------------------- 提示补丁

class PatchError(RuntimeError):
    """提示补丁无法应用。**必须中断**，不能降级成「没改」继续跑。

    理由：补丁没打上时 cand 与 base 完全一致，ΔQ 必然≈0，于是这条假设会被记成
    「无效」，台账与经验库都被假证据污染。静默失败比崩溃危险得多。
    """


def apply_prompt_patch(sandbox: Sandbox, patch: dict) -> dict:
    """在沙箱副本上插入一段提示文本，返回可审计的 diff 摘要。"""
    import difflib

    fname = str(patch.get("file") or "")
    anchor = str(patch.get("anchor") or "")
    insert = str(patch.get("insert") or "")
    if not fname or not anchor or not insert:
        raise PatchError(f"补丁字段不完整：{sorted(patch)}")
    p = sandbox.path / fname
    if not p.exists():
        raise PatchError(f"目标文件不存在：{p}")
    old_text = p.read_text(encoding="utf-8", errors="replace")
    n = old_text.count(anchor)
    if n != 1:
        raise PatchError(f"锚点在 {fname} 出现 {n} 次（必须恰好 1 次）：{anchor!r}")
    new_text = old_text.replace(anchor, insert + anchor, 1)
    p.write_text(new_text, encoding="utf-8")

    diff = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                     fromfile=f"a/{fname}", tofile=f"b/{fname}",
                                     lineterm=""))
    added = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    return {"file": fname, "anchor": anchor, "added": added, "removed": removed,
            "diff": "\n".join(diff)}


# ---------------------------------------------------------------- 一轮 probe

@dataclass
class ProbeOutcome:
    base: list[dict] = field(default_factory=list)
    cand: list[dict] = field(default_factory=list)
    comparison: Comparison | None = None
    failures: list[dict] = field(default_factory=list)
    stopped: str = ""
    n_planned: int = 0
    n_actual: int = 0
    tokens: int = 0
    out_dir: str = ""
    sandboxes: list[Sandbox] = field(default_factory=list)
    #: L2 提示补丁的实际 diff（None 表示本候选是 L1 参数或零改动）
    patch_applied: dict | None = None

    def to_dict(self) -> dict:
        return {"base": self.base, "cand": self.cand,
                "comparison": self.comparison.to_dict() if self.comparison else None,
                "failures": self.failures, "stopped": self.stopped,
                "n_planned": self.n_planned, "n_actual": self.n_actual,
                "tokens": self.tokens, "out_dir": self.out_dir,
                "patch_applied": self.patch_applied,
                "sandboxes": [s.to_dict() for s in self.sandboxes]}


def _stop_loss(deltas: list[float], cfg: Settings) -> bool:
    """Stage 1 只做止损：均值为负且幅度超过 1.5 个标准误。"""
    if len(deltas) < 2:
        return False
    mean = sum(deltas) / len(deltas)
    var = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
    se = (var ** 0.5) / (len(deltas) ** 0.5)
    return mean < 0 and abs(mean) > 1.5 * max(se, 1e-9)


def probe_candidate(cand: Any, tasks: list[dict], out_dir: str | Path,
                    cfg: Settings | None = None, budget: Budget | None = None,
                    n: int | None = None, keep_sandbox: bool = True,
                    fake_profiles: dict[str, str] | None = None,
                    stage1_tasks: int = 4, stage1_reps: int = 2) -> ProbeOutcome:
    """跑完两臂并返回配对比较。fake_profiles 仅供 selftest 使用。"""
    cfg = cfg or CFG
    budget = budget or Budget.from_settings(cfg)
    n = n or cfg.n_default
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outcome = ProbeOutcome(n_planned=n, out_dir=str(out))
    overrides = cand.env_overrides() if hasattr(cand, "env_overrides") else {}

    if fake_profiles is None and cfg.fake_mode:
        raise RuntimeError("fake 模式必须显式提供 fake_profiles（避免假结果混入真实轮次）")

    base_sb = create_sandbox(out / "sandbox-base", "base", keep=keep_sandbox)
    cand_sb = create_sandbox(out / "sandbox-cand", "cand", keep=keep_sandbox)
    outcome.sandboxes = [base_sb, cand_sb]

    # L2：提示补丁只打在 cand 副本上，且在**跑任何 run 之前**打一次。
    # 为什么必须先打一次而不是每次 run 前都打：patch 是「锚点前插入」，
    # 打两次就会插两遍（锚点仍存在），候选实际改了两处而指纹只有一处 —— 归因就错了。
    if getattr(cand, "is_prompt", False):
        outcome.patch_applied = apply_prompt_patch(cand_sb, cand.patch)

    def _profile(arm: str) -> str | None:
        return (fake_profiles or {}).get(arm)

    def _one(arm: str, task: dict, rep: int) -> RunOutcome:
        sb = base_sb if arm == "base" else cand_sb
        ov = {} if arm == "base" else overrides
        ro = run_one(sb, task, rep, arm, out / arm / f"{task['task_id']}-r{rep}",
                     overrides=ov, cfg=cfg, fake_profile=_profile(arm))
        outcome.tokens += ro.tokens
        if ro.tokens:
            try:
                budget.spend(ro.tokens, getattr(cand, "candidate_id", None))
            except BudgetExceeded as e:
                outcome.stopped = f"budget: {e}"
        if not ro.ok:
            outcome.failures.append(ro.to_dict())
        return ro

    # ---- Stage 1 ----
    s1 = tasks[:stage1_tasks]
    for task in s1:
        for rep in range(min(stage1_reps, n)):
            b = _one("base", task, rep)
            if outcome.stopped.startswith("budget"):
                break
            c = _one("cand", task, rep)
            if outcome.stopped.startswith("budget"):
                break
            if b.ok and c.ok:
                outcome.base.append(b.to_score())
                outcome.cand.append(c.to_score())
    if outcome.stopped.startswith("budget"):
        outcome.n_actual = len(outcome.base)
        outcome.comparison = _compare_with_failures(outcome, cfg)
        _persist(outcome, out)
        return outcome

    deltas = [c["Q"] - b["Q"] for b, c in zip(outcome.base, outcome.cand)]
    if _stop_loss(deltas, cfg):
        outcome.stopped = "stop_loss"
        outcome.n_actual = len(outcome.base)
        outcome.comparison = _compare_with_failures(outcome, cfg)
        _persist(outcome, out)
        return outcome

    # ---- Stage 2：补齐到 n ----
    for task in tasks:
        have = {s["rep"] for s in outcome.base if s["task_id"] == task["task_id"]}
        for rep in range(n):
            if rep in have:
                continue
            b = _one("base", task, rep)
            if outcome.stopped.startswith("budget"):
                break
            c = _one("cand", task, rep)
            if outcome.stopped.startswith("budget"):
                break
            if b.ok and c.ok:
                outcome.base.append(b.to_score())
                outcome.cand.append(c.to_score())
        if outcome.stopped.startswith("budget"):
            break

    outcome.n_actual = len(outcome.base)
    outcome.comparison = _compare_with_failures(outcome, cfg)
    _persist(outcome, out)
    return outcome


def _compare_with_failures(outcome: ProbeOutcome, cfg: Settings) -> Comparison:
    """比较前把失败/超时的 run 计入 missing/timeout —— 缺一次就判 INCOMPLETE_EVIDENCE。"""
    cmp = compare(outcome.base, outcome.cand, cfg)
    if outcome.failures:
        cmp.missing_runs = sum(1 for f in outcome.failures if not f.get("timeout"))
        cmp.timeout_runs = sum(1 for f in outcome.failures if f.get("timeout"))
    return cmp


def _persist(outcome: ProbeOutcome, out: Path) -> None:
    write_json(out / "probe.json", outcome.to_dict())
    write_json(out / "base" / "scores.json", outcome.base)
    write_json(out / "cand" / "scores.json", outcome.cand)


def cleanup(outcome: ProbeOutcome) -> None:
    for sb in outcome.sandboxes:
        sb.keep = False
        sb.cleanup()
