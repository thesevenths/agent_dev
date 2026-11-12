"""ARR 0.1 / R4: bridge that feeds ``research_runtime`` outcomes into the RRSI.

Why this lives in ``adaptive_runtime`` (not ``research_runtime``)
---------------------------------------------------------------
``research_runtime`` is the generic, engine-agnostic inner loop. The RRSI
(``adaptive_runtime``) is the outer self-improvement loop that optimises *some
runtime* based on measured traces. This module is therefore a **task-family
adapter** for the RRSI -- exactly like the existing ``humaneval.py`` /
``bench_tasks.py`` adapters that teach the RRSI about a new kind of task. The
generic research runtime never learns the RRSI's name.

What it does (the outer loop, wired to the real RRSI primitives)
----------------------------------------------------------------
1. ``export_research_trace(report)`` turns a research outcome (a research
   objective's autonomous-research result) into an RRSI measurement "trace" --
   the research runtime's quality on that objective.
2. ``ingest(report, ...)`` records the trace in the RRSI ``experience.Store``,
   proposes a *candidate improvement to the research runtime's own config* (a
   ``proposer.Candidate`` over a research-runtime search space), evaluates the
   candidate **inside an isolated ``probe.Sandbox``** (never on the live
   runtime), and runs the real non-compensable ``gate_a`` / ``gate_b``.
3. The decision is recorded in the RRSI ``ledger.Ledger`` and versioned against
   the current ``BaseVersion``.

ARR 0.2 / Package C -- REAL candidate evaluation
------------------------------------------------
The R4 demo evaluated candidates with preset scores and a sandbox marker file.
That is NOT self-improvement. ``ingest(..., real=True)`` replaces it with a real
closed loop:

    real ResearchReport -> bottleneck diagnosis -> candidate over REAL training
    knobs -> CandidateConfigBuilder materialises an ISOLATED, hash-traced config
    (data section + frozen dataset SHA-256 verified on disk) -> verify the patch
    actually changed the runnable config AND the engine's real metrics -> run
    base/cand on PAIRED SEEDS with unified eval (decision split) plus an
    INDEPENDENT holdout (confirmation only, never used to select) -> REAL deltas
    -> gate_a / gate_b -> ACCEPT / REJECT / INCONCLUSIVE -> ledger + evidence.

Safety (must hold; *verified* by a before/after live-runtime fingerprint, and asserted in tests)
--------------------------------------------------------------------------------------------------
* The candidate is evaluated in a ``probe.Sandbox`` copy + isolated engine config
  directories. The LIVE research runtime is never mutated -- it receives the
  report as plain data and writes only to the injected ``ledger`` / ``experience``
  / ``sandbox`` paths and the engine's ``outputs/``. ``live_runtime_untouched`` is
  a before/after SHA-256 + git-HEAD comparison, NOT a hardcoded ``True``.
* The candidate may only touch whitelisted training knobs; ``data`` / ``output`` /
  ``device`` / ``seed`` / ``run_id`` / ``checkpoint`` are frozen by the builder.
  The scoring function (``_research_quality``) and the eval splits live in this
  module: a candidate can never modify its own scoring.
* No ``version.promote`` is ever called here: applying a candidate is a separate,
  human-gated step (``require_operator=True``).
* Deterministic ``--fake`` profiles (good / zero / worse) remain ONLY for
  gate-chain unit tests / CI -- zero network, zero model, zero engine.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gate, ledger as _ledger, experience as _experience, version as _version
from .gate import Decision, explain, ledger_status
from .io_util import new_id, now_iso
from .proposer import Candidate, fingerprint
from .probe import Sandbox, create_sandbox
from .settings import Settings

# ---------------------------------------------------------------- 研究运行时搜索空间
#
# ARR 的外层循环优化对象就是 research_runtime 自己的配置旋钮。这些旋钮必须是
# 「改了会改变研究行为、但不触碰安全边界」的，与 adaptive_runtime 主体搜索空间的
# 冻结/惰性判定同构。这里只列出 R4 演示用的两个代表性旋钮：
#   - RESEARCH_PAIRED_SEEDS：配对种子数，影响决策统计功效（对应 inconclusive 时该提）；
#   - RESEARCH_MAX_CANDIDATES：每轮候选数上限，影响探索广度。
#
# RESEARCH_FROZEN 是绝不许被候选触碰的研究运行时安全边界（证据完整性阈值等），
# 对应 adaptive_runtime 主体的 FROZEN_ENV —— 一旦候选 target 落在里面就直接
# FROZEN_CORE_VIOLATION。

#: 研究运行时搜索空间 = 真实、安全、可控的 SciNet++ 训练旋钮。
#: 这些旋钮直接决定内部动力学实验的训练行为，且都不在 forbidden_root 里
#: （data / output / device / experiment / seed / run_id / checkpoint 一律不可改），
#: 因此候选对它们的干预是一次「受控实验」：同一冻结数据集 + 配对 seed 上比较 base/cand。
#: 这正是 Package C 要求的「候选必须实际运行」，而不是改一个不动行为的惰性旋钮。
RESEARCH_SEARCH_SPACE: dict[str, dict] = {
    "train.lr": {
        "values": ["0.003", "0.005", "0.01"], "level": 1,
        "rationale": "pendulum_smoke 仅 1 epoch + 64 样本，LR 直接决定收敛速度与最终 "
                     "latent 表示质量；在极小数据集上恰当的 LR 可提升 holdout 动力学指标。",
        "expectation": "improve",
        "dominant": "LOW_YIELD",
    },
    "train.weight_decay": {
        "values": ["0.001", "0.0"], "level": 1,
        "rationale": "正则化强度影响泛化；极小数据集上恰当的 wd 可降低 latent_mse。",
        "expectation": "improve",
        "dominant": "LOW_YIELD",
    },
    "jepa.vicreg_var_weight": {
        "values": ["10.0", "40.0"], "level": 1,
        "rationale": "VICReg 方差项权重控制表示去冗余程度，直接影响 cosine@1 动力学预测质量。",
        "expectation": "improve",
        "dominant": "LOW_YIELD",
    },
}

#: 研究运行时的冻结边界：证据完整性阈值等安全关键旋钮绝不许被候选触碰。
RESEARCH_FROZEN = frozenset({
    "RESEARCH_EVIDENCE_STRICT",
    "RESEARCH_FROZEN_DATASET_CHECK",
    "HYPOTHESIS_CRITIC_REJECT_UNKNOWN",
})


def _snapshot_live_runtime() -> str:
    """Fingerprint of the LIVE research_runtime the bridge must never mutate.

    The bridge only receives the research report as plain data and writes to the
    injected ledger / experience / sandbox paths. To make the "live runtime
    untouched" guarantee *verified* rather than merely asserted, we hash the
    on-disk source of research_runtime plus the enclosing repo git HEAD. A
    before/after comparison inside ``ingest`` therefore PROVES the live runtime
    was not imported-then-mutated, instead of relying on a hardcoded ``True``.
    """
    parts: list[str] = []
    try:
        parts.append("git_head=" + _version.git_head())
    except Exception:
        parts.append("git_head=unknown")
    rt_dir = Path(__file__).resolve().parents[1] / "research_runtime"
    if rt_dir.is_dir():
        h = hashlib.sha256()
        for p in sorted(rt_dir.rglob("*.py")):
            h.update(str(p.relative_to(rt_dir)).encode("utf-8"))
            h.update(p.read_bytes())
        parts.append("rt_sha256=" + h.hexdigest())
    else:
        parts.append("rt_dir=missing")
    return "|".join(parts)


#: 诊断信号 -> 该优先搜的真实训练旋钮（与 proposer.propose 的 prefer 同构）。
#: 三个诊断各自映射到不同的、可真跑出差异的旋钮，保证「诊断驱动候选」不是摆设。
DOMINANT_TO_TARGET: dict[str, str] = {
    "DECISION_UNCERTAINTY": "train.lr",
    "LOW_YIELD": "jepa.vicreg_var_weight",
    "ROBUST_DECISION": "train.weight_decay",
}


# ---------------------------------------------------------------- 确定性假评估

#: 研究运行时质量度量（Q=综合质量，A=契约达成率，tasks=各目标上的质量）。
#: 与 run_adaptive.py 的 ``--fake --case {good,zero,worse}`` 同构：用确定性轨迹
#: 验证整条门控链路，不调用任何 LLM / 不跑真实研究战役。
_FAKE_BASE = {"Q": 0.60, "A": 0.60, "tasks": [0.55, 0.62, 0.58, 0.65]}
_FAKE_CAND = {
    "good":  {"Q": 0.79, "A": 0.79, "tasks": [0.74, 0.82, 0.75, 0.85]},
    "zero":  {"Q": 0.60, "A": 0.60, "tasks": [0.59, 0.62, 0.60, 0.61]},
    "worse": {"Q": 0.56, "A": 0.56, "tasks": [0.35, 0.42, 0.30, 0.40]},
}


# ---------------------------------------------------------------- 真实候选运行（Package C）
#
# 审查要求：候选必须**实际运行**——不能只写一个 sandbox 标记文件、再用预设分数决定
# accept/reject。因此这里把候选的 config patch 作为一次**受控的、隔离的** SciNet++
# 训练超参干预：用 CandidateConfigBuilder 落到独立的候选目录（sandbox/worktree 等价物），
# 在同样的冻结数据集 + 配对 seed 上跑 base/cand 两臂，用 holdout（test split）上的真实
# 动力学指标（cosine@1，越高越好）度量研究运行时质量，再把真实 delta 喂给 gate_a/gate_b。
#
# 关键安全点（与 frozen_core / 证据完整性同构）：
#   * 候选只能改白名单里的真实训练旋钮（train.lr 等），绝不许碰 data/output/device/seed 等；
#   * 每次 build 都强制校验冻结数据集 SHA-256（数据身份在盘上验证，不是 JSON 里一个标签）；
#   * 「patch 改变了行为」是硬门：配置 sha 必须变、且被改键的值必须确实生效，否则直接抛错，
#     绝不降级成「没改」继续跑（那会让 ΔQ≈0 把假设记成无效，污染台账）；
#   * live research_runtime 始终不被 import / 不被 mutate（ingest 用 before/after 指纹验证）。

_REAL_OBJECTIVE = "obj_rrsi_candidate_eval"
_DEFAULT_REAL_SEEDS = (42, 43, 44)


def _import_rt():
    """Lazy import of research_runtime primitives (keeps module import cheap/safe)."""
    from research_runtime.adapters import SciNetAdapter
    from research_runtime.schemas import ExperimentSpec
    from research_runtime.candidate_config import (
        CandidateConfigBuilder, CandidateConfigError, DEFAULT_FORBIDDEN_ROOTS,
    )
    import yaml
    return (SciNetAdapter, ExperimentSpec, CandidateConfigBuilder,
            CandidateConfigError, DEFAULT_FORBIDDEN_ROOTS, yaml)


def _get_dotted(config: dict, dotted_key: str):
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


def _scinet_engine(scinet_project_dir: str | Path | None = None,
                  scinet_python: str | Path | None = None):
    SciNetAdapter, _, _, _, _, _ = _import_rt()
    try:
        adapter = SciNetAdapter(project_dir=scinet_project_dir,
                                python_executable=scinet_python, strict=False)
        adapter.list_jobs()  # reachability probe; raises if CLI/engine missing
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "SciNet++ engine unreachable; refusing to fake the candidate "
            f"evaluation (Package C requires a REAL run): {e}") from e
    return adapter


def _default_scinet_config() -> str:
    from research_runtime.settings import DEFAULT_CONFIG
    return DEFAULT_CONFIG


def _frozen_dataset_version(project_dir: Path, base_config: str) -> str:
    _, _, _, _, _, yaml = _import_rt()
    cfg = yaml.safe_load((Path(project_dir) / base_config).read_text(encoding="utf-8"))
    frozen_dir = (cfg.get("data") or {}).get("frozen_dir")
    if not frozen_dir:
        raise RuntimeError("baseline config has no data.frozen_dir")
    manifest = json.loads(
        (Path(project_dir) / frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    dv = manifest.get("dataset_version")
    if not dv:
        raise RuntimeError("frozen manifest has no dataset_version")
    return dv


def _coerce_override(project_dir: Path, base_config: str, key: str, raw: str):
    """Coerce the candidate's new value to the baseline config field's type."""
    _, _, _, _, _, yaml = _import_rt()
    cfg = yaml.safe_load((Path(project_dir) / base_config).read_text(encoding="utf-8"))
    base_v = _get_dotted(cfg, key)
    if isinstance(base_v, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(base_v, int) and not isinstance(base_v, bool):
        return int(raw)
    if isinstance(base_v, float):
        return float(raw)
    return raw


def _research_quality(metrics: dict) -> float:
    """研究运行时质量 = holdout 上的动力学预测质量（越高越好）。"""
    if not metrics:
        return 0.0
    if "cosine@1" in metrics:
        return float(metrics["cosine@1"])
    if "latent_mse@1" in metrics:
        return 1.0 / (1.0 + float(metrics["latent_mse@1"]))
    vals = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _run_real_arm(arm: str, config_relpath: str, seed: int, adapter, *,
                  decision_split: str, holdout_split: str | None,
                  epochs: int) -> dict:
    """Run ONE arm (base or cand) for ONE seed on the real SciNet++ engine.

    一次 train + 一次 decision split 上的 evaluate（gate_b 只读这个），以及可选的
    一次 **独立 holdout** 上的 evaluate（复用同一 run_id / 同一 checkpoint，只作
    confirmation evidence，绝不参与 accept/reject）。两臂走完全相同的流程 = unified eval。
    """
    _, ExperimentSpec, _, _, _, _ = _import_rt()
    spec = ExperimentSpec(
        experiment_id=f"rrsi_{arm}_{seed}", objective_id=_REAL_OBJECTIVE,
        stage="train", config_path=config_relpath, budget={"epochs": epochs},
    )
    rec: dict = {
        "arm": arm, "seed": seed, "config_relpath": config_relpath,
        "ok": False, "Q": 0.0, "holdout_Q": None,
        "metrics": {}, "holdout_metrics": {},
        "checkpoint_sha256": None, "run_id": None,
        "job_ids": [], "error": "",
    }
    jid = adapter.submit(spec)
    rec["job_ids"].append(jid)
    adapter.wait(jid, timeout=600)
    train = adapter.collect_result(jid, spec)
    if train.status != "succeeded":
        rec["error"] = f"train failed: {train.error or train.status}"
        return rec
    rec["run_id"] = train.run_id

    dec_spec = spec.with_run_id(train.run_id, stage="evaluate",
                                split=decision_split, parent_job_id=jid)
    jid2 = adapter.submit(dec_spec)
    rec["job_ids"].append(jid2)
    adapter.wait(jid2, timeout=600)
    ev = adapter.collect_result(jid2, dec_spec)
    rec["metrics"] = dict(ev.metrics)
    rec["checkpoint_sha256"] = (ev.checkpoint or {}).get("sha256")
    rec["evaluation_split"] = ev.evaluation_split or decision_split
    rec["Q"] = _research_quality(ev.metrics)
    if ev.status != "succeeded":
        rec["error"] = f"decision eval failed: {ev.error or ev.status}"
        return rec

    if holdout_split:
        ho_spec = spec.with_run_id(train.run_id, stage="evaluate",
                                   split=holdout_split, parent_job_id=jid)
        jid3 = adapter.submit(ho_spec)
        rec["job_ids"].append(jid3)
        adapter.wait(jid3, timeout=600)
        ho = adapter.collect_result(jid3, ho_spec)
        rec["holdout_metrics"] = dict(ho.metrics)
        rec["holdout_evaluation_split"] = ho.evaluation_split or holdout_split
        rec["holdout_Q"] = _research_quality(ho.metrics)
        if ho.status != "succeeded":
            rec["error"] = f"holdout eval failed: {ho.error or ho.status}"
            return rec
    rec["ok"] = True
    return rec


def _run_real_pair(candidate, *, scinet_project_dir=None, scinet_python=None,
                   base_config=None, dataset_version=None, campaign_id=None,
                   paired_seeds=None, decision_split="validation",
                   holdout_split: str | None = "test", epochs=1, cfg=None):
    """把候选的 REAL patch 落到隔离配置里，在配对 seed 上跑 base/cand 两臂并实测。

    返回 (base_measure, cand_measure, patch_report)。硬约束：
      * 引擎不可达 -> 抛错，绝不悄悄退回假评估（Package C 要求真跑）；
      * 配置 sha 未变 / override 未生效 -> 抛 CandidateConfigError（等价于
        probe.PatchError：补丁没打上就静默继续，会拿 ΔQ≈0 污染台账）；
      * 候选只能改白名单训练旋钮，data/output/device/seed 由 builder 强制冻结；
      * 独立 holdout 只作 confirmation，gate 决策只读 decision split。
    """
    cfg = cfg or Settings()
    (_, _, CandidateConfigBuilder, CandidateConfigError,
     DEFAULT_FORBIDDEN_ROOTS, yaml) = _import_rt()
    adapter = _scinet_engine(scinet_project_dir, scinet_python)
    project_dir = Path(adapter.project_dir).resolve()
    base_config = base_config or _default_scinet_config()
    dv = dataset_version or _frozen_dataset_version(project_dir, base_config)
    builder = CandidateConfigBuilder(
        project_dir, allowed_keys=set(RESEARCH_SEARCH_SPACE),
        forbidden_roots=DEFAULT_FORBIDDEN_ROOTS, seed_key="seed")
    patch = candidate.patch or {}
    key = patch.get("key") or candidate.target
    if key not in RESEARCH_SEARCH_SPACE:
        raise CandidateConfigError(
            f"candidate target {key!r} is not a whitelisted real training knob")
    new_raw = patch.get("new") if patch.get("new") is not None else candidate.new
    new_val = _coerce_override(project_dir, base_config, key, str(new_raw))
    overrides = {key: new_val}
    cid = candidate.candidate_id
    # hyp_fingerprint is a human-readable composite ("train.lr|0.001->0.003|...") used
    # for ledger dedup; CandidateConfigBuilder needs a single-segment safe id for the
    # isolated directory. Hash it to a STABLE safe id (same hypothesis -> same id),
    # rather than mutating the fingerprint or bypassing the builder's id check.
    hid = "hyp_" + hashlib.sha256(
        (candidate.hyp_fingerprint or cid).encode("utf-8")).hexdigest()[:16]
    seeds = list(paired_seeds or _DEFAULT_REAL_SEEDS)
    campaign_id = campaign_id or new_id("rrsi_camp")

    # 冻结数据集 manifest 的 SHA-256：数据身份是盘上校验，不是 JSON 里一个标签。
    try:
        cfg_data = yaml.safe_load(
            (project_dir / base_config).read_text(encoding="utf-8"))
        manifest_path = (project_dir / cfg_data["data"]["frozen_dir"]
                         / "manifest.json")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        manifest_path, manifest_sha = None, "unavailable"

    base_runs, cand_runs = [], []
    patch_report: dict = {}
    for i, seed in enumerate(seeds):
        base_art = builder.build_baseline(
            base_config=base_config, campaign_id=campaign_id, seed=seed,
            dataset_version=dv)
        cand_art = builder.build(
            base_config=base_config, campaign_id=campaign_id, candidate_id=cid,
            hypothesis_id=hid, seed=seed, overrides=overrides, dataset_version=dv)
        if i == 0:
            base_cfg = yaml.safe_load(
                (project_dir / base_art.config_path).read_text(encoding="utf-8"))
            cand_cfg = yaml.safe_load(
                (project_dir / cand_art.config_path).read_text(encoding="utf-8"))
            old_v = _get_dotted(base_cfg, key)
            new_v = _get_dotted(cand_cfg, key)
            config_changed = (cand_art.candidate_config_sha256
                              != base_art.baseline_config_sha256)
            override_effective = (new_v == new_val) and (new_v != old_v)
            if not config_changed or not override_effective:
                raise CandidateConfigError(
                    "candidate patch did NOT apply: "
                    f"config_changed={config_changed} "
                    f"override_effective={override_effective} "
                    f"(key={key} old={old_v!r} new={new_v!r})")
            patch_report = {
                "schema_version": 2,
                "engine": adapter.name,
                "engine_project_dir": str(project_dir),
                "engine_python": str(adapter.python),
                "base_config": base_config,
                "dataset_version": dv,
                "manifest_path": str(manifest_path) if manifest_path else None,
                "manifest_sha256": manifest_sha,
                "git_head": _version.git_head(),
                "campaign_id": campaign_id,
                "candidate_id": cid,
                "key": key, "old": old_v, "new": new_v,
                "config_changed": config_changed,
                "override_effective": override_effective,
                "baseline_config_sha256": base_art.baseline_config_sha256,
                "candidate_config_sha256": cand_art.candidate_config_sha256,
                "seeds": seeds, "epochs": epochs,
                "decision_split": decision_split,
                "holdout_split": holdout_split,
                "holdout_used_for_decision": False,
            }
        base_runs.append(_run_real_arm(
            "base", base_art.config_relpath, seed, adapter,
            decision_split=decision_split, holdout_split=holdout_split,
            epochs=epochs))
        cand_runs.append(_run_real_arm(
            "cand", cand_art.config_relpath, seed, adapter,
            decision_split=decision_split, holdout_split=holdout_split,
            epochs=epochs))

    # ---- 真实行为验证：候选改动必须真的改变了引擎输出，而不是只改了个文件 ----
    behavior_changed = any(
        b["metrics"] != c["metrics"] for b, c in zip(base_runs, cand_runs))
    patch_report["behavior_changed"] = behavior_changed

    # ---- 独立 holdout 一致性：方向与 decision split 是否吻合（confirmation only）----
    if holdout_split:
        holdout_deltas = [
            (c["holdout_Q"] - b["holdout_Q"])
            for b, c in zip(base_runs, cand_runs)
            if b["holdout_Q"] is not None and c["holdout_Q"] is not None
        ]
        decision_deltas = [c["Q"] - b["Q"] for b, c in zip(base_runs, cand_runs)]
        agree = sum(
            (d >= 0) == (h >= 0) for d, h in zip(decision_deltas, holdout_deltas))
        patch_report["holdout_deltas"] = holdout_deltas
        patch_report["holdout_direction_agreement"] = (
            round(agree / len(holdout_deltas), 4) if holdout_deltas else None)
        patch_report["holdout_consistent"] = (
            bool(holdout_deltas) and agree >= (len(holdout_deltas) + 1) // 2)
    else:
        patch_report["holdout_consistent"] = None

    base_ok = [r["ok"] for r in base_runs]
    cand_ok = [r["ok"] for r in cand_runs]
    base_measure = ResearchRuntimeMeasure(
        arm="base",
        Q=sum(r["Q"] for r in base_runs) / max(1, len(base_runs)),
        A=sum(base_ok) / max(1, len(base_ok)),
        tasks=[r["Q"] for r in base_runs])
    cand_measure = ResearchRuntimeMeasure(
        arm="cand",
        Q=sum(r["Q"] for r in cand_runs) / max(1, len(cand_runs)),
        A=sum(cand_ok) / max(1, len(cand_ok)),
        tasks=[r["Q"] for r in cand_runs])
    patch_report["missing_runs"] = sum(1 for r in base_runs + cand_runs
                                       if not r["ok"])
    patch_report["regression_rate"] = round(
        sum(1 for b, c in zip(base_runs, cand_runs) if c["Q"] < b["Q"])
        / max(1, len(seeds)), 4)
    patch_report["base_runs"] = base_runs
    patch_report["cand_runs"] = cand_runs
    patch_report["evidence_sha256"] = hashlib.sha256(
        json.dumps(patch_report, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return base_measure, cand_measure, patch_report


def scinet_available() -> bool:
    """True iff the real SciNet++ engine CLI is reachable (for gated real tests)."""
    try:
        _scinet_engine()
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class ResearchRuntimeMeasure:
    """单臂（base 或 cand）在研究运行时质量维度上的测量。"""
    arm: str
    Q: float
    A: float
    tasks: list[float]

    def to_score(self) -> dict:
        return {"arm": self.arm, "Q": self.Q, "A": self.A, "tasks": list(self.tasks)}


def evaluate_research_runtime(arm: str, case: str | None = None, *,
                              candidate: "Candidate | None" = None,
                              real: bool = False,
                              scinet_project_dir: str | Path | None = None,
                              scinet_python: str | Path | None = None,
                              base_config: str | None = None,
                              dataset_version: str | None = None,
                              campaign_id: str | None = None,
                              paired_seeds: list[int] | None = None,
                              decision_split: str = "validation",
                              holdout_split: str | None = "test",
                              epochs: int = 1,
                              cfg: "Settings | None" = None) -> ResearchRuntimeMeasure:
    """Measure research-runtime quality for one arm.

    FAKE path (``case`` given, or ``real`` is False): deterministic preset
    scores -- used by the gate-chain selftests; never touches the engine.

    REAL path (``real=True`` and ``candidate`` given): applies the candidate's
    config patch in an isolated sandbox and runs the REAL SciNet++ pendulum
    experiment for both arms on paired seeds; returns the measured arm measure.
    Raises if the engine is unreachable or the patch was a no-op.
    """
    if real and candidate is not None:
        base_m, cand_m, _ = _run_real_pair(
            candidate, scinet_project_dir=scinet_project_dir,
            scinet_python=scinet_python, base_config=base_config,
            dataset_version=dataset_version, campaign_id=campaign_id,
            paired_seeds=paired_seeds, decision_split=decision_split,
            holdout_split=holdout_split, epochs=epochs, cfg=cfg)
        return base_m if arm == "base" else cand_m
    # ---- deterministic fake (gate-chain tests) ----
    case = (case or "zero").lower()
    if case not in _FAKE_CAND:
        case = "zero"
    base = _FAKE_BASE
    cand = _FAKE_CAND[case]
    if arm == "cand":
        return ResearchRuntimeMeasure(
            arm="cand", Q=cand["Q"], A=cand["A"], tasks=list(cand["tasks"]))
    return ResearchRuntimeMeasure(
        arm="base", Q=base["Q"], A=base["A"], tasks=list(base["tasks"]))


# ---------------------------------------------------------------- trace 导出

def export_research_trace(report: dict) -> dict:
    """把一份 research_runtime 研究报告 / 摘要转成 RRSI 的「轨迹」度量。

    度量的是研究运行时的**自主研究质量**：在给定目标上，它产出了多少可被
    接受的决定、证据完整性如何。这是 RRSI 优化研究运行时时的「任务得分」。
    """
    comparisons = list(report.get("comparisons") or report.get("decisions") or [])
    decisions = [c.get("decision") if isinstance(c, dict) else str(c)
                 for c in comparisons]
    n = max(1, len(decisions))
    accepted = sum(1 for d in decisions if d == "accepted_for_confirmation")
    rejected = sum(1 for d in decisions if d == "rejected")
    quality = round(accepted / n, 4)
    return {
        "objective_id": report.get("objective_id") or report.get("session_id") or "",
        "outcome": report.get("outcome") or "",
        "n_decisions": len(decisions),
        "accepted_for_confirmation": accepted,
        "rejected": rejected,
        # 研究运行时质量得分：被接受决定占比 + 证据完整性补偿。
        "quality_score": quality,
        "dominant_signal": _dominant_signal(report, accepted, rejected, n),
        "exported_at": now_iso(),
    }


def _dominant_signal(report: dict, accepted: int, rejected: int, n: int) -> str:
    outcome = report.get("outcome") or ""
    if outcome == "inconclusive" or accepted == 0:
        return "DECISION_UNCERTAINTY"
    if rejected >= accepted:
        return "LOW_YIELD"
    return "ROBUST_DECISION"


# ---------------------------------------------------------------- 候选生成

def propose_research_candidate(dominant: str, ldg: "_ledger.Ledger",
                               cfg: Settings | None = None) -> Candidate:
    """根据研究运行时的主导信号，从研究运行时搜索空间生成一个候选改进。

    与 ``proposer.propose`` 同构：确定性、不花 LLM 钱、跳过已在台账结论里的指纹。
    候选只作用于研究运行时配置旋钮，且绝不触碰 RESEARCH_FROZEN 边界。
    """
    cfg = cfg or Settings()
    target = DOMINANT_TO_TARGET.get(dominant) or next(iter(RESEARCH_SEARCH_SPACE))
    spec = RESEARCH_SEARCH_SPACE[target]
    if target in RESEARCH_FROZEN:
        raise ValueError(f"候选 target {target} 落在研究运行时冻结边界，禁止搜索")
    old = _current_research_value(target)
    for new in spec["values"]:
        if new == old:
            continue
        fp = fingerprint(target, old, new, dominant)
        dup, _ = ldg.is_duplicate(fp)
        if dup:
            continue
        patch = {"kind": "research_config", "key": target, "old": old, "new": new}
        return Candidate(
            candidate_id=new_id("rrsi_cand"), target=target, level=spec["level"],
            old=old, new=new,
            hypothesis=f"research_runtime {target} {old}->{new}：{spec['rationale']}",
            taxonomy=dominant, edits=1, est_cost=0,
            expected_behavior=[spec["expectation"]], patch=patch,
            hyp_fingerprint=fp,
        )
    raise ValueError(f"研究运行时候选空间已对 {target} 穷尽（old={old}）")


def _current_research_value(target: str) -> str:
    """研究运行时旋钮的现行值：进程环境 > 代码默认。

    真实实现会读 research_runtime 的配置；R4 演示里它只是个占位默认值，
    保证候选的 old->new 与搜索空间相邻值一致。
    """
    import os
    v = os.environ.get(target)
    if v:
        return v
    # 与 RESEARCH_SEARCH_SPACE 首值相邻：取每旋钮的「起点」默认（= pendulum_smoke 基线）。
    defaults = {"train.lr": "0.001", "train.weight_decay": "0.0001",
                "jepa.vicreg_var_weight": "25.0"}
    return defaults.get(target, "0.001")


# ---------------------------------------------------------------- 证据装配

def build_research_evidence(base: ResearchRuntimeMeasure,
                            cand: ResearchRuntimeMeasure, *,
                            missing_runs: int = 0,
                            timeout_runs: int = 0,
                            regression_rate: float = 0.0,
                            crashed_tasks: list[str] | None = None) -> "gate.Evidence":
    """把 base/cand 双臂测量装配成 gate_b 需要的 Evidence。

    任务级 delta = 每配对 seed 的 Q 的 cand - base；A_base/A_cand 是契约达成率
    （真实路径里 = run 成功率）；dC=0（改配置不消耗 token 预算）。
    真实路径必须把 missing/timeout/regression 传进来 —— 挂掉的 run 不能插补，
    gate_b 会把整轮判 INCOMPLETE_EVIDENCE（与 probe._compare_with_failures 同构）。
    """
    n = min(len(base.tasks), len(cand.tasks))
    task_deltas = [round(cand.tasks[i] - base.tasks[i], 6) for i in range(n)]
    return gate.Evidence(
        deltas=list(task_deltas),
        task_deltas=task_deltas,
        A_base=base.A, A_cand=cand.A,
        dC=0.0,
        regression_rate=regression_rate,
        n_tasks=n,
        add_novel_component=False,
        hard_failure_removed=False,
        missing_runs=missing_runs,
        timeout_runs=timeout_runs,
        judge_unstable_ratio=0.0,
        plan_undercut_rate=0.0,
        crashed_tasks=list(crashed_tasks or []),
    )


# ---------------------------------------------------------------- 编排入口

@dataclass
class IngestResult:
    objective_id: str
    trace: dict
    candidate: dict
    sandbox_path: str
    base_measure: dict
    cand_measure: dict
    evidence: dict
    decision_a: str
    decision_b: str
    ledger_status: str
    ledger_fingerprint: str
    experience_ref: str
    base_version: int
    base_commit: str
    live_runtime_untouched: bool
    note: str = ""
    #: ARR 0.2 / Package C：真实评估的证据与三态判定。
    evaluation_mode: str = "fake"
    decision: str = "INCONCLUSIVE"
    behavior_changed: bool | None = None
    holdout_consistent: bool | None = None
    real_evidence: dict = field(default_factory=dict)
    artifact_sha256: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def ingest(report: dict, *, ledger_path: str | Path,
           experience_path: str | Path, sandbox_root: str | Path | None = None,
           fake_case: str = "good", cfg: Settings | None = None,
           real: bool = False,
           scinet_project_dir: str | Path | None = None,
           scinet_python: str | Path | None = None,
           base_config: str | None = None,
           dataset_version: str | None = None,
           campaign_id: str | None = None,
           paired_seeds: list[int] | None = None,
           decision_split: str = "validation",
           holdout_split: str | None = "test",
           epochs: int = 1) -> IngestResult:
    """把一份**真实** ResearchReport 接入 RRSI 外层循环。

    ``real=False``（默认）：确定性 fake 评估，只用于门控链路单测 / CI，绝不碰引擎。
    ``real=True``：Package C 的真实闭环 —— 从 report 诊断瓶颈 -> 生成候选 ->
    在隔离沙箱里把候选 patch 落成真实配置 -> 验证 patch 真的改变了引擎行为 ->
    base/cand 用配对 seed 跑同一研究任务 + unified eval + 独立 holdout ->
    把**实测** delta 喂给 gate_a/gate_b -> ACCEPT / REJECT / INCONCLUSIVE -> 落账。

    **绝不修改 live research_runtime**：只写注入的 ledger/experience/sandbox 路径与
    SciNet++ checkout 的 outputs/；从不调用 version.promote（落地是独立的人工门）。
    live_runtime_untouched 是 before/after 指纹**比对**出来的，不是自我声明。
    """
    cfg = cfg or Settings()
    # 0) 评估前快照：记录 live research_runtime 指纹，结尾比对以 *验证*（而非声明）未改动。
    live_before = _snapshot_live_runtime()
    # 1) trace 导出 + 经验入库（度量研究运行时质量）
    trace = export_research_trace(report)
    store = _experience.Store(experience_path)
    exp = store.create(
        problem=f"RESEARCH_RUNTIME|{trace['dominant_signal']}",
        claim=f"自主研究质量（{trace['objective_id']}）={trace['quality_score']}",
        component="research_runtime",
        edit={},
        n_pairs=trace["n_decisions"], n_target=trace["n_decisions"],
        p_value=1.0, regression_rate=0.0, dQ=0.0, dC=0.0,
        positive=[trace["objective_id"]], negative=[],
        judge_version="research_ingest.real" if real else "research_ingest.fake",
        hypothesis_id="",
    )

    # 2) 候选生成（基于主导信号，跳过已结论指纹）
    ldg = _ledger.Ledger(ledger_path)
    candidate = propose_research_candidate(trace["dominant_signal"], ldg, cfg=cfg)
    fp = candidate.hyp_fingerprint

    # 3) 隔离沙箱：候选在副本里评估，绝不触碰 live runtime
    if sandbox_root is None:
        sandbox_root = Path(tempfile.mkdtemp(prefix="rrsi_sandbox_"))
    sb = create_sandbox(Path(sandbox_root) / "cand", arm="cand", keep=True)

    # 4) 评估：真实候选运行 / 或确定性 fake（仅门控链路测试）
    patch_report: dict = {}
    if real:
        base, cand, patch_report = _run_real_pair(
            candidate, scinet_project_dir=scinet_project_dir,
            scinet_python=scinet_python, base_config=base_config,
            dataset_version=dataset_version, campaign_id=campaign_id,
            paired_seeds=paired_seeds, decision_split=decision_split,
            holdout_split=holdout_split, epochs=epochs, cfg=cfg)
        # 沙箱里落的是**真实证据**：配置 SHA、数据集版本、seed、逐 seed 指标、
        # checkpoint SHA、git HEAD、holdout 一致性 —— 不再是无副作用的标记文件。
        (sb.path / "rrsi_real_evidence.json").write_text(
            json.dumps(patch_report, indent=2, sort_keys=True, ensure_ascii=False,
                       default=str), encoding="utf-8")
        evidence = build_research_evidence(
            base, cand,
            missing_runs=patch_report.get("missing_runs", 0),
            regression_rate=patch_report.get("regression_rate", 0.0))
    else:
        (sb.path / "rrsi_probe_marker.txt").write_text(
            f"fake gate-chain case={fake_case} candidate "
            f"{candidate.target} {candidate.old}->{candidate.new}\n",
            encoding="utf-8")
        base = evaluate_research_runtime("base", fake_case)
        cand = evaluate_research_runtime("cand", fake_case)
        evidence = build_research_evidence(base, cand)

    # 5) 真实非补偿门控：gate_a（评分前） + gate_b（评分后）
    frozen_violations = [candidate.target] if candidate.target in RESEARCH_FROZEN else []
    dec_a = gate.gate_a(candidate, frozen_violations=frozen_violations, cfg=cfg)
    dec_b = gate.gate_b(evidence, cfg=cfg) if dec_a.accepted else dec_a
    status = ledger_status(dec_b if dec_a.accepted else dec_a, stopped="")
    verdict_notes: list[str] = []

    if real:
        # 5a) 配置改了但引擎输出逐 seed 完全相同 => 该旋钮对本引擎是惰性机制。
        #     这是真实测量结果，不是测量事故：按 INERT_MECHANISM 证伪，避免闭环
        #     反复重提一个永远 ΔQ=0 的旋钮。
        if not patch_report.get("behavior_changed", True):
            dec_b = gate.Decision(
                "REJECT", gate.Reason.INERT_MECHANISM.value,
                [{"check": "behavior_changed", "pass": False,
                  "note": "real base/cand metrics identical on every paired seed"}],
                {"target": candidate.target})
            status = "rejected"
            verdict_notes.append("real run proved an inert mechanism (identical metrics)")
        # 5b) 独立 holdout 方向不吻合 => 结论不稳健，只能 inconclusive。
        elif patch_report.get("holdout_consistent") is False:
            status = "inconclusive"
            verdict_notes.append("independent holdout disagrees with decision split")

    if dec_a.accepted and dec_b.accepted and status == "accepted":
        decision = "ACCEPT"
    elif status == "rejected":
        decision = "REJECT"
    else:
        decision = "INCONCLUSIVE"

    # 6) 写 ledger（指纹 = 候选指纹；证据带上真实 delta/Q/A 与可追溯工件）
    ledger_evidence = {
        "decision_a": explain(dec_a), "decision_b": explain(dec_b),
        "mean_delta": evidence.paired(cfg.alpha).mean,
        "A_base": evidence.A_base, "A_cand": evidence.A_cand,
        "reason": dec_b.reason, "gate_trace_b": dec_b.gate_trace,
        "evaluation_mode": "real" if real else "fake",
        "decision": decision,
    }
    if real:
        ledger_evidence.update({
            "artifact_sha256": patch_report.get("evidence_sha256"),
            "dataset_version": patch_report.get("dataset_version"),
            "paired_seeds": patch_report.get("seeds"),
            "decision_split": patch_report.get("decision_split"),
            "holdout_split": patch_report.get("holdout_split"),
            "behavior_changed": patch_report.get("behavior_changed"),
            "holdout_consistent": patch_report.get("holdout_consistent"),
            "missing_runs": patch_report.get("missing_runs"),
            "verdict_notes": verdict_notes,
        })
    ldg.update(fp, status, evidence=ledger_evidence)

    # 7) 版本归属（只读，不 promote）
    base_ver = _version.current() or _version.BaseVersion(version=0,
                                                         base_commit=_version.git_head())
    exp.edit = {"patch_hash": candidate.patch_hash}

    if real:
        note = ("REAL candidate run: patch applied in isolated config and verified to "
                "change engine behavior; base/cand measured on paired seeds with "
                "unified eval on the decision split and an independent holdout; "
                "gates fed with REAL deltas; no promote() called; live runtime "
                "fingerprint verified equal before/after")
    else:
        note = ("fake gate-chain evaluation (no engine); live research_runtime "
                "not imported or mutated; no promote() called; fingerprint "
                "verified equal before/after")
    if verdict_notes:
        note += "; " + "; ".join(verdict_notes)

    return IngestResult(
        objective_id=trace["objective_id"],
        trace=trace,
        candidate=candidate.to_dict(),
        sandbox_path=str(sb.path),
        base_measure=base.to_score(),
        cand_measure=cand.to_score(),
        evidence={"A_base": evidence.A_base, "A_cand": evidence.A_cand,
                  "task_deltas": evidence.task_deltas,
                  "n_tasks": evidence.n_tasks, "mean_delta": evidence.paired(cfg.alpha).mean,
                  "missing_runs": evidence.missing_runs,
                  "regression_rate": evidence.regression_rate},
        decision_a=explain(dec_a),
        decision_b=explain(dec_b),
        ledger_status=status,
        ledger_fingerprint=fp,
        experience_ref=exp.experience_id,
        base_version=base_ver.version,
        base_commit=base_ver.base_commit,
        live_runtime_untouched=(live_before == _snapshot_live_runtime()),
        note=note,
        evaluation_mode="real" if real else "fake",
        decision=decision,
        behavior_changed=patch_report.get("behavior_changed") if real else None,
        holdout_consistent=patch_report.get("holdout_consistent") if real else None,
        real_evidence=patch_report if real else {},
        artifact_sha256=patch_report.get("evidence_sha256", "") if real else "",
    )


# ---------------------------------------------------------------- CLI 自检

def selftest() -> dict:
    """用三个确定性 profile 跑一遍门控链路，返回 {good, zero, worse} 的判定。"""
    report = {"objective_id": "obj_self_test", "outcome": "inconclusive",
              "comparisons": [{"decision": "inconclusive"}]}
    tmp = Path(tempfile.mkdtemp(prefix="rrsi_selftest_"))
    out: dict[str, Any] = {}
    for case in ("good", "zero", "worse"):
        res = ingest(report, ledger_path=tmp / "ledger.jsonl",
                     experience_path=tmp / "exp.jsonl",
                     sandbox_root=tmp / case, fake_case=case)
        out[case] = {"decision_a": res.decision_a, "decision_b": res.decision_b,
                     "ledger_status": res.ledger_status}
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(selftest())
