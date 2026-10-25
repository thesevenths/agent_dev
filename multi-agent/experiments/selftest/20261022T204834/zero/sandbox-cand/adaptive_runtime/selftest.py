"""端到端自检：两个阴性对照 + 一个阳性对照 + 冻结断言。

这是 P0 唯一的强制通过项（V3 §16）。
判 ACCEPT 说明三种可能之一：δ 错了、judge 有偏、base/candidate 路径不对称 ——
这三种都无法靠后续实验补救。

用法：
    python -m adaptive_runtime.run_adaptive selftest
"""
from __future__ import annotations

import os
from pathlib import Path

from . import frozen, token_probe
from .evaluator import score_trace
from .gate import Reason, gate_b
from .io_util import now_stamp, write_json
from .probe import cleanup, probe_candidate
from .proposer import falsification_candidate, zero_change_candidate
from .settings import CFG, PATHS
from .trace_export import export_from_log

#: 自检至少需要 2 个**不同**任务：gate_b 的 min_tasks 规则要求 n_tasks>=2 才能做 t 检验。
#: 自检不能给自己开后门 —— 它必须走和生产完全一致的判定路径。
_FAKE_TASKS = [
    {"task_id": "SELF-001", "query": "自检任务 A：市场分析并生成报告"},
    {"task_id": "SELF-002", "query": "自检任务 B：行业对比并生成报告"},
]
_FAKE_TASK = _FAKE_TASKS[0]


def _run_case(name: str, base_profile: str, cand_profile: str,
              workdir: Path) -> dict:
    cand = zero_change_candidate()
    outcome = probe_candidate(
        cand, _FAKE_TASKS, workdir / name,
        n=2, stage1_reps=1, stage1_tasks=2, keep_sandbox=False,
        fake_profiles={"base": base_profile, "cand": cand_profile})
    cmp = outcome.comparison
    assert cmp is not None, "probe 未产生 comparison"
    ev = cmp.evidence()
    decision = gate_b(ev)
    cleanup(outcome)
    return {"case": name, "base": base_profile, "cand": cand_profile,
            "n": ev.paired().n, "mean_delta": round(ev.paired().mean, 4),
            "delta_crit": round(ev.paired().delta_crit, 4),
            "decision": decision.decision, "reason": decision.reason,
            "gate_trace": decision.gate_trace}


def selftest(verbose: bool = True) -> dict:
    PATHS.ensure()
    root = PATHS.experiments / "selftest" / now_stamp()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    # ---- 对照 1：零改动必须 WITHIN_NOISE ----
    c1 = _run_case("zero", "base", "zero", root)
    results.append(c1)
    check("negative_control_zero", c1["decision"] == "REJECT" and c1["reason"] == Reason.WITHIN_NOISE.value,
          f"{c1['decision']}/{c1['reason']} mean={c1['mean_delta']} δ={c1['delta_crit']}")

    # ---- 对照 2：故意变差必须被判负 ----
    c2 = _run_case("worse", "base", "worse", root)
    results.append(c2)
    # CONTRACT_FLOOR 比 REGRESSION 更具体（它先命中契约地板），同样是「判负」
    ok2 = c2["decision"] == "REJECT" and c2["reason"] in (
        Reason.REGRESSION.value, Reason.WITHIN_NOISE.value, Reason.CONTRACT_FLOOR.value)
    check("negative_control_worse", ok2, f"{c2['decision']}/{c2['reason']} mean={c2['mean_delta']}")

    # ---- 敏感度：base vs base 也必须 WITHIN_NOISE ----
    c3 = _run_case("base_vs_base", "base", "base", root)
    results.append(c3)
    check("sensitivity_base_vs_base",
          c3["decision"] == "REJECT" and c3["reason"] == Reason.WITHIN_NOISE.value,
          f"{c3['decision']}/{c3['reason']}")

    # ---- 阳性对照：真实改进必须能被接受（否则尺子只会说 NO）----
    c4 = _run_case("good", "base", "good", root)
    results.append(c4)
    check("positive_control_good", c4["decision"] == "ACCEPT",
          f"{c4['decision']}/{c4['reason']} mean={c4['mean_delta']}")

    # ---- 冻结：文件侧 ----
    if not (PATHS.frozen_manifest).exists():
        frozen.freeze()
    v = frozen.verify()
    check("frozen_files_intact", not v, "; ".join(str(x) for x in v[:3]))

    # ---- 冻结：env 侧（改 .env/环境变量必须被抓到）----
    key = "AGENT_CRITIC_MIN_LEN"
    old = os.environ.get(key)
    try:
        os.environ[key] = "1"
        v2 = frozen.verify()
        hit = [x for x in v2 if x.kind == "env_changed" and x.target == key]
        check("frozen_env_detects_override", bool(hit), f"violations={[str(x) for x in v2[:3]]}")
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old

    # ---- token 埋点可安装 ----
    installed = token_probe.install(memory_only=True)
    check("token_probe_installable", installed or token_probe.is_installed(), "依赖 langchain-openai")

    # ---- 一个真正的「零改动」候选：env 相同 ----
    z = zero_change_candidate()
    check("zero_candidate_has_no_env", z.env_overrides() == {}, str(z.env_overrides()))
    f = falsification_candidate()
    check("falsification_lowers_budget", f.new == "12" and f.old == "60")

    passed = all(c["pass"] for c in checks)
    report = {"passed": passed, "checks": checks, "cases": results,
              "root": str(root), "stage": CFG.stage}
    write_json(root / "selftest.json", report)

    if verbose:
        print()
        print(f"  自检{'通过' if passed else '未通过'}：{sum(c['pass'] for c in checks)}/{len(checks)}")
        print(f"  产物：{root / 'selftest.json'}")
    return report


def main() -> int:
    return 0 if selftest(verbose=True)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
