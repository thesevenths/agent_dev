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
from .proposer import falsification_candidate, zero_change_candidate, Candidate
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


def positive_path(verbose: bool = True) -> dict:
    """正向路径自检：ACCEPT → PROMOTE → LAND → HOLDOUT PASS，以及 REJECT → 不晋升 → 回滚。

    零模型成本：用 fake 的 ``version`` 模块避免真实写 ``base_state`` / 动 git；落地全部打到
    临时 root，绝不碰生产 ``.env`` / ``supervisor.py``。这补上了 ``auto --fake`` 故意跳过
    promote/land 那段盲区 —— 真实 6 轮跑从没晋升过，所以「晋升→落地→换基线重测」这条链
    从未被验证过。证明的是**引擎的正向闭环真的能走通**，不是证明某次改进有效。
    """
    from types import SimpleNamespace
    from . import autopromote, landing
    from .landing import wrap_code_patch, CODE_PATCHES, land_code, revert_code
    from .io_util import sha256_text

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    # 临时 root：把生产关键文件拷过去，落地只发生在 tmp，生产纹丝不动。
    # 必须带上冻结校验对象（agents/planutil/runlog/graph），否则 frozen.verify(tmp)
    # 会因文件缺失误报 symbol_missing。
    tmp = PATHS.experiments / "selftest_pp" / now_stamp()
    tmp.mkdir(parents=True, exist_ok=True)
    for _f in (".env", "prompt.py", "agents.py", "planutil.py", "runlog.py",
               "graph.py", "supervisor.py"):
        _src = PATHS.root / _f
        if _src.exists():
            (tmp / _f).write_text(_src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    # fake version：promote / restore 只记内存，不碰真实状态 / git
    class FakeVersion:
        def __init__(self):
            self.cur = SimpleNamespace(version=1, tag="base/v1")
            self.promoted: list[str] = []
            self.restored: list[str] = []

        def current(self):
            return self.cur

        def promote(self, evidence, operator, note="", require_operator=False):
            nxt = self.cur.version + 1
            self.cur = SimpleNamespace(version=nxt, tag=f"base/v{nxt}")
            self.promoted.append(self.cur.tag)
            return self.cur

        def restore(self, prev, note="", operator="auto:adaptive"):
            self.cur = prev
            self.restored.append(prev.tag)
            return prev

        def refresh_snapshot(self, ver=None, base_commit=None):
            return ver or self.cur

        def git_head(self):
            return "fakehead"

    fv = FakeVersion()
    real_version = autopromote.version
    autopromote.version = fv
    try:
        cfg = CFG

        # ---- 1. 好候选：gate 接受 + holdout 正增益 → 判定允许晋升 ----
        good = Candidate(candidate_id="pp-good", target="AGENT_PARSE_TRIES", level=1,
                         old="3", new="5", hypothesis="正向路径自检：好候选",
                         taxonomy="PLANNING_FAILURE",
                         patch={"kind": "env", "key": "AGENT_PARSE_TRIES", "old": "3", "new": "5"})
        _pairs = [SimpleNamespace(delta=10.0), SimpleNamespace(delta=8.0), SimpleNamespace(delta=12.0)]
        _per = {"T1": 10.0, "T2": 8.0, "T3": 12.0}
        _holdout = SimpleNamespace(pairs=_pairs, per_task=_per)
        pol = autopromote.evaluate(cand=good, dec_b_accepted=True, stopped="",
                                   holdout_cmp=_holdout, regression=None,
                                   frozen_violations=[], promotes_done=0, cfg=cfg)
        check("pp_good_policy_allowed", pol.allowed, f"blocking={pol.blocking}")
        check("pp_good_holdout_evidence",
              any(c.name == "holdout_evidence" and c.passed for c in pol.checks))

        # ---- 2. PROMOTE → LAND（打到 tmp）：验证生产内容真的会变 ----
        prev_base = fv.current()  # base/v1，晋升前的基线
        rec = autopromote.promote_and_land(good, policy=pol, holdout_summary={"mean_delta": 10.0},
                                          cfg=cfg, root=tmp)
        _env = (tmp / ".env").read_text(encoding="utf-8", errors="replace")
        check("pp_land_wrote_env", "AGENT_PARSE_TRIES=5" in _env, "托管块未写入临时 .env")
        check("pp_version_promoted", fv.promoted == ["base/v2"], f"promoted={fv.promoted}")

        # ---- 3. 换基线后 holdout 重测：真的更好 → 不回滚 ----
        chk = autopromote.verify_after_promote(
            holdout_prev_mean=70.0, holdout_new_mean=80.0, tol=cfg.holdout_tol,
            prev_tag="base/v1", prev_by_task={"T1": 70, "T2": 70, "T3": 70},
            new_by_task={"T1": 80, "T2": 80, "T3": 80}, cfg=cfg)
        check("pp_holdout_pass", bool(chk.get("ok")), f"drift={chk.get('drift')}")
        check("pp_no_rollback_on_pass",
              "AGENT_PARSE_TRIES=5" in (tmp / ".env").read_text(encoding="utf-8", errors="replace"))

        # ---- 4. 坏候选：holdout 负增益 → 判定不允许晋升（REJECT 阻断 promote）----
        worse = Candidate(candidate_id="pp-worse", target="AGENT_PARSE_TRIES", level=1,
                          old="3", new="2", hypothesis="正向路径自检：坏候选",
                          taxonomy="PLANNING_FAILURE",
                          patch={"kind": "env", "key": "AGENT_PARSE_TRIES", "old": "3", "new": "2"})
        _hb = SimpleNamespace(pairs=[SimpleNamespace(delta=-10.0)] * 3,
                              per_task={"T1": -10.0, "T2": -10.0, "T3": -10.0})
        pol_bad = autopromote.evaluate(cand=worse, dec_b_accepted=True, stopped="",
                                       holdout_cmp=_hb, regression=None,
                                       frozen_violations=[], promotes_done=0, cfg=cfg)
        check("pp_worse_policy_blocked", not pol_bad.allowed, f"blocking={pol_bad.blocking}")
        _raised = False
        try:
            autopromote.promote_and_land(worse, policy=pol_bad, holdout_summary={}, cfg=cfg, root=tmp)
        except RuntimeError:
            _raised = True
        check("pp_rejected_blocks_promote", _raised, "被拒候选不应进入 promote")

        # ---- 5. 晋升后换基线退步 → 自动回滚（rollback）----
        chk2 = autopromote.verify_after_promote(
            holdout_prev_mean=70.0, holdout_new_mean=50.0, tol=cfg.holdout_tol,
            prev_tag="base/v1", prev_by_task={"T1": 70, "T2": 70, "T3": 70},
            new_by_task={"T1": 50, "T2": 50, "T3": 50}, cfg=cfg)
        check("pp_holdout_regress_detected", not chk2.get("ok"), f"drift={chk2.get('drift')}")
        autopromote.rollback(prev_base, reason="自检：换基线后退步", cfg=cfg, root=tmp)
        _env_after = (tmp / ".env").read_text(encoding="utf-8", errors="replace")
        check("pp_rollback_removed_landing", "AGENT_PARSE_TRIES=5" not in _env_after,
              "回滚后托管块仍在")
        check("pp_version_restored", fv.restored == ["base/v1"], f"restored={fv.restored}")

        # ---- 6. 代码级候选也能 land + rollback（结构性编辑空间可用性）----
        _name = "selftest_demo_guard"
        _insert = ("def _rrsi_selftest_demo():  # 自检 fixture，不构成行为变更\n"
                   "    return None")
        CODE_PATCHES[_name] = {"file": "supervisor.py",
                               "anchor": "def supervisor(state: AgentState) -> Dict[str, Any]:",
                               "insert": _insert}
        _wrapped = wrap_code_patch(_name, _insert)
        _ph = sha256_text(_wrapped)[:16]
        _sup = PATHS.root / "supervisor.py"
        if _sup.exists():
            (tmp / "supervisor.py").write_text(
                _sup.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            land_code(_name, "supervisor.py",
                      "def supervisor(state: AgentState) -> Dict[str, Any]:", _insert,
                      version_tag="base/v2", patch_hash=_ph, root=tmp)
            _sup_after = (tmp / "supervisor.py").read_text(encoding="utf-8", errors="replace")
            check("pp_code_landed", _wrapped in _sup_after, "代码块未插入临时 supervisor.py")
            _ok_rev = revert_code("supervisor.py", _wrapped, root=tmp)
            check("pp_code_rolled_back",
                  _ok_rev and _wrapped not in (tmp / "supervisor.py").read_text(
                      encoding="utf-8", errors="replace"))
            # rollback_to 也能按 hash 反查并拆掉代码块
            landing.rollback_to("base/v1", root=tmp)
            check("pp_code_rollback_to", _wrapped not in (tmp / "supervisor.py").read_text(
                encoding="utf-8", errors="replace"))
        else:
            check("pp_code_landed", False, "supervisor.py 不存在，跳过")
        # 清理 fixture 注册，避免污染 landing.CODE_PATCHES
        CODE_PATCHES.pop(_name, None)
    finally:
        autopromote.version = real_version
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    passed = all(c["pass"] for c in checks)
    return {"passed": passed, "checks": checks, "stage": CFG.stage}


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

    # ---- 正向路径：ACCEPT → PROMOTE → LAND → HOLDOUT PASS，以及 REJECT → 不晋升 → 回滚 ----
    # 真实 6 轮跑从没晋升过（0 次 promote），而 ``auto --fake`` 又故意跳过 promote/land，
    # 所以晋升→落地→换基线重测这条链从未被验证。这里用 fake version + 临时 root 零成本补上。
    pp = positive_path(verbose=verbose)
    for c in pp["checks"]:
        checks.append(c)

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
