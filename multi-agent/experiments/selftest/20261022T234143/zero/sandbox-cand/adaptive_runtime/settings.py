"""阈值、路径与阶段开关（唯一真源）。

所有可调参数都集中在这里，且全部可以用 ``ADAPTIVE_*`` 环境变量覆盖。
原则：代码里不出现魔法数字，阈值改动必须能在 meta.json 里被追溯。

注意：本模块**不 import 项目根目录的 config.py**，也不 import langchain，
因此纯 stdlib 即可导入 —— 这让单元测试不依赖项目 venv。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------- 路径

#: 项目根目录（multi-agent/），本包位于 <root>/adaptive_runtime/
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    experiments: Path = ROOT / "experiments"
    runs: Path = ROOT / "experiments" / "runs"
    tasks: Path = ROOT / "experiments" / "tasks"
    worktrees: Path = ROOT / "experiments" / "worktrees"
    quarantine: Path = ROOT / "experiments" / "quarantine"
    reports: Path = ROOT / "experiments" / "reports"
    ledger: Path = ROOT / "experiments" / "ledger"
    experience: Path = ROOT / "experiments" / "experience"
    calibration: Path = ROOT / "experiments" / "calibration.json"
    frozen_manifest: Path = ROOT / "experiments" / "frozen_manifest.json"
    usage: Path = ROOT / "experiments" / "usage"
    #: Regression 集：只用于否决、永不参与演化的固定任务集（V3 §13.4）
    regression: Path = ROOT / "experiments" / "regression"
    log_dir: Path = ROOT / "log"

    def ensure(self) -> None:
        for p in (self.experiments, self.runs, self.tasks, self.worktrees,
                  self.quarantine, self.reports, self.ledger, self.experience,
                  self.usage, self.regression):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# ---------------------------------------------------------------- 工具

def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v is not None and v.strip() != "" else default


def _env_float(key: str, default: float, lo: float | None = None,
               hi: float | None = None) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    if lo is not None and v < lo:
        return lo
    if hi is not None and v > hi:
        return hi
    return v


def _env_int(key: str, default: int, lo: int | None = None,
             hi: int | None = None) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(float(raw))
    except ValueError:
        return default
    if lo is not None and v < lo:
        return lo
    if hi is not None and v > hi:
        return hi
    return v


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# ---------------------------------------------------------------- 阶段

class Stage:
    """阶段标识。P1/P2 的能力在代码里已就位，但由阶段开关关闭。"""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


# ---------------------------------------------------------------- 阈值

@dataclass
class Settings:
    """所有判定阈值。

    取值依据见 docs/adaptive_agent_runtime_final_plan.md §9.3 与附录 A。
    """

    # ---- 阶段 ----
    stage: str = field(default_factory=lambda: _env_str("ADAPTIVE_STAGE", Stage.P0).upper())

    # ---- 主目标：L1 合成分 ----
    #: 契约达成率地板。语义是「相对 base 不能退步」，不是绝对门槛：
    #: 实际判定用 floor = min(A_FLOOR, A_base) - A_TOL（见 gate.py 说明）。
    a_floor: float = field(default_factory=lambda: _env_float("ADAPTIVE_A_FLOOR", 0.85, 0.0, 1.0))
    a_tol: float = field(default_factory=lambda: _env_float("ADAPTIVE_A_TOL", 0.05, 0.0, 1.0))
    #: Q 的三项权重（必须和为 1）
    w_a: float = field(default_factory=lambda: _env_float("ADAPTIVE_W_A", 0.50, 0.0, 1.0))
    w_e: float = field(default_factory=lambda: _env_float("ADAPTIVE_W_E", 0.30, 0.0, 1.0))
    w_c: float = field(default_factory=lambda: _env_float("ADAPTIVE_W_C", 0.20, 0.0, 1.0))

    # ---- 反刷分（reward hacking 防线）----
    #: 低于这个字节数的 .md/.txt 不算「报告」，按「无产物」处理（E=0、C 打折）。
    #: 起因：一次真实冒烟 run 里把一句闲聊存成 md，被判 E=0/C=1 → Q=70。
    min_report_bytes: int = field(default_factory=lambda: _env_int("ADAPTIVE_MIN_REPORT_BYTES", 1200, 0, 10_000_000))
    #: 产物结构分在 C 里的占比：C = c0 * ((1-s) + s*structure)
    structure_share: float = field(default_factory=lambda: _env_float("ADAPTIVE_STRUCT_SHARE", 0.5, 0.0, 1.0))
    #: cand 计划步数 / base 计划步数 低于此比例，且 A 反而上升 → 判「靠削减计划提分」
    plan_undercut_ratio: float = field(default_factory=lambda: _env_float("ADAPTIVE_PLAN_UNDERCUT", 0.6, 0.0, 1.0))

    # ---- 质量缺陷通道（诊断层的第二条信号来源）----
    #: 证据率低于此值 → 产出 EVIDENCE_FAILURE。诊断层若只看「跑崩」，
    #: 就永远发现不了「跑完了但每个数字都没来源」这类质量缺陷 —— 而后者往往是 Q 最大失分项。
    evidence_floor: float = field(default_factory=lambda: _env_float("ADAPTIVE_EVIDENCE_FLOOR", 0.30, 0.0, 1.0))
    #: 完整性低于此值 → 产出 COMPLETENESS_FAILURE。
    completeness_floor: float = field(default_factory=lambda: _env_float("ADAPTIVE_COMPLETENESS_FLOOR", 0.70, 0.0, 1.0))
    #: 关键数字总数低于此值时不判证据缺陷（样本太少，E 的方差不可信）。
    evidence_min_claims: int = field(default_factory=lambda: _env_int("ADAPTIVE_EVIDENCE_MIN_CLAIMS", 8, 0, 100000))

    # ---- 统计 ----
    n_default: int = field(default_factory=lambda: _env_int("ADAPTIVE_N", 8, 1, 200))
    #: 做 t 检验所需的最少**不同任务**数。低于它只能判「证据不足」，不能判「有效」。
    #: 设为 2 是数学下限（df>=1）；实际有意义的检出能力要看 required_n()。
    min_tasks_for_t: int = field(default_factory=lambda: _env_int("ADAPTIVE_MIN_TASKS", 2, 2, 200))
    k_calib: int = field(default_factory=lambda: _env_int("ADAPTIVE_K_CALIB", 6, 2, 30))
    alpha: float = field(default_factory=lambda: _env_float("ADAPTIVE_ALPHA", 0.05, 0.001, 0.4))
    power: float = field(default_factory=lambda: _env_float("ADAPTIVE_POWER", 0.80, 0.5, 0.99))
    #: 配对差值 sd 的目标上限（用于 calibrate 通过判据与 required_n 估算）。
    #:
    #: 为什么从 6.0 调整为 15.0：同一任务（REP-001）在相同代码上连续执行两次，Q 分别为
    #: 73.02 与 48.57 —— 单次运行的波动达 24 分。配对差值会因两臂共享任务难度而
    #: 部分抵消，但无法压到 6。沿用 6 会使 required_n 低估约 (15/6)² ≈ 6 倍，
    #: 即规划时假定 n=8 足够，实际需要 n≈40。
    #: 15.0 是「单 run σ≈24 + 配对部分抵消」的保守折中取值，**必须由 calibrate 用
    #: 真实 null 差值的实测 sd 替换**（calibration.json 的 sd_diff_null）。
    sd_diff_target: float = field(default_factory=lambda: _env_float("ADAPTIVE_SD_TARGET", 15.0, 0.0, 100.0))
    #: 单任务「显著变差」的判定阈值（Q 百分点）
    regression_min_points: float = field(
        default_factory=lambda: _env_float("ADAPTIVE_REGRESSION_MIN", 5.0, 0.1, 100.0))

    # ---- 次要约束 ----
    r_max: float = field(default_factory=lambda: _env_float("ADAPTIVE_R_MAX", 0.15, 0.0, 1.0))
    c_save_min: float = field(default_factory=lambda: _env_float("ADAPTIVE_C_SAVE_MIN", 0.03, 0.0, 1.0))
    beta0: float = field(default_factory=lambda: _env_float("ADAPTIVE_BETA0", 0.05, 0.0, 1.0))
    beta1: float = field(default_factory=lambda: _env_float("ADAPTIVE_BETA1", 0.015, 0.0, 1.0))
    judge_unstable_max: float = field(
        default_factory=lambda: _env_float("ADAPTIVE_JUDGE_UNSTABLE_MAX", 0.20, 0.0, 1.0))

    # ---- 预算 ----
    #: 依据实测重定：本仓库一个真实研报 run ≈ 60 万 token（58 次调用、18+ 分钟）。
    #: 原值 800 万 / 400 万 会使 n=8 的候选（估算 980 万）全部被 Gate A 以 COST_CAP 拒绝，
    #: 外部表现为「闭环从不 accept」，实际原因是预算配置低于单轮真实成本。
    #: 现按「一轮 n=8 ≈ 980 万」预留约 1.5 倍余量；可用 ADAPTIVE_COST_CAP_* 覆盖。
    cost_cap_round: int = field(
        default_factory=lambda: _env_int("ADAPTIVE_COST_CAP_ROUND", 15_000_000, 1, 10 ** 9))
    cost_cap_cand: int = field(
        default_factory=lambda: _env_int("ADAPTIVE_COST_CAP_CAND", 12_000_000, 1, 10 ** 9))

    # ---- 演化 ----
    edit_budget_p0: int = field(default_factory=lambda: _env_int("ADAPTIVE_EDIT_BUDGET", 1, 1, 5))
    #: 允许生成候选的最高层级：1=参数, 2=提示, 3=结构
    max_edit_level: int = field(default_factory=lambda: _env_int("ADAPTIVE_MAX_EDIT_LEVEL", 1, 1, 3))
    #: 单轮候选数（V3 定为 1）
    candidates_per_round: int = field(
        default_factory=lambda: _env_int("ADAPTIVE_CANDIDATES", 1, 1, 5))

    # ---- 诊断 ----
    unknown_max: float = field(default_factory=lambda: _env_float("ADAPTIVE_UNKNOWN_MAX", 0.40, 0.0, 1.0))

    # ---- 经验 ----
    exp_min_confidence: float = field(
        default_factory=lambda: _env_float("ADAPTIVE_EXP_MIN_CONF", 0.50, 0.0, 1.0))
    exp_retrieve_top: int = field(default_factory=lambda: _env_int("ADAPTIVE_EXP_TOPK", 3, 1, 10))

    # ---- L2 rubric ----
    l2_enabled: bool = field(default_factory=lambda: _env_bool("ADAPTIVE_L2", False))
    judge_temperature: float = field(
        default_factory=lambda: _env_float("ADAPTIVE_JUDGE_TEMP", 0.0, 0.0, 2.0))

    # ---- 运行 ----
    #: 假跑模式：不调用 LLM，用确定性轨迹验证整条流水线（自检专用）
    fake_mode: bool = field(default_factory=lambda: _env_bool("ADAPTIVE_FAKE", False))
    #: 使用真实 git worktree（默认关闭：沙箱复制更快且不污染主仓库）
    use_git_worktree: bool = field(
        default_factory=lambda: _env_bool("ADAPTIVE_USE_GIT_WORKTREE", False))
    #: 单次真实 run 的软超时（秒）
    run_timeout: int = field(default_factory=lambda: _env_int("ADAPTIVE_RUN_TIMEOUT", 1800, 30, 86400))

    # ------------------------------------------------------------ 派生

    def normalized_weights(self) -> tuple[float, float, float]:
        total = self.w_a + self.w_e + self.w_c
        if total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return (self.w_a / total, self.w_e / total, self.w_c / total)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CFG = Settings()

#: 权重和不为 1 时归一化并告警（只告警一次，避免每轮刷屏）
_WEIGHT_WARNED = False


def validate() -> list[str]:
    """返回配置问题列表（空列表表示通过）。CLI 启动时调用。"""
    global _WEIGHT_WARNED
    problems: list[str] = []
    total = CFG.w_a + CFG.w_e + CFG.w_c
    if abs(total - 1.0) > 1e-6:
        if not _WEIGHT_WARNED:
            _WEIGHT_WARNED = True
        problems.append(f"权重和={total:.4f}≠1，将按比例归一化")
    if CFG.stage not in (Stage.P0, Stage.P1, Stage.P2):
        problems.append(f"未知阶段 {CFG.stage!r}，回落 P0")
        CFG.stage = Stage.P0
    if CFG.cost_cap_cand > CFG.cost_cap_round:
        problems.append("单候选预算大于单轮预算，按单轮预算截断")
        CFG.cost_cap_cand = CFG.cost_cap_round

    # ---- 检出能力提示：n 过小时，δ 会大到任何真实改进都无法过门 ----
    # 该情形不属于配置错误，而是本轮不具备检出能力；须在执行前明确告知，避免数小时无效运行。
    try:
        from .stats import t_crit
        import math as _math
        sd = CFG.sd_diff_target
        if CFG.n_default >= 2 and sd > 0:
            delta_crit = t_crit(CFG.n_default - 1, CFG.alpha) * sd / _math.sqrt(CFG.n_default)
            if delta_crit > 20.0:
                problems.append(
                    f"ADAPTIVE_N={CFG.n_default} 时噪声门限 δ≈{delta_crit:.1f} 分"
                    f"（sd 取 {sd:.1f}）—— 小于该值的改进无法检出；"
                    f"要检出 ΔQ≈10 分建议 n≥{_required_n_hint(sd, 10.0)}")
        else:
            problems.append("ADAPTIVE_N<2，无法做配对检验")
    except Exception:
        pass
    return problems


def _required_n_hint(sd: float, effect: float) -> int:
    from .stats import required_n
    try:
        return int(required_n(sd, effect, CFG.power, CFG.alpha))
    except Exception:
        return 0
