"""演化层：候选生成（P0 只开放 L1 参数，L2/L3 由 max_edit_level 关闭）。

搜索空间取自**真实参数**，而非规格草案（V2）中的占位名称（如 MAX_REPLAN / REVIEW_THRESHOLD 等）。
判据只有一条：**改它会不会改变判分口径**。会 → 冻结；不会 → 可搜。

两条来自实测的结论已写入搜索空间：
- ``AGENT_CRITIC_MIN_LEN`` 是惰性机制（默认 4 字符，注释明确「不是质量门槛」）→ 不搜；
- ``AGENT_MAX_ITERATIONS_CODE`` 已是全局 15 的 4 倍仍产生 28 次 recursion 失败 →
  只作为**证伪候选**，用来把「调大预算就能修好」这个假设彻底否掉。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

from .budget import estimate_round_tokens
from .diagnosis import Taxonomy
from .io_util import new_id, now_iso, sha256_text
from .ledger import Ledger, fingerprint
from .settings import CFG, Settings

#: 真实搜索空间。level: 1=参数 2=提示 3=结构
SEARCH_SPACE: dict[str, dict] = {
    "AGENT_MAX_ITERATIONS_CODE": {
        "values": ["60", "80", "100"], "level": 1,
        "taxonomies": (Taxonomy.COMPLETION_FAILURE, Taxonomy.EXECUTION_FAILURE),
        "rationale": "已是全局 15 的 4 倍仍 28 次失败 → 预期无效，仅作证伪",
        "expectation": "falsify",
    },
    "AGENT_MAX_ITERATIONS": {
        "values": ["15", "25", "40"], "level": 1,
        "taxonomies": (Taxonomy.COMPLETION_FAILURE, Taxonomy.EXECUTION_FAILURE),
        "rationale": "爬虫类任务的全局 ReAct 预算；未单独配置时落在这里",
        "expectation": "improve",
    },
    "AGENT_PARSE_TRIES": {
        "values": ["2", "3", "5"], "level": 1,
        "taxonomies": (Taxonomy.EXECUTION_FAILURE, Taxonomy.COMPLETION_FAILURE),
        "rationale": "结构化解析重试次数，影响计划/路由的稳定性",
        "expectation": "improve",
    },
    "AGENT_REPLAN_ARTIFACT_MAX": {
        "values": ["2", "3", "5"], "level": 1,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "产物缺失时触发 replan 的门槛（弱效应）",
        "expectation": "weak",
    },
    "AGENT_REPLAN_NOOP_MAX": {
        "values": ["2", "3"], "level": 3,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "改它等于改控制流，P2 才允许", "expectation": "structure",
    },
    "AGENT_REPLAN_OBS_WINDOW": {
        "values": ["20", "10"], "level": 3,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "观测窗口，改它等于改控制流", "expectation": "structure",
    },
    "AGENT_PLAN_GROW_MAX": {
        "values": ["1", "2"], "level": 3,
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "rationale": "计划增长上限，改它等于改控制流", "expectation": "structure",
    },
}

#: ---- L2 提示层补丁（P1，由 ``ADAPTIVE_MAX_EDIT_LEVEL`` 打开）----
#:
#: 为什么是「模板」而不是「让 LLM 现写新提示」：L2 的验收口径是
#: **1 处 + ≤15 行 diff**（方案 §10.2）。模板让改动可枚举、可 diff、可回滚，
#: 并且能把「同一位置同一意图只跑一次」做成硬约束（避免反复重写同一段提示）。
#: LLM 现写会把 diff 大小、锚点唯一性、泄漏三件事全变成概率事件 —— 那是 P2 的事。
#:
#: 每条 = 一条**可证伪**假设：某个真实失败模式由某段提示缺失导致。
#: ``anchor`` 必须在目标文件里**恰好出现一次**，否则 patch 直接失败（不静默跳过）。
PROMPT_PATCHES: dict[str, dict] = {
    "prompt.py::coder.stop-on-repeat": {
        "level": 2,
        "values": ["on"],
        "taxonomies": (Taxonomy.COMPLETION_FAILURE, Taxonomy.EXECUTION_FAILURE),
        "anchor": "- WRITE-ONCE, RUN-ONCE:",
        "insert": (
            "- REPEAT-ERROR HARD STOP: if two consecutive executions fail with the SAME error\n"
            "  (same exception type or same message), do NOT try a third variation. Stop, and\n"
            "  either (a) fall back to a simpler approach that needs no failing capability, or\n"
            "  (b) emit your mandatory final summary explaining precisely what is blocked.\n"
            "  Iterating on a deterministic error only burns the ReAct budget.\n"
        ),
        "rationale": "recursion 类失败占 step-failure 的 78%，提示缺少「同一错误不得重试」的硬约束",
        "expectation": "improve",
    },
    "prompt.py::coder.save-skeleton-first": {
        "level": 2,
        "values": ["on"],
        "taxonomies": (Taxonomy.EXECUTION_FAILURE, Taxonomy.COMPLETION_FAILURE),
        "anchor": "- DELIVERABLE DONE = STOP POLISHING:",
        "insert": (
            "- SAVE SKELETON FIRST: before writing a long report or a large script, write a minimal\n"
            "  complete version to disk (headings + one real data table), verify the file exists,\n"
            "  then extend it in place. A truncated generation that leaves NO file on disk scores 0;\n"
            "  a short file that is on disk already satisfies the deliverable contract.\n"
        ),
        "rationale": "truncated 类失败的共同后果是「无产物」，而 contract 只要求产物存在且非空",
        "expectation": "improve",
    },
    "prompt.py::supervisor.keep-done-steps": {
        "level": 2,
        "values": ["on"],
        "taxonomies": (Taxonomy.PLANNING_FAILURE,),
        "anchor": "Your Core Responsibilities:",
        "insert": (
            "REPLAN DISCIPLINE: when you replan, NEVER restate or reopen a step whose status is\n"
            "already \"done\". Re-emit the full plan but carry completed steps through verbatim, and\n"
            "only add/modify the remaining ones. A replan that flips a done step back to pending\n"
            "re-runs finished work and exhausts the budget.\n\n"
        ),
        "rationale": "replan 空转（_REPLAN_NOOP_MAX / PLAN_GROW_MAX 的存在本身）说明计划状态机在往回退",
        "expectation": "improve",
    },
    "prompt.py::coder.require-sources": {
        "level": 2,
        "values": ["on"],
        "taxonomies": (Taxonomy.EVIDENCE_FAILURE, Taxonomy.REVIEW_FAILURE),
        "anchor": "- When generating data analysis reports, follow these guidelines:",
        "insert": (
            "- SOURCE EVERY NUMBER: every quantitative claim in the report (size, growth rate,\n"
            "  margin, price, count, date) MUST be immediately followed by its source — either an\n"
            "  inline URL, a markdown link, or a 「（来源：X）」 / 「(Source: X)」 marker on the SAME\n"
            "  paragraph. If you computed a figure yourself from other data, write the derivation\n"
            "  source (e.g. 「（来源：由表 2 计算）」). Numbers without a source are treated as unverified\n"
            "  and lower the evidence score, so do not pad the text with bare figures.\n"
        ),
        "rationale": "真实 run 的 E 只有 0.10（268 个关键数字仅 27 个带来源），证据率是 Q 的最大失分项",
        "expectation": "improve",
    },
}

#: 结构层的**代码默认值**，只当「现行值读不到」时的保底起点。
#: 注意：它不是当前生效值。落地（landing）之后真实起点已经变了，而候选必须从
#: **现行值**开始搜索，否则会出现两种错：
#: (a) 生产已是 25 却反复提「15→25」⇒ cand 臂与基线完全一致，ΔQ 必然≈0，
#:     台账把「已上线的改进」记成「无效假设」；
#: (b) 改动被回滚后起点仍占位在落地值上 ⇒ 搜索方向反了。
#: 因此搜索时必须走 ``current_value()``。
CURRENT_DEFAULTS: dict[str, str] = {
    "AGENT_MAX_ITERATIONS_CODE": "60",
    "AGENT_MAX_ITERATIONS": "15",
    "AGENT_PARSE_TRIES": "3",
    "AGENT_REPLAN_ARTIFACT_MAX": "3",
    "AGENT_REPLAN_NOOP_MAX": "2",
    "AGENT_REPLAN_OBS_WINDOW": "20",
    "AGENT_PLAN_GROW_MAX": "1",
}

#: L2 补丁并入统一搜索空间：这样 propose / 探索 / 剪枝 / gate 全部不用分叉。
#: ``values=["on"]`` 是刻意的 —— 提示补丁是「加/不加」的二值开关，没有中间档。
for _name, _spec in PROMPT_PATCHES.items():
    SEARCH_SPACE[_name] = {**_spec, "is_prompt": True}

#: L2 候选最多允许的 diff 行数（方案 §10.2：1 处 + ≤15 行）
MAX_PROMPT_DIFF_LINES = 15


def _check_prompt_patches() -> list[str]:
    """导入期自检：锚点必须在目标文件里**恰好出现一次**，diff 必须 ≤15 行。

    为什么要在导入期就查：如果锚点写错，patch 会静默不生效，于是候选的 ΔQ≈0，
    被记成「该假设无效」—— 台账就被假证据污染了。宁可启动即报错。
    """
    problems: list[str] = []
    root = Path(__file__).resolve().parent.parent
    # 结构自检：CURRENT_DEFAULTS 只能是「参数名 → 字符串值」。
    # 曾经把一个补丁 dict 误插进这个字典 —— Python 不检查泛型标注，于是它静默变成
    # 一个不在 PROMPT_PATCHES 里的孤立表达式，补丁永远不生效而且不报错。
    for k, v in CURRENT_DEFAULTS.items():
        if not isinstance(v, str):
            problems.append(f"CURRENT_DEFAULTS[{k!r}] 不是字符串（是 {type(v).__name__}）")
    for name in PROMPT_PATCHES:
        if name not in SEARCH_SPACE:
            problems.append(f"{name} 未进入 SEARCH_SPACE（插入位置可能写错）")
    for name, spec in PROMPT_PATCHES.items():
        fname = name.split("::", 1)[0]
        p = root / fname
        if not p.exists():
            problems.append(f"{name}: 目标文件 {fname} 不存在")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        n = text.count(spec["anchor"])
        if n != 1:
            problems.append(f"{name}: 锚点出现 {n} 次（必须恰好 1 次）")
        lines = [ln for ln in spec["insert"].splitlines() if ln.strip()]
        if len(lines) > MAX_PROMPT_DIFF_LINES:
            problems.append(f"{name}: 新增 {len(lines)} 行 > {MAX_PROMPT_DIFF_LINES}")
    return problems


PROMPT_PATCH_PROBLEMS = _check_prompt_patches()

#: 泄漏检测：候选文本里出现这些就不许进 probe
_LEAK_PATTERNS = (
    re.compile(r"\b(FRS|CAL|REG|OOD)-\d{2,}\b"),
    re.compile(r"task_id\s*=="),
    re.compile(r"if\s+task(_id)?\s*=="),
)


@dataclass
class Candidate:
    candidate_id: str
    target: str
    level: int
    old: Any
    new: Any
    hypothesis: str
    taxonomy: str = ""
    edits: int = 1
    est_cost: int = 0
    expected_behavior: list[str] = field(default_factory=list)
    patch: dict = field(default_factory=dict)
    base_commit: str = ""
    created_by: str = "proposer"
    hyp_fingerprint: str = ""
    #: 是否由 exploration 强制产生（V3 §10.4）：这类候选的预期本就可能是无效，
    #: 判 WITHIN_NOISE 不代表假设错，只是「覆盖式采样」的正常结果。
    exploration: bool = False
    created_at: str = field(default_factory=now_iso)

    # ---- 便于 gate 使用 ----
    @property
    def is_env(self) -> bool:
        return self.patch.get("kind") == "env"

    @property
    def is_prompt(self) -> bool:
        return self.patch.get("kind") == "prompt"

    @property
    def patch_hash(self) -> str:
        """改动内容的指纹。用内容算，不用 candidate_id —— 后者每次运行都变，没法跨轮对齐。"""
        return sha256_text(self.diff_text())[:16]

    def env_overrides(self) -> dict[str, str]:
        return {self.target: str(self.new)} if self.is_env else {}

    def diff_text(self) -> str:
        if self.is_env:
            return f"{self.target}: {self.old} -> {self.new}"
        if self.is_prompt:
            head = f"{self.patch.get('file')} @ {self.patch.get('anchor')!r}\n"
            return head + str(self.patch.get("insert") or "")
        return str(self.patch.get("diff") or self.patch.get("new") or "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_env"] = self.is_env
        d["is_prompt"] = self.is_prompt
        d["patch_hash"] = self.patch_hash
        return d


def leakage_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pat in _LEAK_PATTERNS:
        m = pat.search(text or "")
        if m:
            hits.append(m.group(0))
    return hits


def current_value(target: str) -> str:
    """某参数的**现行生效值**：进程环境 > .env 托管块 > .env 用户区 > 代码默认。

    读不到（不在搜索空间里 / 无落地层）时退回 ``CURRENT_DEFAULTS``。
    这是「搜索空间跟着已落地的改动走」的唯一机制：不读现行值，闭环就会
    在同一个参数上反复提已上线的那个值。
    """
    if target in PROMPT_PATCHES:
        return "live" if prompt_is_live(target) else ""
    try:
        from . import landing
        v = landing.live_value(target)
    except Exception:
        v = None
    if v is None or str(v).strip() == "":
        return str(CURRENT_DEFAULTS.get(target, ""))
    return str(v).strip()


def prompt_is_live(name: str, *, root: Path | None = None) -> bool:
    """某 L2 补丁是否已经在生产文件里（正文逐字存在）。

    必须查：已落地的补丁再提一次会在同一锚点前**插两遍**（锚点仍唯一，
    apply_prompt_patch 不报错），于是候选实际改了两处而指纹只有一处。
    """
    spec = PROMPT_PATCHES.get(name)
    if not spec:
        return False
    r = root or Path(__file__).resolve().parent.parent
    p = r / str(name).split("::", 1)[0]
    if not p.exists():
        return False
    try:
        return str(spec.get("insert") or "") in p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _adjacent(values: list[str], old: str) -> list[str]:
    """优先尝试相邻值（改动最小），再试远端值。"""
    try:
        i = values.index(str(old))
    except ValueError:
        return list(values)
    order = [i + 1, i - 1, i + 2, i - 2]
    out = []
    for j in order:
        if 0 <= j < len(values):
            out.append(values[j])
    for v in values:
        if v not in out:
            out.append(v)
    return out


def propose(dominant: str, ledger: Ledger, cfg: Settings | None = None,
            avg_run_tokens: int = 600_000, n_pairs: int | None = None,
            limit: int | None = None, exclude_targets: Iterable[str] | None = None,
            prefer_pair: tuple[str, int] | None = None) -> list[Candidate]:
    """按失败模式生成候选。确定性规则，不花 LLM 钱。

    ``exclude_targets``：已进入 dormant 的组件（V3 §10.6），不再由它生成候选。
    ``prefer_pair``：强制探索时指定的 (失败模式 × 层)（V3 §10.4），优先覆盖它。
    """
    cfg = cfg or CFG
    n_pairs = n_pairs or cfg.n_default
    est = estimate_round_tokens(avg_run_tokens, n_pairs)
    out: list[Candidate] = []
    skip = set(exclude_targets or ())

    keys = [k for k in SEARCH_SPACE if k not in skip]
    forced: list[str] = []
    if prefer_pair is not None:
        tax, lv = prefer_pair
        forced = [k for k in keys
                  if SEARCH_SPACE[k]["level"] == lv and tax in SEARCH_SPACE[k]["taxonomies"]]
    ordered = forced + [k for k in keys
                        if k not in forced and dominant in SEARCH_SPACE[k]["taxonomies"]]
    ordered += [k for k in keys if k not in ordered]

    for target in ordered:
        spec = SEARCH_SPACE[target]
        if spec["level"] > cfg.max_edit_level:
            continue
        old = current_value(target)
        if spec.get("is_prompt") and old == "live":
            continue          # 已上线的补丁：再提就是插两遍，必须跳过而不是当新假设
        for new in _adjacent(list(spec["values"]), old):
            if new == old:
                continue
            fp = fingerprint(target, old, new, dominant)
            dup, prev = ledger.is_duplicate(fp)
            if dup:
                continue
            is_prompt = bool(spec.get("is_prompt"))
            if is_prompt:
                patch = {"kind": "prompt", "file": target.split("::", 1)[0],
                         "anchor": spec["anchor"], "insert": spec["insert"],
                         "old": "", "new": new}
            else:
                patch = {"kind": "env", "key": target, "old": old, "new": new}
            cand = Candidate(
                candidate_id=new_id("cand"),
                target=target, level=spec["level"], old=old, new=new,
                hypothesis=f"{target} {old}→{new} 可缓解 {dominant}（{spec['rationale']}）",
                taxonomy=dominant, edits=1, est_cost=est,
                expected_behavior=[spec.get("expectation", "improve")],
                patch=patch,
                hyp_fingerprint=fp,
                # 探索候选：它不是为了改善当前主导模式，而是为了覆盖被冷落的组合
                exploration=bool(target in forced and dominant not in spec["taxonomies"]),
            )
            out.append(cand)
            break
        if len(out) >= (limit or cfg.candidates_per_round):
            break
    return out


def falsification_candidate(taxonomy: str = Taxonomy.COMPLETION_FAILURE,
                            avg_run_tokens: int = 600_000,
                            cfg: Settings | None = None) -> Candidate:
    """阴性对照 2：故意变差（把 code agent 的预算砍到全局默认值以下）。"""
    cfg = cfg or CFG
    return Candidate(
        candidate_id=new_id("neg"),
        target="AGENT_MAX_ITERATIONS_CODE", level=1, old="60", new="12",
        hypothesis="阴性对照：预算砍到 12 应使完成率下降（用于验证比较方向没反）",
        taxonomy=taxonomy, est_cost=estimate_round_tokens(avg_run_tokens, cfg.n_default),
        expected_behavior=["worse_expected"],
        patch={"kind": "env", "key": "AGENT_MAX_ITERATIONS_CODE", "old": "60", "new": "12"},
        created_by="selftest",
    )


def zero_change_candidate(cfg: Settings | None = None) -> Candidate:
    """阴性对照 1：零改动（只改一行注释，env 不变）。"""
    cfg = cfg or CFG
    return Candidate(
        candidate_id=new_id("zero"),
        target="__comment__", level=1, old="", new="",
        hypothesis="阴性对照：零改动必须被判 WITHIN_NOISE",
        taxonomy=Taxonomy.UNKNOWN, edits=1, est_cost=0,
        expected_behavior=["no_change"],
        patch={"kind": "comment", "file": "adaptive_runtime/__init__.py",
               "old": "# no-op", "new": "# no-op "},
        created_by="selftest",
    )
