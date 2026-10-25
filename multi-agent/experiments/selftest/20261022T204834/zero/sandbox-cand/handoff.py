"""子 Agent 任务下发 + 产物落盘（agent 间 handoff 可靠通道）。

- TMP_DIR：所有数据/报告文件写到 <project>/tmp，避免污染仓库根目录，并写入 AGENT_TMP_DIR 供
  tools.py 的 create_file/read_file/str_replace 解析相对路径；
- _step_assignment_text：构造"本步任务指令"，告诉子 agent 它在第几步、要干什么、上游产物在哪、
  并禁止重跑上游工作；
- _save_artifact：把本步最终输出落盘 tmp/，作为下游 handoff 通道（不受 context 窗口限制）。
原定义位于 agent.py:86-91（TMP_DIR）+ 1316-1363（两个函数），拆分时整体迁入。
"""
from pathlib import Path
import os
import re
import logging
from datetime import datetime

from planutil import _normalize_plan, _goal_text, _parse_target_agent
from context import _date_context_str

logger = logging.getLogger(__name__)

# === 统一产物输出目录：所有数据/报告文件写到 <project>/tmp，避免污染仓库根目录 ===
PROJECT_ROOT = Path(__file__).resolve().parent
TMP_DIR = PROJECT_ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
# 供 tools.py 的 create_file/read_file/str_replace 解析相对路径（延迟读取，晚于本模块导入也没关系）
os.environ.setdefault("AGENT_TMP_DIR", str(TMP_DIR))


def _failure_hint(kind: str) -> str:
    """把失败 kind 翻译成"下一个 sub agent 能据以改进行动"的可操作要求。

    这是"失败 → 改进"闭环的价值所在：只报告 reason（如 GraphRecursionError）毫无用处，必须告诉下游
    **这一次具体该怎么做才能绕开同一个坑**（对应 2026-10-02 step3/4/5 连续三次撞同一根因的教训）。
    hint 是纯数据（不带 step/agent 上下文），便于 step_failures 记录与 prompt 渲染共用同一真源。
    """
    k = str(kind or "").strip().lower()
    if k == "recursion":
        return (
            "上一次在这一步耗尽了 ReAct 步数预算（AGENT_MAX_ITERATIONS），执行在你停止的那一刻被硬截断——"
            "即使文件已经落盘，也会因为没走到'写最终总结'而被判失败（随后整步会被标记为 failed 并重做）。\n"
            "预算换算：一次工具往返 = 2 个 super-step（LLM 决策 1 + 工具执行 1），所以预算只够约一半次数的"
            "工具往返。本轮务必压缩往返次数：\n"
            "  1) 先一次性写完整脚本，再一次性跑通，不要边写边改多轮 str_replace；\n"
            "  2) 【最常见死法】脚本一旦执行成功并回显了交付物路径，就**立即收尾**——"
            "不要再回头做打磨性修改（改措辞、改配色、改表格排版、补注释、加条件判断）。"
            "这类 str_replace 对交付物没有实质影响，却一次吃掉一个完整往返，是撞上限的头号原因；\n"
            "  3) shell 命令直接用 Windows 绝对路径（如 python \"F:\\agent\\multi-agent\\tmp\\x.py\"），"
            "不要用 /f/agent/... 这类 Unix 风格路径（Windows cmd 不认，第一次必然失败、白费一次往返）；\n"
            "  4) 能用一个工具做完的事不要拆成两个（多个 read_file 可以放在同一轮并行发出）；\n"
            "  5) 最关键：在预算耗尽之前，必须用一段**不带任何 tool_calls 的最终总结**收尾，"
            "并在其中明确写出落盘文件的绝对路径。这一步的优先级高于任何美化。"
        )
    if k == "truncated":
        # 注意：这一步在记录里是"被截断但产物已落盘并通过质量门"，plan 里标的是 completed，
        # 下游不需要重做。hint 的作用是防止后续步骤误以为"上游没做完"而重复劳动。
        return (
            "上一次在这一步因 ReAct 步数预算耗尽被截断，但它要的交付物当时**已经落盘并通过质量门**，"
            "因此该步已按完成处理（不是失败）。本轮**不要重做**它已经产出的部分：\n"
            "  1) 直接读取已落盘的产物文件继续使用，需要补充的只做增量；\n"
            "  2) 如果你判断产物不完整（例如只有图没有报告、或内容明显残缺），"
            "请只补齐缺失的那一部分，并说明补了什么；\n"
            "  3) 仍须注意：一次工具往返 = 2 个 super-step，跑通后立刻收尾，不要做打磨性修改。"
        )
    if k == "exception":
        return (
            "上一次在这一步连续异常重试 3 次仍失败（通常是工具参数格式错误、文件被占用或环境不可用）。"
            "本轮请先确认参数格式与文件编码（Windows 下注意 GBK/UTF-8 与路径反斜杠），改用更简单的一次性调用；"
            "若确属环境不可用（命令不存在/服务不通），请明确说明缺什么、为什么无法完成，"
            "不要重复同一条已经失败过的路径。"
        )
    if k == "critic":
        return (
            "上一次的产出未通过质量门。最常见的三个原因：\n"
            "  a) 本步是交付型任务，但最终回复里没有给出落盘文件的绝对路径；\n"
            "  b) 内容过于空泛（只回一句'已完成/见文件'），下游拿不到真正需要的数据与结论；\n"
            "  c) 声称完成的文件其实根本没写成功。\n"
            "本轮请务必：确认文件确实已写入磁盘 → 在最终回复里明确给出它的绝对路径 → "
            "并把关键数据/结论直接写出来。"
        )
    return "请针对上述不合格原因逐条修正后重新产出，不要重复同样的问题。"


# 渲染失败档案时向前回溯的步数：supervisor 常把失败步**合并/改写**成后续某一步
# （2026-10-02 线上：step4 合并了失败的 step3+4，step5 又合并了 3+4+5），若只对齐本步
# 就会看不到被合并进来的前序失败原因，所以必须向前回溯。
_FAIL_LOOKBACK = int(os.environ.get("AGENT_FAIL_LOOKBACK", "2"))
_FAIL_RENDER_MAX = int(os.environ.get("AGENT_FAIL_RENDER_MAX", "3"))


def _recent_step_failures(state: dict, cur: int, lookback: int | None = None,
                          limit: int | None = None) -> list:
    """取与本步相关的失败记录：本步 + 往前 lookback 步（覆盖 supervisor 合并/改写场景）。

    返回按发生顺序排列的记录列表（最旧的在前），最多 limit 条最近的。
    cur/step 均为 1-based 步号。
    """
    lb = _FAIL_LOOKBACK if lookback is None else lookback
    lm = _FAIL_RENDER_MAX if limit is None else limit
    try:
        cur = int(cur or 0)
    except (TypeError, ValueError):
        return []
    if cur <= 0:
        return []
    out = []
    for f in (state.get("step_failures") or []):
        if not isinstance(f, dict):
            continue
        try:
            n = int(f.get("step"))
        except (TypeError, ValueError):
            continue
        if cur - lb <= n <= cur:
            out.append(f)
    return out[-max(1, lm):]


def _render_step_failures(state: dict, cur: int) -> str:
    """把"上一步/本步为什么失败 + 这次该怎么改"渲染成派发 / 再规划可直接消费的文本块。

    无相关记录时返回空串（零回归：正常路径的 prompt 一字不变）。整段异常一律 swallow ——
    失败原因只是提示通道，绝不能反过来让派发本身崩掉。
    """
    try:
        fs = _recent_step_failures(state, cur)
        if not fs:
            return ""
        lines = [
            "PREVIOUS ATTEMPT(S) ON THIS STEP FAILED — read carefully, this is the single most "
            "important constraint for this step:"
        ]
        for f in fs:
            n = f.get("step")
            same = (int(n) == int(cur)) if str(n).isdigit() else False
            head = ("FAILED at THIS step" + (f" (attempt #{f.get('attempt')})" if f.get("attempt") else "")
                    if same else f"FAILED at step {n} (merged into your current step)")
            lines.append("")
            lines.append(f"- {head} | kind={f.get('kind')} | agent={f.get('agent')} | at={f.get('at')}")
            if f.get("reason"):
                lines.append(f"  WHY it failed: {f.get('reason')}")
            # hint 优先取记录里存的（与失败当刻的文案一致）；老记录没有则按 kind 现算，
            # 保证无论数据是哪个版本写入的，下游都能拿到可执行的要求。
            _hint = f.get("hint") or _failure_hint(f.get("kind"))
            if _hint:
                lines.append(f"  HOW TO AVOID IT THIS TIME: {_hint}")
        lines.append("")
        lines.append("Do NOT repeat the same mistake. If the same failure reoccurs, explicitly state "
                     "what you changed and what still blocks you.")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"[handoff] render step failures skipped ({e})")
        return ""


def _step_assignment_text(state: dict, agent_name: str) -> str:
    """构造"本步任务指令"：告诉子 agent 它在计划的第几步、要干什么、上游产物在哪、不许重做上游工作。"""
    plan = _normalize_plan(state.get("execution_plan") or [])
    cur = state.get("current_step", 0)
    step = plan[cur - 1] if (plan and 0 < cur <= len(plan)) else None
    lines = [_date_context_str(), ""]
    if step:
        lines += [
            f"[Supervisor assignment] You are executing step {cur}/{len(plan)} of an approved multi-agent plan.",
            f"Overall goal: {state.get('plan_goal') or _goal_text(state)}",
            f"This step is assigned to: {step.get('description', '')}",
            f"Step title: {step.get('title', '')}",
        ]
    else:
        lines.append("[Supervisor assignment] No plan step recorded; answer the user's latest request directly.")
    artifacts = state.get("artifacts") or []
    if artifacts:
        lines.append("")
        lines.append("Upstream results are ALREADY available as persisted files (read them with read_file if you need the full data):")
        for p in artifacts[-5:]:
            lines.append(f"  - {p}")
    # 执行者对账：plan 文本写着 "→ chat_agent"，supervisor 却把本步派给了别的 agent。
    # 这是设计允许的（再规划可以改当前步执行者），但历史上它导致过"发邮件的活被 code_agent
    # 用 dir + read_file 应付过去、critic 还判 PASS"（2026-10-02 step5）。故必须让执行者
    # 明确知道自己是被换上来的，并自查是否具备完成该动作的工具 —— 没有工具就如实说，
    # 不要用"读了一下文件"冒充完成了本步。
    if step and agent_name:
        _planned = _parse_target_agent(step)
        # 规范化必须同时处理两种写法：plan 文本里解析出的是 "chat_agent"（下划线），
        # 而 agent.name 是 "ChatAgent"（驼峰、无下划线）—— 只 replace("_agent") 会把
        # "chatagent" 原样留下，导致"执行者其实一致"被误判成替换而反复插入提示。
        def _n(s):
            t = str(s or "").strip().lower().replace("_agent", "")
            if t.endswith("agent") and len(t) > len("agent"):
                t = t[: -len("agent")]
            return t

        if _planned and _n(_planned) != _n(agent_name):
            lines += [
                "",
                f"[EXECUTOR SUBSTITUTION] The plan text assigns this step to '{_planned}', but you "
                f"({agent_name}) are executing it. Before you start: confirm you actually have the tool "
                f"required by this step's action. If you do NOT, say so explicitly and stop — do NOT "
                f"substitute a read/describe action for the required one.",
            ]
    # 失败回溯：把"本步/前序步为什么失败 + 这次该怎么改"插在任务定义之后、HARD RULES 之前——
    # 必须在 sub agent 调第一个工具之前就看到，否则它会重蹈覆辙（徒增一轮工具往返）。
    fail_ctx = _render_step_failures(state, cur)
    if fail_ctx:
        lines += ["", fail_ctx]
    lines += [
        "",
        "HARD RULES for this step:",
        "1. Do NOT repeat work already done by upstream agents (e.g. do NOT re-run the same web search / re-crawl). "
        "Their outputs are in this conversation and in the files listed above—consume them as your INPUT.",
        "2. Output ONLY the deliverable for THIS step, then stop. Do not answer parts of the goal belonging to other steps.",
        "3. Save substantial results/reports/data to a file and state the file path in your reply.",
        f"4. All files you write must go to this directory: {TMP_DIR}",
    ]
    return "\n".join(lines)


def _save_artifact(agent_name: str, step_idx: int, content: str) -> str | None:
    """把本步的最终输出落盘到 tmp/，作为下游 handoff 的可靠通道（不受 context 窗口限制）。"""
    try:
        if not content or not str(content).strip():
            return None
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_agent = re.sub(r"[^A-Za-z0-9_.-]", "", agent_name or "agent")
        path = TMP_DIR / f"{ts}__step{step_idx}__{safe_agent}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))
        logger.info(f"artifact saved: {path}")
        return str(path)
    except Exception as e:
        logger.warning(f"save artifact failed: {e}")
        return None
