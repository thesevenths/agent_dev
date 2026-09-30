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

from planutil import _normalize_plan, _goal_text
from context import _date_context_str

logger = logging.getLogger(__name__)

# === 统一产物输出目录：所有数据/报告文件写到 <project>/tmp，避免污染仓库根目录 ===
PROJECT_ROOT = Path(__file__).resolve().parent
TMP_DIR = PROJECT_ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
# 供 tools.py 的 create_file/read_file/str_replace 解析相对路径（延迟读取，晚于本模块导入也没关系）
os.environ.setdefault("AGENT_TMP_DIR", str(TMP_DIR))


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
