"""当前日期 / 交易时段 / 数据时效上下文（纯函数，零内部依赖）。

根治：模型无实时时钟，必须显式注入"今天/昨天"与交易时段，否则会把"昨天"映射到训练记忆
里的某次事件，或在盘后按"早盘/实时"去规划与检索（数据滞后十几小时却无提示）。
原定义位于 agent.py:97-188，拆分时整体迁入。
"""
from datetime import datetime, timedelta
import os
import re
from typing import Optional


_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _date_context_str() -> str:
    """返回 [System context] 串，含今天/昨天的精确日期，注入给各 agent 以解析'昨天/上周/本月'等相对时间。

    除日期外还注入**当前时刻（HH:MM）与市场交易时段**：模型没有时钟，只给日期会让它
    在盘后（如 22:54）仍按"早盘/实时"去规划与检索，抓到 10:04 的盘中快照却当成当前行情
    做走势研判——数据滞后十几小时而全链路无任何提示。给时刻 + 时段，模型才能问对口径
    （收盘/收评 vs 盘中）。
    """
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return (
        f"[System context] Today's date is {today.strftime('%Y-%m-%d')} ({_WEEKDAY_CN[today.weekday()]}). "
        f"Yesterday was {yesterday.strftime('%Y-%m-%d')} ({_WEEKDAY_CN[yesterday.weekday()]}). "
        f"Current local time is {today.strftime('%H:%M')}. "
        f"Use these EXACT dates to resolve any relative time expression (昨天/上周/本月/近期) in the user request. "
        f"Do NOT guess or invent the year/month/day.\n"
        f"{_market_session_note(today)}\n"
        f"[Data freshness rule] Any time-series data you retrieve or pass downstream MUST state its "
        f"as-of timestamp as a line 'AS_OF: YYYY-MM-DD HH:MM'. Before using such data, compare it with the "
        f"current local time above; if the data is from today but hours old, label it explicitly as "
        f"'as of HH:MM' and NEVER present it as current/real-time."
    )


# === 数据时效性 ===
# 交易时段表（名称 -> (开盘时,分), (收盘时,分)）。24h 市场（如 crypto）不列入，按"始终在市"处理。
_MARKET_SESSIONS = {"A股": ((9, 30), (15, 0))}
# "当日数据但滞后多少小时"才告警。历史数据（昨天及更早）不告警——用户要的就是历史口径，
# 若一并告警会把合法历史数据淹在误报里。
DATA_FRESHNESS_MAX_HOURS = float(os.environ.get("DATA_FRESHNESS_MAX_HOURS", "2"))


def _market_session_note(now: datetime) -> str:
    """给出当前时刻相对各市场的状态（盘前/盘中/已收盘）与取数口径建议。"""
    parts = []
    for name, ((oh, om), (ch, cm)) in _MARKET_SESSIONS.items():
        mins = now.hour * 60 + now.minute
        open_m, close_m = oh * 60 + om, ch * 60 + cm
        weekend = now.weekday() >= 5
        if weekend:
            state, tip = "休市（周末）", "取最近一个交易日的**收盘**数据"
        elif mins < open_m:
            state, tip = "盘前（未开盘）", "取上一交易日的**收盘**数据，不要取盘中数据"
        elif mins <= close_m:
            state, tip = "盘中", "可取实时/盘中数据，但必须标注 as-of 时刻（盘中数据会随时变化）"
        else:
            state, tip = ("已收盘", f"取当日**收盘/收评**数据；不要再把早盘/盘中快照当作当前行情"
                                    f"（距收盘已 {((mins - close_m) // 60)} 小时，会严重滞后）")
        parts.append(f"{name}：{state} → {tip}")
    return "[Market session] " + "；".join(parts)


_AS_OF_RE = re.compile(
    r"AS_OF\s*[:=]\s*(?:(?P<date>\d{4}-\d{2}-\d{2})\s+)?(?P<time>\d{1,2}:\d{2})", re.IGNORECASE
)


def _data_freshness_check(text: str, now: datetime | None = None) -> str | None:
    """检查产出文本中 AS_OF 标记的数据时刻，返回告警文案；无需告警则返回 None。

    设计取舍：**只校验"as_of 是今天但已滞后数小时"的情形**。
    用户明确要历史数据（如"昨天收盘"）时数据本就陈旧，告警属于误报，
    会把真正的时效问题淹没掉，故历史数据不告警。
    依赖模型按 prompt 规则输出 'AS_OF:' 标记；未输出则本函数静默返回 None（优雅降级）。
    """
    if not text:
        return None
    m = _AS_OF_RE.search(text)
    if not m:
        return None
    now = now or datetime.now()
    date_s = m.group("date") or now.strftime("%Y-%m-%d")
    try:
        as_of = datetime.strptime(f"{date_s} {m.group('time')}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    if as_of.date() != now.date():
        return None  # 历史口径，合法，不告警
    delta_h = (now - as_of).total_seconds() / 3600.0
    if delta_h > DATA_FRESHNESS_MAX_HOURS:
        return (
            f"[数据时效] 产出标注的数据时刻 AS_OF {as_of.strftime('%Y-%m-%d %H:%M')} "
            f"距今已 {delta_h:.1f} 小时（阈值 {DATA_FRESHNESS_MAX_HOURS}h），"
            f"属当日滞后数据，不得作为当前/实时行情使用，下游引用时必须显式标注 as-of 时刻。"
        )
    if delta_h < -0.5:
        return f"[数据时效] 产出标注的数据时刻 {as_of.strftime('%Y-%m-%d %H:%M')} 晚于当前时间 {now.strftime('%H:%M')}，请核对时钟或数据源。"
    return None
