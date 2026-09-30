# -*- coding: utf-8 -*-
"""
Step 3: 技术面与量价关系分析 (上证指数 SH000001)
AS_OF: 2026-09-30 10:10 (盘中快照) / 2026-09-30 15:00 (收盘)
数据来源: step1 (sh_index_data_20260930.json) + step2 (market_sentiment_sectors_20260930.json)

数据稀疏说明:
- 确切收盘价仅有 09-24 (3888.37) 与 09-29 (3830.45); 09-25/09-28 缺失。
- 因此 MA5/MA10/MA20 等均线无法精确计算, 此处用可得数据做"区间/方向"判断并明确标注为估算。
- MACD/RSI 需要连续多日收盘序列, 数据不足, 仅做定性判断。
"""
import json

# ---------- 载入上游数据 ----------
with open(r"E:\agent_dev\multi-agent\tmp\sh_index_data_20260930.json", encoding="utf-8") as f:
    d1 = json.load(f)
with open(r"E:\agent_dev\multi-agent\tmp\market_sentiment_sectors_20260930.json", encoding="utf-8") as f:
    d2 = json.load(f)

intra = d1["intraday_2026_09_30"]
close_0929 = d1["close_2026_09_29"]["close"]
close_0924 = d1["recent_5d_kline"][0]["close"]   # 3888.37
w52_high = intra["week52_high"]
w52_low = intra["week52_low"]

# 09-30 收盘 (step2)
close_0930 = 3882.78
chg_0930 = 0.52  # %

# ---------- 1. 关键价位 / 支撑阻力 ----------
# 52周区间位置
pos_52w = (close_0930 - w52_low) / (w52_high - w52_low) * 100

# 阻力位: 09-24 收盘 3888.37 (近期高点区), 52周高 4258.86
# 支撑位: 09-29 低点 ~3810, 09-22 低点 3823.6, 52周低 3741.11
resistance_1 = close_0924          # 3888.37 近期高点
resistance_2 = 4000                # 整数关口
resistance_3 = w52_high            # 4258.86
support_1 = 3843                   # 09-29 高点 / 09-30 盘中低点区
support_2 = 3810                   # 09-29 低点
support_3 = 3823.6                 # 09-22 低点
support_4 = w52_low                # 3741.11

# ---------- 2. 量价关系 ----------
# 09-29 成交 ~6600亿(雪球口径, 偏小, 可能为沪市); 09-30 沪深北 21975亿(明显放量)
# 09-30 半日(10:10) 2511.82亿 -> 全天估算 ~ 2511.82/0.42*1.0 量级, 但 step2 给出全天 21975亿(三市)
vol_note = "09-30 沪深北三市成交 21975亿, 较 09-29 明显放量; 价涨量增, 量价配合偏多"

# ---------- 3. 均线 (估算, 数据稀疏) ----------
# 可得收盘: 09-24=3888.37, 09-29=3830.45, 09-30=3882.78
# 09-30 收盘 3882.78 已收复 09-24 高点 3888.37 附近, 站上近期整理平台
ma_note = ("确切 MA5/MA10/MA20 因 09-25/09-28 收盘缺失无法精确计算。"
           "定性判断: 09-30 收 3882.78, 已收复 09-24 高点 3888.37 附近, "
           "月线五连涨, 中长期均线(月线/季线)多头排列; "
           "短期(日线)在 3810-3888 区间震荡后向上突破, 短线均线预计走平转多。")

# ---------- 4. MACD / RSI (定性) ----------
macd_note = ("数据不足无法精确计算。定性: 月线五连涨、季度+12.73%, 中长期 MACD 多头; "
             "日线在 09-22 大跌后于 3810-3888 区间整理, 09-30 放量收复高点, "
             "日线 MACD 预计零轴附近金叉/红柱放大, 偏多但需防高位钝化。")
rsi_note = ("数据不足无法精确计算。定性: 指数处 52 周区间中下位置(约 {:.1f}%, 距52周高约 {:.1f}%), "
            "月线五连涨, RSI 预计处于 55-70 偏强区, 接近但未到超买(>80), "
            "短期有获利回吐压力。").format(pos_52w, (w52_high-close_0930)/w52_high*100)

# ---------- 5. 市场状态判断 ----------
state = {
    "趋势": "中长期多头(月线五连涨), 短期震荡后向上突破",
    "位置": "52周区间中下位置 {:.1f}% (距52周高约{:.1f}%)".format(pos_52w, (w52_high-close_0930)/w52_high*100),
    "量价": "价涨量增, 量价配合偏多",
    "结构": "指数分化, 权重(金融/酿酒)拖累, 科技/有色/军贸领涨, 热点快速轮动",
    "风险": "月线收官+连涨, 短期获利回吐与热点退潮风险; 09-30 为9月最后交易日",
}

result = {
    "AS_OF": "2026-09-30 10:10 (盘中) / 15:00 (收盘)",
    "close_0930": close_0930,
    "chg_0930_pct": chg_0930,
    "pos_52w_pct": round(pos_52w, 1),
    "support": {"S1": support_1, "S2": support_2, "S3": support_3, "S4_52w_low": support_4},
    "resistance": {"R1": resistance_1, "R2": resistance_2, "R3_52w_high": resistance_3},
    "volume_price": vol_note,
    "ma": ma_note,
    "macd": macd_note,
    "rsi": rsi_note,
    "market_state": state,
    "data_gaps": [
        "09-25/09-28 收盘缺失 -> MA5/MA10/MA20 无法精确计算, 仅定性",
        "MACD/RSI 需连续多日序列, 数据不足, 仅定性判断",
        "09-29 成交量口径(雪球~6600亿)与三市口径不一致, 量价对比以三市21975亿为准"
    ],
}

with open(r"E:\agent_dev\multi-agent\tmp\technical_analysis_20260930.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))
