"""A/B：同一批记忆下，AGENT_MEMORY_MIN_SCORE=0.25 vs 0.58 的召回差异。

做法：临时 DB 里塞 3 条与 query 相关的 + 8 条完全无关的，
      比较两个阈值分别把哪些召回了（query 只与第 1 条强相关）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, r"F:\agent\multi-agent")
os.environ["AGENT_MEMORY_DB"] = r"F:\agent\multi-agent\tmp\_calib\ab.db"
os.environ.setdefault("AGENT_LONGTERM_MEMORY", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(r"F:\agent\multi-agent\.env")

MIN = os.environ["AGENT_MEMORY_MIN_SCORE"]

import longterm  # noqa: E402

RELEVANT = [
    ("评估 joern 与 tree-sitter 两种调用链提取方案在 nacos 上的召回率", "decision"),
    ("GuardFox 的 sink 驱动扫描基线是 OWASP BenchmarkJava", "fact"),
    ("CWE-89 检测范围从 big-4 SQL sinks 扩展到任意 raw SQL string 的 DB 抽象层", "fact"),
]
NOISE = [
    ("红烧肉要先把五花肉焯水再炒糖色", "note"),
    ("今天下午有阵雨，出门记得带伞", "note"),
    ("下周去成都的机票改签到周五晚上", "note"),
    ("猫砂盆该换了，顺便买点罐头", "note"),
    ("会议室预约系统升级到 v3，周五晚上停机", "note"),
    ("新版报销单需要附上电子发票 PDF", "note"),
    ("咖啡机除垢剂放在茶水间第二个柜子", "note"),
    ("季度团建定在十月中旬，地点待定", "note"),
    ("打印机墨盒型号是 CF258A", "note"),
    ("楼下便利店八点后生鲜打折", "note"),
    ("周三下午三点和客户开需求评审会", "note"),
    ("git 仓库的 master 分支改名成 main 了", "note"),
    ("饮水机滤芯每三个月换一次", "note"),
    ("出差报销标准按城市等级分三档", "note"),
    ("新同事的工位在靠窗第三排", "note"),
    ("电梯年检通知贴在公告栏", "note"),
    ("年会抽奖的一等奖是扫地机器人", "note"),
    ("快递柜取件码每天凌晨刷新", "note"),
    ("食堂周三供应饺子，排队很久", "note"),
    ("公司停车位需要提前一天在小程序上预约", "note"),
    ("防火墙策略变更需要走变更评审流程", "note"),
    ("生产环境发版窗口定在每周二凌晨", "note"),
    ("监控告警阈值调整要同步给值班同学", "note"),
    ("办公区空调温度统一设定在 26 度", "note"),
    ("打印机局域网 IP 段改成 10.20 开头了", "note"),
    ("体检报告出来了，各项指标都正常", "note"),
    ("投影仪遥控器电池该换了", "note"),
    ("绿萝浇水频率改成一周两次", "note"),
    ("楼下健身房新开了一家，年卡有折扣", "note"),
    ("周末想去看场电影，还没定看什么", "note"),
]

QUERY = "我用哪个工具对比过调用链提取？"

if os.environ.get("AB_SEED", "1") == "1" and not os.path.exists(os.environ["AGENT_MEMORY_DB"]):
    for t, ty in RELEVANT + NOISE:
        longterm.remember(t, mtype=ty, tags=[], importance=0.5)

hit = longterm.recall(QUERY)
lines = [ln for ln in hit.splitlines() if ln.strip()]
relevant_texts = {t for t, _ in RELEVANT}
n_rel = sum(1 for ln in lines if any(t[:14] in ln for t in relevant_texts))
print(f"MIN_SCORE={MIN}  召回 {len(lines)} 条（库内 {len(RELEVANT) + len(NOISE)} 条，真相关 3 条）"
      f"  其中相关 {n_rel} 条 / 噪声 {len(lines) - n_rel} 条")
for ln in lines:
    tag = "相关" if any(t[:14] in ln for t in relevant_texts) else "噪声"
    print(f"    [{tag}] {ln}")
