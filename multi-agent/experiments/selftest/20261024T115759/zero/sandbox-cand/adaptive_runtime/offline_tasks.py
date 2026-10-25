"""离线任务集生成：只依赖 ``offline_data/*.csv`` 的本地分析任务。

出题的两条硬约束
----------------
1. **绝不要求实时信息**。一旦要求「最新股价/近期新闻」，supervisor 就会排爬虫步，
   离线语料里没有的答案会让 run 变成「检索失败 → 兜底」，测的是网络而不是 harness。
2. **必须留下可改进的分数空间**。Q = 0.5·A + 0.3·E + 0.2·C：
   - A：计划步是否过 gate → 要求「落盘报告文件 + 报告绝对路径」，交付物契约才可判定；
   - E：关键数字是否带来源 → 要求「每个数字就地标注来源与聚合口径」，
     基线提示词下模型常常写出一堆裸数字，E 有实打实的上升空间；
   - C：产物完整性 → 每题强制 3-4 个子问题，只答一半就扣得出来。

   没有这两条，闭环会看到「base 已接近满分」，任何候选都落在噪声内 —— 那不是
   「没有改进」，那是「这道题没有可改进的空间」，两者必须区分开。

族分布按 ``regression`` 的规格反推：≥12 任务 / ≥6 族 / 每族 ≥2。这里给 18 / 6 / 3，
多出的余量用于三路划分（evolve / ID held-out / OOD）后每层仍不为空。
"""
from __future__ import annotations

import json
from pathlib import Path

from .io_util import now_stamp, write_json
from .offline_data import ensure_datasets, manifest
from .settings import PATHS

DATA = "offline_data"
#: 所有任务共用的收尾要求（写成一句话，避免每题手工复制出错）
_TAIL = (
    "输出要求：(1) 把完整分析写成一份 Markdown 报告文件保存到磁盘，并在最终回复里给出该文件的"
    "绝对路径；(2) 报告里每个关键数字都必须就地标注来源，格式为"
    "「（来源：offline_data/xxx.csv 的 <列名>，<聚合口径与样本行数>）」，自己推导出来的比值要写"
    "「（来源：由前述两行计算）」；(3) 报告必须含一级标题、至少一个数据表格、以及结尾的结论段落；"
    "(4) 全部结论只能来自上述本地 CSV 文件，禁止联网检索最新信息。"
)

TASKS: list[dict] = [
    # ---------------- 族 1：单变量分布 ----------------
    {"task_id": "OFF-STAT-01", "family": "student_stats",
     "query": f"分析 {DATA}/students.csv 中 math_score 的分布："
              f"报告样本行数、均值、标准差、中位数、Q1、Q3、最小值与最大值；"
              f"再用 IQR 规则（Q1-1.5·IQR 与 Q3+1.5·IQR）给出离群行的数量与占比。{_TAIL}"},
    {"task_id": "OFF-STAT-02", "family": "student_stats",
     "query": f"对比 {DATA}/students.csv 中 math_score 与 reading_score 两列："
              f"分别给出均值、标准差、中位数，并计算两列的皮尔逊相关系数；"
              f"说明哪个变量的离散程度更大（用变异系数）。{_TAIL}"},
    {"task_id": "OFF-STAT-03", "family": "student_stats",
     "query": f"基于 {DATA}/students.csv，把 attendance_pct 按 20% 一档分箱，"
              f"报告每档的行数与该档 math_score 均值，并指出从哪一档开始均值出现明显跳升。{_TAIL}"},

    # ---------------- 族 2：分组比较 ----------------
    {"task_id": "OFF-GRP-01", "family": "student_group",
     "query": f"在 {DATA}/students.csv 中按 tutoring（0/1）分组比较 math_score："
              f"给出两组的样本量、均值、标准差与均值差，并判断该差值相对于合并标准差属于"
              f"小/中/大效应。{_TAIL}"},
    {"task_id": "OFF-GRP-02", "family": "student_group",
     "query": f"在 {DATA}/students.csv 中按 grade（G10/G11/G12）与 region（A/B/C/D）"
              f"交叉分组，报告每个单元格行数与 reading_score 均值；找出均值最高与最低的组合，"
              f"并标注其样本量。{_TAIL}"},
    {"task_id": "OFF-GRP-03", "family": "student_group",
     "query": f"在 {DATA}/students.csv 中，把学生按 study_hours_per_week 分为低(<2.5)、"
              f"中(2.5-4.5)、高(>4.5) 三组，报告三组的 math_score 均值与样本量，"
              f"并估算从低组到高组的分数增益。{_TAIL}"},

    # ---------------- 族 3：相关与驱动因素 ----------------
    {"task_id": "OFF-COR-01", "family": "student_corr",
     "query": f"计算 {DATA}/students.csv 中 study_hours_per_week、sleep_hours_per_night、"
              f"attendance_pct、screen_hours_per_day 四个变量分别与 math_score 的皮尔逊相关系数，"
              f"按绝对值排序，指出哪个变量与 math_score 关系最强。{_TAIL}"},
    {"task_id": "OFF-COR-02", "family": "student_corr",
     "query": f"用 {DATA}/students.csv 检验屏幕时间与成绩的关系：报告 screen_hours_per_day 与 "
              f"math_score、reading_score 的相关系数；再按 screen_hours_per_day > 5 与 <= 5 "
              f"分两组报告两组的样本量与成绩均值差。{_TAIL}"},
    {"task_id": "OFF-COR-03", "family": "student_corr",
     "query": f"在 {DATA}/students.csv 上做一次简要驱动因素分析：给出 math_score 与各自变量"
              f"（study_hours_per_week / attendance_pct / tutoring / screen_hours_per_day）的相关"
              f"矩阵要点，并用每单位变化对应的分数变化（简单线性回归斜率）说明哪个变量最值得投入。{_TAIL}"},

    # ---------------- 族 4：销售趋势与季节性 ----------------
    {"task_id": "OFF-TRD-01", "family": "sales_trend",
     "query": f"分析 {DATA}/regional_sales.csv 的月度营收：报告 2024 全年与 2025 全年的营收合计、"
              f"同比增速，以及销量在 11-12 月相对其余月份的放大倍数。{_TAIL}"},
    {"task_id": "OFF-TRD-02", "family": "sales_trend",
     "query": f"基于 {DATA}/regional_sales.csv，按 product_line 汇总：报告每条产品线的销量、"
              f"营收合计与平均单价，并指出营收占比最高的产品线及其占比。{_TAIL}"},
    {"task_id": "OFF-TRD-03", "family": "sales_trend",
     "query": f"用 {DATA}/regional_sales.csv 比较四个区域（N/S/E/W）：报告各区域营收合计、"
              f"占比与月均营收，并找出营收最高的区域及其峰值月份。{_TAIL}"},

    # ---------------- 族 5：数据质量与异常（可发现的植入缺陷）----------------
    {"task_id": "OFF-QUA-01", "family": "sales_quality",
     "query": f"在 {DATA}/regional_sales.csv 中按 product_line 计算退货率（returns/units），"
              f"报告各产品线的退货率；再找出退货率异常偏高的单一「月份×区域×产品线」组合，"
              f"给出其退货率与同产品线其余月份均值退货率的对比。{_TAIL}"},
    {"task_id": "OFF-QUA-02", "family": "sales_quality",
     "query": f"核查 {DATA}/regional_sales.csv 的内部一致性：逐行验证 revenue 是否等于 "
              f"units × unit_price（给出偏差超过 0.5% 的行数与最大偏差），"
              f"并报告 returns 是否大于 units 的非法行数。{_TAIL}"},
    {"task_id": "OFF-QUA-03", "family": "sales_quality",
     "query": f"在 {DATA}/regional_sales.csv 中定位单月放量异常：计算每个「区域×产品线」的月度"
              f"销量相对其自身 24 个月均值的倍数，报告倍数最大的前三个组合及其倍数与对应月份的"
              f"退货率，并判断该放量更像真实需求还是渠道压货。{_TAIL}"},

    # ---------------- 族 6：产品留存与断点 ----------------
    {"task_id": "OFF-RET-01", "family": "app_retention",
     "query": f"分析 {DATA}/app_metrics.csv 的 retention_d7：报告前 30 周与后 30 周的均值、"
              f"标准差与差值；再算出全部 60 周中低于整体均值 2 个标准差以上的周数。{_TAIL}"},
    {"task_id": "OFF-RET-02", "family": "app_retention",
     "query": f"在 {DATA}/app_metrics.csv 中寻找留存断点：比较 W30-W36 与 W37-W43 的 "
              f"retention_d7、churn_rate、arpu_usd 三个指标的均值，判断断点出现在哪一周，"
              f"并量化前后差异。{_TAIL}"},
    {"task_id": "OFF-RET-03", "family": "app_retention",
     "query": f"用 {DATA}/app_metrics.csv 建立留存与商业价值的关系：报告 retention_d7 与 arpu_usd "
              f"的相关系数、retention_d7 与 churn_rate 的相关系数；再按 retention_d7 高于/低于中位数"
              f"分两组，报告两组的 arpu_usd 均值与 cohort_size 总量。{_TAIL}"},
]


def build_tasks() -> list[dict]:
    out: list[dict] = []
    for t in TASKS:
        out.append({"task_id": t["task_id"], "query": t["query"], "family": t["family"],
                    "source": "offline_data", "requires_web": False})
    return out


def families(tasks: list[dict] | None = None) -> dict[str, int]:
    tasks = tasks if tasks is not None else build_tasks()
    dist: dict[str, int] = {}
    for t in tasks:
        dist[t["family"]] = dist.get(t["family"], 0) + 1
    return dist


def write_tasks_file(root: Path | None = None, force_data: bool = False) -> Path:
    """生成数据集 + 写出 ``experiments/tasks/offline_local.json``。"""
    root = root or PATHS.root
    ensure_datasets(root, force=force_data)
    tasks = build_tasks()
    out = PATHS.tasks / "offline_local.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {
        "created_at": now_stamp(),
        "source": "offline_tasks",
        "requires_web": False,
        "count": len(tasks),
        "data_manifest": manifest(root),
        "families": families(tasks),
        "tasks": tasks,
    })
    return out


if __name__ == "__main__":
    p = write_tasks_file(force_data=True)
    print(json.dumps({"path": str(p), "families": families()}, ensure_ascii=False))
