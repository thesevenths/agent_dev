"""离线数据集生成器：为「零联网」的演化闭环提供固定数据。

为什么需要它
------------
闭环要回答「agent 有没有变强」，必须在**可重复**的任务上配对比较。而带联网检索的任务
有两个致命问题：

1. **烧 tavily 免费额度**（超了要付费），跑一次多轮 benchmark 就是几百次检索；
2. **注入不可控噪声**：同一个 query 今天的搜索结果和明天不一样，报告内容随之变化，
   sd(Δ) 被外部世界的不稳定性抬高 —— 于是 δ 变大，真实改进检不出来。

所以这里生成三份**确定性**数据集（纯 stdlib + 固定 seed，跨版本可复现），
所有离线任务只读这些本地文件，不产生任何网络调用。

不依赖 numpy：numpy 的 RandomState/Generator 跨版本不保证同序列，
而闭环要的是「今天和上周生成的 CSV 一字不差」。
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

#: 数据目录必须在项目根内 —— probe 的沙箱只复制 ``multi-agent/`` 下的条目，
#: 放在仓库其它位置（如 ../langGraph/）的任务在沙箱里会找不到文件。
DATA_DIR_NAME = "offline_data"

SEED = 20261009


def data_dir(root: Path | None = None) -> Path:
    return (root or Path(__file__).resolve().parent.parent) / DATA_DIR_NAME


# ---------------------------------------------------------------- 学生行为习惯

def _students(n: int = 400) -> list[dict]:
    rng = random.Random(SEED + 1)
    rows: list[dict] = []
    for i in range(n):
        grade = rng.choice(["G10", "G11", "G12"])
        region = rng.choice(["A", "B", "C", "D"])
        study = round(rng.uniform(0.5, 7.0), 1)
        sleep = round(rng.uniform(4.5, 9.5), 1)
        attend = rng.randint(55, 100)          # 出勤率 %
        screen = round(rng.uniform(0.5, 8.0), 1)
        tutoring = 1 if rng.random() < 0.32 else 0
        # 植入真实可发现的结构：学习时长与出勤是主因，屏幕时间轻微负向，辅导有正效应
        math_ = (38 + 5.2 * study + 0.28 * attend + 7.5 * tutoring
                 - 0.9 * screen + rng.gauss(0, 5.5))
        read_ = (35 + 3.4 * study + 0.31 * attend + 5.0 * tutoring
                 + 1.1 * sleep + rng.gauss(0, 5.0))
        math_ = max(20, min(100, round(math_)))
        read_ = max(20, min(100, round(read_)))
        rows.append({
            "student_id": f"S{i + 1:04d}",
            "grade": grade,
            "region": region,
            "study_hours_per_week": f"{study:.1f}",
            "sleep_hours_per_night": f"{sleep:.1f}",
            "attendance_pct": str(attend),
            "screen_hours_per_day": f"{screen:.1f}",
            "tutoring": str(tutoring),
            "math_score": str(math_),
            "reading_score": str(read_),
        })
    return rows


# ---------------------------------------------------------------- 区域月度销售

def _sales(n: int = 312) -> list[dict]:
    """2024-01 .. 2025-12 × 4 区域 × 3 产品线中抽样，植入趋势 + E 区 Q4 脉冲 + 高退货异常。"""
    rng = random.Random(SEED + 2)
    rows: list[dict] = []
    months = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]
    base_price = {"Hardware": 480.0, "Service": 120.0, "Accessory": 45.0}
    base_units = {"Hardware": 900, "Service": 2100, "Accessory": 5400}
    idx = 0
    for mi, month in enumerate(months):
        for region in ("N", "S", "E", "W"):
            for line in ("Hardware", "Service", "Accessory"):
                idx += 1
                if idx > n:
                    break
                trend = 1.0 + 0.014 * mi                       # 整体温和增长
                season = 1.18 if month.endswith(("-11", "-12")) else 1.0
                pulse = 1.65 if (region == "E" and month == "2025-03") else 1.0
                units = int(base_units[line] * trend * season * pulse * rng.uniform(0.85, 1.15))
                price = round(base_price[line] * trend * rng.uniform(0.96, 1.04), 2)
                revenue = round(units * price, 2)
                # 退货率：Service 低，Accessory 高；2025-03 的 E 区 Hardware 异常高（可发现的缺陷）
                ret_rate = {"Hardware": 0.031, "Service": 0.008, "Accessory": 0.062}[line]
                if region == "E" and month == "2025-03" and line == "Hardware":
                    ret_rate = 0.174
                returns = int(units * ret_rate * rng.uniform(0.9, 1.1))
                rows.append({
                    "month": month,
                    "region": region,
                    "product_line": line,
                    "units": str(units),
                    "unit_price": f"{price:.2f}",
                    "revenue": f"{revenue:.2f}",
                    "returns": str(returns),
                })
    return rows


# ---------------------------------------------------------------- 应用留存指标

def _app_metrics(n: int = 60) -> list[dict]:
    """60 周：留存基线 + 第 37 周起的版本回归（可发现的断点）。"""
    rng = random.Random(SEED + 3)
    rows: list[dict] = []
    for w in range(1, n + 1):
        cohort = rng.randint(8000, 26000)
        reg = 0.062 if w >= 37 else 0.0        # 37 周起 d7 留存回归 6.2 个百分点
        ret7 = round(min(0.72, max(0.18, 0.415 - reg + rng.gauss(0, 0.021))), 4)
        dau = int(cohort * ret7 * rng.uniform(1.4, 2.2))
        churn = round(min(0.35, max(0.03, 0.175 + reg + rng.gauss(0, 0.014))), 4)
        arpu = round(2.6 + 5.4 * ret7 + rng.gauss(0, 0.18), 3)
        tickets = int(cohort * churn * rng.uniform(0.05, 0.09))
        rows.append({
            "week": f"W{w:02d}",
            "cohort_size": str(cohort),
            "dau": str(dau),
            "retention_d7": f"{ret7:.4f}",
            "churn_rate": f"{churn:.4f}",
            "arpu_usd": f"{arpu:.3f}",
            "support_tickets": str(tickets),
        })
    return rows


DATASETS: dict[str, list[dict]] = {
    "students.csv": None,          # 占位，ensure_datasets 里填
    "regional_sales.csv": None,
    "app_metrics.csv": None,
}


def ensure_datasets(root: Path | None = None, force: bool = False) -> list[Path]:
    """幂等地把三份数据集写进 ``<root>/offline_data/``。返回文件路径列表。"""
    d = data_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    payloads = {
        "students.csv": _students(),
        "regional_sales.csv": _sales(),
        "app_metrics.csv": _app_metrics(),
    }
    out: list[Path] = []
    for name, rows in payloads.items():
        p = d / name
        if p.exists() and not force:
            out.append(p)
            continue
        with p.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        out.append(p)
    return out


def manifest(root: Path | None = None) -> dict:
    """数据集摘要（写进任务文件，便于日后核对「数据有没有偷偷变了」）。"""
    import hashlib
    d = data_dir(root)
    out: dict[str, dict] = {}
    for p in sorted(d.glob("*.csv")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        with p.open(encoding="utf-8") as fh:
            n = max(0, sum(1 for _ in fh) - 1)
        out[p.name] = {"rows": n, "sha256_16": h}
    return {"dir": str(d), "seed": SEED, "files": out}


if __name__ == "__main__":
    ensure_datasets(force=True)
    print(manifest())
