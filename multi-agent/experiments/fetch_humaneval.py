"""一次性下载 OpenAI HumanEval 数据集到 ``experiments/humaneval/HumanEval.jsonl``（需联网）。

HumanEval 是公开单文件（164 题 ≈ 80KB），不依赖 Docker，正好用来证明 RRSI 引擎的
正路径（failure → proposal → candidate → accept → promote → land → H1 → holdout）。

用法：
    python experiments/fetch_humaneval.py
    # 或自定义落盘位置：
    python experiments/fetch_humaneval.py --dest /path/to/HumanEval.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_runtime.humaneval import fetch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 HumanEval 数据集（免 Docker 基准）")
    ap.add_argument("--dest", default=None, help="输出路径（默认 experiments/humaneval/HumanEval.jsonl）")
    args = ap.parse_args()
    out = fetch(args.dest)
    print(f"已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
