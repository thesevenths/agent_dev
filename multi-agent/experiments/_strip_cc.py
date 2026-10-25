"""临时：定位并清除源码里的不可见控制字符（原子写回）。

用法：python experiments/_strip_cc.py <相对路径...>
不带参数时只扫描、不修改。
"""
import os
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bad(line: str) -> list[str]:
    out = []
    for ch in line:
        if ch in ("\t", "\n", "\r"):
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn"):
            out.append(f"U+{ord(ch):04X}")
    return out


def main() -> int:
    targets = sys.argv[1:]
    fix = bool(targets)
    files = [ROOT / t for t in (targets or [
        "adaptive_runtime/diag_batch.py",
    ])]
    for f in files:
        src = f.read_text(encoding="utf-8")
        lines = src.splitlines(keepends=True)
        hits = []
        for i, ln in enumerate(lines, 1):
            codes = bad(ln)
            if codes:
                hits.append((i, codes, ln.strip()[:60]))
        print(f"{f.relative_to(ROOT)}: {len(hits)} 行含控制字符".encode("ascii", "replace").decode())
        for h in hits:
            print("   ", h[0], h[1], h[2])
        if not (fix and hits):
            continue
        for i, codes, _ in hits:
            lines[i - 1] = "".join(
                c for c in lines[i - 1]
                if c in ("\t", "\n", "\r")
                or unicodedata.category(c) not in ("Cc", "Cf")
            )
        tmp = f.with_suffix(f.suffix + ".tmp")
        tmp.write_text("".join(lines), encoding="utf-8")
        os.replace(tmp, f)
        left = sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if bad(ln))
        print(f"    -> 已修复，剩余 {left} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
