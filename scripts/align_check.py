# -*- coding: utf-8 -*-
"""对齐诊断：把每行落格结果打印成固定宽度的格子，与原始 OCR 框文本并排对照。

⚠️ 直接调用生产代码的 `_page_tokens`，不另写一套估计逻辑 ——
诊断脚本一旦和生产代码走不同的路，验证就没有意义了。

用法：
    .venv/Scripts/python.exe scripts/align_check.py <图片路径> [更多图片...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

COLS = 40
BAR_NO = {"", "26", "28", "31", "33", "35", "37", "39", "41", "43", "45", "47", "24", "5"}


def show(path: Path) -> None:
    import extract.omr as omr

    raw = omr._normalize(path.read_bytes(), omr.MAX_IMAGE_WIDTH)
    items = omr._ocr_items(raw)
    rows = omr._group_rows(items)
    rows_tokens, used = omr._page_tokens(path.read_bytes())

    # 复算 left / cell_w 仅用于打印参考值（与生产代码同一套函数）
    left = omr._page_left_from_boxes(rows)
    rc = [omr._row_chars(r, left=left) for r in rows]
    rc = [c if (c and c[-1][0] - c[0][0] + c[-1][2] >= omr.MIN_ROW_SPAN) else [] for c in rc]
    cws = [omr._estimate_cell_width([q[0] for q in c]) for c in rc if len(c) >= 4]
    cws = [w for w in cws if w >= 8]
    cw = sorted(cws)[len(cws) // 2] if cws else 0

    print(f"══ {path.name}  left={left} cell_w={cw:.1f} rows={used} ══")

    # rows_tokens 已经过滤掉无效行，按顺序对回原行用于打印原文
    valid = [r for r, c in zip(rows, rc) if c]
    for row, toks in zip(valid, rows_tokens):
        yc = int(np.mean([it["yc"] for it in row]))
        n = max(COLS, max((t["slot"] for t in toks), default=0) + 1)
        grid = ["·"] * n
        for t in toks:
            grid[t["slot"]] = t["ch"]
        raws = [it["text"] for it in sorted(row, key=lambda x: x["x0"])
                if it["text"].strip() not in BAR_NO]
        print(f"  y{yc:4d} |{''.join(grid)}|")
        print(f"         {' '.join(raws)[:130]}")
    print()


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        root = BASE / "data" / "uploads" / "2026" / "10"
        paths = [root / n for n in (
            "bfadb43f95444919983b151054d93553.jpg",
            "467981d25f24471bbc9547cabdc7879a.jpg",
        ) if (root / n).exists()]
    for p in paths:
        show(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
