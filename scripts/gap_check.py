# -*- coding: utf-8 -*-
"""真空隙检测：找出「格位为空但物理宽度够放一个字」的位置，即 OCR 漏字。

为什么需要这个
--------------
落格对齐修好之后，剩下的错位全部来自**字符漏读**：OCR 少读一个字，
这一组的后续字符整体左移一格，听起来就是「音节对不上」。
漏字有个非常干净的几何指纹：

    同一行相邻两个字符的中心距 = k × 格宽（k 为整数，表示中间空了 k-1 格）

    若 k-1 == 0（相邻）      → 没有空隙，正常
    若 k-1 == 1（空 1 格）   → **可疑**：谱面极少出现单格空洞，
                               多半是中间那个字被漏读了
    若 k-1 >= 2              → 组间留白，正常

所以本脚本只统计「空 1 格」的候选点，并按样本汇总，用来估计漏字的
真实发生率，以及漏字是否集中在某些 x 位置（能反推是哪一列难读）。

用法（在 handpan-backend 目录下）：
    .venv/Scripts/python.exe scripts/gap_check.py            # 全部样本
    .venv/Scripts/python.exe scripts/gap_check.py 49         # 单个样本
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

DB = BASE / "data" / "handpan.db"


def main() -> int:
    import extract.omr as omr

    want = [int(x) for x in sys.argv[1:] if x.isdigit()]
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, original_file_url FROM scores "
        "WHERE status='done' AND source_type='image' AND midi_data_json IS NOT NULL "
        "ORDER BY id"
    ).fetchall()
    if want:
        rows = [r for r in rows if r["id"] in want]

    total_gap1 = 0
    total_rows = 0
    col_hist: Counter[int] = Counter()
    per_sample: list[tuple[int, str, int, int]] = []

    for r in rows:
        url = r["original_file_url"] or ""
        name = url.split("/")[-1].strip('"[]')
        p = BASE / "data" / "uploads" / "2026" / "10" / name
        if not p.exists():
            continue
        try:
            rows_tokens, _n = omr._page_tokens(p.read_bytes())
        except Exception as exc:  # noqa: BLE001
            print(f'{r["id"]:>3} 失败: {type(exc).__name__}: {str(exc)[:60]}')
            continue
        g1 = 0
        for toks in rows_tokens:
            slots = sorted(t["slot"] for t in toks)
            for a, b in zip(slots, slots[1:]):
                gap = b - a - 1
                if gap == 1:
                    g1 += 1
                    col_hist[a % omr.SLOTS_PER_BAR] += 1
        total_gap1 += g1
        total_rows += len(rows_tokens)
        per_sample.append((r["id"], name[:20], len(rows_tokens), g1))

    print(f'\n有效样本 {len(per_sample)} 个 / 共 {total_rows} 行')
    print(f'「空 1 格」候选点合计：{total_gap1} 处'
          f'（平均每行 {total_gap1 / max(1, total_rows):.2f} 处）\n')
    print(f'{"id":>3}  {"行数":>4}  {"空1格":>5}  文件名')
    for sid, name, nrow, g1 in per_sample:
        print(f'{sid:>3}  {nrow:>4}  {g1:>5}  {name}')
    if col_hist:
        print("\n空 1 格的列分布（小节内第几格 → 次数）：")
        for k in sorted(col_hist):
            print(f'  第 {k:>2} 格  {"█" * min(60, col_hist[k]):<60} {col_hist[k]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
