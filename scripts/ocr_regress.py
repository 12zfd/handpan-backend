# -*- coding: utf-8 -*-
"""OCR 回归基线：跑同一组真机输入，打印可比对的结构化结果。

用途
----
任务 #44（微信图片_20261028155834_63_223 两张）是真机上真实跑过的一单，
已知的产出是：31 行 / 994 格 / 63 小节 / 735 音符（260 有音高 + 475 无音高）。
本脚本把这张 Monument 重跑一遍，输出同口径的指标，用来确认改动
「有没有变差」。

用法（在 handpan-backend 目录下）：
    .venv/Scripts/python.exe scripts/ocr_regress.py
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# 真机输入：任务 #44 的两张上传件（与 onprem 上跑的是同一批字节）
PAGES = [
    BASE / "data" / "uploads" / "2026" / "10" / "467981d25f24471bbc9547cabdc7879a.jpg",
    BASE / "data" / "uploads" / "2026" / "10" / "fa8571a62aa549c280d8457dc289589b.jpg",
]


def main() -> int:
    from extract.omr import image_to_tab, _Timing

    raws = [p.read_bytes() for p in PAGES]
    print("pages:", [str(p.name) + " " + str(p.stat().st_size) + "B" for p in PAGES])

    t = _Timing()
    t0 = time.time()
    tab = image_to_tab(raws, timing=t)
    wall = int((time.time() - t0) * 1000)

    events = tab.get("events") or []
    bars = tab.get("bars")
    print()
    print("── 结果 ──────────────────────────────────")
    print("rows       :", (tab.get("stats") or {}).get("rows"))
    print("slots      :", (tab.get("stats") or {}).get("chars"))
    print("events     :", len(events))
    print("  pitched  :", sum(1 for e in events if e.get("kind") != "stroke"))
    print("  taps     :", sum(1 for e in events if e.get("kind") == "stroke"))
    print("bars       :", bars)
    print("wall_ms    :", wall)
    print("timing     :", dict(t))
    print("── warnings ─────────────────────────────")
    for w in tab.get("warnings") or []:
        print("  ·", w)

    print()
    print("── 真机基线（#44）────────────────────────")
    print("  rows=31  slots=994  events=735  pitched=260  taps=475  bars=63")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
