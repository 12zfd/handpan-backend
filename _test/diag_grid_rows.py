# -*- coding: utf-8 -*-
"""诊断：网格通道每行落格详情 —— 定位字符→事件转化的损耗。"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import io  # noqa: E402

from extract import omr  # noqa: E402
from extract.ocr_raster import recognize_raster_grid  # noqa: E402
from extract.beams import detect_beam_segments  # noqa: E402


def main() -> None:
    with open(os.path.join(_HERE, "original_right.png"), "rb") as f:
        raw = f.read()
    proc_raw = omr._normalize(raw, omr.MAX_IMAGE_WIDTH)
    grid_items = recognize_raster_grid(proc_raw)
    print(f"grid items: {len(grid_items)}")
    gray = np.array(Image.open(io.BytesIO(proc_raw)).convert("L"))
    segments = detect_beam_segments(gray)
    rows = omr._group_rows(grid_items)
    print(f"rows: {len(rows)}")
    left_hint = omr._page_left_from_boxes(rows)
    print(f"left_hint={left_hint}")

    def _pass(cell_w):
        out = []
        for row in rows:
            chars = omr._row_chars(row, cell_w=cell_w, left=left_hint)
            out.append(chars)
        return out

    rows_chars = _pass(None)
    cws = []
    for ch in rows_chars:
        if len(ch) >= 4:
            w = omr._estimate_cell_width([c[0] for c in ch])
            if w >= 8.0:
                cws.append(w)
    cell_w_hint = sorted(cws)[len(cws) // 2] if cws else None
    print(f"cell_w candidates={sorted(round(w,1) for w in cws)}  hint={cell_w_hint}")

    if cell_w_hint:
        rows2 = [omr._resolve_overlaps(row, cell_w_hint) for row in rows]
        rows_chars = [omr._row_chars(row, cell_w=cell_w_hint, left=left_hint) for row in rows2]
    for i, (row, chars) in enumerate(zip(rows, rows_chars)):
        toks = omr._row_tokens(row, segments, cell_w_hint=cell_w_hint,
                               left_hint=left_hint, chars=chars,
                               require_dts=False)
        txt = "".join(t["ch"] for t in toks)
        slots = [t["slot"] for t in toks]
        print(f"行{i}: items={len(row)} chars={len(chars)} toks={len(toks)} "
              f"slot范围={min(slots) if slots else '-'}~{max(slots) if slots else '-'}")
        print(f"     {txt!r}")
        if slots and max(slots) >= omr.ROW_SLOTS - 2:
            print(f"     ⚠️ 接近/达到 ROW_SLOTS={omr.ROW_SLOTS} 上限")


if __name__ == "__main__":
    main()
