# -*- coding: utf-8 -*-
"""OCR 落格诊断 v2：直接走 _page_tokens 新逻辑，打印每行 slot 落位。

用法：.venv/Scripts/python.exe scripts/ocr_diag.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

PAGES = [
    BASE / "data" / "uploads" / "2026" / "10" / "467981d25f24471bbc9547cabdc7879a.jpg",
    BASE / "data" / "uploads" / "2026" / "10" / "fa8571a62aa549c280d8457dc289589b.jpg",
]


def main() -> int:
    from extract import omr
    from extract.omr import (_group_rows, _normalize, _ocr_items, _row_tokens,
                             MAX_IMAGE_WIDTH, MIN_ROW_SPAN)
    from extract.beams import detect_beam_segments

    for pi, page in enumerate(PAGES):
        raw = _normalize(page.read_bytes(), MAX_IMAGE_WIDTH)
        items = _ocr_items(raw)
        gray = np.array(Image.open(io.BytesIO(raw)).convert("L"))
        segments = detect_beam_segments(gray)
        rows = _group_rows(items)
        print(f"══ 页 {pi+1}：items={len(items)}  rows={len(rows)} ══")
        for ri, row in enumerate(rows):
            try:
                x0s = [float(it["x0"]) for it in row]
                x1s = [float(it["x1"]) for it in row]
                span = max(x1s) - min(x0s) if x0s else 0
            except Exception:  # noqa: BLE001
                span = -1
            toks = _row_tokens(row, segments)
            if not toks:
                reason = "span" if span < MIN_ROW_SPAN else "无谱字符"
                print(f"  行{ri:02d} [跳过-{reason}] span={span:5.0f} "
                      f"texts={[it['text'][:14] for it in row][:4]}")
                continue
            grid = "".join(
                next((t["ch"] for t in toks if t["slot"] == s), "·")
                for s in range(omr.ROW_SLOTS)
            )
            print(f"  行{ri:02d} span={span:5.0f} 字符={len(toks):02d} |{grid}|")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
