# -*- coding: utf-8 -*-
"""V4：连通域法提取字符格（方案 A 落地形态候选）。

V3 的教训：整行带做列投影会把上标/下标八度点、符杠、小节线一起切出来，
且裸格裁剪太窄（12px 宽 → rec 压成 9×48）大部分读空。

V4 改用 **连通域（connected components）**：
    1. 二值化（<150，水印圈 ~200+ 不入域）
    2. cv2.connectedComponentsWithStats
    3. 按几何分类：
       - 符杠：宽高比 > 3 且宽 > 40（长横线）→ 丢弃（符杠另有检测器）
       - 小节线/边框：高宽比 > 3（长竖线）→ 丢弃
       - 八度点：面积 < 主字符面积中位数 × 0.5 → 丢弃
       - 其余 = 主字符 → 裁剪（+3px padding，宽度拉到 ≥ 高度的 0.55）→ rec
    4. 按行分组（yc 聚类）输出每行文本

同时验证两张图：
    - original_right.png   （书页拍照，目标场景）
    - qa_sample_long.png   （应用截图，防误伤：det 通道的主战场）
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from extract.ocr_raster import _EngineAdapter, _load_engine  # noqa: E402

SCORE_ALPHABET = set("DXxTtSs0123456789")


def extract_char_components(gray: np.ndarray) -> tuple[list[dict], dict]:
    """二值化 + 连通域 → 主字符候选框列表。

    返回 [{"x0","x1","y0","y1","yc","area","w","h"}, ...] 与统计信息。
    """
    import cv2

    binimg = (gray < 150).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comps.append({"x0": int(x), "y0": int(y), "w": int(w), "h": int(h),
                      "area": int(area), "x1": int(x + w), "y1": int(y + h),
                      "yc": y + h / 2.0})

    # 第一轮：长横线（符杠）与长竖线（小节线/边框）
    chars = []
    beams = 0
    vlines = 0
    for c in comps:
        if c["w"] >= 40 and c["w"] >= 3 * c["h"]:
            beams += 1
            continue
        if c["h"] >= 40 and c["h"] >= 3 * c["w"]:
            vlines += 1
            continue
        chars.append(c)

    # 第二轮：面积过滤（八度点等小件）。
    # 主字符面积取中位数 —— 一页里绝大多数连通域都是主字符。
    areas = sorted(c["area"] for c in chars)
    med = areas[len(areas) // 2] if areas else 0
    dots = 0
    main = []
    for c in chars:
        if c["area"] < max(30, med * 0.45):
            dots += 1
            continue
        main.append(c)
    stats_info = {"components": n - 1, "beams": beams, "vlines": vlines,
                  "dots_dropped": dots, "median_area": med, "main": len(main)}
    return main, stats_info


def group_rows(comps: list[dict], tol_ratio: float = 0.55) -> list[list[dict]]:
    if not comps:
        return []
    hs = sorted(c["h"] for c in comps)
    tol = hs[len(hs) // 2] * 0.9 * tol_ratio + hs[len(hs) // 2] * 0.45
    tol = max(tol, 8)
    rows: list[list[dict]] = []
    for c in sorted(comps, key=lambda t: t["yc"]):
        if rows and abs(c["yc"] - rows[-1][0]["yc"]) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    for r in rows:
        r.sort(key=lambda t: t["x0"])
    return rows


def make_crop(im: Image.Image, c: dict) -> np.ndarray:
    """字符裁剪：+3px padding；太窄的（比高瘦）左右扩到 0.55×高。"""
    pad = 3
    x0 = max(0, c["x0"] - pad)
    x1 = min(im.width, c["x1"] + pad)
    y0 = max(0, c["y0"] - pad)
    y1 = min(im.height, c["y1"] + pad)
    w, h = x1 - x0, y1 - y0
    min_w = int(h * 0.55)
    if w < min_w:
        grow = (min_w - w) / 2.0
        x0 = max(0, int(x0 - grow))
        x1 = min(im.width, int(x1 + grow + 1))
    return np.array(im.crop((x0, y0, x1, y1)))


def run_on(path: str, engine: _EngineAdapter) -> None:
    print(f"\n{'=' * 60}\n=== V4 连通域: {os.path.basename(path)} ===")
    with open(path, "rb") as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGB")
    gray = np.array(im.convert("L"))

    t0 = time.time()
    comps, info = extract_char_components(gray)
    print(f"连通域: {info}  ({time.time() - t0:.2f}s)")

    rows = group_rows(comps)
    print(f"行数: {len(rows)}")
    crops = []
    meta = []
    for ri, row in enumerate(rows):
        for c in row:
            crops.append(make_crop(im, c))
            meta.append((ri, c))
    print(f"字符格总数: {len(crops)}")

    t0 = time.time()
    results = engine.recognize(crops)
    dt = time.time() - t0
    print(f"rec 耗时: {dt:.1f}s")

    texts_by_row = ["" for _ in rows]
    n_empty = 0
    for (ri, c), res in zip(meta, results):
        text = str(res[0]).replace(" ", "") if res and res[0] else ""
        conf = float(res[1]) if res and len(res) > 1 and res[1] is not None else 0.0
        if not text:
            n_empty += 1
            continue
        texts_by_row[ri] += text
    for ri, t in enumerate(texts_by_row):
        print(f"  行{ri} ({len(rows[ri])}格): {t!r}")
    joined = "".join(texts_by_row)
    print(f"  空读: {n_empty}/{len(crops)}")
    print(f"  D={joined.count('D')}  x/X={joined.count('x') + joined.count('X')}"
          f"  S/s={joined.count('S') + joined.count('s')}"
          f"  digits={sum(joined.count(c) for c in '123456789')}"
          f"  total={len(joined)}")


def main() -> None:
    engine = _EngineAdapter(_load_engine())
    run_on(os.path.join(_HERE, "original_right.png"), engine)
    run_on(os.path.join(_ROOT, "data", "qa_sample_long.png"), engine)


if __name__ == "__main__":
    main()
