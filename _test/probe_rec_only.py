# -*- coding: utf-8 -*-
"""方案 A 前提验证：跳过 det，按行切片直接喂 rec 模型。

背景
----
真实原谱（书页拍照，D Kurd 9音《晴天》）上 OCR 只认出后半截：
    - `D` 全部丢失（det 把它当噪点滤掉）
    - `x` 变 `X` 或整体丢失
    - 行首整组丢失
取证明白：问题出在 det 层（把原谱放大反而框数 60→48，单行裁切放大 → 0）。

本脚本做三组对比（同一张真实原谱）：
    V1 基线     —— 现有 det+rec 全流程（recognize_raster）
    V2 整行rec  —— 像素投影找行 → 整行裁剪 → 直接喂 rec（一行一个文本串）
    V3 逐格rec  —— 行带内做列投影切出每个字符格 → 逐格裁剪 → 批量喂 rec
                   （既拿字符、又拿每格 x 坐标，是可落地的形态）

判据：V2/V3 的 `D` 与 `x` 召回是否显著高于 V1。
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

from extract.ocr_raster import _EngineAdapter, _load_engine, recognize_raster  # noqa: E402

IMG_PATH = os.path.join(_HERE, "original_right.png")


def detect_row_bands(gray: np.ndarray, min_h: int = 12, merge_gap: int = 9) -> list[tuple[int, int]]:
    """暗像素行投影 → 连续暗带 = 谱面行（含上方八度点与下方符杠）。

    与行与行之间约 30-40px 的空白相比，行内上标与主体只隔几像素，
    merge_gap=9 能把「八度点 + 字符 + 符杠」合成一个行带，
    又不会把两行谱并成一行。
    """
    dark = (gray < 150).sum(axis=1)
    thresh = max(3, int(dark.max() * 0.06))
    bands: list[list[int]] = []
    in_band = False
    start = 0
    for y, d in enumerate(dark):
        if d > thresh and not in_band:
            in_band, start = True, y
        elif d <= thresh and in_band:
            in_band = False
            if y - start >= min_h:
                bands.append([start, y])
    if in_band:
        bands.append([start, len(dark)])
    merged: list[list[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(a, b) for a, b in merged]


def split_columns(band: np.ndarray, dark_thresh: int = 150,
                  col_min_dark: int = 3, min_w: int = 4,
                  merge_gap: int = 3) -> list[tuple[int, int]]:
    """行带内列投影切字符格。

    关键：符杠是贴在字符下方的 1-2px 细横线，对列投影只贡献 1-2 个暗像素；
    字符笔画对一列贡献 ≥4 个暗像素。col_min_dark=3 即可把符杠滤掉，
    只留字符本体 → 相邻字符之间（哪怕同一组内）也能切开。
    """
    dark = (band < dark_thresh).sum(axis=0)
    cols = dark >= col_min_dark
    clusters: list[list[int]] = []
    in_c = False
    start = 0
    for x, on in enumerate(cols):
        if on and not in_c:
            in_c, start = True, x
        elif not on and in_c:
            in_c = False
            if x - start >= min_w:
                clusters.append([start, x])
    if in_c:
        clusters.append([start, len(cols)])
    merged: list[list[int]] = []
    for c in clusters:
        if merged and c[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    return [(a, b) for a, b in merged]


def count_chars(texts: list[str]) -> dict:
    joined = "".join(texts)
    return {
        "D": joined.count("D"),
        "x": joined.count("x") + joined.count("X"),
        "S": joined.count("S") + joined.count("s"),
        "digits": sum(joined.count(c) for c in "123456789"),
        "total": len(joined),
    }


def main() -> None:
    with open(IMG_PATH, "rb") as f:
        img_bytes = f.read()
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    gray = np.array(im.convert("L"))
    print(f"图片: {IMG_PATH}  尺寸={im.size}")

    engine = _EngineAdapter(_load_engine())

    # ── V1 基线：现有 det + rec 全流程 ─────────────────────────────
    print("\n=== V1 基线 det+rec（recognize_raster） ===")
    t0 = time.time()
    items = recognize_raster(img_bytes)
    print(f"框数={len(items)}  耗时={time.time() - t0:.1f}s")
    v1_texts = [str(it["text"]) for it in items]
    v1 = count_chars(v1_texts)
    print(f"字符统计: {v1}")
    for it in items[:20]:
        print(f"   y={int(it['yc']):4d} x={int(it['x0']):4d}  {it['text']!r}")

    # ── 找行带 ────────────────────────────────────────────────────
    bands = detect_row_bands(gray)
    print(f"\n行带（暗像素投影）: {len(bands)} 条")
    for a, b in bands:
        print(f"   y {a:4d} ~ {b:4d}  (h={b - a})")

    # ── V2 整行 → rec ─────────────────────────────────────────────
    print("\n=== V2 整行裁剪 → 直接 rec ===")
    crops = []
    pad = 6
    for (y0, y1) in bands:
        crop = np.array(im.crop((0, max(0, y0 - pad), im.width, min(im.height, y1 + pad))))
        crops.append(crop)
    t0 = time.time()
    results = engine.recognize(crops)
    print(f"耗时={time.time() - t0:.1f}s")
    v2_texts = []
    for (y0, y1), res in zip(bands, results):
        text = str(res[0]) if res and res[0] else ""
        conf = float(res[1]) if res and len(res) > 1 and res[1] is not None else 0.0
        v2_texts.append(text)
        print(f"   y={y0:4d} conf={conf:.2f}  {text!r}")
    v2 = count_chars(v2_texts)
    print(f"字符统计: {v2}")

    # ── V3 行带内列切分 → 逐格 rec ────────────────────────────────
    print("\n=== V3 行带内列投影切格 → 逐格 rec ===")
    cell_crops: list[np.ndarray] = []
    cell_meta: list[tuple[int, int, int]] = []  # (band_idx, x0, x1)
    for bi, (y0, y1) in enumerate(bands):
        band = gray[y0:y1]
        cols = split_columns(band)
        print(f"   行{bi} y={y0}: {len(cols)} 格  x范围={[(a, b) for a, b in cols][:24]}")
        for (x0, x1) in cols:
            crop = np.array(im.crop((x0, max(0, y0 - 2), x1, min(im.height, y1 + 2))))
            cell_crops.append(crop)
            cell_meta.append((bi, x0, x1))
    print(f"总格数={len(cell_crops)}")
    t0 = time.time()
    cell_results = engine.recognize(cell_crops)
    print(f"耗时={time.time() - t0:.1f}s")
    # 按行汇总
    v3_texts_by_band: list[str] = ["" for _ in bands]
    for (bi, x0, x1), res in zip(cell_meta, cell_results):
        text = str(res[0]).replace(" ", "") if res and res[0] else ""
        conf = float(res[1]) if res and len(res) > 1 and res[1] is not None else 0.0
        v3_texts_by_band[bi] += text
        if text:
            print(f"   行{bi} x={x0:4d}-{x1:4d} conf={conf:.2f}  {text!r}")
    v3 = count_chars(v3_texts_by_band)
    print("按行拼接:")
    for bi, t in enumerate(v3_texts_by_band):
        print(f"   行{bi}: {t!r}")
    print(f"字符统计: {v3}")

    # ── 汇总对比 ─────────────────────────────────────────────────
    print("\n=== 汇总：D / x 召回对比 ===")
    print(f"{'方案':<6}{'D':>4}{'x/X':>5}{'S':>4}{'digits':>7}{'total':>7}")
    for name, st in (("V1 det", v1), ("V2 整行", v2), ("V3 逐格", v3)):
        print(f"{name:<6}{st['D']:>4}{st['x']:>5}{st['S']:>4}{st['digits']:>7}{st['total']:>7}")


if __name__ == "__main__":
    main()
