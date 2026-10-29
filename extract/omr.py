"""谱面截图 → note_events（字符谱 OCR 通道，支持多页合并）。

为什么不是通用 OMR：
网上/App 里的手碟谱绝大多数不是五线谱，而是**字符网格**：
    D    = Ding（中心音）
    1-9  = 音位
    X    = 无音高敲击（手击边 / 闷音）
    T    = 拍击 / 泛音
这类谱本质是"能 OCR 的文字"，通用五线谱 OMR（oemer 等）反而识别不了。

关键做法：**按坐标落格，不是按顺序拼接**。
    OCR 会把相邻几个字符合并成一个文本框（如 "TXXDXDXD"），如果直接首尾相接，
    一旦某行少读/多读一个字符，后面整首都会错位。
    这里先估计单字符宽度，再用每个文本框的 x 起点算出它落在第几格，逐格填。

多页：一次最多 5 张，按上传顺序首尾相接；每张图从新的一行（小节线）开始。
"""
from __future__ import annotations

import io
import os
import re
import statistics
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_ENGINE = None
UPSCALE = 2.0            # 手机截图字太小，放大后再识别，召回率约翻倍
SLOTS_PER_BAR = 16       # 原谱一行 = 2 小节 × 16 格（4 拍 × 16 分音符）
BARS_PER_ROW = 2
MAX_PAGES = 5

SCORE_CHARS = re.compile(r"^[0-9DXxTtSsDd|Il]{1,}$")
BAR_NUMBER = re.compile(r"^\d{2,}$")


def _engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except ImportError as exc:
        raise RuntimeError("未安装 OCR 引擎：pip install rapidocr-onnxruntime") from exc
    _ENGINE = RapidOCR()
    return _ENGINE


def _ocr_items(arr) -> list[dict]:
    """numpy 图像 → [{"text","x0","x1","yc","h"}, ...]，兼容 rapidocr 两种 API。"""
    out = _engine()(arr)
    items: list[dict] = []

    def push(box, text):
        if not text:
            return
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
        except Exception:
            return
        items.append({"text": str(text), "x0": min(xs), "x1": max(xs),
                      "yc": sum(ys) / len(ys), "h": max(ys) - min(ys)})

    if isinstance(out, tuple):               # rapidocr-onnxruntime: (result, elapse)
        for row in out[0] or []:
            if len(row) >= 2:
                push(row[0], row[1])
        return items

    boxes = getattr(out, "boxes", None)      # rapidocr v3
    txts = getattr(out, "txts", None)
    if boxes is not None and txts is not None:
        for i, box in enumerate(boxes):
            push(box, txts[i])
        return items

    if isinstance(out, list):
        for row in out:
            if len(row) >= 2:
                push(row[0], row[1])
    return items


def _run_ocr(path: Path) -> list[dict]:
    """调试入口：读文件 → 放大 → OCR。"""
    img = Image.open(path).convert("RGB")
    if UPSCALE != 1.0:
        img = img.resize((int(img.width * UPSCALE), int(img.height * UPSCALE)), Image.LANCZOS)
    return _ocr_items(np.array(img))


def _group_rows(items: list[dict]) -> list[list[dict]]:
    if not items:
        return []
    heights = sorted(it["h"] for it in items if it["h"] > 0)
    tol = (heights[len(heights) // 2] if heights else 20) * 0.6
    rows: list[list[dict]] = []
    for it in sorted(items, key=lambda t: t["yc"]):
        if rows and abs(it["yc"] - rows[-1][0]["yc"]) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    for r in rows:
        r.sort(key=lambda t: t["x0"])
    return rows


def _clean(text: str) -> str:
    """清掉误识别的分隔符，只保留有效谱字符。"""
    out = []
    for ch in text.replace(" ", ""):
        if ch in "DXxTtSs" or ch in "123456789":
            out.append(ch)
        # 其它（| I l . 等）一律丢弃
    return "".join(out)


def _beam_lines(gray: "np.ndarray", item: dict) -> int:
    """数一个文本框下方有几条横线（符杠）。

    规则（用户定义）：一条横线 = 占 2 格，两条 = 占 1 格。
    这里只负责"看到几条"，占几格由上层解释。
    判据：文字下方一条横带里，若某一行有 ≥45% 宽度是暗像素，就算一条线。
    """
    h = max(item["h"], 1.0)
    y0 = int(item["yc"] + h * 0.45)
    y1 = int(item["yc"] + h * 1.65)
    x0 = max(0, int(item["x0"]))
    x1 = int(item["x1"])
    if x1 - x0 < 8 or y1 - y0 < 3:
        return 0
    band = gray[y0:y1, x0:x1] < 150
    if band.size == 0:
        return 0
    frac = band.mean(axis=1)
    runs = 0
    prev = False
    for v in frac > 0.45:
        if v and not prev:
            runs += 1
        prev = v
    return min(runs, 2)


def _row_tokens(row: list[dict], gray: "np.ndarray") -> list[dict]:
    """一行文本框 → [{"ch": 字符, "beams": 横线条数}, ...]，按 x 排序。"""
    cells: list[tuple[float, str, int]] = []
    for it in row:
        t = it["text"].replace(" ", "").strip()
        if not t or BAR_NUMBER.match(t):
            continue
        if not SCORE_CHARS.match(t):
            continue
        cleaned = _clean(t)
        if cleaned:
            cells.append((it["x0"], cleaned, _beam_lines(gray, it)))

    cells.sort(key=lambda c: c[0])
    return [{"ch": ch, "beams": b} for _x, s, b in cells for ch in s]


def _page_tokens(raw: bytes) -> tuple[list[dict], int]:
    """单张图 → 逐行落格后的 [{ch, beams}]，并统计符杠分布。"""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if UPSCALE != 1.0:
        img = img.resize((int(img.width * UPSCALE), int(img.height * UPSCALE)), Image.LANCZOS)
    arr = np.array(img)
    gray = np.array(img.convert("L"))

    rows = _group_rows(_ocr_items(arr))

    tokens: list[dict] = []
    used_rows = 0
    row_len = SLOTS_PER_BAR * BARS_PER_ROW

    for row in rows:
        toks = _row_tokens(row, gray)
        if not toks:
            continue
        used_rows += 1
        # 不补位：时值由横线决定，补位会虚增格数、把节奏拉长
        tokens.extend(toks)

    return tokens, used_rows


def image_to_tab(raw: bytes | list[bytes]) -> dict:
    """raw 可以是单张 bytes，也可以是按顺序排列的多张（最多 5 张）。

    时值**以输入图为准**：
        一条横线 = 占 2 格，两条 = 占 1 格，无横线 = 1 格。
    实测这张谱按此规则共 376 格 ≈ 23.5 小节（原谱 22），而"每字 1 格"只有 19 小节，
    说明横线确实代表时值，所以不再人为把每行压回 32 格。
    """
    engine_name = os.environ.get("OMR_ENGINE", "tab").lower().strip()
    pages = raw if isinstance(raw, list) else [raw]

    if engine_name in ("off", "none"):
        return {"events": [], "bars": 12, "bpm": 120, "warnings": [], "stats": {}}

    pages = pages[:MAX_PAGES]
    bpm = 120.0
    sec_per_slot = 60.0 / bpm / 4
    from score.note_mapper import D_KURD_9

    midi_of = {x["pos"]: x["midi"] for x in D_KURD_9}
    DUR_OF = {0: 1, 1: 2, 2: 1}

    events: list[dict] = []
    beam_dist = {0: 0, 1: 0, 2: 0}
    cursor = 0
    total_rows = 0
    rests = 0
    unpitched = 0
    empty_pages = 0

    for pi, page in enumerate(pages):
        toks, rows = _page_tokens(page)
        total_rows += rows
        if not toks:
            empty_pages += 1
            continue
        if pi and cursor % SLOTS_PER_BAR:
            # 每张图从整小节开始，避免跨图串拍
            cursor += SLOTS_PER_BAR - cursor % SLOTS_PER_BAR

        for tok in toks:
            ch = tok["ch"]
            beams = max(0, min(2, int(tok.get("beams") or 0)))
            if ch == ".":
                rests += 1
                cursor += 1
                continue

            beam_dist[beams] = beam_dist.get(beams, 0) + 1
            dur = DUR_OF[beams]
            start = round(cursor * sec_per_slot, 3)
            cursor += dur

            if ch in "XxTtSs":
                unpitched += 1
                events.append({"start": start, "pitch": None, "kind": "stroke",
                               "label": "S", "beams": beams, "dur": dur, "amp": 0.8})
                continue
            if ch in "Dd":
                events.append({"start": start, "pitch": 50, "kind": "ding",
                               "label": "D", "beams": beams, "dur": dur, "amp": 0.9})
                continue
            midi = midi_of.get(int(ch)) if ch.isdigit() else None
            if midi is not None:
                events.append({"start": start, "pitch": midi,
                               "beams": beams, "dur": dur, "amp": 0.9})

    if not events:
        return {
            "events": [], "bars": 12, "bpm": bpm,
            "stats": {"chars": 0, "pages": len(pages)},
            "warnings": ["没能从这张图里读出字符谱。请确认：① 截图清晰不解屏；② 谱面是 D / T / X + 数字的字符谱"],
        }

    bars = max(1, (cursor + SLOTS_PER_BAR - 1) // SLOTS_PER_BAR)
    pitched = sum(1 for e in events if e.get("kind") != "stroke")

    head_txt = f"从 {len(pages)} 张图里" if len(pages) > 1 else "从图里"
    warnings = [
        f"{head_txt}读出 {total_rows} 行：{pitched} 个有音高，{unpitched} 个无音高敲击",
        "时值按输入图的横线排：一条占 2 格、两条占 1 格、无横线占 1 格",
        f"排布结果 {cursor} 格 ≈ {bars} 小节（应与原谱一致；有偏差说明横线检测有误，请对照原图校对）",
    ]
    if beam_dist[1] + beam_dist[2]:
        warnings.append(f"符杠：一条 {beam_dist[1]} 个、两条 {beam_dist[2]} 个")
    else:
        warnings.append("未检测到符杠横线（字太细），本页按每字 1 格排版")
    if len(pages) > 1:
        warnings.append(f"{len(pages)} 张图已按上传顺序首尾相接")
    if empty_pages:
        warnings.append(f"其中 {empty_pages} 张没读出内容，已跳过")

    return {
        "events": events,
        "bars": min(bars, 240),
        "bpm": bpm,
        "cells": SLOTS_PER_BAR,
        "warnings": warnings,
        "stats": {"chars": cursor, "pitched": pitched, "unpitched": unpitched,
                  "rests": rests, "rows": total_rows, "pages": len(pages)},
    }
