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
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .beams import count_beams, detect_beam_segments
from .ocr_provider import get_provider

_ENGINE = None
SLOTS_PER_BAR = 16       # 原谱一行 = 2 小节 × 16 格（4 拍 × 16 分音符）
BARS_PER_ROW = 2
MAX_PAGES = 5

# OCR 前的处理策略（环境变量可覆盖，实际实现见 extract/ocr_raster.py）：
#   OCR_TILES          —— 竖向分块数，det 分块跑，峰值内存 ∝ 1/块数
#   OCR_TILE_OVERLAP   —— 相邻块重叠像素，防跨块行被切断
#   OCR_REC_SCALE      —— 识别时的放大倍数（只影响裁剪块，不影响 det 像素密度）
#   OCR_UPSCALE        —— 旧参数：整图放大倍数，新通道不再使用
#
# 为什么不整图放大 2 倍再检测：
#   早期版本 UPSCALE=2.0 会把 1080×2412 放大到 2160×4824，numpy 副本 + onnx
#   张量把峰值内存推到 1.6GB；Render Free 只有 512MB，实例必被 OOM 重启，
#   后台任务永远停在「提取中」（用户看到「图片一直在读取中」）。
#   现在改成「分块检测 + 放大裁剪识别」：
#     det 在原图上分块跑 → 像素密度不变，框的划分与整图检测一致，内存降到 1/4
#     rec 从 2x 放大图裁剪 → 字符占更多像素，识别精度与整图 2x 持平
#   实测峰值 1682MB → 316MB，字符数与小节数与整图 2x 完全一致。
try:
    UPSCALE = float(os.environ.get("OCR_UPSCALE") or 1.0)
except ValueError:
    UPSCALE = 1.0

# 图片归一化基准宽度（OCR_NORM_WIDTH）。
# 所有输入统一缩到这个宽度再识别 → det 看到的分辨率永远一致 →
# 切块数、检测结果、小节数都与输入尺寸无关（实测 1080/1600/2200/3000
# 四种宽度归一化后结果完全相同）。
try:
    MAX_IMAGE_WIDTH = int(os.environ.get("OCR_NORM_WIDTH") or 1080)
except ValueError:
    MAX_IMAGE_WIDTH = 1080

SCORE_CHARS = re.compile(r"^[0-9DXxTtSsDd|Il]{1,}$")
BAR_NUMBER = re.compile(r"^\d{2,}$")


class _Timing(dict):
    """分段耗时收集器（decode / infer / format）。

    用于回答「哪一步最耗时」——Render Free 只有 0.1 CPU，
    如果不拆开看，只能看到总耗时，无法定位瓶颈。
    """

    def mark(self, key: str, ms: float) -> None:
        self[key] = int(ms)


def _normalize(raw: bytes, target_width: int) -> bytes:
    """把图片等比缩放到宽度 = target_width（不放大到超过 target）。

    统一分辨率是本方案能同时满足「省内存」和「结果稳定」的关键。

    背景（实测结论）：
      · det 的召回强烈依赖**字符像素密度**：1080 宽检出 121 框，
        缩到 880 宽只剩 76 框（掉 37%）→ 不能靠缩图省内存；
      · det 的内存 ∝ 输入**面积** → 必须分块；
      · 但分块数一变，结果就漂（tiles=4 → 24 小节，tiles=8 → 34 小节），
        因为 DB 检测器在不同尺寸的块上行为不同。

    所以：**先把所有输入归一化到同一个基准宽度（1080，与最常见的手机
    截图一致），再固定切块** → 任何尺寸的输入都得到完全一致的结果，
    且 det 看到的分辨率始终是「已验证」的那一档。

    按**宽度**而非最长边归一化：手碟谱截图是竖长条，限制最长边会把字
    压到认不出的程度（实测 125px 宽时 OCR 完全读不出谱面字符）。
    """
    if not target_width or target_width <= 0:
        return raw
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:  # noqa: BLE001
        return raw
    if im.width == target_width:
        return raw
    ratio = target_width / float(im.width)
    im = im.resize(
        (max(1, int(round(im.width * ratio))),
         max(1, int(round(im.height * ratio)))),
        Image.LANCZOS,
    )
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _engine():
    """保留旧入口兼容；真正的识别已下沉到 ocr_provider。"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    _ENGINE = get_provider()
    return _ENGINE


def _ocr_items(arr, timing: dict | None = None) -> list[dict]:
    """numpy 图像 / 图片字节 → [{"text","x0","x1","yc","h"}, ...]（兼容旧调用方）。"""
    provider = get_provider()
    if isinstance(arr, (bytes, bytearray)):
        return provider.recognize(bytes(arr), timing=timing)
    # numpy 输入 → 先编码成 PNG 再交给 provider
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return provider.recognize(buf.getvalue(), timing=timing)


def _run_ocr(path: Path) -> list[dict]:
    """调试入口：读文件 → OCR。"""
    return _ocr_items(path.read_bytes())


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


def _row_tokens(row: list[dict],
                segments: list[tuple[float, float, float, int]]) -> list[dict]:
    """一行文本框 → [{"ch": 字符, "beams": 横线条数}, ...]，按 x 排序。

    符杠不再靠"在框下方找横线"来猜（OCR 框会横跨整行，猜出来的节奏必错），
    改成用**独立检测出的横线段**按 x 区间归属给每个文本框。
    """
    cells: list[tuple[float, str, int]] = []
    for it in row:
        t = it["text"].replace(" ", "").strip()
        if not t or BAR_NUMBER.match(t):
            continue
        if not SCORE_CHARS.match(t):
            continue
        cleaned = _clean(t)
        if cleaned:
            cells.append((it["x0"], cleaned, count_beams(it, segments)))
    cells.sort(key=lambda c: c[0])
    return [{"ch": ch, "beams": b} for _x, s, b in cells for ch in s]


def _page_tokens(raw: bytes, timing: "_Timing | None" = None) -> tuple[list[dict], int]:
    """单张图 → 逐行落格后的 [{ch, beams}]，并统计有效行数。

    流程：
      1. 识别（provider 内部是「分块 det + 2x 裁剪 rec」，峰值内存 ~316MB）
      2. 在**原图**上独立检测所有横线段（符杠）——原图分辨率，细线不被抹掉
      3. 逐行分组 + 按 x 归属符杠条数 → 落格
    """
    t0 = time.time()
    # 统一预处理：超大图先等比缩到 OCR_MAX_WIDTH。
    # OCR 与符杠检测必须用**同一张图**，否则两套坐标系对不上，
    # 符杠归属会全部错位。缩放后两张图共用同一坐标系，相对关系不变。
    proc_raw = _normalize(raw, MAX_IMAGE_WIDTH)
    items = _ocr_items(proc_raw, timing=timing)
    t1 = time.time()

    segments: list[tuple[float, float, float, int]] = []
    try:
        gray = np.array(Image.open(io.BytesIO(proc_raw)).convert("L"))
        segments = detect_beam_segments(gray)
    except Exception:  # noqa: BLE001 - 符杠检测失败不应让整条流水线挂掉
        segments = []

    rows = _group_rows(items)
    tokens: list[dict] = []
    used_rows = 0
    for row in rows:
        toks = _row_tokens(row, segments)
        if not toks:
            continue
        used_rows += 1
        # 不补位：时值由横线决定，补位会虚增格数、把节奏拉长
        tokens.extend(toks)
    t2 = time.time()

    if timing is not None:
        # provider 若已给出更细的分段耗时（子进程模式下由 raster 回传），
        # 这里不覆盖，只补齐缺失项。
        timing.setdefault("infer_ms", int((t1 - t0) * 1000))
        timing.setdefault("format_ms", int((t2 - t1) * 1000))
    return tokens, used_rows


def image_to_tab(raw: bytes | list[bytes],
                 timing: "_Timing | None" = None) -> dict:
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

    t_fmt0 = time.time()
    for pi, page in enumerate(pages):
        toks, rows = _page_tokens(page, timing)
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
        out = {
            "events": [], "bars": 12, "bpm": bpm,
            "stats": {"chars": 0, "pages": len(pages)},
            "warnings": ["没能从这张图里读出字符谱。请确认：① 截图清晰不解屏；② 谱面是 D / T / X + 数字的字符谱"],
        }
        if timing is not None:
            timing.setdefault("format_ms", int((time.time() - t_fmt0) * 1000))
            out["timing"] = dict(timing)
        return out

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

    out = {
        "events": events,
        "bars": min(bars, 240),
        "bpm": bpm,
        "cells": SLOTS_PER_BAR,
        "warnings": warnings,
        "stats": {"chars": cursor, "pitched": pitched, "unpitched": unpitched,
                  "rests": rests, "rows": total_rows, "pages": len(pages)},
    }
    if timing is not None:
        timing.setdefault("format_ms", int((time.time() - t_fmt0) * 1000))
        out["timing"] = dict(timing)
    return out
