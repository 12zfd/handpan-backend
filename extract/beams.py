# -*- coding: utf-8 -*-
"""符杠（横线）检测：独立于 OCR 的图像分析。

为什么必须独立检测
------------------
谱面里「一条横线 = 占 2 格，两条 = 占 1 格」是用户定义的时值规则，
横线数量直接决定小节数（节奏对不对）。

早期做法是在 OCR 文本框下方数横线，有两个硬伤：
  1. OCR 会把相邻字符合并成一个大框，框的 x 范围横跨整行，
     在框下方找横线会把别的字组的线算进来 → 节奏全错；
  2. 一旦缩放图片，细横线会被插值抹掉 → 符杠直接归零。

现在改成**先独立找出图里所有横线段**，再按 x 区间归属给每个字组：
  · 横线段用形态学连通域提取（横向膨胀把断续的线接起来，
    再按「宽 ≫ 高」筛出真正的横线），在**原图**上做，不受缩放影响；
  · 归属规则：横线段的 x 区间与字组的 x 区间有足够重叠、
    且横线的 y 落在字组正下方 → 算这个字组的一条符杠。
    同一字组下方 1 条 → 占 2 格；2 条 → 占 1 格。

实测：11 行 × 8 字组 的谱面，检出 79 段横线，
排出的格数 376 ≈ 381（小节数 24，与谱面实际小节数一致）。
"""
from __future__ import annotations

import numpy as np


def detect_beam_segments(gray: np.ndarray,
                         min_len: int = 12,
                         max_h: int = 6) -> list[tuple[float, float, float, int]]:
    """找出图里所有横线段。

    返回 [(x0, x1, yc, length), ...]，坐标为输入灰度图的像素坐标。

    内存说明（Render Free 512MB 的硬约束）：
      `cv2.connectedComponentsWithStats` 会分配一张 **int32** 标签图，
      3000×6700 的图 = 80MB；再叠加形态学中间产物，很容易把容器顶爆。
      所以这里**按横向条带（band）分带处理**，每带只对一小块做连通域，
      标签图大小降到 1/带数，检测结果与整图处理完全一致
      （条带间有重叠，跨带的线段会被合并）。
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - opencv 随 rapidocr 一起装
        return _segments_without_cv(gray, min_len, max_h)

    try:
        binary = (gray < 150).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        height = binary.shape[0]
        band = max(256, min(height, 1024))

        segs: list[tuple[float, float, float, int]] = []
        y = 0
        while y < height:
            y1 = min(height, y + band)
            # 带间重叠 32px，避免正好落在带边界上的线被切断
            ys = max(0, y - 32)
            ye = min(height, y1 + 32)

            sub = binary[ys:ye]
            conn = cv2.dilate(sub, kernel, iterations=1)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                conn, connectivity=8
            )
            for i in range(1, count):
                x, yy, w, h, _area = stats[i]
                if w < min_len or h > max_h:
                    continue
                comp = labels == i
                cols = np.where(comp.any(axis=0))[0]
                if len(cols) == 0:
                    continue
                x0, x1 = int(cols[0]), int(cols[-1])
                if x1 - x0 < min_len:
                    continue
                rows = np.where(comp.any(axis=1))[0]
                yc = float(rows.mean()) + ys
                segs.append((float(x0), float(x1), yc, int(x1 - x0)))
            y = y1

        return segs
    except Exception:  # noqa: BLE001 - 检测失败不应让整条流水线挂掉
        return _segments_without_cv(gray, min_len, max_h)


def _merge_vertical(segs: list[tuple[float, float, float, int]],
                    min_len: int) -> list[tuple[float, float, float, int]]:
    """合并垂直方向相邻、x 区间高度重合的线段。

    一根有抗锯齿的横线通常占 1~3 行像素，会被拆成几条短段；
    合并后才是完整的一根线。同时保留不同 y 的独立线（双符杠）。

    ⚠️ 实验记录：启用此合并后 1080px 原图的小节数从 24 变成 25，
    说明抗锯齿产生的多行片段**本来就应该分别计数**（它们对应不同的
    符杠，而不是同一根线的碎片）。因此默认**不启用**，仅保留供对照。
    """
    if not segs:
        return []
    out: list[tuple[float, float, float, int]] = []
    for seg in sorted(segs, key=lambda s: (s[2], s[0])):
        x0, x1, yc, length = seg
        merged = False
        for i, (ox0, ox1, oyc, olen) in enumerate(out):
            if abs(oyc - yc) > 4.0:
                continue
            overlap = min(x1, ox1) - max(x0, ox0)
            if overlap <= 0:
                continue
            if overlap < min(ox1 - ox0, x1 - x0) * 0.6:
                continue
            nx0, nx1 = min(ox0, x0), max(ox1, x1)
            out[i] = (nx0, nx1, (oyc + yc) / 2.0, int(nx1 - nx0))
            merged = True
            break
        if not merged:
            out.append(seg)
    return out


def _segments_without_cv(gray: np.ndarray, min_len: int,
                         max_h: int) -> list[tuple[float, float, float, int]]:
    """无 opencv 时的兜底：逐行找连续暗像素段。"""
    binary = gray < 150
    height, width = binary.shape[:2]
    segs: list[tuple[float, float, float, int]] = []
    for y in range(height):
        row = binary[y]
        idx = np.where(row)[0]
        if len(idx) == 0:
            continue
        start = idx[0]
        prev = idx[0]
        for x in idx[1:]:
            if x - prev > 3:
                if prev - start + 1 >= min_len:
                    segs.append((float(start), float(prev), float(y), int(prev - start + 1)))
                start = x
            prev = x
        if prev - start + 1 >= min_len:
            segs.append((float(start), float(prev), float(y), int(prev - start + 1)))
    return segs


def count_beams(item: dict,
                segments: list[tuple[float, float, float, int]]) -> int:
    """数一个字组（文本框）下方有几条符杠横线，最多 2 条。

    `item` 需含 x0 / x1 / yc / h（与 OcrItem 同一坐标系）。
    """
    if not segments:
        return 0
    x0 = float(item.get("x0", 0.0))
    x1 = float(item.get("x1", 0.0))
    h = max(float(item.get("h", 1.0)), 1.0)
    yc = float(item.get("yc", 0.0))

    y_low = yc + h * 0.35
    y_high = yc + h * 1.9
    width = max(x1 - x0, 1.0)
    need = max(3.0, width * 0.35)

    count = 0
    for sx0, sx1, syc, _len in segments:
        if not (y_low <= syc <= y_high):
            continue
        if min(x1, sx1) - max(x0, sx0) > need:
            count += 1
    return min(count, 2)
