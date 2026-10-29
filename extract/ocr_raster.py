# -*- coding: utf-8 -*-
"""分块检测 + 放大裁剪识别：本地 OCR 通道的省内存实现。

为什么不用「整图 2x 检测」
------------------------
早期版本把手机截图整体放大 2 倍（1080×2412 → 2160×4824）再检测，
DB 检测的中间特征图 + onnxruntime arena 把峰值内存推到 **1.6GB**。
Render Free 只有 512MB，实例必被 OOM 重启，后台任务永远停在「提取中」
（用户看到「图片一直在读取中」）。

本模块的做法
------------
    det —— 在**原始分辨率**上把图竖向切成若干块，逐块检测。
           像素密度不变 → 框的划分与整图检测一致；
           单块面积小 → 内存断崖下降（这是内存能降下来的唯一原因）。
    rec —— 从 **2x 放大图**上按 det 框裁剪后识别。
           rec 的输入固定 48px 高，从 2x 图裁剪能让字符占更多像素，识别更准。

实测（1080×2412 手机截图，测试用谱面）：
    整图 2x  → 峰值 1682MB，309 字符，24 小节，9.5s
    分 4 块  → 峰值  316MB，310 字符，24 小节，6.5s     ← 本方案
（字符/小节数与整图 2x 一致，内存降到 1/5，速度快 1.5 倍）

对外契约
--------
返回的框坐标一律换算回**原图像素坐标系**，上层 `omr.py` 的落格、
符杠、排版逻辑完全不用感知本模块的存在。
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


# 分块数（竖向）。4 块时单块约 1080×603，实测内存/召回的最佳平衡点。
TILES = _env_int("OCR_TILES", 4)
# 相邻块的重叠像素，防止跨块的那一行被切成两半。
TILE_OVERLAP = _env_int("OCR_TILE_OVERLAP", 120)
# 识别用的放大倍数（只影响裁剪块，不影响 det 的像素密度）。
REC_SCALE = _env_float("OCR_REC_SCALE", 1.0)
# 单个检测框的识别置信度下限。
MIN_SCORE = _env_float("OCR_MIN_SCORE", 0.5)
# 目标块高（像素）。块高 ≈ 600 时 det 的等效放大倍数一致 → 结果最稳。
TARGET_BLOCK_H = _env_int("OCR_BLOCK_H", 600)


def _row_valleys(gray: np.ndarray,
                 win: int = 15,
                 ratio: float = 0.35) -> list[int]:
    """行投影低谷：行与行之间空白处的 y 坐标。

    谱面的每一行是「字符 + 下划线符杠」组成的横条，行与行之间有空白。
    用行投影（每行暗像素计数）平滑后找明显低于局部均值的低谷，
    就是行间空白 —— 在这里切块，**永远不会把一行切成两半**。

    为什么必须这样做：det 在不同尺寸的块上行为不同，固定高度切块会
    切断行 → det 在半行上产出碎片框 → 音符虚高、小节数漂移
    （实测 tiles=4 → 24 小节，tiles=8 → 34 小节）。
    """
    if gray.ndim != 2 or gray.shape[0] < win * 2:
        return []
    dark = (gray < 150).sum(axis=1).astype(np.float64)
    kernel = np.ones(win) / win
    smooth = np.convolve(dark, kernel, mode="same")
    cand = np.where(dark < smooth * ratio)[0]
    groups: list[list[int]] = []
    for y in cand:
        if groups and y - groups[-1][-1] <= 5:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])
    return sorted(int(round(float(np.mean(g)))) for g in groups)


def _plan_blocks(gray: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    """规划分块：等高切分 + 重叠。

    为什么不用「低谷切块」（在行与行之间的空白处切）：
      实验过，行带切块 + DP 均衡切点能让**框数**几乎等于整图检测
      （120 vs 121），但**小节数从 24 掉到 23** —— 少了一个小节，
      而 24 才是谱面真值（谱面左侧小节编号 24~47）。
      原因：切块后行被拆成两半，det 在半行上会产出一个**只覆盖半行的框**，
      rec 识别它时字符被截断 → 落格时少一格 → 少一个小节。
      固定高度切块 + 足够重叠（≥ 一行高）能保证每行至少有一块是完整的，
      实测 311 音符 / 24 小节，与谱面真值吻合。

    保留低谷检测作为**块边界微调**的备选（_row_valleys 仍可用），
    但默认走等高切分 —— 结果更准。
    """
    ntiles = max(1, TILES)
    if TARGET_BLOCK_H > 0:
        auto = int(round(height / float(TARGET_BLOCK_H)))
        if auto >= 1:
            ntiles = max(1, min(12, auto))
    if ntiles <= 1:
        return [(0, height)]

    tile_h = max(1, height // ntiles)
    overlap = max(0, TILE_OVERLAP)
    spans: list[tuple[int, int]] = []
    for ti in range(ntiles):
        y0 = max(0, ti * tile_h - overlap)
        y1 = min(height, (ti + 1) * tile_h + overlap)
        if y1 - y0 >= 20:
            spans.append((y0, y1))
    return spans or [(0, height)]


def _load_engine():
    """加载 RapidOCR 引擎。

    兼容 rapidocr-onnxruntime 的两代 API：
        1.2.x（本地验证版）：engine.text_detector / engine.text_recognizer
        1.4.x（新版）：      engine.text_det      / engine.text_rec
    两者签名一致（返回 (结果, 耗时)），只是属性名不同。
    """
    from rapidocr_onnxruntime import RapidOCR  # type: ignore

    return RapidOCR()


class _EngineAdapter:
    """把两代 RapidOCR 的属性名差异统一掉。

    1.2.x: engine.text_detector(img)  → (boxes, elapse)
            engine.text_recognizer(crops) → ([(text, conf), ...], elapse)
            engine.get_crop_img_list(img, boxes) → [crop, ...]
    1.4.x: engine.text_det(img) / engine.text_rec(crops)，其余一致。
    """

    def __init__(self, engine):
        self._e = engine
        self._det = getattr(engine, "text_detector", None)
        self._rec = getattr(engine, "text_recognizer", None)
        if self._det is None:
            self._det = getattr(engine, "text_det", None)
        if self._rec is None:
            self._rec = getattr(engine, "text_rec", None)
        if self._det is None or self._rec is None:
            raise RuntimeError(
                "RapidOCR API 不兼容：找不到检测/识别入口"
                f"（text_detector/text_det={self._det is not None}, "
                f"text_recognizer/text_rec={self._rec is not None}）"
            )

    def detect(self, img):
        out = self._det(img)
        return out[0] if isinstance(out, tuple) else out

    def recognize(self, crops):
        out = self._rec(crops)
        return out[0] if isinstance(out, tuple) else out

    def crop(self, img, boxes):
        return self._e.get_crop_img_list(img, boxes)


def _dedup(items: list[dict]) -> list[dict]:
    """去掉重叠块里重复识别到的同一个框。

    同一个字组可能同时落在相邻两块里，各被检测一次 → 文本与位置几乎相同。

    判据：**文本相同 + 垂直位置接近 + 左边界接近**。
    曾经试过改成「纯几何重叠率」判据（不看文本），结果**过度去重**：
    1080px 原图从 311 音符 / 24 小节掉到 300 音符 / 23 小节，
    而 24 小节才是谱面真值（谱面左侧小节编号 24~47）。
    所以坚持用文本判据 —— 它更保守，保留的是真框。
    """
    items.sort(key=lambda it: (it["yc"], it["x0"]))
    out: list[dict] = []
    for it in items:
        dup = False
        for d in out:
            if (d["text"] == it["text"]
                    and abs(d["yc"] - it["yc"]) < max(8.0, d["h"] * 0.6)
                    and abs(d["x0"] - it["x0"]) < 12.0):
                dup = True
                break
        if not dup:
            out.append(it)
    return out


def recognize_raster(image_bytes: bytes, timing: dict | None = None) -> list[dict]:
    """识别一张谱面截图，返回原图坐标系的 OcrItem 列表。

    OcrItem = {"text", "x0", "x1", "yc", "h"}（与 ocr_provider 的契约一致）

    `timing` 若传入 dict，会写入分段耗时：
        decode_ms —— 图片解码（PIL 打开 + 转 numpy）
        infer_ms  —— 模型推理（det + rec）
    """
    import time as _time

    t_dec = _time.time()
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"图片解码失败：{exc}") from exc

    width, height = im.size
    if width <= 0 or height <= 0:
        return []

    base = np.array(im)
    gray = np.array(im.convert("L"))
    if timing is not None:
        timing["decode_ms"] = int((_time.time() - t_dec) * 1000)
    t_inf = _time.time()

    scale = REC_SCALE if REC_SCALE and REC_SCALE != 1.0 else 1.0
    if scale == 1.0:
        rec_img = base
    else:
        rec_img = np.array(
            im.resize((max(1, int(width * scale)), max(1, int(height * scale))),
                      Image.LANCZOS)
        )

    engine = _EngineAdapter(_load_engine())
    spans = _plan_blocks(gray, height, width)

    items: list[dict] = []
    for (y0, y1) in spans:
        strip = base[y0:y1]
        if strip.size == 0:
            continue

        try:
            boxes = engine.detect(strip)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"本地 OCR 检测失败：{exc}") from exc
        if boxes is None or len(boxes) == 0:
            continue

        # det 框（原图系，块内坐标）→ 放大图坐标
        boxes_rec = np.asarray(boxes, dtype=np.float32).copy()
        boxes_rec[:, :, 0] *= scale
        boxes_rec[:, :, 1] *= scale
        boxes_rec[:, :, 1] += y0 * scale

        try:
            crops = engine.crop(rec_img, boxes_rec)
            results = engine.recognize(crops)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"本地 OCR 识别失败：{exc}") from exc

        for box, res in zip(boxes_rec, results):
            try:
                text = str(res[0]).replace(" ", "").strip()
                score = float(res[1])
            except Exception:  # noqa: BLE001
                continue
            if not text or score < MIN_SCORE:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            # 换算回**原图**坐标（放大倍数要还原）
            items.append({
                "text": text,
                "x0": min(xs) / scale,
                "x1": max(xs) / scale,
                "yc": (sum(ys) / len(ys)) / scale,
                "h": (max(ys) - min(ys)) / scale,
            })

    if timing is not None:
        timing["infer_ms"] = int((_time.time() - t_inf) * 1000)
    return _dedup(items)
