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
# det 专用图宽度（0 = 不分离）。
#
# 实测结论：**不要用这个参数**。
# 把 det 图从 1080 宽缩到 900/800 宽，音符数从 311 掉到 257/288，
# 小节数从 24 掉到 20/19 —— 谱面里的字符本来就小，缩 det 图等于
# 直接丢字。而 det 的 `limit_type='min'` 语义决定：
#     if min(h, w) < limit_side_len(736):  ratio = 736 / min(h, w)   ← 放大
#     else:                                ratio = 1                    ← 不缩放
# 分块后块高 ≈ 723（< 736）→ 放大倍数只有 1.018 倍，
# 几乎不浪费算力（实测 limit=736 与 limit=603 的框数完全相同：167）。
# 所以 det 图保持原分辨率即可，此参数留作实验开关（生产值 0）。
DET_WIDTH = _env_int("OCR_DET_WIDTH", 0)


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

    建 session **之前**先调 ort_tune：把 onnxruntime 的 intra_op 线程数
    限制为 1 —— 默认开满物理核会让 80% 的 CPU 烧在自旋等待上
    （实测 threads=16 → 50.7 CPU 秒，threads=1 → 10.7 CPU 秒）。
    Render Free 只有 0.1 CPU，这一步是把 150s 压进 60s 的关键。
    """
    from .ort_tune import tune_ort

    tune_ort()

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


def recognize_raster(image_bytes: bytes, timing: dict | None = None,
                     engine=None) -> list[dict]:
    """识别一张谱面截图，返回原图坐标系的 OcrItem 列表。

    OcrItem = {"text", "x0", "x1", "yc", "h"}（与 ocr_provider 的契约一致）

    `timing` 若传入 dict，会写入分段耗时：
        decode_ms —— 图片解码（PIL 打开 + 转 numpy）
        infer_ms  —— 模型推理（det + rec）
    `engine` 传入已建好的 _EngineAdapter 时复用（网格通道与 det 通道
    共享同一个引擎，省一次 ~1s 的模型加载）。
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

    # ── det 图：可与 rec 图分离（det 更便宜，rec 精度不降）
    det_w = DET_WIDTH if (DET_WIDTH and DET_WIDTH > 0 and DET_WIDTH < width) else width
    if det_w == width:
        det_base = base
        det_h = height
        back = 1.0                      # det 坐标 → rec 坐标 的放大倍数
    else:
        det_h = max(1, int(round(height * det_w / float(width))))
        det_base = np.array(
            Image.fromarray(base).resize((det_w, det_h), Image.LANCZOS)
        )
        back = width / float(det_w)

    engine = engine if engine is not None else _EngineAdapter(_load_engine())
    spans = _plan_blocks(gray, det_h, det_w)

    items: list[dict] = []
    for (y0, y1) in spans:
        strip = det_base[y0:y1]
        if strip.size == 0:
            continue

        try:
            boxes = engine.detect(strip)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"本地 OCR 检测失败：{exc}") from exc
        if boxes is None or len(boxes) == 0:
            continue

        # det 框（det 图坐标）→ rec 图坐标
        boxes_rec = np.asarray(boxes, dtype=np.float32).copy()
        boxes_rec[:, :, 0] *= scale * back
        boxes_rec[:, :, 1] *= scale * back
        boxes_rec[:, :, 1] += y0 * scale * back

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


# ═════════════════════════════════════════════════════════════════
# 方案 A：网格通道（跳过 det，连通域找字符格 → 逐格 rec）
#
# 为什么需要第二条通道
# --------------------
# 真实书页拍照谱（如 D Kurd 9音《晴天》，米色纸底 + 水印圈 + 八度点）
# 上，det 通道系统性失败（_test/probe_rec_only.py 实测）：
#     - `D` 召回 0/4，`x` 大量丢失，后 4 行整行丢（框数 57 且集中在
#       上半页）；放大反而更糟（整页 60 框 → 48 框，单行放大 → 0 框）。
#     - 整行直接喂 rec（V2）字符召回反而极好（conf 0.83~0.86）——
#       瓶颈在 det，不在 rec。
#
# 本通道的取巧：谱面字符是**纸上孤立的墨块**，不需要 det 找框 ——
# 二值化 + 连通域就能精确拿到每个字符的包围盒，再逐格喂 rec：
#     实测（original_right.png，_test/probe_v4_components.py）：
#     D 召回 0 → 4/4（100%），8 个谱面行全部还原，x/S 基本齐全。
#
# 与 det 通道的关系
# -----------------
#     - 输出契约完全一致（OcrItem），omr.py 的落格机制两通道共用；
#     - 上层路由见 omr._page_tokens：网格有效行 > det 有效行时用网格；
#     - 应用截图（纯白底、橙色音符）会被照片守卫拦下，恒走 det 通道，
#       原有校准结果零影响。
# ═════════════════════════════════════════════════════════════════

# 二值化阈值（灰度 < 阈值算墨迹）。水印圈 / 纸纹 ~200+，不会入域。
GRID_BIN_THRESHOLD = _env_int("OCR_GRID_BIN", 150)
# 面积过滤：连通域面积 < 全页主字符面积中位数 × 该系数 → 八度点/杂点，丢。
GRID_AREA_FRACTION = _env_float("OCR_GRID_AREA_FRAC", 0.45)
# 行过滤：一行主字符高度中位数 < 全页最高行 × 该系数 → 八度点行，丢。
GRID_ROW_H_RATIO = _env_float("OCR_GRID_ROW_H", 0.75)
# 一行至少这么多个字符才算谱面行（八度点行通常 ≤ 8 个）。
GRID_MIN_ROW_CHARS = _env_int("OCR_GRID_MIN_CHARS", 4)
# 照片守卫：近白像素（灰度 ≥ 248）占比 ≥ 该阈值 → 判为应用截图，
# 本通道直接让路（恒走 det 通道）。OCR_GRID_FORCE=1 可强制跑（测试用）。
GRID_PHOTO_WHITE_FRAC = _env_float("OCR_GRID_WHITE_FRAC", 0.60)

# rec 常把 ×（U+00D7）读出来、全角数字也偶现，统一折算。
_GRID_CHAR_MAP = str.maketrans({
    "×": "x", "Ⅹ": "x", "✕": "x", "✖": "x",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})


def _looks_like_photo(gray: np.ndarray) -> bool:
    """书页拍照谱 vs 应用截图的廉价判别：近白像素占比。

    米色纸底的照片近白占比 ≈ 0；应用截图（白底）≈ 0.9+。
    """
    try:
        return float((gray >= 248).mean()) < GRID_PHOTO_WHITE_FRAC
    except Exception:  # noqa: BLE001
        return False


def _grid_clean(text: str) -> str:
    """网格通道单字符归一：全角→半角、×→x，去空白。"""
    return str(text).translate(_GRID_CHAR_MAP).replace(" ", "").strip()


def _grid_components(gray: np.ndarray) -> tuple[list[dict], dict]:
    """二值化 + 连通域 → 主字符候选框（八度点/符杠/小节线已滤除）。"""
    import cv2  # rapidocr 的传递依赖，缺失时网格通道整体让路

    binimg = (gray < GRID_BIN_THRESHOLD).astype(np.uint8)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    comps: list[dict] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area <= 0:
            continue
        comps.append({"x0": x, "y0": y, "w": w, "h": h,
                      "area": area, "x1": x + w, "y1": y + h,
                      "yc": y + h / 2.0})
    if not comps:
        return [], {"components": 0}

    # 高度中位数：横线/竖线是少数派，中位数落在主字符上
    hs = sorted(c["h"] for c in comps)
    med_h = hs[len(hs) // 2]

    # 第一轮：长横线（符杠）与长竖线（小节线/纸边框）。
    # 用相对阈值（×中位数），字符 '1'（细高但高度 ≈ 中位数）不受误伤。
    cands: list[dict] = []
    beams = vlines = 0
    for c in comps:
        if c["w"] >= 4 * c["h"] and c["w"] >= 1.6 * med_h:
            beams += 1
            continue
        if c["h"] >= 4 * c["w"] and c["h"] >= 1.6 * med_h:
            vlines += 1
            continue
        cands.append(c)

    # 第二轮：面积过滤（八度点等小件）
    areas = sorted(c["area"] for c in cands)
    med_area = areas[len(areas) // 2] if areas else 0
    main: list[dict] = []
    dots = 0
    floor = max(20, med_area * GRID_AREA_FRACTION)
    for c in cands:
        if c["area"] < floor:
            dots += 1
            continue
        main.append(c)
    return main, {"components": n - 1, "beams": beams, "vlines": vlines,
                  "dots_dropped": dots, "median_h": med_h,
                  "median_area": med_area, "main": len(main)}


def _grid_group_rows(comps: list[dict]) -> list[list[dict]]:
    """按 yc 聚行（与 omr._group_rows 同思路，容差取高度中位数 × 0.6）。"""
    if not comps:
        return []
    hs = sorted(c["h"] for c in comps)
    tol = max(10.0, hs[len(hs) // 2] * 0.6)
    rows: list[list[dict]] = []
    for c in sorted(comps, key=lambda t: (t["yc"], t["x0"])):
        if rows and abs(c["yc"] - rows[-1][0]["yc"]) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    for r in rows:
        r.sort(key=lambda t: t["x0"])
    return rows


def _grid_make_crop(im: Image.Image, c: dict) -> np.ndarray:
    """字符裁剪：+3px padding；太窄的左右扩到 ≥ 0.55×高。

    V3 的教训：12px 宽的裸格裁剪被 rec 压成 9×48，大面积读空；
    左右扩宽后逐格召回立刻上来。
    """
    pad = 3
    x0 = max(0, c["x0"] - pad)
    x1 = min(im.width, c["x1"] + pad)
    y0 = max(0, c["y0"] - pad)
    y1 = min(im.height, c["y1"] + pad)
    min_w = int((y1 - y0) * 0.55)
    if x1 - x0 < min_w:
        grow = (min_w - (x1 - x0)) / 2.0
        x0 = max(0, int(x0 - grow))
        x1 = min(im.width, int(x1 + grow) + 1)
    return np.array(im.crop((x0, y0, x1, y1)))


def recognize_raster_grid(image_bytes: bytes, timing: dict | None = None,
                          engine=None) -> list[dict]:
    """方案 A「网格通道」：跳过 det，连通域找字符格 → 逐格 rec。

    返回与 det 通道同构的 OcrItem 列表（**每个字符一个 item**），
    上层 omr.py 的 `_group_rows → _column_tokens → _row_tokens`
    落格机制原样复用，仅对网格行放宽 D/T/S 硬门槛（require_dts=False）。

    适用面：书页拍照谱（米色/纸纹底）。应用截图会被照片守卫拦下
    （返回空列表，上层自动走 det 通道），本通道失败也不致命。
    """
    import time as _time

    t0 = _time.time()
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:  # noqa: BLE001
        return []
    if im.width <= 0 or im.height <= 0:
        return []
    gray = np.array(im.convert("L"))

    def _finish(items: list[dict]) -> list[dict]:
        if timing is not None:
            timing["grid_ms"] = int((_time.time() - t0) * 1000)
        return items

    # 照片守卫：应用截图不是本通道的战场（橙字在灰度下不可见是天然
    # 保险，但白底判别是显式保险 —— 蓝色低声部在灰度下是暗的）。
    if not _looks_like_photo(gray) and not _env_int("OCR_GRID_FORCE", 0):
        return _finish([])

    try:
        comps, _stats = _grid_components(gray)
    except Exception:  # noqa: BLE001 —— cv2 缺失等，网格通道整体让路
        return _finish([])

    # 行过滤：八度点行（矮、少）整行丢，谱面行（高）保留
    rows = _grid_group_rows(comps)
    if not rows:
        return _finish([])
    row_h = [float(np.median([c["h"] for c in r])) for r in rows]
    h_max = max(row_h)
    kept: list[dict] = []
    for r, rh in zip(rows, row_h):
        if len(r) < GRID_MIN_ROW_CHARS:
            continue
        if rh < h_max * GRID_ROW_H_RATIO:
            continue
        kept.extend(r)
    if not kept:
        return _finish([])

    crops = [_grid_make_crop(im, c) for c in kept]
    try:
        eng = engine if engine is not None else _EngineAdapter(_load_engine())
        results = eng.recognize(crops)
    except Exception:  # noqa: BLE001 —— rec 失败，让路给 det 通道
        return _finish([])

    # 逐格 conf 普遍偏低（0.2~0.6），不能照搬 det 通道的 0.5 门槛：
    # 丢一个字符 = 丢一个音，宁可交给用户在编辑器里改。
    items: list[dict] = []
    for c, res in zip(kept, results):
        text = _grid_clean(res[0]) if res and res[0] else ""
        if not text:
            continue
        items.append({
            "text": text,
            "x0": float(c["x0"]),
            "x1": float(c["x1"]),
            "yc": float(c["yc"]),
            "h": float(c["h"]),
        })
    return _finish(items)
