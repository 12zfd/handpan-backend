"""OpenCV 网格切分 + 单元格级轻量 OCR（兜底通道，绝不整页 OCR）。

定位：主通道（det 通道 / 网格连通域通道）整页都读不出有效行时，
用**纯几何方法**把谱面切成一个个小单元格，再逐格调轻量 OCR 认数字。
这样即使整页 OCR 失效（内存不够、引擎崩溃、截屏底色干扰），
只要图上有网格线，就还能把音挖出来。

流程（对应需求 1~3）：
  1. 灰度 → 自适应二值化（adaptiveThreshold），并用 HSV 饱和度把
     彩色底（App 截图的黄底/蓝底）从墨迹里剔除；
  2. morphologyEx 开运算分别提取水平线 / 垂直线，二者相交得交叉点，
     聚类出列边界（x）与行带（y），切成小单元格；
  3. 每个单元格内按 Y 中线分上下两半 —— 上半是上轨道（右手），
     下半是下轨道（左手），分别 OCR，同一列上下都有字 = 和弦；
  4. OCR 引擎按可用性自动降级：pytesseract → easyocr → 放弃该格。
     引擎缺失只影响识别率，**绝不抛异常**（需求 4 的前提）。

输出与 extract/omr.py::_page_tokens 的 rows_tokens 完全同构：
  [[{slot, ch, beams, voice}, ...], ...] —— 外层按行带、内层按格位。
omr 的行对齐落格会自动把每行放进正确的 32 格行宽里。
"""
from __future__ import annotations

import io
import os
import threading

import numpy as np
from PIL import Image

# ── 可调参数（环境变量可覆盖）─────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default

MIN_COLS = _env_int("GRIDOCR_MIN_COLS", 4)          # 至少要切出这么多列才认
MAX_COLS = _env_int("GRIDOCR_MAX_COLS", 80)
MAX_ROWS = _env_int("GRIDOCR_MAX_ROWS", 60)
MIN_CELL_INK = _env_int("GRIDOCR_MIN_INK", 10)      # 单元格最少墨迹像素（细笔画的 1/7 只有 10 来个）
CELL_TARGET_H = _env_int("GRIDOCR_TARGET_H", 40)    # OCR 前把字符块归一化到这个高
MIN_CHARS = _env_int("GRIDOCR_MIN_CHARS", 2)        # 全图最少认出几个字才启用兜底

# OCR 结果清洗：常见形近字 → 谱面字符（D=ding，S=slap，0-9=音位）
_CHAR_FIX = {
    "O": "0", "o": "0", "Q": "0", "U": "0",
    "l": "1", "I": "1", "i": "1", "|": "", "!": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4",
    "s": "S", "$": "S", "5": "5",
    "b": "6", "G": "6",
    "T": "7", "?": "7",
    "B": "8",
    "g": "9", "q": "9",
    "d": "D", "p": "D", "P": "D",
}

_LAST_INFO: dict = {}
_INFO_LOCK = threading.Lock()


def last_info() -> dict:
    """最近一次兜底识别的元信息（供上层拼 warnings）。"""
    with _INFO_LOCK:
        return dict(_LAST_INFO)


# ── OCR 引擎按可用性自动降级 ─────────────────────────────────────
_ENGINE_NAME: str | None = None
_ENGINE_CHECKED = False
_EASY_READER = None


def _pick_engine() -> str | None:
    """pytesseract（需要系统 tesseract-ocr）→ easyocr（纯 Python 但重）→ None。

    引擎探测结果进程内缓存；探测失败只是没有 OCR 引擎，
    网格切分依然可用（空格不产音），绝不 raise。
    """
    global _ENGINE_NAME, _ENGINE_CHECKED, _EASY_READER
    if _ENGINE_CHECKED:
        return _ENGINE_NAME
    _ENGINE_CHECKED = True

    try:
        import pytesseract  # noqa: F401
        from pytesseract import TesseractNotFoundError
        try:
            pytesseract.image_to_string(
                Image.new("L", (32, 32), 255), config="--psm 10")
            _ENGINE_NAME = "pytesseract"
            return _ENGINE_NAME
        except TesseractNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass
    except ImportError:
        pass

    try:
        import easyocr  # noqa: F401
        _EASY_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ENGINE_NAME = "easyocr"
        return _ENGINE_NAME
    except Exception:  # noqa: BLE001 —— 模型下载失败/内存不足都归为无引擎
        _EASY_READER = None

    _ENGINE_NAME = None
    return None


def _ocr_char(img: Image.Image) -> str:
    """单字符小块 → 1 个谱面字符（认不出返回 ""）。

    psm 10（单字符）优先，认不出再试 psm 8（单词）——
    小字号 / 抗锯齿字符在 psm 10 下经常直接给空，psm 8 能多捞一批。
    """
    engine = _pick_engine()
    if engine is None:
        return ""
    try:
        if engine == "pytesseract":
            import pytesseract
            txt = ""
            for psm in (10, 8):
                txt = pytesseract.image_to_string(
                    img, lang="eng",
                    config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789DdSs",
                ).strip()
                if txt:
                    break
        else:
            arr = np.array(img.convert("RGB"))
            res = _EASY_READER.readtext(arr, allowlist="0123456789DS", detail=0)
            txt = "".join(res) if res else ""
    except Exception:  # noqa: BLE001 —— 单格失败不影响其它格
        return ""
    for ch in (txt or "").strip():
        fixed = _CHAR_FIX.get(ch, ch)
        if fixed in "0123456789DS":
            return fixed
    return ""


# ── OpenCV 几何切分 ──────────────────────────────────────────────
def _to_gray(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """bytes → (gray, colored_mask)。colored_mask=True 的像素是彩色底（非墨迹）。"""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    rgb = np.array(im)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    # 近似饱和度：彩底（黄底截图）mx-mn 大；纸面/黑字 mx-mn 小
    colored = (mx - mn) > 70
    gray = np.array(im.convert("L"))
    return gray, colored


def _cluster(vals: list[int], gap: int) -> list[int]:
    """把排好序的坐标聚成簇，返回每簇的均值。"""
    if not vals:
        return []
    vals = sorted(vals)
    groups: list[list[int]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(sum(g) / len(g)) for g in groups]


def recognize_grid(raw: bytes) -> list[list[dict]] | None:
    """单张图 → rows_tokens（与 omr._page_tokens 输出同构），失败返回 None。

    任何一步失败都是 return None，绝不抛异常（pipeline 层还有第二道兜底）。
    """
    try:
        return _recognize_grid_impl(raw)
    except Exception:  # noqa: BLE001
        return None


def _recognize_grid_impl(raw: bytes) -> list[list[dict]] | None:
    import cv2

    gray, colored = _to_gray(raw)
    h, w = gray.shape[:2]
    if w < 120 or h < 80:
        return None

    # ── 1. 自适应二值化（暗字=白），彩色底直接判为背景
    binimg = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    binimg[colored] = 0

    # ── 2. 形态学提取横线 / 竖线
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, (w // 40) | 1), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, (h // 40) | 1)))
    h_lines = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, vk)
    if h_lines.sum() < 255 * 50 or v_lines.sum() < 255 * 50:
        return None                      # 线太少：这不是网格谱，走上层兜底

    # 行带：横线 y 聚类，相邻两条横线夹出一个谱面行
    row_ys = _cluster([int(y) for y in np.nonzero(h_lines.sum(axis=1) > 255 * (w // 10))[0]],
                      max(6, h // 120))
    # 列边界：竖线 x 聚类
    col_xs = _cluster([int(x) for x in np.nonzero(v_lines.sum(axis=0) > 255 * (h // 10))[0]],
                      max(6, w // 120))
    if len(col_xs) < MIN_COLS or len(row_ys) < 2:
        return None
    col_xs = col_xs[:MAX_COLS + 1]
    if len(row_ys) > MAX_ROWS + 1:
        row_ys = row_ys[:MAX_ROWS + 1]

    # 线掩码（膨胀后从单元格墨迹里抹掉，防止线被 OCR 认成 1/7）
    line_mask = cv2.dilate(
        cv2.bitwise_or(h_lines, v_lines),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0

    engine = _pick_engine()
    med_w = float(np.median(np.diff(col_xs))) if len(col_xs) > 2 else float(w)

    rows_tokens: list[list[dict]] = []
    chars_found = 0

    for ri in range(len(row_ys) - 1):
        y0, y1 = row_ys[ri], row_ys[ri + 1]
        band_h = y1 - y0
        if band_h < 24:                  # 太矮不是谱行（装饰线/页眉）
            continue
        mid = y0 + band_h // 2
        row_toks: list[dict] = []
        n_cols = min(len(col_xs) - 1, MAX_COLS)
        for ci in range(n_cols):
            x0, x1 = col_xs[ci], col_xs[ci + 1]
            cell_w = x1 - x0
            if cell_w < 8 or cell_w > med_w * 4:
                continue                 # 过宽过窄都是非谱面竖线
            pad = max(2, cell_w // 10)
            cx0, cx1 = x0 + pad, x1 - pad
            if cx1 - cx0 < 6:
                continue
            # ── 3. 单元格内按 Y 分上下轨道，分别提墨、OCR
            for voice, (ty0, ty1) in (
                ("upper", (y0 + band_h // 12, mid)),
                ("lower", (mid, y1 - band_h // 12)),
            ):
                if ty1 - ty0 < 8:
                    continue
                cell = binimg[ty0:ty1, cx0:cx1].copy()
                cell[line_mask[ty0:ty1, cx0:cx1]] = 0
                ys, xs = np.nonzero(cell)
                if len(ys) < MIN_CELL_INK:
                    continue
                bx0, bx1 = xs.min(), xs.max()
                by0, by1 = ys.min(), ys.max()
                crop = cell[by0:by1 + 1, bx0:bx1 + 1]
                ch_img = Image.fromarray(255 - crop.astype(np.uint8))  # 黑字白底
                scale = max(2, CELL_TARGET_H // max(1, ch_img.height))
                if scale > 1:
                    ch_img = ch_img.resize(
                        (ch_img.width * scale, ch_img.height * scale), Image.LANCZOS)
                ch_img = ch_img.point(lambda v: 255 if v > 200 else 0)
                # 四周补白，psm 10 单字符识别需要呼吸空间
                padded = Image.new("L", (ch_img.width + 16, ch_img.height + 16), 255)
                padded.paste(ch_img, (8, 8))
                ch = _ocr_char(padded)
                if not ch:
                    continue
                chars_found += 1
                row_toks.append({"slot": ci, "ch": ch, "beams": 0, "voice": voice})
        if row_toks:
            rows_tokens.append(row_toks)

    with _INFO_LOCK:
        _LAST_INFO.clear()
        _LAST_INFO.update({
            "engine": engine or "none",
            "rows": len(rows_tokens),
            "cols": len(col_xs) - 1,
            "chars": chars_found,
        })

    if chars_found < MIN_CHARS or not rows_tokens:
        return None                      # OCR 一个字都没认出 → 上层空白兜底
    return rows_tokens
