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


def _find_tesseract() -> str | None:
    """定位 tesseract 可执行文件。

    pytesseract 默认直接调 "tesseract"，依赖**子进程 PATH**。
    Windows 上 tesseract 往往只是装了、没进 PATH（甚至只装了 GUI 版），
    于是 pytesseract 抛 FileNotFoundError → _pick_engine 判定「无引擎」
    → 全部字符识别为空 → 整页解析退化成 None（空白谱）。
    这里主动找一遍：先找 PATH，再找 Windows 常见安装目录，最后回落到
    pytesseract 自带的 tesseract_cmd。
    """
    import shutil
    import glob

    exe = shutil.which("tesseract")
    if exe:
        return exe

    # Windows 常见安装目录（UB-Mannheim / 官方 installer 默认路径）
    if os.name == "nt":
        patterns = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                         "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                         "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs",
                         "Tesseract-OCR", "tesseract.exe"),
        ]
        for p in patterns:
            if p and os.path.exists(p):
                return p
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),):
            hits = glob.glob(os.path.join(base, "Tesseract*", "tesseract.exe"))
            if hits:
                return hits[0]

    try:
        import pytesseract
        cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
        if cmd and os.path.exists(cmd):
            return cmd
    except Exception:  # noqa: BLE001
        pass
    return None


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
        tesseract_exe = _find_tesseract()
        if tesseract_exe:
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe
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
            for psm in (10, 8, 6):
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
def _erase_circles(binimg: np.ndarray) -> np.ndarray:
    """擦掉带圈数字（①⑤③…）的外圈，只留圈里的数字给 OCR。

    手碟谱常用 ①-⑨ 标音位，RapidOCR / tesseract 对带圈字形基本全瞎。
    判据：外轮廓有且只有一个子轮廓（洞）、外框近正方形且 14~70px、
    子轮廓（圈内内容）bbox 明显小于外框（数字 0/8 的洞几乎和外面一样大，
    会被这条排除）、外圈面积接近外框（圆是凸的）。
    """
    import cv2
    try:
        cnts, hier = cv2.findContours(binimg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(cnts) == 0:
            return binimg
        hier = hier[0]
        out = binimg.copy()
        n = len(cnts)
        children: dict[int, list[int]] = {}
        for j in range(n):
            children.setdefault(int(hier[j][3]), []).append(j)
        for i in range(n):
            if hier[i][3] == -1 or len(children.get(i, [])) != 1:
                continue                      # 只处理「有且仅有一个洞」的外圈
            x, y, w0, h0 = cv2.boundingRect(cnts[i])
            if w0 < 14 or h0 < 14 or w0 > 70 or h0 > 70:
                continue
            ar = w0 / float(h0)
            if ar < 0.65 or ar > 1.55:
                continue
            if cv2.contourArea(cnts[i]) < 0.5 * w0 * h0:
                continue                      # 圈是凸的；破损圈不擦
            cx, cy, cw, ch = cv2.boundingRect(cnts[children[i][0]])
            if cw > 0.72 * w0 or ch > 0.72 * h0:
                continue                      # 数字 0/8 的洞几乎和外面一样大
            cv2.drawContours(out, [cnts[i]], -1, 0, thickness=3)
        return out
    except Exception:  # noqa: BLE001 —— 擦圈失败不影响切格
        return binimg


def _band_columns(v_lines: np.ndarray, y0: int, y1: int, w: int) -> list[int]:
    """在单个行带内独立找竖直节拍线 → 列边界。

    谱册（多行系统排版）里节拍线只跨本行带，全页累计根本够不到门槛 ——
    这是《天空之城》这类谱册整页空白的根因。每行带用自己的竖线，
    互不对齐也能切。
    """
    band_h = y1 - y0
    prof = v_lines[y0:y1, :].sum(axis=0)
    thr = 255 * max(8, band_h // 3)
    cand = [int(x) for x in np.nonzero(prof > thr)[0]]
    return _cluster(cand, max(6, w // 200))


def _cluster_row(binimg: np.ndarray, h_lines: np.ndarray, full_v_cols: list[int],
                 y0: int, y1: int, w: int) -> list[dict]:
    """墨迹簇兜底切分：行带内没有竖线刻度时，直接按列墨迹聚簇 → 逐簇 OCR。

    《天空之城》这类谱册的节拍刻度在形态学竖线输出里根本不存在，
    只能放弃「找线切格」，改为「找字切格」：扣除横线和页边框后，
    列墨迹画像上的每个连续簇就是一个字符（或音名组合），逐簇交给 OCR。
    """
    ya, yb = max(0, y0), min(binimg.shape[0], y1)
    band = binimg[ya:yb, :].copy()
    # 抹掉横线与贯穿全页的竖线（页边框/系统分隔线）
    band[h_lines[ya:yb, :] > 0] = 0
    for x in full_v_cols:
        band[:, max(0, x - 6):x + 7] = 0
    col = band.sum(axis=0) / 255
    nz = col > 6
    runs: list[list[int]] = []
    x = 0
    while x < len(nz):
        if nz[x]:
            s = x
            while x < len(nz) and nz[x]:
                x += 1
            runs.append([s, x])
        else:
            x += 1
    if not runs:
        return []
    min_gap = max(12, w // 80)
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    max_w = w // 12          # 超过这个宽度的簇多半是标题/和弦名整行，不认
    mid = y0 + (y1 - y0) // 2
    toks: list[dict] = []
    for ci, (s, e) in enumerate(merged):
        if e - s < 8 or e - s > max_w:
            continue
        crop_full = band[:, max(0, s - 2):e + 2]
        ys, xs = np.nonzero(crop_full)
        if len(ys) < MIN_CELL_INK or ys.max() - ys.min() < 8:
            continue
        crop = crop_full[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ch_img = Image.fromarray(255 - crop.astype(np.uint8))
        scale = max(2, CELL_TARGET_H // max(1, ch_img.height))
        if scale > 1:
            ch_img = ch_img.resize((ch_img.width * scale, ch_img.height * scale), Image.LANCZOS)
        ch_img = ch_img.point(lambda v: 255 if v > 200 else 0)
        padded = Image.new("L", (ch_img.width + 16, ch_img.height + 16), 255)
        padded.paste(ch_img, (8, 8))
        ch = _ocr_char(padded)
        if not ch:
            continue
        if ch == "0":
            continue             # 手碟音位是 1-9/D/S；'0' 多是小节辅助数字
        voice = "upper" if (int(ys.min()) + ya) < mid else "lower"
        toks.append({"slot": ci, "ch": ch, "beams": 0, "voice": voice})
    return toks if len(toks) >= 2 else []


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

    # 擦掉带圈数字的外圈（①-⑨ → 裸数字），必须在切格/OCR 之前
    binimg = _erase_circles(binimg)

    # 行带：横线 y 聚类，相邻两条横线夹出一个谱面行
    row_ys = _cluster([int(y) for y in np.nonzero(h_lines.sum(axis=1) > 255 * (w // 10))[0]],
                      max(6, h // 120))
    # 全页对齐的列边界（仅作行带内找不到竖线时的兜底）
    col_xs = _cluster([int(x) for x in np.nonzero(v_lines.sum(axis=0) > 255 * (h // 10))[0]],
                      max(6, w // 120))
    col_xs_full = list(col_xs)       # 全页贯穿竖线（页边框等），供墨迹簇兜底扣除
    if len(row_ys) < 2:
        return None
    col_xs = col_xs[:MAX_COLS + 1] if len(col_xs) >= MIN_COLS else []
    if len(row_ys) > MAX_ROWS + 1:
        row_ys = row_ys[:MAX_ROWS + 1]

    # 线掩码（膨胀后从单元格墨迹里抹掉，防止线被 OCR 认成 1/7）
    line_mask = cv2.dilate(
        cv2.bitwise_or(h_lines, v_lines),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0

    engine = _pick_engine()

    rows_tokens: list[list[dict]] = []
    chars_found = 0
    cols_total = 0

    for ri in range(len(row_ys) - 1):
        y0, y1 = row_ys[ri], row_ys[ri + 1]
        band_h = y1 - y0
        if band_h < 24:                  # 太矮不是谱行（装饰线/页眉）
            continue
        mid = y0 + band_h // 2
        # 行带内独立找列边界；找不到再用全页对齐的（老网格截图谱）
        cols = _band_columns(v_lines, y0, y1, w)
        if len(cols) - 1 < 3:
            cols = col_xs
        n_cols = min(len(cols) - 1, MAX_COLS)
        if n_cols < 3:
            # 墨迹簇兜底：行带内既没有自己的竖线、全页列也对不上时，
            # 直接按列墨迹聚簇切字（谱册无竖线刻度的布局）
            row_toks = _cluster_row(binimg, h_lines, col_xs_full, y0, y1, w)
            if row_toks:
                chars_found += len(row_toks)
                cols_total += len(row_toks)
                rows_tokens.append(row_toks)
            continue
        cols_total += n_cols
        row_toks: list[dict] = []
        med_w = float(np.median(np.diff(cols))) if len(cols) > 2 else float(w)
        for ci in range(n_cols):
            x0, x1 = cols[ci], cols[ci + 1]
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
            "cols": cols_total // max(1, len(rows_tokens)),
            "chars": chars_found,
        })

    if chars_found < MIN_CHARS or not rows_tokens:
        return None                      # OCR 一个字都没认出 → 上层空白兜底
    return rows_tokens
