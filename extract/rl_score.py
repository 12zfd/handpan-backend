"""R/L 行谱专用解析器 —— 针对「带圈数字 + R/L 双行 + 底行数字」的谱册版式。

版式（抖音手碟谱册 / 《天空之城》这类）：
  每个系统（system）自上而下：
    1. 节拍梁（粗横杠 + 符干，十六分音符网格）
    2. R 行：带圈数字 ①-⑨ / 字母 D —— 右手旋律
    3. 分隔线（全宽横线）
    4. L 行：带圈数字 —— 左手伴奏
    5. 底行：普通数字（低音音位，0=休止）+ 组下划线
  每系统 2 小节 × 16 个十六分音符格；底行 32 个数字就是 32 个槽位的天然网格。

解析策略（几何优先，OCR 只认小字符）：
  1. 全宽横线聚类 → 每系统 3 条参考线（梁/分隔线/下划线行）；
  2. 圆环轮廓检测（外轮廓恰 1 洞，且洞必须**居中**——这正是带圈数字与
     D 的空碗的区别：D 的碗是偏右上的豁口，洞中心明显偏离圈心）；
  3. 洞内墨量占比（ink density）区分「真·带圈数字」（洞里有成形的数字，
     墨量足）与「D 碗里的噪点 / 0·6·9 的小洞」（墨量很少）。有墨且占比够
     → 圈内 OCR 认数字；墨量不够 → 整圈 OCR 只认 D/S；
  4. 底行连通域 → 按 x 聚成 32 组 → 每组中心 = 槽位中心，主分量 OCR 认数字；
  5. R/L 音符与底行共用同一套 32 槽位网格（统一 argmin 就近落格），保证
     输入图列位与导出列位一一对齐；同 (槽位, 声部) 去重，杜绝重复计数。

输出与 omr._page_tokens 的 rows_tokens 同构：
  [[{slot, ch, beams, voice}, ...], ...] —— 每系统一行（32 槽 = 2 小节）。
任何一步失败都 return None（上层还有 grid_ocr 兜底与空白谱兜底），绝不抛异常。
"""
from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image

_INFO_LOCK = threading.Lock()
_LAST_INFO: dict = {}


def last_info() -> dict:
    """最近一次解析的元信息（供上层拼 warnings）。"""
    with _INFO_LOCK:
        return dict(_LAST_INFO)


# ── 几何辅助 ─────────────────────────────────────────────────────

def _merge_runs(vals: list[int], gap: int) -> list[list[int]]:
    """把排序后的坐标聚成连续簇。"""
    if not vals:
        return []
    out: list[list[int]] = [[vals[0]]]
    for v in vals[1:]:
        if v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def _find_rings(binimg: np.ndarray, w: int) -> list[dict]:
    """检测圆环轮廓。返回 [{cx, cy, x0, y0, x1, y1, hole, ink, density}]。

    判据（层层收紧，专治 D 空碗误判）：
      · 外轮廓恰有 1 个子轮廓（洞）；
      · 外框近正方形（圈，不是字母）；
      · 洞 bbox 在外框 0.45~0.97 之间（带圈数字的环形空洞几乎和圈一样大）；
      · ★洞必须**居中**：洞心与外框心偏移 < 0.25 边长。D 的碗是偏右上的
        豁口，洞心明显偏，直接被这一条排除 → 不会当成带圈数字。
    洞内墨量密度（ink / 洞面积）用于后续区分「真·带圈数字（墨足量）」与
    「D 碗噪点 / 0·6·9 小洞（墨很少）」。
    """
    import cv2
    cnts, hier = cv2.findContours(binimg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings: list[dict] = []
    if hier is None or len(cnts) == 0:
        return rings
    hier = hier[0]
    children: dict[int, list[int]] = {}
    for j in range(len(cnts)):
        children.setdefault(int(hier[j][3]), []).append(j)
    lo, hi = max(22, w // 90), w // 24          # 圈径范围（2160px 图 → 24~90）
    for i in range(len(cnts)):
        if hier[i][3] != -1 or len(children.get(i, [])) != 1:
            continue
        x, y, w0, h0 = cv2.boundingRect(cnts[i])
        if w0 < lo or h0 < lo or w0 > hi or h0 > hi:
            continue
        ar = w0 / float(h0)
        if ar < 0.7 or ar > 1.45:
            continue
        cx0, cy0, cw0, ch0 = cv2.boundingRect(cnts[children[i][0]])
        if cw0 < 0.45 * w0 or cw0 > 0.97 * w0 or ch0 < 0.45 * h0 or ch0 > 0.97 * h0:
            continue
        # ★ 居中约束：洞心必须贴近圈心（D 的碗偏右上，会被这一条排除）。
        #   放宽到 0.33，避免误伤手绘时略偏心的真·带圈数字。
        ring_cx, ring_cy = x + w0 / 2.0, y + h0 / 2.0
        hole_cx, hole_cy = cx0 + cw0 / 2.0, cy0 + ch0 / 2.0
        if abs(hole_cx - ring_cx) > 0.33 * w0 or abs(hole_cy - ring_cy) > 0.33 * h0:
            continue
        pad = max(1, min(cw0, ch0) // 12)
        inner = binimg[cy0 + pad:cy0 + ch0 - pad, cx0 + pad:cx0 + cw0 - pad]
        ink = int((inner > 0).sum()) if inner.size else 0
        hole_area = max(1, (cw0 - 2 * pad) * (ch0 - 2 * pad))
        rings.append({
            "cx": int(ring_cx), "cy": int(ring_cy),
            "x0": x, "y0": y, "x1": x + w0, "y1": y + h0,
            "hole": (cx0, cy0, cw0, ch0),
            "ink": ink, "density": ink / float(hole_area),
        })
    return rings


def _ocr_crop(binimg: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              target_h: int = 52) -> str:
    """二值图裁块 → 黑字白底 → 放大 → 单字符 OCR（复用 grid_ocr 的级联）。"""
    from .grid_ocr import _ocr_char
    crop = binimg[max(0, y0):y1, max(0, x0):x1]
    ys, xs = np.nonzero(crop)
    if len(ys) < 12:
        return ""
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray(255 - crop.astype(np.uint8))
    scale = max(2, target_h // max(1, img.height))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    img = img.point(lambda v: 255 if v > 200 else 0)
    padded = Image.new("L", (img.width + 16, img.height + 16), 255)
    padded.paste(img, (8, 8))
    return _ocr_char(padded)


def _count_holes(crop: np.ndarray) -> int:
    """数字块内部封闭孔洞数（背景被墨完全包围）。

    0=无孔(1/2/3/5/7)  1=6/0/9/闭4  2=8。
    用于纠正 5↔6 形近混淆：'6' 有底环孔，'5' 无孔——这是几何特征，
    比 tesseract 对印刷小字的随机 OCR 稳定得多。
    """
    if crop.size == 0:
        return 0
    import cv2
    cnts, hier = cv2.findContours(
        (crop > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return 0
    hier = hier[0]
    hh, ww = crop.shape[:2]
    total_ink = int((crop > 0).sum())
    min_hole = max(4, total_ink * 0.04)      # 过滤噪点小孔
    holes = 0
    for j in range(len(cnts)):
        if hier[j][3] == -1:
            continue  # 外轮廓，不是孔
        if cv2.contourArea(cnts[j]) < min_hole:
            continue
        # 只认完全被墨包围的孔：贴到裁块边缘的「伪孔」（数字与圈之间的空隙）不计。
        # 这一步对带圈数字尤其重要——否则圈与数字之间的缝隙会被数成假孔。
        bx, by, bw, bh = cv2.boundingRect(cnts[j])
        if bx <= 0 or by <= 0 or bx + bw >= ww or by + bh >= hh:
            continue
        holes += 1
    return holes


def _ocr_digit(binimg: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               target_h: int = 52):
    """二值图裁块 → 单字符 OCR + 内部孔洞数（同 _ocr_crop，但多返回 holes）。"""
    from .grid_ocr import _ocr_char
    crop = binimg[max(0, y0):y1, max(0, x0):x1]
    ys, xs = np.nonzero(crop)
    if len(ys) < 12:
        return "", 0
    cy0, cy1 = int(ys.min()), int(ys.max()) + 1
    cx0, cx1 = int(xs.min()), int(xs.max()) + 1
    tight = crop[cy0:cy1, cx0:cx1]
    holes = _count_holes(tight)
    img = Image.fromarray(255 - tight.astype(np.uint8))
    scale = max(2, target_h // max(1, img.height))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    img = img.point(lambda v: 255 if v > 200 else 0)
    padded = Image.new("L", (img.width + 16, img.height + 16), 255)
    padded.paste(img, (8, 8))
    return _ocr_char(padded), holes


def _merge_split_groups(groups: list[list]) -> list[list]:
    """合并被切开的数字（一个数字的墨被拆成两块 → 被误当成两列）。

    判据：相邻两组的间距 < 0.55 × 中位间距。谱册列距是节拍网格
    （实测 ≈56px），而同一个数字被切开的两块间距只有十几 px。
    不过这一关，列数会虚高（实测 34），整个槽位网格随之错位。
    """
    if len(groups) < 2:
        return groups
    groups = [list(g) for g in groups]
    while len(groups) > 1:
        centers = [g[0][0] for g in groups]
        gaps = [centers[k + 1] - centers[k] for k in range(len(centers) - 1)]
        thr = 0.55 * float(np.median(gaps))
        hit = next((k for k, gp in enumerate(gaps) if gp < thr), None)
        if hit is None:
            break
        groups = groups[:hit] + [groups[hit] + groups[hit + 1]] + groups[hit + 2:]
    return groups


def _dp_assign_slots(xs: list[float], model: list[float], n_slots: int,
                     skip_cost: float = 45.0) -> list[tuple[int, int]]:
    """单调 DP：把观测到的列心分配到槽位 0..n_slots-1。

    槽位序号必须严格递增（列是从左到右排的）；某个观测可以被跳过
    （代价 skip_cost），用来容忍多出来的碎块。
    """
    n = len(xs)
    a, p, q, r = model

    def pred(i: int) -> float:
        return a + p * i + q * (i // 4) + r * (i // 16)

    INF = float("inf")
    # cost[g][s] = 处理完前 g 个观测、槽位 < s 已被占用时的最小代价
    cost = [[INF] * (n_slots + 1) for _ in range(n + 1)]
    prev = [[None] * (n_slots + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for g in range(n):
        for sl in range(n_slots + 1):
            c = cost[g][sl]
            if c == INF:
                continue
            if c + skip_cost < cost[g + 1][sl]:          # 跳过当前观测
                cost[g + 1][sl] = c + skip_cost
                prev[g + 1][sl] = (g, sl, -1)
            if sl < n_slots:                              # 当前观测落到槽位 sl
                c2 = c + abs(xs[g] - pred(sl))
                if c2 < cost[g + 1][sl + 1]:
                    cost[g + 1][sl + 1] = c2
                    prev[g + 1][sl + 1] = (g, sl, sl)
    s = min(range(n_slots + 1), key=lambda k: cost[n][k])
    if cost[n][s] == INF:
        return []
    assign: list[tuple[int, int]] = []
    g = n
    while g > 0:
        step = prev[g][s]
        if step is None:
            break
        pg, ps, assigned = step
        if assigned >= 0:
            assign.append((pg, assigned))
        g = pg
        s = ps
    assign.reverse()
    return assign


def _fit_beat_grid(xs: list[float], n_slots: int = 32, iters: int = 8):
    """节拍感知网格拟合：把观测列心单调分配到 0..n_slots-1。

    模型  x(i) = a + p·i + q·(i//4) + r·(i//16)
      p    → 六十四分音符的基本列距
      i//4 → 每 4 格的节拍边界留白
      i//16 → 每小节线的留白
    先用 DP 分配，再最小二乘回代参数，迭代收敛。
    """
    n = len(xs)
    if n == 0:
        return None, []
    if n == 1:
        return [float(xs[0]), 0.0, 0.0, 0.0], [(0, 0)]
    model = [float(xs[0]), float(np.median(np.diff(xs))), 0.0, 0.0]
    assign = [(i, i) for i in range(n)]
    for _ in range(iters):
        assign = _dp_assign_slots(xs, model, n_slots)
        if len(assign) < 3:
            break
        s_idx = np.array([sl for _, sl in assign], dtype=float)
        y = np.array([xs[g] for g, _ in assign], dtype=float)
        A = np.stack([np.ones_like(s_idx), s_idx,
                      np.floor(s_idx / 4.0), np.floor(s_idx / 16.0)], axis=1)
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:  # noqa: BLE001
            break
        model = [float(v) for v in sol]
    return model, assign


def _slot_anchors(xs: list[float], n_slots: int = 32) -> list[float]:
    """由底行观测到的列心，构建 n_slots 个槽位的 x 锚点。

    ★ 底行的 32 个数字就是 32 个槽位的天然网格 —— 每个槽位一个数字，
    它的**真实 x 位置**直接就是该槽位的锚点。比「均匀 linspace 网格」
    准得多（实测 R/L 落格残差 5.7px vs 10.7px），因为排版在节拍边界和
    小节线处留白，列距本来就不均匀。

    若观测列数 ≠ 32（个别槽空缺 / 粘连），退到节拍感知模型补齐。
    """
    xs = sorted(float(v) for v in xs)
    if not xs:
        return []
    if len(xs) == n_slots:
        return xs
    model, assign = _fit_beat_grid(xs, n_slots)
    if model is None:
        return [float(v) for v in np.linspace(xs[0], xs[-1], n_slots)]
    a, p, q, r = model
    anchors: list[float | None] = [None] * n_slots
    for g, sl in assign:
        if 0 <= sl < n_slots:
            anchors[sl] = xs[g]
    for i in range(n_slots):
        if anchors[i] is None:
            anchors[i] = a + p * i + q * (i // 4) + r * (i // 16)
    return [float(v) for v in anchors]


def _fix_digit(ch: str, holes: int, crop_h: int) -> str:
    """用孔洞数纠正 5↔6 形近混淆（带圈数字 / 底行数字通用）。

    · 恰好 1 个孔且字高足够 → 必不是 5（5 无孔），判为 6；
    · 无孔且字高足够（环本应可见，crop_h≥30）→ 判为 5。
      小字（带圈数字）阈值可能破环，故小字不做 6→5，避免误伤真 6。
    """
    if ch == "5" and holes == 1 and crop_h >= 20:
        return "6"
    if ch == "6" and holes == 0 and crop_h >= 30:
        return "5"
    return ch


# ── 主解析 ───────────────────────────────────────────────────────

def parse_rl_score(raw: bytes, debug: bool = False) -> list[list[dict]] | None:
    """单页 R/L 行谱 → rows_tokens；不是这种版式或解析失败返回 None。

    debug=True 时每个 token 带 "_src"（"bass"/"R"/"L"），仅供诊断脚本区分
    底行 / 右手行 / 左手行，不影响生产消费。
    """
    try:
        return _parse_impl(raw, debug=debug)
    except Exception:  # noqa: BLE001
        return None


def _parse_impl(raw: bytes, debug: bool = False) -> list[list[dict]] | None:
    import cv2
    from .grid_ocr import _to_gray

    gray, colored = _to_gray(raw)
    h, w = gray.shape[:2]
    if w < 500 or h < 400:
        return None
    binimg = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    binimg[colored] = 0

    # ── 1. 全宽横线 → 系统参考线（梁 0.67 / 分隔线 0.90 / 下划线行 0.76）
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, (w // 30) | 1), 1))
    h_lines = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, hk)
    cov = h_lines.sum(axis=1) / 255.0 / w
    line_ys = [int(y) for y in np.nonzero(cov > 0.55)[0]]
    staff = [int(np.mean(g)) for g in _merge_runs(line_ys, 8)]
    if len(staff) < 3:
        return None

    # 系统分组：取每 3 条为一组（梁、分隔线、下划线行），间距必须相近
    systems: list[tuple[int, int, int]] = []
    i = 0
    while i + 2 < len(staff):
        yb, ys_, yu = staff[i], staff[i + 1], staff[i + 2]
        d1, d2 = ys_ - yb, yu - ys_
        if 40 <= d1 <= h // 5 and 40 <= d2 <= h // 4 and abs(d1 - d2) < max(30, d1):
            systems.append((yb, ys_, yu))
            i += 3
        else:
            i += 1
    if len(systems) < 1:
        return None

    rings = _find_rings(binimg, w)

    rows_tokens: list[list[dict]] = []
    n_bass = n_note = n_fail = 0
    for (yb, ys_, yu) in systems:
        row = _parse_system(binimg, h_lines, rings, yb, ys_, yu, w, debug=debug)
        if row is None:
            continue
        toks, nb, nn, nf = row
        n_bass += nb
        n_note += nn
        n_fail += nf
        if toks:
            rows_tokens.append(toks)
    with _INFO_LOCK:
        _LAST_INFO.clear()
        _LAST_INFO.update({
            "engine": "rl",
            "rows": len(rows_tokens),
            "cols": (n_bass // max(1, len(rows_tokens))) if rows_tokens else 0,
            "chars": n_bass + n_note,
            "mode": "rl",
            "notes": n_note,
            "bass": n_bass,
            "ocr_miss": n_fail,
        })
    if not rows_tokens:
        return None
    # 至少要认出一点东西才算这个版式
    if n_bass + n_note < 6:
        return None
    return rows_tokens


def _parse_system(binimg, h_lines, rings, yb, ys_, yu, w, debug: bool = False):
    """解析单个系统 → (tokens, 底行数, 音符数, OCR失败数) | None。"""
    import cv2
    tokens: list[dict] = []
    zone_h = yu - ys_

    # ── 2. 底行数字：连通域 → 按 x 聚组 → 主分量（最高的）OCR
    by0, by1 = ys_ + int(zone_h * 0.40), yu - 6
    zone = binimg[by0:by1, :].copy()
    # 抹掉组下划线 / 长横线（≥35px 的横向连通）
    zk = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    zone[cv2.morphologyEx(zone, cv2.MORPH_OPEN, zk) > 0] = 0
    n, labels, stats, cents = cv2.connectedComponentsWithStats(zone)
    comps = []
    for k in range(1, n):
        x, y, ww, hh, area = stats[k]
        if ww < 8 or hh < 10 or hh > zone_h or area < 20:
            continue
        if ww > 90:                       # 残余长横线/连线
            continue
        comps.append((cents[k][0], cents[k][1] + by0, x, y, ww, hh))
    if len(comps) < 16:
        return None
    comps.sort(key=lambda c: c[0])
    groups: list[list] = [comps[0:1]]
    for c in comps[1:]:
        if c[0] - groups[-1][-1][0] < max(14, w // 150):
            groups[-1].append(c)
        else:
            groups.append([c])
    # ★ 把被切开的数字合并回一列（否则列数虚高，槽位网格整体错位）
    groups = _merge_split_groups(groups)
    if not (28 <= len(groups) <= 36):
        return None                       # 不是这个版式的槽位密度

    # ★ 槽位锚点用底行数字的**真实列心**，不是均匀 linspace：
    #   排版在节拍边界 / 小节线处留白，列距本来就不均匀。
    anchors = _slot_anchors([g[0][0] for g in groups], 32)
    if len(anchors) < 32:
        return None
    anchor_arr = np.array(anchors, dtype=float)

    def slot_of(cx: float) -> int:
        """统一就近落格：R/L 与底行共用同一套 32 槽位网格（argmin）。"""
        return int(np.clip(np.argmin(np.abs(anchor_arr - cx)), 0, 31))

    by_slot: dict[int, list] = {}
    for g in groups:
        main = max(g, key=lambda c: c[5])
        si = slot_of(main[0])
        cur = by_slot.get(si)
        if cur is None or main[5] > cur[5]:
            by_slot[si] = main
    bass: list[tuple[int, str]] = []
    miss = 0
    bass_holes: dict[int, int] = {}
    for si in range(32):
        main = by_slot.get(si)
        if main is None:
            bass.append((si, ""))
            miss += 1
            continue
        _, _, x, y, ww, hh = main
        ch, holes = _ocr_digit(binimg, x, by0 + y, x + ww, by0 + y + hh, target_h=52)
        ch = _fix_digit(ch, holes, hh)
        bass.append((si, ch))
        bass_holes[si] = holes
        if not ch:
            miss += 1
    n_bass = 0
    for si, (_, ch) in enumerate(bass):
        if ch and ch != "0" and ch in "0123456789":
            tok = {"slot": si, "ch": ch, "beams": 0, "voice": "lower", "lane": "bass"}
            if debug:
                tok["_src"] = "bass"
                tok["_holes"] = bass_holes.get(si, 0)
            tokens.append(tok)
            n_bass += 1

    # ── 3. R 行 / L 行音符：带圈数字 + 非圈 D/S 字母
    #    每个圈先圈内 OCR，墨量够(密度≥0.06)且认出 1-9 才当带圈数字；
    #    否则整圈 OCR 只认 D/S；都认不出就丢弃。按声部+槽位去重。
    y_mid = ys_ + int(zone_h * 0.5)
    n_note = 0
    seen: set[tuple[int, str]] = set()

    # 先把圈按声部分到 R / L
    def voice_of(cy: int) -> str | None:
        if yb + 16 <= cy <= ys_ - 4:
            return "upper"
        if ys_ + 4 <= cy <= ys_ + int(zone_h * 0.45):
            return "lower"
        return None

    for r in rings:
        voice = voice_of(r["cy"])
        if voice is None:
            continue
        hx, hy, hw, hh2 = r["hole"]
        pad = max(1, min(hw, hh2) // 14)
        inner, holes = _ocr_digit(binimg, hx + pad, hy + pad, hx + hw - pad, hy + hh2 - pad, target_h=52)
        whole = _ocr_crop(binimg, r["x0"], r["y0"], r["x1"], r["y1"], target_h=52)
        # 圈内/整圈任一认出合法字符（1-9 或 D/S）即可；优先取数字，
        # 没数字再取 D/S。D 的空碗因洞偏右上已被居中约束排除，不会进 rings。
        ch = ""
        inner_ok = inner in "123456789DS"
        whole_ok = whole in "123456789DS"
        if inner_ok and whole_ok:
            ch = inner if inner in "123456789" else whole
        elif inner_ok:
            ch = inner
        elif whole_ok:
            ch = whole
        if ch in "56":
            # 5↔6 用孔洞数定夺：带圈数字小字不反向修（避免破环误伤真 6）
            ch = _fix_digit(ch, holes, hh2 - 2 * pad)
        if not ch or ch == "0":
            continue
        slot = slot_of(r["cx"])
        key = (slot, voice)
        if key in seen:
            continue                        # 同槽同声部去重
        seen.add(key)
        tok = {"slot": slot, "ch": ch, "beams": 0, "voice": voice, "lane": voice}
        if debug:
            tok["_src"] = "R" if voice == "upper" else "L"
            tok["_holes"] = int(holes)
        tokens.append(tok)
        n_note += 1

    # 非圈的 D/S 字母：连通域（避开已认的圈）
    for voice, (ztop, zbot) in (
        ("upper", (yb + 16, ys_ - 4)),
        ("lower", (ys_ + 4, ys_ + int(zone_h * 0.45))),
    ):
        ztop_i, zbot_i = max(0, ztop), min(binimg.shape[0], zbot)
        zimg = binimg[ztop_i:zbot_i, :].copy()
        zimg[h_lines[ztop_i:zbot_i, :] > 0] = 0
        n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(zimg)
        for k in range(1, n2):
            x, y, ww, hh2, area = st2[k]
            if not (22 <= ww <= 58 and 30 <= hh2 <= 60 and area > 120):
                continue
            cx, cy = ce2[k][0], ce2[k][1] + ztop_i
            if any(abs(cx - r["cx"]) < 28 and abs(cy - r["cy"]) < 28 for r in rings):
                continue                  # 圈的内部字 / 圈本身
            ch = _ocr_crop(binimg, x, ztop_i + y, x + ww, ztop_i + y + hh2, target_h=52)
            if ch not in ("D", "S"):
                continue
            slot = slot_of(cx)
            key = (slot, voice)
            if key in seen:
                continue
            seen.add(key)
            tok = {"slot": slot, "ch": ch, "beams": 0, "voice": voice, "lane": voice}
            if debug:
                tok["_src"] = "R" if voice == "upper" else "L"
            tokens.append(tok)
            n_note += 1

    tokens.sort(key=lambda t: (t["slot"], t.get("voice") != "upper"))
    return tokens, n_bass, n_note, miss
