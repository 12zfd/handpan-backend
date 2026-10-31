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
# 每行的**占位宽度**（行对齐的基石）。
# 原谱一行理论上是 2 小节 × 16 格 = 32 格，但实测行尾常有一组
# 超出小节线的字符（如 `… SXXDXT654` 尾部比 32 格多 2~3 格），
# 硬卡 32 会把行尾整组截掉（实测第 0/3/4 行末尾 T654 只剩 T）。
# 留 8 格余量给行尾溢出，行首仍严格按各自的物理 x 落格，
# 所以多出来的余量只吸收溢出、不会让后面的行整体右移。
ROW_SLOTS = SLOTS_PER_BAR * BARS_PER_ROW + 8
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
# 小节号：纯数字，1~2 位（原谱行首编号 24、26、28…；末行会有 5 这种 1 位）。
# ⚠️ 千万不能用 ^\d{2,}$ —— 谱面里的音符组大量是纯数字
#    （9876 / 7876 / 5654 / 3456 …），会被整组当成小节号丢掉，
#    这正是「音节对不上」的头号原因（实测单页 16 组被误杀）。
#    真正的判据是**位置**：小节号独占谱面最左侧一列（x0 远小于谱面左边界），
#    见 _is_bar_number()。
BAR_NUMBER = re.compile(r"^\d{1,3}$")

# 有效谱面行的最小 x 跨度（归一化 1080 宽坐标系）。
# 一行 32 格 × ~23px ≈ 700px，即使右半漏读也有 350px 左右；
# 指法图例、竖排小字这类异物通常 < 300px。
try:
    MIN_ROW_SPAN = float(os.environ.get("OCR_MIN_ROW_SPAN") or 300)
except ValueError:
    MIN_ROW_SPAN = 300.0


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


def _resolve_overlaps(row: list[dict],
                      cell_w: float | None = None) -> list[dict]:
    """把一行里 x 区间互相重叠的框，裁决成**互不重叠**的一组框。

    这是「音节挤在一起」的第三大根因，也是最难的一个 ——
    OCR 在同一行会产出三种互相矛盾的输出（原图逐字校对过）：

      y=650（原图：`2 x x | T x x D | x D x D | T`）
        'TxXD'  x0=210 w=106   ← 第一节的正确识别
        'TXXD'  x0=211 w=104   ← 同一节的第二次识别（起点宽都几乎一样）
      两个都必须只留一个，否则这节 8 个字符挤进 4 格。

      y=1271（原图：`D x x x | T x x D | x x D x | T x`）
        'DxXXTxXD' x0=114 w=206 ← 把第一节 DxxX 与第二节 TxxD 合并成一个框
        'Dxxx'     x0=115 w= 96 ← 第一节的独立识别
        'TXXD'     x0=216 w=100 ← 第二节的独立识别
      这里**合并框与两个短框可以并存**：合并框若按框宽均分，
      8 个字符正好铺满 206px（≈ 两个 4 格宽），与两个短框位置一致；
      但若三个都参与 `_row_chars`，字符会被重复计入 12 个。

    裁决算法（按 x 扫描，贪心选取）：
      1. 所有框按 x0 升序，依次尝试加入；
      2. 若有框与**已选中**的框 x 重叠超过较窄者宽度的 25%，
         则两者二选一：保留「字符密度更接近格宽」的那个
         （|框宽/字符数 - 格宽| 更小者胜），因为密度吻合说明
         这两个框确实各自完整覆盖了一段真谱；
      3. 密度都不吻合时保留覆盖 x 更宽的那个（信息更多）。

    这样 y=650 留下 'TxXD'（密度 26.5 vs 26）丢掉 'TXXD'（26.0），
    y=1271 留下合并框 'DxXXTxXD'（密度 25.75）并丢掉两个短框，
    两种情形都得到「一 x 段一组」的正确结果。
    """
    if len(row) < 2:
        return row
    cw = cell_w if (cell_w and cell_w > 8.0) else 0.0

    def _n(it: dict) -> int:
        return max(1, len(_clean(it.get("text", ""))))

    def _density(it: dict) -> float:
        return (float(it["x1"]) - float(it["x0"])) / _n(it)

    def _score(it: dict) -> tuple[float, float]:
        """越小越优先：先比密度偏离格宽的程度，再比覆盖宽度（取负）。"""
        d = _density(it)
        dev = abs(d - cw) if cw > 0 else 0.0
        return (dev, -(float(it["x1"]) - float(it["x0"])))

    def _overlap(a: dict, b: dict) -> float:
        inter = min(float(a["x1"]), float(b["x1"])) - max(float(a["x0"]), float(b["x0"]))
        if inter <= 0:
            return 0.0
        narrow = min(float(a["x1"]) - float(a["x0"]), float(b["x1"]) - float(b["x0"]))
        return inter / max(1.0, narrow)

    kept: list[dict] = []
    for it in sorted(row, key=lambda t: float(t["x0"])):
        clash = [k for k in kept if _overlap(it, k) > 0.25]
        if not clash:
            kept.append(it)
            continue
        # 与所有冲突者比，自己是更优就替换掉它们
        mine = _score(it)
        if all(mine <= _score(k) for k in clash):
            kept = [k for k in kept if k not in clash]
            kept.append(it)
        # 否则丢掉自己
    kept.sort(key=lambda t: float(t["x0"]))
    return kept


def _detect_two_column(items: list[dict]) -> bool:
    """判断一页是不是**双栏排版**（左栏一列谱、右栏一列谱并排）。

    为什么要判这个：
      手机截图里同一张图常是双栏（实测 ca1b5cb7 这张 1080x2412 就是：
      左栏 `D xx TxxD xDxD T 7`、右栏 `D xx TxxD xDxD T 5` 并排）。
      双栏下 OCR 读出的框顺序是**跨栏交错**的，按 y 分组会把左右两栏
      串成一行，落格从根上错乱。这种图目前无法自动还原成正确的单栏
      序列，只能识别出来并如实告知，而不是输出一份错位的谱。

    判据（实测区分度很高）：
      逐行看相邻谱框的**最大水平间隙**。单栏谱的组间距最多 1~2 格
      （实测 bfadb43f 全页只有 1 行超过 150px 的间隙，是偶然的行尾留白）；
      双栏谱每行中间都有一条「整栏间距」（实测 ca1b5cb7 有 10 行
      超过 150px，最大 354px）。
      → 大间隙行数 >= 4 判为双栏。
    """
    rows = _group_rows(items)
    big = 0
    for row in rows:
        xs = sorted((float(it["x0"]), float(it["x1"])) for it in row
                    if SCORE_CHARS.match(it["text"].replace(" ", "").strip()))
        if len(xs) < 3:
            continue
        gap = max(xs[k + 1][0] - xs[k][1] for k in range(len(xs) - 1))
        if gap > 150.0:
            big += 1
    return big >= 4


def _is_polluted_row(items: list[dict]) -> bool:
    """判定一行文本框是否被 OCR **连同下划线读成整行乱码**。

    为什么要判这个（实测 44/45 张历史图都中招，每张 1~2 行）：
      谱面里「双下划线」的小节会让 OCR 把下划线当成字符并吞进同一个框。
      实测 bfadb43f 的 y=827：
          x0=117 x1=385  'DxxDxxTxTxxDxxDx'   ← 268px / 15 字符
          x0=407 x1=677  'DxxDxxDxxDxxDxxD'   ← 270px / 15 字符
      原图这一行其实是 `DxxD xxTx TxxD xxDx | DxxD xxDx xDxx DxxD`，
      15 个字符里约 5 个是下划线变的。按正常逻辑会产出 30 个假音符，
      该小节整段变噪音 —— 这是「音节对不上」里最刺耳的一类。

    判据（三条同时满足，实测对正常行零误伤）：
      1. 框数 <= 3        正常谱面行 6~9 个框
      2. 平均框内 >= 6 字  正常框 3~4 字
      3. 主导字符占比 >= 60% 正常行 D/T/X 混合，不可能单字符刷屏

    ⚠️ 第 1 条必需：实测 ca1b5cb7 的 y=1966 也是「4 框 / 32 字 / 平均 8 /
       X 占 22」，但原图确实是疏排的 `D x x x T x x D x x D x T x x x`，
       属正常行。区别就在框数：正常疏排 4 框，污染行只塌成 2 框。
    """
    texts = []
    for it in items:
        t = "".join(ch for ch in str(it.get("text", "")).replace(" ", "")
                    if ch in "DXxTtSs123456789")
        if t:
            texts.append(t)
    if not texts:
        return False
    n_box = len(texts)
    n_ch = sum(len(t) for t in texts)
    if n_box > 3 or n_ch < 8:
        return False
    if n_ch / n_box < 6.0:
        return False
    joined = "".join(texts)
    top = max(joined.count(c) for c in set(joined))
    return top / n_ch >= 0.6


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
        r.sort(key=lambda t: float(t["x0"]))
    # 重叠框裁决要等全页格宽估出来才能做（密度判据依赖格宽），
    # 所以这里只分组、不去重，由 `_page_tokens` 在拿到 cell_w 后调用
    # `_resolve_overlaps`。
    return rows


# 最近一次 _page_tokens 是否判定为双栏（供 image_to_tab 拼告警文案）
_LAST_TWO_COLUMN = False

# 最近一次 image_to_tab 累计清空的「污染行」数（供告警文案）
_LAST_DROPPED_ROWS = 0


def _clean(text: str) -> str:
    """清掉误识别的分隔符，只保留有效谱字符。"""
    out = []
    for ch in text.replace(" ", ""):
        if ch in "DXxTtSs" or ch in "123456789":
            out.append(ch)
        # 其它（| I l . 等）一律丢弃
    return "".join(out)


def _row_chars(row: list[dict],
               cell_w: float | None = None,
               left: float | None = None) -> list[tuple[float, str, float]]:
    """一行文本框 → [(字符中心x, 字符, 字符宽), ...]，按 x 排序。

    OCR 会把相邻几个字符合并成一个文本框（如 "TXXDXDXD"）。
    框内每个字符的真实 x 未知，用**全页格宽**（cell_w）推进最准：
    谱面字符严格等距排列，组内间隔恒等于 1 格宽。

    `left` 传入时，位于谱面左边界**左侧**（小节号列）的纯数字框会被丢弃。
    这是唯一可靠的小节号判据 —— 靠文本格式判断会误杀音符组。
    """
    chars: list[tuple[float, str, float]] = []
    for it in row:
        t = it["text"].replace(" ", "").strip()
        if not t:
            continue
        if not SCORE_CHARS.match(t):
            continue
        # 小节号判定：纯数字 + 位于谱面左边界左侧（含少量容差）
        if BAR_NUMBER.match(t) and _is_bar_number(it, left):
            continue
        cleaned = _clean(t)
        if not cleaned:
            continue
        x0 = float(it["x0"])
        x1 = float(it["x1"])
        n = len(cleaned)
        if n == 1:
            chars.append(((x0 + x1) / 2.0, cleaned, max(x1 - x0, 1.0)))
            continue
        # 框内字符按等距排布。核心模型是**左端锚定 + 格宽推进**：
        #     cx0 = x0 + pitch/2      （第一格中心 = 框左沿 + 半格）
        #     cx_i = cx0 + i * pitch
        #
        # pitch 的取法（按顺序判断）：
        #   1) 没有格宽 → 用框宽均分（退路）；
        #   2) 框明显宽于 n 格（> 1.3 倍）→ 合并了多组，必须铺满整框，
        #      否则后面的字符挤在左半边、整行短掉（实测 y=1271 的
        #      'DxXXTxXD' 8 字符 206px）；
        #   3) 框明显**密于**格宽（< 0.78 倍）→ 真·密集连打（一拍四音），
        #      字符本来就落在相邻格/半格上，按框宽均分；
        #   4) 其余（0.78~1.3 倍）→ 用格宽。这一步是关键：
        #      OCR 框右沿普遍带留白（实测 '2XX' 框宽 91px，3 字符只占
        #      3×24=72px，右边多 19px，比例 1.26）。若按框宽均分会把
        #      3 字符铺成 30px 间距，第 2 个字符落在第 1、2 格之间，
        #      四舍五入后整组右移一格 —— 实测 y=650 因此变成
        #      '2X·XTXXD'（本该 '2XX·TXXD'）。用格宽推进才对齐。
        span = x1 - x0
        if not (cell_w and cell_w > 0):
            pitch = span / n if n else 1.0
        elif span > cell_w * n * 1.3:
            pitch = span / n          # 合并框
        elif span < cell_w * n * 0.78:
            pitch = span / n          # 密集连打
        else:
            pitch = cell_w            # 正常单组：左端锚定 + 格宽
        cx0 = x0 + pitch / 2.0
        for i, ch in enumerate(cleaned):
            chars.append((cx0 + i * pitch, ch, pitch))
    chars.sort(key=lambda c: c[0])
    return chars


def _is_bar_number(item: dict, left: float | None) -> bool:
    """纯数字框是否是小节号。

    判据只有位置：小节号独占谱面最左侧一列，位于谱面左边界
    （第一个音符的 x）左边至少大半个格宽。谱面里的纯数字音符组
    （9876 / 5654 …）全部落在左边界右侧，不会被误判。
    """
    if left is None or left <= 0:
        return False
    x0 = float(item.get("x0", 0.0))
    x1 = float(item.get("x1", 0.0))
    # 整框都在左边界左侧（留 6px 容差）才算小节号
    return x1 < left - 6.0 or (x0 + x1) / 2.0 < left - 10.0


def _estimate_cell_width(xs: list[float]) -> float:
    """估计单格宽度（相邻字符中心之差 = 1 格）。

    谱面结构（实测确认）：
      · 组内字符连续排列，间距**稳定等于 1 格宽**（实测 23~25px @1080 宽）；
      · 组与组之间有节拍留白，间距是格宽的整数倍（51px≈2格、242px≈10格）。

    陷阱：OCR 把一组字符合成一个框后，`_row_chars` 按「框宽/字数」均分，
    若框比实际字符窄（漏读框内部分字符）会产出 2.4px、13.2px 这类伪间距。
    所以不能取「最小间距」（会被伪间距带偏），而应取
    **间距直方图里最密集的一簇**的中位数 —— 真格宽出现的次数最多。
    """
    diffs = [b - a for a, b in zip(xs, xs[1:]) if b - a > 8.0]
    if not diffs:
        return 0.0
    # 按 4px 桶聚类，取成员最多的一簇（真格宽恒在一档，伪间距散落各处）
    buckets: dict[int, list[float]] = {}
    for d in diffs:
        buckets.setdefault(int(d // 4), []).append(d)
    best = max(buckets.values(), key=len)
    # 与最优簇相邻的桶也并入（跨越桶边界的情况）
    key = int(sorted(best)[0] // 4)
    merged = list(best)
    for k in (key - 1, key + 1):
        if k in buckets and len(buckets[k]) >= len(best) * 0.5:
            merged.extend(buckets[k])
    return sorted(merged)[len(merged) // 2]


def _row_left_margin(rows_chars: list[list[tuple[float, str, float]]]) -> float | None:
    """全页统一的谱面左边界（像素）。

    实测：同一页所有谱面行的首字符 x 几乎相同
    （本机样本 12 行中有 9 行在 125~128px，另 2 行因漏读首组偏到 225）。
    用**众数**而非最小值/中位数：漏读首组的行是少数派，众数天然忽略它们。
    """
    vals = []
    for chars in rows_chars:
        if chars:
            vals.append(round(chars[0][0]))
    if not vals:
        return None
    # 按 8px 桶做众数，避免浮点抖动
    buckets: dict[int, list[float]] = {}
    for v in vals:
        buckets.setdefault(v // 8, []).append(v)
    best = max(buckets.values(), key=len)
    return float(sorted(best)[len(best) // 2])


def _page_left_from_boxes(rows: list[list[dict]]) -> float | None:
    """只看文本框位置，估出谱面左边界（第一个音符格的 x0）。

    做法：取所有**多字符**谱框 x0 的众数峰里**最靠左**的那个。

    实测 x0 直方图（本机主样本，12 行 / 106 框）：
        x0≈ 96 桶  n=11  → 小节号列（26/28/31/33/35/37…）  ← 纯数字短框
        x0≈112 桶  n=10  → 音符列（DXXX / DxxX / 2XX …）    ← 谱面左边界
    「最靠左的多字符框」恒落在音符列，因为小节号是纯数字短框（2 位），
    而首组音符是 DXXX/2XX/T156 这类 3~4 字符框。

    为什么不沿用「取最靠右的强峰」：
      每行的第二、第三组也会形成 x0 峰（实测 208/320/424/544/648/736/
      864 各有 3~8 个框），取最右会跑到 653 这种行中位置，
      整页音符左移十几个格，比偏移半格更糟。

    为什么用 x0 而不是字符中心：
      左边界 = 第一格左沿，落格公式用 (中心x - left) / cell_w，
      left 应约等于「第一格中心 - 半格」= 第一格 x0。
    """
    cands: list[float] = []
    fallback: list[float] = []
    for row in rows:
        for it in row:
            t = it["text"].replace(" ", "").strip()
            if not t or not SCORE_CHARS.match(t):
                continue
            fallback.append(float(it["x0"]))
            # 纯数字短框（1~2 位）= 小节号候选，排除
            if t.isdigit() and len(t) <= 2:
                continue
            cands.append(float(it["x0"]))
    if not cands:
        cands = fallback
    if not cands:
        return None
    buckets: dict[int, list[float]] = {}
    for v in cands:
        buckets.setdefault(int(v) // 8, []).append(v)
    top = max(len(v) for v in buckets.values())
    # 强峰（≥ 最大峰 60%）里取最靠左 —— 那就是第一组音符列。
    # 60% 让第二/三组的峰（通常只有 3~8 个）被排除，只剩首组峰。
    key = min(k for k, v in buckets.items() if len(v) >= top * 0.6)
    mem = buckets[key]
    return float(sorted(mem)[len(mem) // 2])


def _row_tokens(row: list[dict],
                segments: list[tuple[float, float, float, int]],
                cell_w_hint: float | None = None,
                left_hint: float | None = None,
                chars: list[tuple[float, str, float]] | None = None) -> list[dict]:
    """一行文本框 → [{"slot": 行内格位, "ch": 字符, "beams": 条数}, ...]

    **按物理坐标落格**，三个关键约定（全部来自对真机图谱面的实测）：

    1. 格宽 = 组内间距（≈23.5px @1080 宽），不是「行跨度 / 32」。
       行尾常有留白、行首偶尔漏读，跨度与格数没有固定换算关系。
    2. 行内格位 = round((x - 左边界) / 格宽)。
       左边界用**全页统一**的谱面左边距（left_hint），不是「本行第一个字符」——
       否则漏读首组的行会把后面所有字符整体左移。
    3. 字符撞格时顺延到下一个空格（不丢字），但绝不跨越
       「本行最后一个字符 + 剩余格位」的范围去污染下一行。

    错误传播被限制在单行内：一行漏读只造成该行局部缺字，
    下一行永远从自己的物理 x 重新落格。
    """
    chars = chars if chars is not None else _row_chars(row, cell_w=cell_w_hint)
    if not chars:
        return []
    xs = [c[0] for c in chars]
    # 谱字符自身跨度：真实谱面行 ≥ 440px；指法图例（D1456789）、
    # 竖排装饰等异物的谱字符挤在 <300px 里，直接丢弃。
    # 注意必须用**谱字符**的跨度而不是整行 items 的跨度 ——
    # 图例行常常和小节号等无关框被聚到同一行，整行跨度会虚高。
    char_span = xs[-1] - xs[0]
    cw_last = chars[-1][2]
    if char_span + cw_last < MIN_ROW_SPAN:
        return []

    # 谱面行判据：**必须含至少一个丁（D/T/S）字符**。
    # ⚠️ 只用跨度过滤不够。实测 y=614 那种小节号残行：
    #      '246' x0=93            '53' x0=550
    #    跨度 = 550+22-93 = 479px > MIN_ROW_SPAN(300)，被当成有效行，
    #    于是 cursor 白推进 ROW_SLOTS(32) 格 —— 后面**每一行都错位一行**，
    #    这正是「输入图片后音节全部对不上」的顶层根因。
    #    谱面每个小节恒以 D/T/S（丁/叮/扫）开头，纯数字行绝不可能是谱面行。
    if not any(c[1].upper() in ("D", "T", "S") for c in chars):
        return []

    # 格宽必须用**全页统一值**。
    # ⚠️ 不能在这里重新估：密集连打行（一拍四音，实测 y=826 的 16 字符
    #    只占 268px，间距 16px）会估出 16px 的「本行格宽」，
    #    整行音符被均匀拉开到 16 格宽的位置，与相邻组的真实格位全错开。
    #    全页格宽 24px 才是谱面的物理格距，密集行只是「一格多音」，
    #    格位仍按 24px 走。
    cell_w = cell_w_hint if (cell_w_hint and cell_w_hint > 8.0) else 0.0
    if cell_w <= 0:
        cell_w = _estimate_cell_width(xs)
    if cell_w <= 0:
        cell_w = max(1.0, char_span / max(1, len(chars)))

    # 左边界：优先用全页众数；本行首字符明显偏离时说明漏读了首组，
    # 此时仍用全页边界，让后面字符回到正确格位（而不是整体左移）。
    x_ref = left_hint if left_hint is not None else xs[0]

    out: list[dict] = []
    beams_of = [(it, count_beams(it, segments)) for it in row]
    last_slot = -1
    for x, ch, _cw in chars:
        # 找到该字符所属的文本框，取它的符杠条数
        beams = 0
        for it, b in beams_of:
            if float(it["x0"]) - 1.0 <= x <= float(it["x1"]) + 1.0:
                beams = b
                break
        slot = int(round((x - x_ref) / cell_w))
        slot = max(0, slot)
        if slot <= last_slot:
            slot = last_slot + 1
        if slot >= ROW_SLOTS:
            break
        out.append({"slot": slot, "ch": ch, "beams": beams})
        last_slot = slot
    return out


def _column_tokens(col: list[dict],
                   segments: list[tuple[float, float, float, int]]
                   ) -> list[list[dict]]:
    """单栏（一列谱面）→ 逐行落格 tokens。

    流程（顺序不能换，每一步都为下一步提供必需的量）：
      0. 只用**框位置**估谱面左边界 left（含量节号会把 left 拉偏）
      1. 第一轮 `_row_chars(cell_w=None)` 框宽均分 → 逐行估格宽 → 取中位数
      2. 用格宽裁决重叠框（重复检测 / 跨组合并），需要格宽才能做
      3. 第二轮 `_row_chars(cell_w=格宽)` 重排字符 → 用首字符 x 众数精修 left
      4. 逐行 `_row_tokens` 落格
    """
    rows = _group_rows(col)
    if not rows:
        return []

    left_hint = _page_left_from_boxes(rows)

    def _row_chars_pass(cell_w: float | None) -> list[list[tuple[float, str, float]]]:
        out: list[list[tuple[float, str, float]]] = []
        for row in rows:
            chars = _row_chars(row, cell_w=cell_w, left=left_hint)
            if not chars:
                out.append([])
                continue
            xs = [c[0] for c in chars]
            if xs[-1] - xs[0] + chars[-1][2] < MIN_ROW_SPAN:
                out.append([])           # 图例/装饰行，后面会被丢弃
                continue
            out.append(chars)
        return out

    rows_chars = _row_chars_pass(None)
    # 全页格宽：逐行估一次，再取这些「行级格宽」的中位数。
    # ⚠️ 不能把所有行的字符 x 混在一起排序再估 —— 跨行相邻会凭空
    # 产生大量中等间距（末字符→下一行首字符），把直方图的峰带偏
    # （实测：混算得 11.35px，真值 23.5px，直接导致全部错位）。
    cws = []
    for ch in rows_chars:
        if len(ch) >= 4:
            w = _estimate_cell_width([c[0] for c in ch])
            if w >= 8.0:
                cws.append(w)
    cell_w_hint = sorted(cws)[len(cws) // 2] if cws else None

    if cell_w_hint:
        rows = [_resolve_overlaps(row, cell_w_hint) for row in rows]
        rows_chars = _row_chars_pass(cell_w_hint)
        polished = _row_left_margin([c for c in rows_chars if c])
        if polished is not None:
            left_hint = polished

    out_rows: list[list[dict]] = []
    for row, chars in zip(rows, rows_chars):
        if not chars:
            continue
        # ── 污染行：整行清空，只**保留占位**
        # 污染行里的字符有一部分是下划线变的（实测 32 字符里约 10 个），
        # 字符本身已经错了，无论怎么裁都拼不回原谱。实测裁剪后仍得到
        # `DxxDxxTxTxx·DxxDxxDxxDx`，而原图是
        # `DxxD xxTx TxxD xxDx | DxxD xxDx xDxx DxxD` —— 组边界全错。
        # 所以整行不放音符，宁可缺一拍（明显、可补录），也不要输出
        # 一串听着像噪音的假音（刺耳、且用户不知道该改哪里）。
        #
        # ⚠️ 但必须推一个空列表进去占住位置：直接跳过会让后面所有行
        #    整体前移一行，小节划分全漂（实测 63 小节 → 73 小节）。
        # ⚠️ 只在最后这一步处理，格宽/左边界估计**照旧让它参与**：
        #    实测把它从格宽中位数里摘出去，中位数 24.2→25.0，
        #    反而让原本正确的 5 行发生退化。
        if _is_polluted_row(row):
            global _LAST_DROPPED_ROWS
            _LAST_DROPPED_ROWS += 1
            out_rows.append([])
            continue
        toks = _row_tokens(row, segments, cell_w_hint=cell_w_hint,
                           left_hint=left_hint, chars=chars)
        if toks:
            out_rows.append(toks)
    return out_rows


def _page_tokens(raw: bytes, timing: "_Timing | None" = None) -> tuple[list[list[dict]], int]:
    """单张图 → 每个有效行的落格 tokens + 有效行数。

    返回 rows_tokens: [[{slot, ch, beams}, ...], ...] —— 外层按行、
    内层按行内格位。调用方按 ROW_SLOTS 给每行分配固定行宽。

    流程：
      1. 识别（provider 内部是「分块 det + 裁剪 rec」，峰值内存 ~316MB）
      2. 在**原图**上独立检测所有横线段（符杠）——原图分辨率，细线不被抹掉
      3. 逐行分组 + 按 x 落格（不再顺序拼接）
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

    # ── 双栏识别：双栏截图无法按几何还原成正确的单栏序列
    #    （左右栏的框被 OCR 跨栏融合，实测所有切分方案都会把
    #    单栏图切坏、或把双栏图切得更乱），这里只记录标记由上层告警。
    global _LAST_TWO_COLUMN
    _LAST_TWO_COLUMN = _detect_two_column(items)
    rows_tokens = _column_tokens(items, segments)
    used_rows = len(rows_tokens)
    t2 = time.time()

    if timing is not None:
        # provider 若已给出更细的分段耗时（子进程模式下由 raster 回传），
        # 这里不覆盖，只补齐缺失项。
        timing.setdefault("infer_ms", int((t1 - t0) * 1000))
        timing.setdefault("format_ms", int((t2 - t1) * 1000))
    return rows_tokens, used_rows


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

    # 每次调用重置污染行计数（多页时跨页累加）
    global _LAST_DROPPED_ROWS
    _LAST_DROPPED_ROWS = 0

    t_fmt0 = time.time()
    for pi, page in enumerate(pages):
        rows_tokens, rows = _page_tokens(page, timing)
        total_rows += rows
        if not rows_tokens:
            empty_pages += 1
            continue
        # 每张图从整行（2 小节）开始，避免跨图串拍
        if cursor % ROW_SLOTS:
            cursor += ROW_SLOTS - cursor % ROW_SLOTS

        # ★ 行对齐落格：每行固定占 ROW_SLOTS 格，行内字符按 x 落到 slot。
        #   一行无论读出多少字符，下一行永远从正确的行首开始 ——
        #   OCR 漏读/误读只造成本行局部缺字，不再把错误传给整首歌。
        #   行内组间空隙自然成为休止（没有字符的 slot 就没有事件）。
        for toks in rows_tokens:
            row_base = cursor
            for tok in toks:
                ch = tok["ch"]
                beams = max(0, min(2, int(tok.get("beams") or 0)))
                if ch == ".":
                    rests += 1
                    continue

                beam_dist[beams] = beam_dist.get(beams, 0) + 1
                dur = DUR_OF[beams]
                pos = row_base + int(tok.get("slot") or 0)
                start = round(pos * sec_per_slot, 3)

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
            cursor = row_base + ROW_SLOTS

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
    if _LAST_TWO_COLUMN:
        warnings.append(
            "这张图是双栏排版（左右两栏谱面并排）。双栏图的字符顺序在 OCR "
            "阶段就是跨栏交错的，自动还原会把音节错位。请改为单栏截图 "
            "（一次只截一栏），或把图旋转/裁成单栏后再传。")
    if _LAST_DROPPED_ROWS:
        warnings.append(
            f"有 {_LAST_DROPPED_ROWS} 行被 OCR 连同下划线读成了乱码"
            "（整行塌成两个大框、一串 x 刷屏），已清空该行内容但保留其"
            "节拍位置。这些行需要对照原图补录，否则会缺几拍。")
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
