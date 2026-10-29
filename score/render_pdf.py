"""谱面出稿：score JSON → PDF（matplotlib）。

版式严格对齐标准样张，参数全部来自 layout.py。
只读 score JSON，不碰识别结果 —— 这是"版面恒定"的保证。
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

from score.layout import CELLS_PER_SUBBAR, COLOR, LINE, NOTE, PAGE, SYSTEM, TEXT

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "assets" / "fonts"

_META = {
    "Title": "Handpan Sheet",
    "Author": "Handpan Sheet Forge",
    "Creator": "Handpan Sheet Forge",
    "Producer": "Handpan Sheet Forge",
    "CreationDate": datetime(2026, 1, 1),
    "ModDate": datetime(2026, 1, 1),
}


def _resolve_cjk_font() -> str:
    candidates = [
        FONT_DIR / "NotoSansSC-Regular.ttf",
        FONT_DIR / "simhei.ttf",
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                return font_manager.FontProperties(fname=str(path)).get_name()
            except Exception:
                continue
    return "DejaVu Sans"


CJK = _resolve_cjk_font()
NUM = "DejaVu Sans"

H = PAGE["h"]
W = PAGE["w"]
INSTRUMENT = "D Kurd 9音"


def _y(from_top: float) -> float:
    """距页顶 mm → matplotlib 坐标（原点左下）。"""
    return H - from_top


def _setup_page():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def _cells_of(score_json: dict) -> int:
    grid = score_json.get("grid") or {}
    try:
        n = int(grid.get("cells") or SYSTEM["cells"])
    except (TypeError, ValueError):
        n = SYSTEM["cells"]
    return max(4, min(n, 24))


def _draw_system(ax, mid_top: float, bars: list[dict], first_bar_no: int, cells: int) -> None:
    """mid_top：该行中线距页顶的 mm。"""
    x0 = PAGE["ml"]
    x1 = W - PAGE["mr"]
    width = x1 - x0
    bar_w = width / SYSTEM["bars"]
    cell_w = bar_w / cells

    mid_c = _y(mid_top)

    # 三级竖线：粗=大节（整行高）、长=小节（每 4 格）、短=格
    for b in range(SYSTEM["bars"]):
        for c in range(cells + 1):
            x = x0 + b * bar_w + c * cell_w
            if c == 0 or c == cells:
                spec = LINE["bar"]
            elif c % CELLS_PER_SUBBAR == 0:
                spec = LINE["long"]
            else:
                spec = LINE["short"]
            half = spec["h"] / 2.0
            ax.plot([x, x], [_y(mid_top + half), _y(mid_top - half)],
                    color=spec["color"], lw=spec["lw"], solid_capstyle="butt")

    ax.plot([x0, x1], [mid_c, mid_c], color=LINE["mid"]["color"], lw=LINE["mid"]["lw"])

    # 小节号：压在粗线上方
    for b in range(SYSTEM["bars"]):
        ax.text(x0 + b * bar_w, _y(mid_top - LINE["bar"]["h"] / 2.0 - 2.2), str(first_bar_no + b),
                fontsize=TEXT["bar_number_size"], ha="center", va="bottom",
                color=COLOR["index"], fontfamily=NUM)

    for b, bar in enumerate(bars[:SYSTEM["bars"]]):
        for ev in bar.get("events") or []:
            slot = int(ev.get("slot") or 0)
            if slot < 0 or slot >= cells:
                continue
            x = x0 + b * bar_w + (slot + 0.5) * cell_w
            for n in ev.get("notes") or []:
                # 两档模型：上档橙、下档蓝（颜色跟档位走）；老数据 hand=mid 视为上档
                hand = n.get("hand") or "upper"
                if hand == "mid":
                    hand = "upper"
                is_upper = hand != "lower"
                label = n.get("label") or (str(n["pos"]) if n.get("pos") else "S")
                color = COLOR["orange"] if is_upper else COLOR["blue"]

                if is_upper:
                    ax.text(x, mid_c + NOTE["gap_mm"], label,
                            fontsize=NOTE["upper_size"], ha="center", va="bottom",
                            color=color, fontfamily=NUM)
                else:
                    ax.text(x, mid_c - NOTE["gap_mm"], label,
                            fontsize=NOTE["lower_size"], ha="center", va="top",
                            color=color, fontfamily=NUM)


def _draw_page(ax, score_json: dict, page: int, total_pages: int = 1) -> None:
    name = (score_json.get("title") or "").strip()
    name = re.sub(r"（等 \d+ 页）$", "", name).strip()   # 多页后缀不进标题
    if name in ("", INSTRUMENT, "Handpan", "D Kurd 9"):
        name = ""
    bpm = (score_json.get("tempo") or {}).get("bpm", 120)
    bars = score_json.get("bars") or []
    cells = _cells_of(score_json)
    per_page = SYSTEM["per_page"] * SYSTEM["bars"]

    if page == 0:
        if name:
            # 歌曲名放最上面、用大字号
            ax.text(W / 2, _y(PAGE["mt"]), name, fontsize=TEXT["title_size"],
                    ha="center", va="top", color=COLOR["title"], fontfamily=CJK)
            # 音阶名作为副标题，小一号
            ax.text(W / 2, _y(PAGE["mt"] + 9.5), INSTRUMENT,
                    fontsize=TEXT["title_size"] * 0.6, ha="center", va="top",
                    color=COLOR["meta"], fontfamily=CJK)
        else:
            ax.text(W / 2, _y(PAGE["mt"]), INSTRUMENT, fontsize=TEXT["title_size"],
                    ha="center", va="top", color=COLOR["title"], fontfamily=CJK)

        ax.text(PAGE["ml"], _y(40.0), f"\u2669 = {bpm:g}",
                fontsize=TEXT["tempo_size"], ha="left", va="top",
                color=COLOR["title"], fontfamily=NUM)

    start = page * per_page
    page_bars = bars[start:start + per_page]
    for i in range(SYSTEM["per_page"]):
        chunk = page_bars[i * SYSTEM["bars"]:(i + 1) * SYSTEM["bars"]]
        if not chunk:
            break
        mid_top = SYSTEM["first_mid_y"] + i * SYSTEM["pitch"]
        _draw_system(ax, mid_top, chunk, start + i * SYSTEM["bars"] + 1, cells)

    # 页脚不输出任何说明文字（用户要求：输出只有谱面本身）。
    # 识别提示仍保留在网页预览里，不印到纸上。
    # 页码只在多页时才出现，单页成品通篇无页脚。
    if total_pages > 1:
        ax.text(W - PAGE["mr"], _y(H - PAGE["mb"] + 4), f"{page + 1} / {total_pages}",
                fontsize=7, ha="right", va="top", color=COLOR["index"], fontfamily=NUM)


def render_score_pdf(score_json: dict) -> bytes:
    bars = score_json.get("bars") or []
    per_page = SYSTEM["per_page"] * SYSTEM["bars"]
    pages = max(1, (len(bars) + per_page - 1) // per_page)

    out = io.BytesIO()
    with PdfPages(out, metadata=_META) as pdf:
        for p in range(pages):
            fig, ax = _setup_page()
            _draw_page(ax, score_json, p, pages)
            pdf.savefig(fig)
            plt.close(fig)

    out.seek(0)
    return out.read()


def render_score_png(score_json: dict, page: int = 0, dpi: int = 110) -> bytes:
    """单页位图：调试与缩略图用。"""
    fig, ax = _setup_page()
    _draw_page(ax, score_json, page, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
