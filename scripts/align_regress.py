# -*- coding: utf-8 -*-
"""全样本对齐回归：拿数据库里所有历史图片任务，对比「改前（库里的产出）」vs「改后（现在跑）」。

为什么要这么比
--------------
用户的原话是「音节始终不对齐」。对齐问题的本质是**每个音符落在第几格**
（slot），而光看总数（多少小节、多少音符）看不出来错位 ——
21 小节 vs 24 小节都可能一样「总数对、位置全偏」。

所以这里的核心指标是
    `first_mismatch`  第几个格位开始与历史产出不一致
    `shift`           整体偏移量（改后 slot - 改前 slot 的众数）
    `aligned_ratio`   两版 slot 序列逐位相同的比例

比对方式：把两版都还原成「格位 → 字符/音符」的序列后做最长公共子序列式
逐位比对（用 raw 的 start→slot 映射，不依赖 bars 展开的差异）。

用法（在 handpan-backend 目录下）：
    .venv/Scripts/python.exe scripts/align_regress.py            # 全部
    .venv/Scripts/python.exe scripts/align_regress.py 49 45 4    # 指定 id
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

DB = BASE / "data" / "handpan.db"


def _slots_of(raw: list, cells: int) -> list[int]:
    """事件序列 → [slot_abs, ...]（绝对拍点，四舍五入到格）。

    兼容两种格式：
      · 库里的历史产出：raw = [[start_sec, midi], ...]
      · 现在的 image_to_tab：events = [{start, kind, ...}, ...]
    """
    out = []
    for ev in raw or []:
        try:
            if isinstance(ev, dict):
                start = float(ev.get("start"))
            else:
                start = float(ev[0])
        except (TypeError, ValueError, IndexError):
            continue
        out.append(int(round(start * 8)))   # 1 格 = 0.125s（tempo 120, 16分）
    return out


def _align(a: list[int], b: list[int]) -> dict:
    """逐位比对两版 slot 序列。"""
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    shift = Counter(b[i] - a[i] for i in range(n)).most_common(1)
    first_bad = next((i for i in range(n) if a[i] != b[i]), -1)
    return {
        "n_cmp": n,
        "same": same,
        "ratio": (same / n) if n else 0.0,
        "shift": shift[0][0] if shift else 0,
        "shift_n": shift[0][1] if shift else 0,
        "first_bad": first_bad,
        "len_a": len(a),
        "len_b": len(b),
    }


def main() -> int:
    import extract.omr as omr

    want = [int(x) for x in sys.argv[1:] if x.isdigit()]
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, title, source_type, original_file_url, midi_data_json "
        "FROM scores WHERE status='done' AND source_type='image' "
        "AND midi_data_json IS NOT NULL ORDER BY id"
    ).fetchall()
    if want:
        rows = [r for r in rows if r["id"] in want]

    print(f"共 {len(rows)} 个历史图片样本\n")
    hdr = (f'{"id":>3} {"改前":>16} {"改后":>16} {"逐位相同":>9} '
           f'{"整体偏移":>8} {"首个不一致":>10}  {"文件名":<24}')
    print(hdr)
    print("─" * len(hdr.encode("gbk", "ignore")))

    worse = []
    for r in rows:
        url = r["original_file_url"] or ""
        name = url.split("/")[-1]
        p = BASE / "data" / "uploads" / "2026" / "10" / name
        if not p.exists():
            print(f'{r["id"]:>3} {"文件缺失":>16}  {name}')
            continue
        try:
            hist = json.loads(r["midi_data_json"])
            cells = int((hist.get("grid") or {}).get("cells") or 16)
            a = _slots_of(hist.get("raw"), cells)
            tab = omr.image_to_tab(p.read_bytes())
            b = _slots_of(tab.get("events"), cells)
            st = _align(a, b)
            bst = tab.get("stats") or {}
            print(f'{r["id"]:>3} {len(a):>6}音/{ (hist.get("stats") or {}).get("bars", 0):>3}节 '
                  f'{len(b):>6}音/{tab.get("bars", 0):>3}节 '
                  f'{st["ratio"]*100:>7.1f}% '
                  f'{st["shift"]:>+5d}({st["shift_n"]:>3}) '
                  f'{str(st["first_bad"]):>10}  {name[:24]:<24}'
                  f'  [改后 pitched={bst.get("pitched")} taps={bst.get("unpitched")}]')
            if st["ratio"] < 0.6:
                worse.append((r["id"], st["ratio"]))
        except Exception as exc:  # noqa: BLE001
            print(f'{r["id"]:>3} 失败: {type(exc).__name__}: {str(exc)[:60]}')
    print()
    if worse:
        print("⚠ 逐位一致率 <60% 的样本：", worse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
