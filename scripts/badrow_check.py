# -*- coding: utf-8 -*-
"""异常行检测：找出「一行被 OCR 读成极少数超宽框」的畸形行。

背景
----
实测 bfadb43f 的 y=827 行：
    x0=117 x1=385  'DxxDxxTxTxxDxxDx'     ← 268px 宽，15 字符
    x0=407 x1=677  'DxxDxxDxxDxxDxxD'     ← 270px 宽，15 字符

原谱这一行实际是 `DxxX TxxD xDxD Txxx | DxxX TxxD xxDx Txxx`，
OCR 把**下划线**读成了 D，把整行拼成一串。这类行按当前逻辑会被
当成 30 个合法音符落格，产生一整行的假音符。

判据（三条同时满足才判定畸形）：
  1. 行内框数 ≤ 3（正常行 6~9 个框）
  2. 平均每框字符数 ≥ 6（正常框 3~4 字符）
  3. 同一字符占比 ≥ 60%（正常谱面 D/T/X 混合，不可能单字符刷屏）

本脚本统计畸形行在全部历史样本里的出现率，用来决定是否要在生产代码里
做整行屏蔽。只读，不改任何数据。

用法：
    .venv/Scripts/python.exe scripts/badrow_check.py
"""
from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

DB = BASE / "data" / "handpan.db"


def scan_row(row: list[dict]) -> tuple[bool, str]:
    """判定一行的文本框是否「被 OCR 污染」。"""
    texts = []
    for it in row:
        t = "".join(ch for ch in it["text"].replace(" ", "")
                    if ch in "DXxTtSs123456789")
        if t:
            texts.append(t)
    if not texts:
        return False, ""
    n_box = len(texts)
    n_ch = sum(len(t) for t in texts)
    if n_box > 3:
        return False, ""
    if n_ch < 8:
        return False, ""
    if n_ch / n_box < 6:
        return False, ""
    cnt = Counter("".join(texts))
    top_ch, top_n = cnt.most_common(1)[0]
    if top_n / n_ch < 0.6:
        return False, ""
    return True, f"boxes={n_box} chars={n_ch} top={top_ch}x{top_n}"


def main() -> int:
    import extract.omr as omr

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, original_file_url FROM scores "
        "WHERE status='done' AND source_type='image' AND midi_data_json IS NOT NULL "
        "ORDER BY id"
    ).fetchall()

    seen: set[str] = set()
    total_rows = 0
    total_bad = 0
    report: list[tuple[int, str, int, int]] = []

    for r in rows:
        url = r["original_file_url"] or ""
        name = url.split("/")[-1].strip('"[]')
        if name in seen:
            continue
        p = BASE / "data" / "uploads" / "2026" / "10" / name
        if not p.exists():
            continue
        seen.add(name)
        try:
            proc = omr._normalize(p.read_bytes(), omr.MAX_IMAGE_WIDTH)
            items = omr._ocr_items(proc)
            grp = omr._group_rows(items)
        except Exception as exc:  # noqa: BLE001
            print(f'{r["id"]:>3} 失败 {type(exc).__name__}: {str(exc)[:50]}')
            continue
        bad = 0
        for row in grp:
            ok, why = scan_row(row)
            if ok:
                bad += 1
                yc = sum(i["yc"] for i in row) / len(row)
                if bad <= 3:
                    print(f'  {name[:16]:<16} y={yc:>4.0f}  {why}')
        total_rows += len(grp)
        total_bad += bad
        report.append((r["id"], name[:16], len(grp), bad))

    print(f'\n唯一图片 {len(report)} 张 / 共 {total_rows} 个行组')
    print(f'畸形行合计 {total_bad} 处（占 {total_bad / max(1, total_rows) * 100:.1f}%）')
    hit = [x for x in report if x[3]]
    print(f'含畸形行的图片：{len(hit)}/{len(report)}')
    for sid, name, nr, nb in hit:
        print(f'   id={sid:<3} {name:<16} {nb}/{nr} 行畸形')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
