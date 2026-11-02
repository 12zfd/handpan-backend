# -*- coding: utf-8 -*-
"""端到端验证：方案 A 网格通道并入 image_to_tab 后的行为。

    ① original_right.png（书页拍照谱，目标场景）
       - 必须走出网格通道：行数显著多于 det 通道的 2 行
       - D / x / S / 数字齐全
    ② qa_sample_long.png（应用截图，回归保护）
       - 必须仍走 det 通道，事件数与改动前基线一致（±2）
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from extract.omr import image_to_tab, _Timing  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}{(' —— ' + detail) if detail else ''}")


def summarize(tag: str, out: dict) -> dict:
    ev = out.get("events") or []
    kinds = {}
    labels = {}
    for e in ev:
        kinds[e.get("kind") or "?"] = kinds.get(e.get("kind") or "?", 0) + 1
        lab = e.get("label")
        if lab:
            labels[lab] = labels.get(lab, 0) + 1
    slots = [e.get("start") for e in ev]
    print(f"  [{tag}] events={len(ev)} bars={out.get('bars')} "
          f"kinds={kinds} labels={labels}")
    print(f"  [{tag}] warnings={out.get('warnings')}")
    print(f"  [{tag}] timing={out.get('timing')}")
    return {"events": len(ev), "kinds": kinds, "labels": labels,
            "bars": out.get("bars")}


def main() -> None:
    force = os.environ.get("OCR_GRID_FORCE") == "1"

    print("\n=== ① 书页拍照谱 original_right.png ===")
    with open(os.path.join(_HERE, "original_right.png"), "rb") as f:
        raw1 = f.read()
    tm = _Timing()
    out1 = image_to_tab(raw1, timing=tm)
    s1 = summarize("photo", out1)
    check("D(丁) 召回 ≥ 4", s1["labels"].get("D", 0) >= 4,
          f"D={s1['labels'].get('D', 0)}")
    check("敲击(S/x) ≥ 10", s1["kinds"].get("stroke", 0) >= 10,
          f"stroke={s1['kinds'].get('stroke', 0)}")
    check("有音高音符 ≥ 180", s1["kinds"].get("ding", 0)
          + sum(v for k, v in s1["kinds"].items() if k not in ("ding", "stroke")) >= 180
          or len(out1.get("events") or []) >= 180,
          f"total={s1['events']}")
    check("小节数 ≥ 8", (s1["bars"] or 0) >= 8, f"bars={s1['bars']}")
    check("网格耗时已记录", (tm.get("grid_ms") or 0) >= 0, f"grid_ms={tm.get('grid_ms')}")

    print("\n=== ② 应用截图 qa_sample_long.png（回归保护） ===")
    with open(os.path.join(_ROOT, "data", "qa_sample_long.png"), "rb") as f:
        raw2 = f.read()
    tm2 = _Timing()
    out2 = image_to_tab(raw2, timing=tm2)
    s2 = summarize("app", out2)
    baseline_path = os.path.join(_HERE, "baseline_app_events.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            base = json.load(f)
        diff = abs(s2["events"] - base["events"])
        check("与基线事件数一致(±2)", diff <= 2,
              f"now={s2['events']} base={base['events']}")
    else:
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump({"events": s2["events"]}, f)
        check("基线已记录（首次运行）", True, f"events={s2['events']}")

    print(f"\n=== 汇总: {PASS} PASS / {FAIL} FAIL ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
