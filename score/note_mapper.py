"""音位映射引擎 —— 本产品的核心。

输入：任意 note_events（视频/OMR/MIDI 都归一成这个结构）
输出：schema 1.1 的 score JSON，只描述"第几小节第几格有哪些音位"

三条设计约束（对应 PRD 6.3 / 技术方案 5.4）：
1. 映射是纯函数：同一个 pitch 永远得到同一个音位，排版才可能恒定。
2. 保留 raw 原始时间戳：改 BPM 重排的唯一依据，不存就重排不了。
3. 音阶表是唯一事实源：换音阶只改 D_KURD_9。
"""
from __future__ import annotations

# ★ 音位表：1–2 蓝，3–9 橙（D-1 已确认：第 9 音独立显示为 9）
D_KURD_9 = [
    {"pos": 1, "name": "D3",  "midi": 50, "color": "blue"},    # Ding
    {"pos": 2, "name": "A3",  "midi": 57, "color": "blue"},
    {"pos": 3, "name": "Bb3", "midi": 58, "color": "orange"},
    {"pos": 4, "name": "C4",  "midi": 60, "color": "orange"},
    {"pos": 5, "name": "D4",  "midi": 62, "color": "orange"},
    {"pos": 6, "name": "E4",  "midi": 64, "color": "orange"},
    {"pos": 7, "name": "F4",  "midi": 65, "color": "orange"},
    {"pos": 8, "name": "G4",  "midi": 67, "color": "orange"},
    {"pos": 9, "name": "A4",  "midi": 69, "color": "orange"},
]

BLUE_POSITIONS = {1, 2}
DEFAULT_CELLS = 12          # 每小节格数，与标准样张一致（12 格 = 三连音八分）
ONSET_MERGE_SEC = 0.06      # 60ms 内视为同一事件（金属泛音溢出）
MAX_SIMULTANEOUS = 2        # 只保留最高 + 最低


def color_of(pos: int) -> str:
    return "blue" if pos in BLUE_POSITIONS else "orange"


def nearest_position(pitch: int) -> tuple[dict, int, int]:
    """在 ±2 个八度内取全场最小半音距离。

    F#3(54) 会落到 F4(65)（距离 1），而不是被"按中心折叠"错判成 A3(57)（距离 3）。
    """
    best: tuple[int, dict, int] | None = None
    # 顺序很重要：先试 shift=0，否则 D4(62) 会先匹配到低八度的 D3(50)，
    # 而严格小于导致后面的精确匹配再也覆盖不上——5、9 号音位会被吞成 1、2。
    for shift in (0, -12, 12, -24, 24):
        cand = pitch + shift
        for item in D_KURD_9:
            d = abs(item["midi"] - cand)
            if best is None or d < best[0]:
                best = (d, item, cand)
    assert best is not None
    return best[1], best[0], best[2]


def map_pitch(pitch: int) -> dict | None:
    item, dist, folded = nearest_position(pitch)
    if dist > 1:          # 差 2 个半音以上视为音阶外，丢弃并计数
        return None
    return {
        "pos": item["pos"],
        "midi": item["midi"],
        "approx": dist == 1,
        "folded_from": pitch if folded != pitch else None,
    }


def group_events(notes: list[dict], threshold: float = ONSET_MERGE_SEC) -> list[list[dict]]:
    notes = sorted(notes, key=lambda n: n["start"])
    events: list[list[dict]] = []
    cur: list[dict] = []
    anchor = 0.0
    for n in notes:
        if not cur:
            cur, anchor = [n], n["start"]
        elif n["start"] - anchor <= threshold:
            cur.append(n)
        else:
            events.append(cur)
            cur, anchor = [n], n["start"]
    if cur:
        events.append(cur)
    return events


def build_score(
    note_events: list[dict],
    bpm: float = 120,
    source: str = "video",
    title: str = "D Kurd 9音",
    bars: int | None = None,
    cells: int = DEFAULT_CELLS,
) -> dict:
    """note_events: [{"start": 秒, "pitch": MIDI}, ...]"""
    cells = max(4, min(int(cells), 24))
    slots_per_beat = max(1, cells // 4)
    bpm = float(bpm)
    sec_per_slot = 60.0 / bpm / slots_per_beat
    approx_count = dropped = over_limit = tap_total = 0
    grid: dict[tuple[int, int], dict] = {}

    for ev in group_events(note_events):
        mapped = []
        tap_count = 0
        tap_label = "S"
        tap_beams = 0
        for n in ev:
            if n.get("pitch") is None:      # S / X / T：无音高敲击，保留时值、不落音位
                tap_count += 1
                tap_label = n.get("label") or "S"
                tap_beams = int(n.get("beams") or 0)
                continue
            m = map_pitch(int(round(n["pitch"])))
            if m is None:
                dropped += 1
            else:
                if n.get("label"):
                    m["label"] = n["label"]
                if n.get("beams"):
                    m["beams"] = int(n["beams"])
                mapped.append(m)
        if not mapped and not tap_count:
            continue

        uniq: dict[int, dict] = {}
        for m in mapped:
            uniq.setdefault(m["pos"], m)
        ordered = sorted(uniq.values(), key=lambda m: m["midi"])

        if len(ordered) > MAX_SIMULTANEOUS:
            over_limit += 1

        # 两档模型（用户规则）：上档橙、下档蓝，不再有中档；S 也可上可下
        #   1 个音      → 上档
        #   2 个音      → 高音上档、低音下档
        #   1 个音 + S  → 音在上档、S 在下档
        #   只有 S      → S 上档
        slot_abs = int(round(ev[0]["start"] / sec_per_slot))
        if slot_abs < 0:
            continue
        bar_idx, slot = divmod(slot_abs, cells)
        cell = grid.setdefault((bar_idx, slot), {"bar": bar_idx, "slot": slot, "notes": []})

        def _lane(hand: str, note: dict) -> None:
            if any(x.get("pos") == note.get("pos") and x.get("kind") == note.get("kind")
                   for x in cell["notes"]):
                return
            note["hand"] = hand
            note["color"] = "orange" if hand == "upper" else "blue"
            cell["notes"].append(note)

        def _from_m(m: dict) -> dict:
            d = {"pos": m["pos"], "midi": m["midi"], "approx": m["approx"]}
            if m.get("label"):
                d["label"] = m["label"]
            return d

        tap_note = {"pos": 0, "midi": 0, "label": tap_label or "S",
                    "kind": "stroke", "approx": False}

        if ordered:
            top = ordered[-1]
            if top["approx"]:
                approx_count += 1
            _lane("upper", _from_m(top))
            if len(ordered) > 1:
                bot = ordered[0]
                if bot["approx"]:
                    approx_count += 1
                _lane("lower", _from_m(bot))
            elif tap_count:
                _lane("lower", dict(tap_note))
        elif tap_count:
            _lane("upper", dict(tap_note))

        if tap_count:
            tap_total += 1

    max_bar = max((b for b, _ in grid), default=-1)
    total_bars = max(max_bar + 1, bars or 0, 1)
    out_bars = []
    for b in range(total_bars):
        events = [grid[(bb, s)] for (bb, s) in sorted(grid) if bb == b]
        out_bars.append({"index": b + 1, "events": events})

    total_notes = sum(len(c["notes"]) for c in grid.values())
    pitched_notes = sum(1 for c in grid.values() for n in c["notes"] if n.get("kind") != "tap")
    warnings = []
    if approx_count:
        warnings.append(f"{approx_count} 个音为近似映射（音阶外音已就近折叠）")
    if dropped:
        warnings.append(f"{dropped} 个音超出音域已丢弃")
    if over_limit:
        warnings.append(f"{over_limit} 处超过双音上限，仅保留最高与最低音")
    if tap_total:
        warnings.append(f"{tap_total} 处无音高敲击（S）已保留在谱面上，仅示节奏")
    if not pitched_notes:
        warnings.append("未识别到音符，请在预览页手动录入")

    return {
        "version": "1.1",
        "title": title,
        "instrument": "D Kurd 9",
        "tempo": {"bpm": round(bpm, 1), "meter": "4/4"},
        "source": {"type": source},
        "grid": {"cells": cells, "bars_per_system": 2},
        "stats": {
            "total_notes": pitched_notes,
            "taps": tap_total,
            "approx": approx_count,
            "dropped": dropped,
            "bars": len(out_bars),
        },
        "warnings": warnings,
        # ★ 原始时间戳：改 BPM 重排的唯一依据（无音高敲击存 null）
        "raw": [
            [round(float(n["start"]), 3),
             None if n.get("pitch") is None else int(round(n["pitch"]))]
            for n in note_events
        ],
        "bars": out_bars,
    }


def requantize(score_json: dict, bpm: float) -> dict:
    """改 BPM 整首重排：只重算时间网格，音位一个都不动。

    build_score 的映射是纯函数，同样的 pitch 必然得到同样的音位，
    所以重排前后音位集合完全一致，只有落格/小节数会变。
    """
    raw = score_json.get("raw") or []
    if not raw:
        raise ValueError("缺少 raw 时间戳，无法重排，请重新提取素材")

    events = [{"start": t, "pitch": p} for t, p in raw]
    rebuilt = build_score(
        events,
        bpm=bpm,
        source=(score_json.get("source") or {}).get("type", "video"),
        title=score_json.get("title", "D Kurd 9音"),
        cells=int((score_json.get("grid") or {}).get("cells") or DEFAULT_CELLS),
        # 不继承旧小节数：BPM 变了，同样时长能装下的小节数本来就会变
    )
    rebuilt["metrics"] = score_json.get("metrics", {})
    return rebuilt


def empty_score(bars: int = 12, title: str = "D Kurd 9音", bpm: float = 120,
                source: str = "image") -> dict:
    """空白谱格：截图未启用自动识别时，给用户一个可手动录入的画布。"""
    return {
        "version": "1.1",
        "title": title,
        "instrument": "D Kurd 9",
        "tempo": {"bpm": bpm, "meter": "4/4"},
        "source": {"type": source},
        "grid": {"cells": DEFAULT_CELLS, "bars_per_system": 2},
        "stats": {"total_notes": 0, "approx": 0, "dropped": 0, "bars": bars},
        "warnings": ["未启用自动识别，请在预览页手动录入音符"],
        "raw": [],
        "bars": [{"index": i + 1, "events": []} for i in range(bars)],
    }
