"""MIDI 文件 → note_events。

这是一条"直连"通道：MIDI 本身就是提取结果，跳过音频识别。
用途：① 手上有现成 MIDI 时直接出谱；② 不装识别引擎也能验证整条排版链路。
"""
from __future__ import annotations

import io

import mido


def midi_to_note_events(raw: bytes) -> tuple[list[dict], float]:
    """返回 (note_events, bpm)。"""
    mid = mido.MidiFile(file=io.BytesIO(raw))
    tpb = mid.ticks_per_beat or 480

    bpm = 120.0
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                bpm = round(mido.tempo2bpm(msg.tempo), 1)
                break
        else:
            continue
        break

    events: list[dict] = []
    for track in mid.tracks:
        ticks = 0
        tempo = mido.bpm2tempo(bpm)
        for msg in track:
            ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                events.append({
                    "start": round(mido.tick2second(ticks, tpb, tempo), 3),
                    "pitch": int(msg.note),
                    "amp": round(msg.velocity / 127, 3),
                })

    events.sort(key=lambda e: e["start"])
    return events, float(bpm)
