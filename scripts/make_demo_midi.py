"""生成一段 D Kurd 风格的示例 MIDI，用来验证整条链路（不需要识别引擎）。

用法：.venv/Scripts/python.exe -m scripts.make_demo_midi
产物：data/demo_kurd.mid（可以直接在网页上传，走 .mid 通道出谱）
"""
from __future__ import annotations

from pathlib import Path

import mido

BPM = 150
TPB = 480
BAR = TPB * 4

# 8 小节循环：上手旋律 + 下手低音
UPPER = [3, 8, 8, 3, 4, 8, 8, 4]
LOWER = [1, None, None, 1, 2, None, None, 2]
MIDI = {1: 50, 2: 57, 3: 58, 4: 60, 5: 62, 6: 64, 7: 65, 8: 67, 9: 69}


def build(path: Path, bars: int = 16) -> Path:
    mid = mido.MidiFile(ticks_per_beat=TPB)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    pending: list[tuple[int, mido.Message]] = []

    def add(pos: int, note: int, vel: int = 96) -> None:
        pending.append((pos, mido.Message("note_on", note=note, velocity=vel)))
        pending.append((pos + 60, mido.Message("note_off", note=note, velocity=0)))

    for b in range(bars):
        base = b * BAR
        k = b % 8
        add(base, MIDI[UPPER[k]], 100)
        if LOWER[k]:
            add(base, MIDI[LOWER[k]], 80)
        add(base + TPB, MIDI[3 if b % 3 == 0 else 8], 92)
        add(base + TPB * 2, MIDI[4 if b % 4 == 1 else 8], 92)
        if b % 4 == 2:
            add(base + TPB * 2, MIDI[2], 76)
        add(base + TPB * 3, MIDI[3 if b % 2 else 7], 92)

        # 八分音符加花
        add(base + TPB // 2, MIDI[8], 70)
        add(base + TPB * 2 + TPB // 2, MIDI[7 if b % 5 == 0 else 8], 70)

    pending.sort(key=lambda x: x[0])
    last = 0
    for tick, msg in pending:
        track.append(msg.copy(time=max(0, tick - last)))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=TPB))

    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "demo_kurd.mid"
    print("已生成：", build(out))
