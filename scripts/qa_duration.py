"""QA：验证「长短(时值)」从 OMR → build_score → PDF 全链路打通，并测峰值内存。

1) 单元：合成 events（带/不带 dur）走 build_score + requantize，断言 dur/dur_sec 落地且 BPM 重排后仍保留。
2) 真机：对 2 张真实历史图跑 image_to_tab → build_score，断言输出音符含 dur，且 dur>1 占比合理。
3) 渲染：用一张含长音的构造谱渲染 PNG，确认延音线不报错。
4) 内存：psutil 采样峰值 RSS（含 onnx C++ 分配）。
"""
from __future__ import annotations
import sys, os, time, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import psutil, gc
PROC = psutil.Process(os.getpid())
_peak = [0.0]
_stop = [False]
def _sampler():
    while not _stop[0]:
        try:
            r = PROC.memory_info().rss
            if r > _peak[0]: _peak[0] = r
        except Exception: pass
        time.sleep(0.05)

def section(t): print("\n" + "="*68 + f"\n{t}\n" + "="*68)

def test_unit():
    section("① 单元：build_score + requantize 的 dur 传播")
    from score.note_mapper import build_score, requantize
    # 合成 events：0.0s 音高62(无符杠 dur=1)，0.5s 音高64 带1符杠(dur=2)，1.0s 带2符杠(dur=4)
    evs = [
        {"start": 0.0, "pitch": 62, "dur": 1},
        {"start": 0.5, "pitch": 64, "dur": 2},
        {"start": 1.0, "pitch": 65, "dur": 4},
        {"start": 1.5, "pitch": None, "kind": "stroke", "label": "S", "dur": 2},
    ]
    sj = build_score(evs, bpm=120, source="image", cells=16)
    # 收集所有音符的 dur
    durs = []
    for b in sj["bars"]:
        for e in b["events"]:
            for n in e["notes"]:
                durs.append((n.get("pos"), n.get("midi"), n.get("dur"), n.get("dur_sec")))
    print("  输出音符 (pos, midi, dur, dur_sec):")
    for d in durs: print("   ", d)
    assert any(n.get("dur")==4 for b in sj["bars"] for e in b["events"] for n in e["notes"]), "dur=4 未落地"
    assert any(n.get("dur_sec")==round(4*60/120/4,3) for b in sj["bars"] for e in b["events"] for n in e["notes"]), "dur_sec 错误"
    # raw 第三元素应为 dur
    assert all(len(r)>=3 and isinstance(r[2],int) for r in sj["raw"]), "raw 未带 dur"
    print("  raw 前几行:", sj["raw"][:4], "...")
    # requantize 后 dur 应保留
    sj2 = requantize(sj, 90)
    durs2 = [(n.get("midi"), n.get("dur")) for b in sj2["bars"] for e in b["events"] for n in e["notes"] if n.get("midi")]
    print("  requantize(90bpm) 后 dur 保留:", durs2)
    assert any(d==4 for _,d in durs2), "requantize 后 dur=4 丢失"
    print("  ✅ 单元通过：dur/dur_sec 落地且 BPM 重排后仍保留")

def test_real():
    section("② 真机：2 张历史图 image_to_tab → build_score")
    from extract.omr import image_to_tab
    from score.note_mapper import build_score
    import glob
    imgs = sorted(glob.glob(os.path.join(ROOT,"data/uploads/2026/**/*.jpg"), recursive=True))[:2]
    th = threading.Thread(target=_sampler, daemon=True); _stop[0]=False; th.start()
    t0=time.time()
    for p in imgs:
        raw = open(p,"rb").read()
        out = image_to_tab(raw)
        evs = out.get("events") or []
        sj = build_score(evs, bpm=out.get("bpm",120), source="image", bars=out.get("bars"), cells=out.get("cells") or 12)
        # 统计输出里 dur>1 的音符数
        longs = sum(1 for b in sj["bars"] for e in b["events"] for n in e["notes"] if (n.get("dur") or 1) > 1)
        alln = sum(1 for b in sj["bars"] for e in b["events"] for n in e["notes"])
        print(f"  {os.path.basename(p)}: events={len(evs)} bars={len(sj['bars'])} 音符={alln} 其中长音(dur>1)={longs}")
        assert longs > 0, "真机未识别出任何长音(dur>1)"
    _stop[0]=True; th.join()
    print(f"  ✅ 真机通过；本次 OMR 峰值RSS={_peak[0]/1024/1024:.1f}MB，耗时 {time.time()-t0:.1f}s")

def test_render():
    section("③ 渲染：构造含长音的谱，渲染 PNG 验证延音线不报错")
    from score.note_mapper import build_score
    from score.render_pdf import render_score_png
    evs = [
        {"start": 0.0, "pitch": 50, "dur": 4},
        {"start": 1.0, "pitch": 62, "dur": 2},
        {"start": 2.0, "pitch": 64, "dur": 1},
        {"start": 3.0, "pitch": None, "kind":"stroke","label":"S","dur":2},
    ]
    sj = build_score(evs, bpm=120, source="image", cells=16)
    png = render_score_png(sj, page=0, dpi=110)
    print(f"  PNG 字节数={len(png)}")
    assert len(png) > 1000, "PNG 渲染异常"
    outp = os.path.join(ROOT, "data", "qa_sample_long.png")
    open(outp,"wb").write(png)
    print(f"  ✅ 渲染通过；样例已写 {outp}")

if __name__ == "__main__":
    test_unit()
    test_real()
    test_render()
    print("\n全部 QA 通过 ✅")
