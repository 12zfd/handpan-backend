"""QA：回归抽样 —— 10 张真实图 + 老数据兼容（2元素 raw / 无 dur 事件）。"""
from __future__ import annotations
import sys, os, time, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    from extract.omr import image_to_tab
    from score.note_mapper import build_score, requantize
    import psutil, threading
    PROC = psutil.Process(os.getpid()); peak=[0.0]; stop=[False]
    def samp():
        while not stop[0]:
            try:
                r=PROC.memory_info().rss
                if r>peak[0]: peak[0]=r
            except Exception: pass
            time.sleep(0.05)
    th=threading.Thread(target=samp, daemon=True); stop[0]=False; th.start()

    imgs = sorted(glob.glob(os.path.join(ROOT,"data/uploads/2026/**/*.jpg"), recursive=True))
    sample = imgs[::max(1,len(imgs)//10)][:10]   # 均匀抽 10 张
    print(f"回归抽样 {len(sample)}/{len(imgs)} 张：")
    fails = 0
    for p in sample:
        raw = open(p,"rb").read()
        try:
            out = image_to_tab(raw)
            evs = out.get("events") or []
            sj = build_score(evs, bpm=out.get("bpm",120), source="image",
                             bars=out.get("bars"), cells=out.get("cells") or 12)
            longs = sum(1 for b in sj["bars"] for e in b["events"] for n in e["notes"] if (n.get("dur") or 1)>1)
            alln = sum(1 for b in sj["bars"] for e in b["events"] for n in e["notes"])
            print(f"  {os.path.basename(p)[:12]}… bars={len(sj['bars']):3d} notes={alln:3d} 长音={longs:3d} "
                  f"warnings={len(out.get('warnings',[]))}")
        except Exception as e:
            fails += 1
            print(f"  ❌ {os.path.basename(p)[:12]}… 失败: {e!r}")
    stop[0]=True; th.join()
    print(f"失败 {fails}/{len(sample)}；峰值RSS={peak[0]/1024/1024:.1f}MB")

    # ── 老数据兼容
    print("\n老数据兼容：")
    evs_old = [{"start":0.0,"pitch":62},{"start":0.5,"pitch":64}]   # 无 dur（视频/MIDI）
    sj = build_score(evs_old, bpm=120, source="video", cells=16)
    durs = [n.get("dur") for b in sj["bars"] for e in b["events"] for n in e["notes"]]
    print(f"  无dur事件 → 默认 dur: {durs}")
    assert all(d==1 for d in durs), "无dur事件默认应为1"
    # 老 raw（2元素）requantize
    sj_old = {"version":"1.1","title":"t","instrument":"D Kurd 9","tempo":{"bpm":120,"meter":"4/4"},
              "source":{"type":"video"},"grid":{"cells":16,"bars_per_system":2},
              "stats":{},"warnings":[],"raw":[[0.0,62],[0.5,64]],"bars":[]}
    sj2 = requantize(sj_old, 100)
    durs2 = [n.get("dur") for b in sj2["bars"] for e in b["events"] for n in e["notes"]]
    print(f"  老 raw(2元素) requantize → dur: {durs2}")
    assert all(d==1 for d in durs2), "老raw requantize 默认应为1"
    print("  ✅ 老数据兼容通过")

if __name__ == "__main__":
    main()
