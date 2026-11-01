"""QA：对真实历史图片跑 OMR，输出识别质量 + 峰值内存。

用法：
  .venv/Scripts/python.exe scripts/qa_omr.py [图片路径...]
不传参数则自动取 data/uploads/2026 下前 3 张 jpg。
"""
from __future__ import annotations
import sys, os, io, time, threading, glob

# ── Windows RSS 采样（能抓到 onnxruntime 的 C++ 分配，tracemalloc 抓不到）
try:
    import ctypes
    from ctypes import wintypes
    _k32 = ctypes.windll.kernel32
    class _PROC_MEM(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    def _rss() -> int:
        p = _k32.GetCurrentProcess()
        m = _PROC_MEM()
        m.cb = ctypes.sizeof(m)
        _k32.GetProcessMemoryInfo(p, ctypes.byref(m), m.cb)
        return m.WorkingSetSize
    HAVE_RSS = True
except Exception:
    HAVE_RSS = False
    def _rss(): return 0

_peak = [0]
_stop = [False]
def _sampler():
    while not _stop[0]:
        v = _rss()
        if v > _peak[0]:
            _peak[0] = v
        time.sleep(0.05)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        cand = sorted(glob.glob(os.path.join(root, "data/uploads/2026/**/*.jpg"), recursive=True))
        paths = cand[:3]
    if not paths:
        print("no images found"); return

    sys.path.insert(0, root)
    from extract.omr import image_to_tab, _Timing
    from score.note_mapper import build_score

    th = threading.Thread(target=_sampler, daemon=True)
    _stop[0] = False
    th.start()
    t0 = time.time()

    for p in paths:
        raw = open(p, "rb").read()
        print("="*70)
        print(f"IMAGE: {os.path.basename(p)}  ({len(raw)} bytes)")
        try:
            tm = _Timing()
            out = image_to_tab(raw, timing=tm)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("OMR FAILED:", repr(e)); continue
        evs = out.get("events") or []
        bd = {}
        for e in evs:
            bd[e.get("beams", 0)] = bd.get(e.get("beams", 0), 0) + 1
        print(f"  rows(total)={out.get('stats',{}).get('rows')}  events={len(evs)}  "
              f"bars={out.get('bars')}  cells={out.get('cells')}")
        print(f"  beam分布(0/1/2下划线)={bd}  时值DUR_OF映射里0:1,1:2,2:1")
        print(f"  warnings({len(out.get('warnings',[]))}):")
        for w in out.get("warnings", []):
            print("    -", w)
        # 跑 build_score，看 dur 是否影响输出
        notes = [e for e in evs if e.get("pitch") is not None]
        sj = build_score(evs, bpm=out.get("bpm",120), source="image",
                         bars=out.get("bars"), cells=out.get("cells") or 12)
        # 检查输出里每个事件的“时值占位”是否区分长短
        print(f"  build_score -> bars={len(sj['bars'])} total_notes={sj['stats']['total_notes']}")
        # 抽样前 20 个事件：start/slot/pitch/dur
        print("  抽样 events[0:24] (start, pitch, beams, dur):")
        for e in evs[:24]:
            print(f"    start={e.get('start')} pitch={e.get('pitch')} beams={e.get('beams')} dur={e.get('dur')}")
        # 关键验证：dur 是否在 build_score 输出里体现？
        # build_score 的 bars[].events[].notes 没有 dur 字段 —— 打印确认
        sample_cell = None
        for b in sj["bars"]:
            if b["events"]:
                sample_cell = b["events"][0]; break
        print(f"  输出单元样例 keys={list(sample_cell.keys()) if sample_cell else None}  (无 dur 即长短未落稿)")

    _stop[0] = True
    th.join()
    print("="*70)
    print(f"TOTAL wall={time.time()-t0:.1f}s  peakRSS={_peak[0]/1024/1024:.1f}MB  (HAVE_RSS={HAVE_RSS})")

if __name__ == "__main__":
    main()
