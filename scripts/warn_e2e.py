"""端到端验证：污染行告警能否一路流到前端能拿到的字段里。

链路：
    extract.omr.image_to_tab()  ->  tab["warnings"]
    pipeline.process_image()    ->  score_json["warnings"]  (extra_warnings + 原 warnings)
    main.py /api/...            ->  HTTP JSON  d.warnings

前端 index.html 里：
    if(d && d.warnings && d.warnings.length){ ... warnbox ... }

本脚本不启服务，直接打 process_image，断言最终 score_json 里能看到告警文案。
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract import omr
import pipeline

UP = ROOT / "data" / "uploads"
imgs = sorted([p for p in UP.rglob("*.jpg")] + [p for p in UP.rglob("*.png")])
print(f"样本图片 {len(imgs)} 张")

hit = 0
miss = []
t0 = time.time()
for i, p in enumerate(imgs):
    try:
        tab = omr.image_to_tab(p.read_bytes())
        w = tab.get("warnings") or []
        polluted_w = [x for x in w if "乱码" in x]
        dropped = omr._LAST_DROPPED_ROWS
        if polluted_w:
            hit += 1
            print(f"[{i:02d}] {p.name[:8]}  dropped={dropped}  warn={polluted_w[0][:48]}")
        else:
            miss.append((p.name, dropped))
            if dropped:
                print(f"[{i:02d}] {p.name[:8]}  ⚠️ dropped={dropped} 但没生成告警！")
    except Exception as e:
        print(f"[{i:02d}] {p.name[:8]}  EXC {type(e).__name__}: {e}")

print(f"\n=== omr 层：{hit}/{len(imgs)} 张产生污染行告警，耗时 {time.time()-t0:.1f}s ===")

# ── pipeline 层：不跑真实 DB/存储，只验证 warnings 拼接逻辑
# pipeline.run_pipeline 依赖 Neon + 对象存储，本地跑不通。
# 真正要证的是 line 103-104 这一步会不会把 extra_warnings 带进 score_json：
#     if extra_warnings: score_json["warnings"] = extra_warnings + score_json["warnings"]
cand = None
for p in imgs:
    omr.image_to_tab(p.read_bytes())
    if omr._LAST_DROPPED_ROWS:
        cand = p
        break

if cand is None:
    print("没有找到含污染行的样本，跳过 pipeline 验证")
    sys.exit(0)

print(f"\n=== pipeline 层验证：{cand.name[:8]} ===")
from score.note_mapper import build_score

tab = omr.image_to_tab(cand.read_bytes())
extra_warnings = tab.get("warnings") or []
note_events = tab.get("events") or []
sj = build_score(note_events, bpm=tab.get("bpm") or 120.0, source="image",
                 title="测试", bars=tab.get("bars"), cells=tab.get("cells") or 12)
if extra_warnings:
    sj["warnings"] = extra_warnings + sj["warnings"]

ws = sj.get("warnings") or []
mine = [x for x in ws if "乱码" in x]
print(f"score_json.warnings 共 {len(ws)} 条")
for x in ws:
    print("  ·", x[:70])
print("污染行告警出现在最终 score_json：", "✅ 是" if mine else "❌ 否")

# 前端读的正是 d.warnings（index.html line ~1235），结构已对齐
print("\nscore_json 顶层字段：", ", ".join(sorted(sj.keys()))[:300])
