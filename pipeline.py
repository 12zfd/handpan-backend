"""转换流水线编排：状态机 + 四步（提取 → 映射 → 排版 → 出稿）。"""
from __future__ import annotations

import logging
import threading
import time
import traceback

from database import SessionLocal
from sqlalchemy.orm.attributes import flag_modified
from extract.audio import bpm_from_audio, video_to_note_events
from extract.midi_file import midi_to_note_events
from extract.omr import image_to_tab, _Timing
from models import Score
from score.note_mapper import build_score, empty_score
from score.render_pdf import render_score_pdf
from storage import download_bytes, parse_keys, upload_bytes

log = logging.getLogger("handpan.pipeline")


# OCR 全局串行闸。
# 本地 OCR（onnxruntime）的 arena 会随连续识别累积、不释放回操作系统；
# 两个任务同时进 OCR 会把峰值内存叠加，Render Free 的 512MB 直接被打爆，
# 实例一重启，后台任务就永远停在「extracting」（用户看到「一直读取中」）。
# 这里把重活串行化：内存水位恒定，代价只是第二个任务晚几秒开始。
_OCR_LOCK = threading.Semaphore(1)



def _with_metrics(score_json: dict, metrics: dict) -> dict:
    """把 metrics 挂到 score_json 上，并返回**全新的外层 dict**。

    为什么必须换外层（这是真实 Neon/Postgres 上实测出来的，不是推理）：
    SQLAlchemy 判定 JSON 列「变没变」用的是**值相等比较**。生产 session 是
    `expire_on_commit=False`，commit 后属性不失效，committed_state 里存的就是
    **同一个对象引用**。于是：
      · 原地改 metrics → 旧值快照被同步污染 → 两者永远相等 → 永不判脏
      · 只换内层 dict 不够：外层还是同一引用 → 外层值依然相等 → 依然不判脏
      · 实测（Neon Postgres，临时表 t_json_probe）：这两种写法都会让
        render_ms / total_ms **静默丢失**
    换一个新的外层 dict 后，外层引用变了 → 值不再相等 → 正常判脏落库。

    `flag_modified` 是第二道保险（见 _set），两者都上，任何写法都安全。
    """
    return {**score_json, "metrics": dict(metrics)}


def _set(db, score: Score, **kw) -> None:
    """写字段并提交。

    `midi_data_json` 是 JSON 列，SQLAlchemy 判定它「变没变」用的是**值相等
    比较**。生产 session 是 `expire_on_commit=False`，commit 后属性不失效，
    committed_state 里存的就是同一个对象引用 → 原地改它会同步污染旧值快照
    → 两者永远相等 → 永不判脏 → 字段**静默丢失**（实测 render_ms / total_ms
    就是这么消失的）。

    两道保险一起上，任何写法都安全：
      1. `flag_modified` —— 显式标脏，绕过值相等比较（实测有效）
      2. 调用方用 `_with_metrics()` 返回**全新外层 dict** —— 外层引用变了，
         值不再相等，正常判脏（实测有效）

    注意：**只换内层 dict 不够**（外层还是同一引用 → 外层值依然相等 → 依然丢）。
    `MutableDict` 也不行 —— 它只追踪顶层 key 变化，嵌套 dict 改动照样丢。
    """
    for k, v in kw.items():
        setattr(score, k, v)
    if "midi_data_json" in kw:
        flag_modified(score, "midi_data_json")
    db.commit()


def _store_pdf(score: Score, pdf: bytes) -> str:
    key = f"pdfs/{score.id}_{int(time.time())}.pdf"
    upload_bytes(key, pdf, content_type="application/pdf")
    return key


def run_pipeline(score_id: int) -> None:
    db = SessionLocal()
    t0 = time.time()
    metrics: dict = {}
    try:
        score = db.get(Score, score_id)
        if score is None:
            return

        keys = parse_keys(score.original_file_url)
        if not keys:
            _set(db, score, status="failed", error_message="原文件丢失，请重新上传")
            return
        # 多图任务按上传顺序整批取回；其他类型只有 1 个 key
        raw = ([download_bytes(k) for k in keys]
               if score.source_type == "image" and len(keys) > 1
               else download_bytes(keys[0]))

        # ---------- ① 提取
        _set(db, score, status="extracting")
        t = time.time()
        note_events: list[dict] = []
        bpm = 120.0
        bars_hint: int | None = None
        cells_hint: int | None = None
        extra_warnings: list[str] = []

        if score.source_type == "video":
            note_events = video_to_note_events(raw)
            bpm = bpm_from_audio(raw) or 120.0
        elif score.source_type == "midi":
            note_events, bpm = midi_to_note_events(raw)
        elif score.source_type == "image":
            t_img = time.time()
            with _OCR_LOCK:
                try:
                    # 分段计时：图片解码 / OCR 推理 / 结果格式化
                    tab = image_to_tab(raw, timing=_Timing())
                except Exception as exc:  # noqa: BLE001
                    # 识别引擎崩了也**绝不抛**：落空结果，让下面的空白兜底接管。
                    # （需求 4：绝不允许因为异常导致用户看到纯空白或任务失败）
                    traceback.print_exc()
                    tab = {"events": [], "bars": None, "bpm": 120.0,
                           "warnings": [f"识别引擎异常（{str(exc)[:120]}），已生成空白谱格，请在预览页手动录入"]}
            note_events = tab.get("events") or []
            bpm = tab.get("bpm") or 120.0
            bars_hint = tab.get("bars")
            cells_hint = tab.get("cells")
            extra_warnings = tab.get("warnings") or []
            tm = tab.get("timing") or {}
            metrics["decode_ms"] = int(tm.get("decode_ms", 0))
            metrics["infer_ms"] = int(tm.get("infer_ms", 0))
            metrics["format_ms"] = int(tm.get("format_ms", 0))
            metrics["ocr_ms"] = int((time.time() - t_img) * 1000)
            log.info(
                "image extract: decode=%sms infer=%sms format=%sms total=%sms",
                metrics["decode_ms"], metrics["infer_ms"],
                metrics["format_ms"], metrics["ocr_ms"],
            )
        metrics["extract_ms"] = int((time.time() - t) * 1000)

        # ---------- ② 映射
        _set(db, score, status="mapping")
        t = time.time()
        if note_events:
            score_json = build_score(note_events, bpm=bpm, source=score.source_type,
                                     title=score.title or "D Kurd 9音", bars=bars_hint,
                                     cells=cells_hint or 12)
            if extra_warnings:
                score_json["warnings"] = extra_warnings + score_json["warnings"]
        elif score.source_type == "image":
            # 一个字符都没读出来（含 OCR 引擎异常 / OpenCV 兜底也没救回）：
            # 强制返回 4/4 拍 8 小节空白手碟谱 + status="fallback" 提示，
            # 用户在预览页手动录入。绝不允许空白/异常透出到前端。
            score_json = empty_score(bars=8, bpm=120, source="image",
                                     title=score.title or "D Kurd 9音")
            score_json["status"] = "fallback"
            score_json["message"] = "未能精准识别，已为你生成空白网格，请手动输入"
            score_json["warnings"] = extra_warnings + [
                "未能精准识别，已为你生成 4/4 拍 8 小节空白网格，请手动输入",
            ]
        else:
            _set(db, score, status="failed",
                 error_message="未识别到有效音符，建议换更清晰的素材或改用手动录入")
            return
        metrics["map_ms"] = int((time.time() - t) * 1000)
        score_json = _with_metrics(score_json, metrics)
        _set(db, score, midi_data_json=score_json)

        # ---------- ③ 排版 + ④ 出稿
        _set(db, score, status="rendering")
        t = time.time()
        pdf = render_score_pdf(score_json)
        metrics["render_ms"] = int((time.time() - t) * 1000)
        score_json = _with_metrics(score_json, metrics)
        _set(db, score, midi_data_json=score_json, pdf_url=_store_pdf(score, pdf), status="done")

        metrics["total_ms"] = int((time.time() - t0) * 1000)
        score_json = _with_metrics(score_json, metrics)
        _set(db, score, midi_data_json=score_json)

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        db.rollback()
        s = db.get(Score, score_id)
        if s is not None:
            # OCR provider 的失败（超时 / 额度耗尽 / 凭据失效…）统一落到
            # status="failed" + error_message，字段结构与既有失败返回完全一致，
            # 前端无需任何改动即可显示。
            _set(db, s, status="failed", error_message=str(exc)[:500])
    finally:
        db.close()


def render_only(score_id: int) -> None:
    """只重新出稿：校对保存、改 BPM 重排之后调用。不重新提取。"""
    db = SessionLocal()
    try:
        s = db.get(Score, score_id)
        if s is None or not s.midi_data_json:
            return
        pdf = render_score_pdf(s.midi_data_json)
        _set(db, s, pdf_url=_store_pdf(s, pdf), status="done", error_message=None)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        db.rollback()
        s = db.get(Score, score_id)
        if s is not None:
            # OCR provider 的失败（超时 / 额度耗尽 / 凭据失效…）统一落到
            # status="failed" + error_message，字段结构与既有失败返回完全一致，
            # 前端无需任何改动即可显示。
            _set(db, s, status="failed", error_message=str(exc)[:500])
    finally:
        db.close()
