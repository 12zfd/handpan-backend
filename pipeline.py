"""转换流水线编排：状态机 + 四步（提取 → 映射 → 排版 → 出稿）。"""
from __future__ import annotations

import time
import traceback

from database import SessionLocal
from extract.audio import bpm_from_audio, video_to_note_events
from extract.midi_file import midi_to_note_events
from extract.omr import image_to_tab
from models import Score
from score.note_mapper import build_score, empty_score
from score.render_pdf import render_score_pdf
from storage import download_bytes, parse_keys, upload_bytes


def _set(db, score: Score, **kw) -> None:
    for k, v in kw.items():
        setattr(score, k, v)
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
            tab = image_to_tab(raw)
            note_events = tab.get("events") or []
            bpm = tab.get("bpm") or 120.0
            bars_hint = tab.get("bars")
            cells_hint = tab.get("cells")
            extra_warnings = tab.get("warnings") or []
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
            # 一个字符都没读出来：给一张空白谱格，让用户手动录入（兜底路径）
            score_json = empty_score(bars=12, bpm=120, source="image",
                                     title=score.title or "D Kurd 9音")
            score_json["warnings"] = extra_warnings + ["未识别到音符，请在预览页手动录入"]
        else:
            _set(db, score, status="failed",
                 error_message="未识别到有效音符，建议换更清晰的素材或改用手动录入")
            return
        metrics["map_ms"] = int((time.time() - t) * 1000)
        score_json["metrics"] = metrics
        _set(db, score, midi_data_json=score_json)

        # ---------- ③ 排版 + ④ 出稿
        _set(db, score, status="rendering")
        t = time.time()
        pdf = render_score_pdf(score_json)
        metrics["render_ms"] = int((time.time() - t) * 1000)
        score_json["metrics"] = metrics
        _set(db, score, midi_data_json=score_json, pdf_url=_store_pdf(score, pdf), status="done")

        metrics["total_ms"] = int((time.time() - t0) * 1000)
        score_json["metrics"] = metrics
        _set(db, score, midi_data_json=score_json)

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        db.rollback()
        s = db.get(Score, score_id)
        if s is not None:
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
            _set(db, s, status="failed", error_message=str(exc)[:500])
    finally:
        db.close()
