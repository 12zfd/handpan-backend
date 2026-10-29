"""版式升级迁移：把旧任务的谱面数据按新网格重排，并重出 PDF。

背景：V1.1 早期每小节 16 格，改为标准样张的 12 格后，旧 JSON 里的 slot 会越界。
好在每个任务都存了 raw 原始时间戳，直接 requantize 就能无损迁移。

用法：.venv/Scripts/python.exe -m scripts.migrate_grid
"""
from __future__ import annotations

from database import SessionLocal
from models import Score
from score.note_mapper import DEFAULT_CELLS, requantize
from score.render_pdf import render_score_pdf
from storage import upload_bytes


def main() -> None:
    db = SessionLocal()
    try:
        rows = db.query(Score).all()
        changed = 0
        for s in rows:
            # 必须复制：SQLAlchemy 的 JSON 列对"原地修改"不会标记脏数据
            j = dict(s.midi_data_json) if s.midi_data_json else None
            if not j:
                continue
            cells = (j.get("grid") or {}).get("cells")
            if cells == DEFAULT_CELLS:
                continue

            if j.get("raw"):
                bpm = float((j.get("tempo") or {}).get("bpm") or 120)
                j = requantize(j, bpm)
            else:
                j.setdefault("grid", {"cells": DEFAULT_CELLS, "bars_per_system": 2})

            s.midi_data_json = j
            pdf = render_score_pdf(j)
            key = f"pdfs/{s.id}_migrated.pdf"
            upload_bytes(key, pdf, content_type="application/pdf")
            s.pdf_url = key
            s.status = "done"
            s.error_message = None
            changed += 1

        db.commit()
        print(f"迁移完成：{changed} / {len(rows)} 个任务")
    finally:
        db.close()


if __name__ == "__main__":
    main()
