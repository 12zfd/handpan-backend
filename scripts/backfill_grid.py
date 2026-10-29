"""补齐旧数据缺失的 grid 字段（早期版本没有这个键）。

判定优先级：
  1) 若该任务有任何 slot >= 12，说明是 16 格制
  2) 否则按来源：image → 16（截图通道固定 16 格），其他 → 12
"""
from __future__ import annotations

from database import SessionLocal
from models import Score


def guess_cells(j: dict) -> int:
    slots = [e.get("slot", 0) for b in (j.get("bars") or []) for e in (b.get("events") or [])]
    if any(s >= 12 for s in slots):
        return 16
    if (j.get("source") or {}).get("type") == "image":
        return 16
    return 12


def main() -> None:
    db = SessionLocal()
    try:
        fixed = 0
        for s in db.query(Score).all():
            j = dict(s.midi_data_json) if s.midi_data_json else None
            if not j:
                continue
            if (j.get("grid") or {}).get("cells"):
                continue
            cells = guess_cells(j)
            j["grid"] = {"cells": cells, "bars_per_system": 2}
            s.midi_data_json = j
            fixed += 1
            print(f"score {s.id}: 补 grid.cells={cells}")
        db.commit()
        print(f"完成：修正 {fixed} 条")
    finally:
        db.close()


if __name__ == "__main__":
    main()
