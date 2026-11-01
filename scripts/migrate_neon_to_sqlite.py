"""Neon(Postgres) → SQLite 数据迁移脚本。

用途：Neon 免费版每月 100 CU-hours，撞额度后整个后端 500。
本脚本把 users / scores 两张表的数据搬到本地 SQLite（Render 持久盘），
让服务在 Neon 不可用时仍能正常工作。

用法（在 handpan-backend 目录下执行）：
    # 1) 导出 Neon 数据到 JSON
    python scripts/export_from_neon.py --out ../_tmp/neon_dump.json
    # 2) 导入到 SQLite
    python scripts/import_to_sqlite.py --dump ../_tmp/neon_dump.json

也可以一步完成：
    python scripts/migrate_neon_to_sqlite.py

注：SQLite 走的是同一套 SQLAlchemy 模型（models.py），字段名完全一致，
    中间通过 JSON 中转，避免同时持有两个 engine 的连接管理问题。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DUMP_PATH = BASE_DIR.parent / "_tmp" / "neon_dump.json"


def _export(out_path: Path) -> int:
    """从 DATABASE_URL 指向的 Postgres 导出 users/scores。"""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from database import DATABASE_URL
    from models import Score, User

    if DATABASE_URL.startswith("sqlite"):
        print("当前 DATABASE_URL 是 SQLite，无需迁移。", file=sys.stderr)
        return 1

    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        users = db.execute(select(User)).scalars().all()
        scores = db.execute(select(Score)).scalars().all()
    finally:
        db.close()

    def iso(v):
        return v.isoformat() if isinstance(v, datetime) else v

    data = {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "password": u.password,
                "created_at": iso(u.created_at),
            }
            for u in users
        ],
        "scores": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "original_file_url": s.original_file_url,
                "midi_data_json": s.midi_data_json,
                "pdf_url": s.pdf_url,
                "status": s.status,
                "error_message": s.error_message,
                "title": s.title,
                "source_type": s.source_type,
                "created_at": iso(s.created_at),
                "updated_at": iso(s.updated_at),
            }
            for s in scores
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已导出 users={len(data['users'])} scores={len(data['scores'])} -> {out_path}")
    return 0


def _import(dump_path: Path, sqlite_path: Path | None = None) -> int:
    """把 dump 导入 SQLite（默认写 data/handpan.db）。"""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from models import Base, Score, User

    if not dump_path.exists():
        print(f"找不到 dump 文件: {dump_path}", file=sys.stderr)
        return 1
    data = json.loads(dump_path.read_text(encoding="utf-8"))

    if sqlite_path is None:
        from database import DATA_DIR
        sqlite_path = DATA_DIR / "handpan.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{sqlite_path.as_posix()}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = Session()
    try:
        existing_u = {u.username for u in db.execute(select(User)).scalars().all()}
        for u in data.get("users", []):
            if u["username"] in existing_u:
                continue
            db.add(User(
                id=u["id"], username=u["username"], password=u["password"],
                created_at=u.get("created_at") or datetime.utcnow(),
            ))
        existing_s = {s.id for s in db.execute(select(Score)).scalars().all()}
        for s in data.get("scores", []):
            if s["id"] in existing_s:
                continue
            db.add(Score(
                id=s["id"], user_id=s["user_id"],
                original_file_url=s["original_file_url"],
                midi_data_json=s.get("midi_data_json"),
                pdf_url=s.get("pdf_url"), status=s.get("status") or "queued",
                error_message=s.get("error_message"), title=s.get("title"),
                source_type=s.get("source_type"),
                created_at=s.get("created_at") or datetime.utcnow(),
                updated_at=s.get("updated_at") or datetime.utcnow(),
            ))
        db.commit()
        nu = db.execute(select(User)).scalars().all()
        ns = db.execute(select(Score)).scalars().all()
    finally:
        db.close()
    print(f"导入完成 -> {sqlite_path}  users={len(nu)} scores={len(ns)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DUMP_PATH), help="导出 JSON 路径")
    ap.add_argument("--dump", default=str(DUMP_PATH), help="导入用 JSON 路径")
    ap.add_argument("--sqlite", default=None, help="目标 SQLite 路径")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--import-only", action="store_true")
    args = ap.parse_args()

    if not args.import_only:
        rc = _export(Path(args.out))
        if rc != 0 and not args.export_only:
            return rc
    if not args.export_only:
        return _import(Path(args.dump), Path(args.sqlite) if args.sqlite else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
