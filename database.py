"""数据库引擎与会话。

本地开发：不配置 DATABASE_URL 时自动落到 SQLite（data/handpan.db），
Render 生产：配置 DATABASE_URL（Internal Database URL）即切到 PostgreSQL。
两种库对上层完全透明，业务代码不需要改。
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from models import Base

BASE_DIR = Path(__file__).resolve().parent
# 与 storage 共用 DATA_DIR：挂载 Render Disk 后，SQLite 也落在持久盘上
DATA_DIR = Path(os.environ.get("DATA_DIR") or (BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(url: str) -> str:
    """Render 给的是 postgresql://，SQLAlchemy 2 + psycopg3 需要显式 driver。"""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = _normalize(
    os.environ.get("DATABASE_URL") or f"sqlite:///{(DATA_DIR / 'handpan.db').as_posix()}"
)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    _kwargs: dict = dict(pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=1800)
    # Render 免费容器没有 IPv6 出网，而 Neon 域名是双栈解析；
    # 显式解析一个 IPv4 交给 psycopg（host 仍用于 TLS 证书校验），避免 it 只试 IPv6 而失败。
    try:
        _host = make_url(DATABASE_URL).host
        _infos = socket.getaddrinfo(_host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if _infos:
            _kwargs["connect_args"] = {"hostaddr": _infos[0][4][0]}
    except OSError:
        pass  # 解析失败就按默认行为连接
    engine = create_engine(DATABASE_URL, **_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
