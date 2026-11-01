"""数据库引擎与会话。

本地开发：不配置 DATABASE_URL 时自动落到 SQLite（data/handpan.db），
Render 生产：配置 DATABASE_URL（Internal Database URL）即切到 PostgreSQL。
两种库对上层完全透明，业务代码不需要改。
"""
from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

from sqlalchemy import create_engine, select
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
    _kwargs: dict = dict(
        pool_pre_ping=True,
        # ★ 连接池水位直接决定 Neon 的 CU-hours 消耗速度。
        # Neon 免费版每项目每月只有 100 CU-hours，compute 必须「5 分钟无连接」才缩到零。
        # 池里闲置的长连接会一直吊着 compute 不缩零 → compute 整天在跑 → 月底必然
        # 撞上 "Your account or project has exceeded the quota"。
        # 单实例 + 低流量场景不需要大池子，pool_size=1 就够。
        pool_size=int(os.environ.get("DB_POOL_SIZE", "1")),
        max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "2")),
        # recycle 必须 ≤ Neon 的 5 分钟缩零窗口，否则连接一直吊着 compute
        pool_recycle=int(os.environ.get("DB_POOL_RECYCLE", "240")),
        pool_timeout=15,
    )
    # Render 免费容器没有 IPv6 出网，而 Neon 域名是双栈解析；
    # 显式解析一个 IPv4 交给 psycopg（host 仍用于 TLS 证书校验），避免 it 只试 IPv6 而失败。
    try:
        _host = make_url(DATABASE_URL).host
        _infos = socket.getaddrinfo(_host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if _infos:
            _ca = {"hostaddr": _infos[0][4][0]}
            # connect_timeout 必须有：Neon 不可达时 psycopg 默认可能挂很久，
            # 启动钩子会卡住，连 /healthz 都探不到。
            _ca["connect_timeout"] = int(os.environ.get("DB_CONNECT_TIMEOUT", "8"))
            _kwargs["connect_args"] = _ca
    except OSError:
        pass  # 解析失败就按默认行为连接
    engine = create_engine(DATABASE_URL, **_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

# 当前实际使用的后端类型（sqlite / postgres），供 /healthz 与启动日志报告
SQLITE_PATH = DATA_DIR / "handpan.db"
# 降级后为 True。做成可变而非常量，因为 Postgres 配额爆掉时可能中途切到 SQLite。
_state = {"sqlite": DATABASE_URL.startswith("sqlite"), "fallback": False}


def db_backend() -> str:
    """当前数据库后端类型：'sqlite' 或 'postgres'。"""
    return "sqlite" if _state["sqlite"] else "postgres"


def is_fallback() -> bool:
    """是否处于「Postgres 不可用 → 降级到 SQLite」的应急状态。"""
    return _state["fallback"]


def init_db() -> None:
    Base.metadata.create_all(engine)


def fallback_to_sqlite(reason: str = "") -> bool:
    """Postgres 不可用（配额超限 / 连接失败）时降级到本地 SQLite。

    SQLite 文件落在 DATA_DIR —— Render 上挂的是 1GB 持久盘，跨重启保留。
    单实例部署下 SQLite 完全够用；多副本部署会因为文件锁冲突失效，
    那种场景应该升级 Neon 计划而不是依赖降级。

    返回 True 表示降级成功。
    """
    if _state["sqlite"]:
        return True  # 本来就是 SQLite
    global engine, SessionLocal
    try:
        engine.dispose()
    except Exception:  # noqa: BLE001
        pass
    url = f"sqlite:///{SQLITE_PATH.as_posix()}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    _state["sqlite"] = True
    _state["fallback"] = True
    try:
        Base.metadata.create_all(engine)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("handpan.db").error("SQLite 降级后建表失败: %s", exc)
        return False
    logging.getLogger("handpan.db").warning(
        "已降级到本地 SQLite (%s)：%s", SQLITE_PATH, reason)
    return True


def get_db():
    """取一个会话。Postgres 连不上时自动降级到 SQLite 并重试本次请求。

    降级发生在「拿到会话」这一刻（路由还没执行 SQL），所以当前这个请求
    就能走 SQLite 完成，不需要等客户端重试。
    """
    db = SessionLocal()
    if not _state["sqlite"]:
        try:
            db.execute(select(1))
        except Exception as exc:  # noqa: BLE001
            db.close()
            if not fallback_to_sqlite(str(exc)[:200]):
                raise
            db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
