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
from sqlalchemy.pool import NullPool

from models import Base

BASE_DIR = Path(__file__).resolve().parent


def _resolve_data_dir() -> Path:
    """定位数据目录，优先级：
    1. 环境变量 DATA_DIR（Render 显式配置，最权威）
    2. /var/data —— Render 持久盘的标准挂载点，存在即用
    3. 代码目录下的 data/（本地开发兜底）

    第 2 条很重要：Render 上如果 DATA_DIR 没配好，写到容器临时盘会导致
    容器一重启 SQLite 文件就没了（数据丢失）。
    """
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env)
    for candidate in (Path("/var/data"),):
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            pass
    return BASE_DIR / "data"


# 与 storage 共用 DATA_DIR：挂载 Render Disk 后，SQLite 也落在持久盘上
DATA_DIR = _resolve_data_dir()
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
    # ★★ Neon 的 CU-hours 与出网流量，全都取决于「连接被怎么持有」★★
    #
    # 坑 1（CU-hours）：SQLAlchemy 的 pool_recycle 只在「取连接时」判定过期，
    #   池里那条启动时开的空闲连接永远不会被回收 —— 它被永久持有；
    #   Neon 只要看到还有连接在，就不会把 compute 缩到零，compute 24 小时在跑，
    #   100 CU-hours 的免费额度 4 天就烧光，之后整个月都是 "exceeded the quota"。
    #   → 改用 NullPool：连接用完即关，绝不留空闲连接。
    #   （Session 在一个事务内始终复用同一条连接，所以每请求只开 1 次连接，
    #     get_db 里那次探活 select(1) 不额外开连接。）
    #
    # 坑 2（出网流量）：出网字节按「Neon → 应用」计，取出来又丢掉的字段照样计费。
    #   → 见 main.py 里 /history、/tasks/{id} 的 load_only()。
    _kwargs: dict = dict(poolclass=NullPool)
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
_state = {"sqlite": DATABASE_URL.startswith("sqlite"), "fallback": False,
          "last_error": ""}


def db_backend() -> str:
    """当前数据库后端类型：'sqlite' 或 'postgres'。"""
    return "sqlite" if _state["sqlite"] else "postgres"


def is_fallback() -> bool:
    """是否处于「Postgres 不可用 → 降级到 SQLite」的应急状态。"""
    return _state["fallback"]


def db_last_error() -> str:
    """最近一次数据库故障的原文（空字符串 = 没有记录过故障）。

    为什么需要它：只看到「已降级」是没法处置的 —— 配额超限、出网额度用尽、
    连不上、认证失败，四种根因的处理方式完全不同，必须把原文暴露出来。
    """
    return _state["last_error"]


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
    _state["last_error"] = reason[:300]
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
            _state["last_error"] = str(exc)[:300]
            if not fallback_to_sqlite(str(exc)[:200]):
                raise
            db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
