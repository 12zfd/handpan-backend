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
from sqlalchemy.pool import NullPool, QueuePool

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
    # ── PostgreSQL 连接策略（生产库 Aiven Free：1 CPU / 1 GB RAM / max_connections=20）──
    #
    # 历史背景：这套代码最早跑在 Neon 免费版，Neon 按「compute 能否缩到 0」计
    # CU-hours —— 池里留一条空闲连接就会让 compute 24h 常驻、4 天烧光额度，
    # 所以当时刻意改成 NullPool（连接用完即关）。
    #
    # 现在生产库是 **Aiven Free**（固定规格，不按连接存活计费），改为可配置：
    #   · 默认 QueuePool(pool_size=5, max_overflow=2)：复用连接，省掉每次 TLS 握手；
    #   · DB_POOL_MODE=nullpool：一键回退「用完即关」。Aiven Free 的
    #     max_connections 只有 20 且 Aiven 自身运维还要占用若干条，一旦改成多
    #     worker / 多副本，池连接会成倍放大；遇到连接数或内存告警时，
    #     **改一个环境变量即可恢复，不用改代码、不用重新发版**。
    #     见 aiven.io/docs/products/postgresql/reference/pg-connection-limits
    #
    # 池化三件套（防「僵死连接」与「请求无限等连接」）：
    #   pool_pre_ping=True —— 取连接前先探活，断线自动重连（Aiven/中间件会静默断空闲连接）
    #   pool_recycle=280   —— 连接活过 280s 主动换掉，早于常见 300s 空闲断线
    #   pool_timeout=10    —— 池空时最多等 10s，超时报错而不是把请求挂死
    _mode = os.environ.get("DB_POOL_MODE", "queue").strip().lower()
    if _mode == "nullpool":
        _kwargs: dict = dict(poolclass=NullPool)
    else:
        _kwargs = dict(
            poolclass=QueuePool,
            pool_size=int(os.environ.get("DB_POOL_SIZE", "5")),
            max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "2")),
            pool_pre_ping=True,
            pool_recycle=int(os.environ.get("DB_POOL_RECYCLE", "280")),
            pool_timeout=int(os.environ.get("DB_POOL_TIMEOUT", "10")),
        )

    # connect_timeout：库不可达时不能让启动钩子 / 请求无限挂住。
    # application_name：在 Aiven 的 pg_stat_activity 里能一眼认出应用连接，
    #   排查「连接泄漏 / idle in transaction」时是关键线索。
    _ca = {
        "connect_timeout": int(os.environ.get("DB_CONNECT_TIMEOUT", "10")),
        "application_name": "handpan-backend",
    }
    # Render 免费容器没有 IPv6 出网，而 Aiven 域名是双栈解析；
    # 显式解析一个 IPv4 交给 psycopg（host 仍用于 TLS 证书校验），避免只试 IPv6 而失败。
    try:
        _host = make_url(DATABASE_URL).host
        _infos = socket.getaddrinfo(_host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if _infos:
            _ca["hostaddr"] = _infos[0][4][0]
    except OSError:
        pass  # 解析失败就按默认行为连接
    _kwargs["connect_args"] = _ca
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


def pool_stats() -> dict:
    """连接池当前状态，供 /healthz 观察「连接是否泄漏」。

    checkedout 长期接近 pool_size+max_overflow，就说明有请求没归还连接。
    NullPool 没有这些方法，返回 mode 即可。
    """
    p = engine.pool
    out = {"mode": type(p).__name__}
    fn = getattr(p, "checkedout", None)
    if callable(fn):
        try:
            out["checkedout"] = fn()          # 正在被请求占用的连接数
        except Exception:  # noqa: BLE001
            pass
    ci = getattr(p, "checkedin", None)          # QueuePool 上是 property，不是方法
    if isinstance(ci, (int, float)):
        out["checkedin"] = ci                  # 池中空闲可复用的连接数
    fn = getattr(p, "size", None)
    if callable(fn):
        try:
            out["pool_size"] = fn()            # 配置的池大小（非当前实际连接数）
        except Exception:  # noqa: BLE001
            pass
    return out


def get_db():
    """取一个会话。Postgres 连不上时自动降级到 SQLite 并重试本次请求。

    降级发生在「拿到会话」这一刻（路由还没执行 SQL），所以当前这个请求
    就能走 SQLite 完成，不需要等客户端重试。
    """
    db = SessionLocal()
    if not _state["sqlite"]:
        try:
            db.execute(select(1))
            # ★ 探活这条 SELECT 会自动 BEGIN 一个事务。若不结束，整个请求期间
            #   连接都停在「idle in transaction」（Aiven 上要等
            #   idle_in_transaction_session_timeout=300s 才被服务端清理）。
            #   此刻还没加载任何 ORM 对象，rollback 无副作用，后续查询会自动重新 BEGIN。
            db.rollback()
        except Exception as exc:  # noqa: BLE001
            _state["last_error"] = str(exc)[:300]
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass
            if not fallback_to_sqlite(str(exc)[:200]):
                raise
            db = SessionLocal()
    try:
        yield db
    except Exception:
        # 路由抛异常时事务可能停在半途 → 显式回滚，保证连接归还池中是干净状态。
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        # 每个请求必关。close 失败只记日志，不能掩盖路由原本的异常。
        try:
            db.close()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("handpan.db").warning("Session.close 失败: %s", exc)
