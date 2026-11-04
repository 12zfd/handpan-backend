"""FastAPI 应用：认证、上传、转换、历史、校对、改 BPM、下载。

接口清单（对应 PRD 七 / 技术方案 4.1）：
  POST   /login
  POST   /upload
  POST   /convert/{score_id}
  GET    /history
  GET    /scores/{score_id}
  PATCH  /scores/{score_id}/notes
  PATCH  /scores/{score_id}/tempo
  GET    /download/{score_id}
  GET    /healthz
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

# 性能日志（图片解码 / 推理 / 格式化 分段耗时）必须能在 Render 日志里看到，
# uvicorn 只配置了自己的 logger，root logger 默认 WARNING 会把 log.info 吞掉。
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.orm import Session, load_only
from sqlalchemy.orm.attributes import flag_modified

from database import dispose_engine, get_db, init_db, pool_stats
from database import SQLITE_PATH, db_backend, db_last_error, fallback_to_sqlite, is_fallback
from models import Score, User
from pipeline import render_only, run_pipeline
from score.note_mapper import requantize
from security import decode_token, hash_password, make_token, verify_password
from storage import dump_keys, download_bytes, local_path, parse_keys, presign_get, upload_bytes

BASE_DIR = Path(__file__).resolve().parent

ALLOWED_EXT = {
    ".mp4": "video", ".mov": "video", ".webm": "video",
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".mid": "midi", ".midi": "midi",
}
MAX_BYTES = 200 * 1024 * 1024
MAX_FILES = 5          # 图片一次最多 5 张（多页谱）

bearer = HTTPBearer(auto_error=False)

app = FastAPI(title="Handpan Sheet Forge", version="1.1.1")

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    # 前端用 fetch + blob 下载，需要能读到文件名
    expose_headers=["Content-Disposition"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局兜底：未捕获异常也必须带上 CORS 头。

    Starlette 中间件顺序为 ServerErrorMiddleware(最外) -> CORSMiddleware ->
    ExceptionMiddleware -> router。未捕获异常由 ServerErrorMiddleware 生成 500，
    它在 CORSMiddleware **外层**，所以 500 响应里没有 Access-Control-Allow-Origin，
    浏览器会把跨域 500 当成网络故障 -> fetch() reject "Failed to fetch"
    -> 前端误报「连不上后端服务」。

    这里显式补上 CORS 头，并返回可读的错误码，让前端能区分
    「数据库挂了」和「真的网络不通」。
    """
    logger = logging.getLogger("handpan.unhandled")
    logger.exception("未处理异常: %s %s", request.method, request.url.path)

    detail = "服务器内部错误"
    code = "internal_error"
    # 数据库故障判定：异常类型优先，字符串兜底。
    # 只靠 str(exc) 匹配会漏 —— DBAPIError 常被 FastAPI 包一层，
    # 原文里未必出现 "operationalerror" 字样。
    text = f"{type(exc).__name__} {exc}".lower()
    is_db_exc = isinstance(exc, (OperationalError, InterfaceError, DBAPIError))
    connection_full = any(k in text for k in (
        "too many connections", "max_connections", "connection pool exhausted",
        "exceeded the quota", "remaining connection slots"))
    if connection_full:
        detail = "数据库连接数已达上限（免费版上限 20 条），已临时切换到本地数据库，请稍后重试"
        code = "db_quota_exceeded"
    elif is_db_exc or any(k in text for k in (
            "operationalerror", "interfaceerror", "connection refused",
            "connection timed out", "server closed the connection",
            "could not connect", "ssl syscall error", "terminating connection")):
        detail = "数据库暂时不可用，已临时切换到本地数据库（历史记录暂不可见），请稍后重试"
        code = "db_unavailable"

    # 数据库类故障 → 立刻降级到 SQLite，让后续请求能正常走通
    if code in ("db_quota_exceeded", "db_unavailable"):
        try:
            fallback_to_sqlite(str(exc)[:200])
        except Exception:  # noqa: BLE001
            pass

    resp = JSONResponse(status_code=500, content={"detail": detail, "code": code})
    origin = request.headers.get("origin")
    if origin and (origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.on_event("startup")
def _startup() -> None:
    # 数据库连不上（配额超限 / 临时不可用）时不能让整个容器起不来，
    # 否则连 /healthz 都探不到，完全没法自愈和诊断。
    log = logging.getLogger("handpan.startup")
    log.info("数据库后端: %s", db_backend())
    try:
        init_db()
        log.info("数据库就绪: %s", db_backend())
    except Exception as exc:  # noqa: BLE001
        log.error("Postgres 初始化失败，尝试降级到 SQLite: %s", exc)
        if fallback_to_sqlite(str(exc)[:200]):
            log.warning("已降级到 SQLite: %s", SQLITE_PATH)
        else:
            log.error("降级失败，服务将在每次请求时重试")


@app.on_event("shutdown")
def _shutdown() -> None:
    # 应用关闭前彻底释放连接池：把池里所有连接关掉，不留给进程退出慢慢回收。
    # Render 每次发版 / 重启都会走到这里；池里残留的连接会白占 Aiven 的
    # max_connections（免费版只有 20 条），也可能把 idle in transaction 状态
    # 带到下一次启动。
    log = logging.getLogger("handpan.shutdown")
    log.info("关闭数据库连接池: %s", pool_stats())
    dispose_engine()
    log.info("数据库连接池已释放")


@app.middleware("http")
async def _cache_headers(request, call_next):
    """HTML 永不缓存（改了前端要能立刻生效）；带 hash 的 assets 长期缓存。"""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "") or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    elif path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


# ------------------------------------------------------------------ Schema
class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class ScoreOut(BaseModel):
    id: int
    title: str | None
    source_type: str | None
    status: str
    error_message: str | None
    has_pdf: bool
    created_at: datetime
    updated_at: datetime


class NotesIn(BaseModel):
    midi_data_json: dict


class TempoIn(BaseModel):
    bpm: float


class TitleIn(BaseModel):
    title: str


class TaskOut(BaseModel):
    """异步任务视图。前端轮询它拿进度；字段与既有 ScoreOut 对齐。"""
    task_id: int
    score_id: int
    status: str
    stage: str
    progress: int
    error_message: str | None
    has_pdf: bool
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------ 认证
def current_user(
    cred: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if cred is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未登录")
    payload = decode_token(cred.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已失效，请重新登录")
    user = db.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号不存在")
    return user


# 与 current_user 等价，但额外允许通过 URL ?token= 传入 JWT。
# 用途：移动端「跳转到原生浏览器下载」是顶层导航，无法携带 Authorization 请求头，
# 只能把 token 放在 URL 里；桌面端 / fetch 仍走 Authorization 头，两种都支持。
def current_user_from_header_or_query(
    cred: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
    token: str | None = Query(default=None, alias="token"),
) -> User:
    raw = cred.credentials if cred is not None else (token or None)
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未登录")
    payload = decode_token(raw)
    if not payload or "sub" not in payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已失效，请重新登录")
    user = db.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号不存在")
    return user


def _fallback_creds() -> dict[str, str]:
    """降级到 SQLite 后的应急账号表。

    格式：FALLBACK_USERS="用户名:密码,用户名2:密码2"
    在 Render 环境变量里配置；未配置则不启用应急登录。
    """
    raw = os.environ.get("FALLBACK_USERS", "")
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if ":" in item:
            u, p = item.split(":", 1)
            if u.strip() and p.strip():
                out[u.strip()] = p.strip()
    return out


def _ensure_fallback_user(db: Session, username: str, password: str) -> User | None:
    """在降级用的 SQLite 里建一个账号（仅在配置了 FALLBACK_USERS 时调用）。"""
    try:
        user = User(username=username, password=hash_password(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("handpan.login").error("创建降级账号失败: %s", exc)
        db.rollback()
        return None


@app.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Annotated[Session, Depends(get_db)]):
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None and is_fallback():
        # 降级到 SQLite 后 users 表是空的。用环境变量里的应急账号兜底，
        # 保证「Neon 挂了」不会变成「谁都登不进来」。
        if _fallback_creds().get(body.username) == body.password:
            user = _ensure_fallback_user(db, body.username, body.password)
    if user is None or not verify_password(body.password, user.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号或密码错误")
    return TokenOut(access_token=make_token(user.id, user.username), username=user.username)


# ------------------------------------------------------------------ 上传
@app.post("/upload")
def upload(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
):
    """图片：一次最多 5 张，按顺序合并成一份谱；视频/MIDI：一次 1 个。"""
    incoming = [f for f in (files or []) if f is not None]
    if file is not None:
        incoming.append(file)

    if not incoming:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "没有收到文件")
    if len(incoming) > MAX_FILES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"最多一次上传 {MAX_FILES} 张图片")

    kinds = []
    for f in incoming:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                "仅支持 mp4 / mov / jpg / jpeg / png / mid")
        kinds.append(ALLOWED_EXT[ext])

    if len(incoming) > 1 and any(k != "image" for k in kinds):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "只有图片可以多张上传；视频 / MIDI 一次一个")

    keys: list[str] = []
    for f in incoming:
        data = f.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                f"{f.filename} 超过 200MB")
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{f.filename} 是空文件")
        ext = Path(f.filename or "").suffix.lower()
        key = f"uploads/{datetime.now():%Y/%m}/{uuid.uuid4().hex}{ext}"
        upload_bytes(key, data, content_type=f.content_type or "application/octet-stream")
        keys.append(key)

    stem = Path(incoming[0].filename or "未命名").stem[:100]
    title = stem if len(keys) == 1 else f"{stem}（等 {len(keys)} 页）"

    score = Score(
        user_id=user.id,
        original_file_url=dump_keys(keys),
        status="queued",
        source_type=kinds[0],
        title=title[:120],
    )
    db.add(score)
    db.commit()
    db.refresh(score)
    # 兼容旧前端：score_id/status/files 三个字段一字未改。
    # task_id 是新增的异步句柄，等价于 score_id（同一个任务）。
    return {"score_id": score.id, "status": score.status, "files": len(keys),
            "task_id": score.id}


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
def create_task(
    background: BackgroundTasks,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
):
    """一步式异步任务：上传即返回 task_id，转换在后台跑，前端轮询 /tasks/{id}。

    与 /upload + /convert 完全等价，只是把两步合成一步、立即返回。
    响应体：{"task_id", "score_id", "status", "files"}
    """
    incoming = [f for f in (files or []) if f is not None]
    if file is not None:
        incoming.append(file)
    if not incoming:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "没有收到文件")
    if len(incoming) > MAX_FILES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"最多一次上传 {MAX_FILES} 张图片")

    kinds = []
    for f in incoming:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                "仅支持 mp4 / mov / jpg / jpeg / png / mid")
        kinds.append(ALLOWED_EXT[ext])
    if len(incoming) > 1 and any(k != "image" for k in kinds):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "只有图片可以多张合并；视频 / MIDI 一次一个")

    keys: list[str] = []
    for f in incoming:
        data = f.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                f"{f.filename} 超过 200MB")
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{f.filename} 是空文件")
        ext = Path(f.filename or "").suffix.lower()
        key = f"uploads/{datetime.now():%Y/%m}/{uuid.uuid4().hex}{ext}"
        upload_bytes(key, data, content_type=f.content_type or "application/octet-stream")
        keys.append(key)

    stem = Path(incoming[0].filename or "未命名").stem[:100]
    title = stem if len(keys) == 1 else f"{stem}（等 {len(keys)} 页）"
    score = Score(
        user_id=user.id,
        original_file_url=dump_keys(keys),
        status="extracting",
        source_type=kinds[0],
        title=title[:120],
    )
    db.add(score)
    db.commit()
    db.refresh(score)

    background.add_task(run_pipeline, score.id)
    return {"task_id": score.id, "score_id": score.id,
            "status": score.status, "files": len(keys)}


# ------------------------------------------------------------------ 异步任务视图
_STAGE_TEXT = {
    "queued": "排队中",
    "extracting": "识别谱面字符",
    "mapping": "映射到手碟音位",
    "rendering": "排版出稿",
    "done": "已完成",
    "failed": "失败",
}
_STAGE_PROGRESS = {
    "queued": 0, "extracting": 30, "mapping": 60, "rendering": 85,
    "done": 100, "failed": 100,
}


@app.get("/tasks/{task_id}", response_model=TaskOut)
def task_status(
    task_id: int,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """轮询任务进度。task_id 与 score_id 是同一个 id（不新增数据表）。

    只 SELECT 用得到的列：midi_data_json 是大字段，整行取出来再丢掉，
    出网流量照样按「Neon → 应用」的字节计 —— 前端 2.5s 轮询一次，
    几分钟就能把一个月的免费出网额度烧穿。
    """
    s = db.scalars(
        select(Score).options(load_only(
            Score.id, Score.status, Score.error_message,
            Score.pdf_url, Score.created_at, Score.updated_at,
        )).where(Score.id == task_id)
    ).first()
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    return TaskOut(
        task_id=s.id,
        score_id=s.id,
        status=s.status,
        stage=_STAGE_TEXT.get(s.status, s.status),
        progress=_STAGE_PROGRESS.get(s.status, 0),
        error_message=s.error_message,
        has_pdf=bool(s.pdf_url),
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


# ------------------------------------------------------------------ 触发转换
@app.post("/convert/{score_id}")
def convert(
    score_id: int,
    background: BackgroundTasks,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    score = db.get(Score, score_id)
    if score is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    if score.status not in ("queued", "failed"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"当前状态 {score.status} 不可重复转换")

    score.status = "extracting"
    score.error_message = None
    db.commit()

    background.add_task(run_pipeline, score_id)
    return {"score_id": score_id, "status": "extracting"}


# ------------------------------------------------------------------ JSON 写入
def _write_json(db: Session, s: Score, payload: dict) -> None:
    """写 `midi_data_json`（JSON 列）并显式标脏。

    SQLAlchemy 判定 JSON 列「变没变」用的是**值相等比较**，而不是对象身份。
    只要我们持有的还是同一个 dict 引用并原地改它，旧值快照会被同步污染，
    两者永远相等 → 永不判脏 → 字段**静默丢失**（实测 render_ms / total_ms
    就是这么消失的）。换成新 dict 也救不了：值仍然相等。

    正解是 `flag_modified`（实测有效）。
    `MutableDict` 也不行 —— 它只追踪顶层 key 变化，嵌套 dict 改动照样丢。
    """
    s.midi_data_json = payload
    flag_modified(s, "midi_data_json")


# ------------------------------------------------------------------ 历史 / 详情
def _to_out(s: Score) -> ScoreOut:
    return ScoreOut(
        id=s.id, title=s.title, source_type=s.source_type, status=s.status,
        error_message=s.error_message, has_pdf=bool(s.pdf_url),
        created_at=s.created_at, updated_at=s.updated_at,
    )


@app.get("/history", response_model=list[ScoreOut])
def history(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(50, ge=1, le=200),
):
    # load_only：只要这 8 个轻字段。midi_data_json 不取 ——
    # 出网流量按字节计，「取出来再丢掉」一样烧额度（Neon 免费版每月出网额度有限）。
    rows = db.scalars(
        select(Score).options(load_only(
            Score.id, Score.title, Score.source_type, Score.status,
            Score.error_message, Score.pdf_url, Score.created_at, Score.updated_at,
        )).order_by(Score.created_at.desc()).limit(limit)
    ).all()
    return [_to_out(s) for s in rows]


@app.get("/scores/{score_id}")
def score_detail(
    score_id: int,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    s = db.get(Score, score_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    return {**_to_out(s).model_dump(), "midi_data_json": s.midi_data_json,
            "files": len(parse_keys(s.original_file_url))}


# ------------------------------------------------------------------ 校对保存
@app.patch("/scores/{score_id}/notes")
def save_notes(
    score_id: int,
    body: NotesIn,
    background: BackgroundTasks,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    s = db.get(Score, score_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")

    payload = dict(body.midi_data_json)
    # 重算统计，避免前端改完统计还是旧的
    notes = [n for b in payload.get("bars", []) for e in b.get("events", []) for n in e.get("notes", [])]
    for n in notes:
        n["color"] = "blue" if int(n["pos"]) <= 2 else "orange"
    payload["stats"] = {
        **(payload.get("stats") or {}),
        "total_notes": len(notes),
        "bars": len(payload.get("bars", [])),
    }
    if len(notes) == 0:
        payload["warnings"] = ["谱面为空，请录入音符"]
    else:
        payload["warnings"] = [w for w in (payload.get("warnings") or []) if "手动录入" not in w and "谱面为空" not in w]

    _write_json(db, s, payload)
    s.status = "rendering"
    db.commit()

    background.add_task(render_only, score_id)
    return {"score_id": score_id, "status": "rendering", "total_notes": len(notes)}


# ------------------------------------------------------------------ 改歌曲名
@app.patch("/scores/{score_id}")
def rename_score(
    score_id: int,
    body: TitleIn,
    background: BackgroundTasks,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """改歌曲名：列表显示与 PDF 大标题都改，并重新出稿。"""
    name = (body.title or "").strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "歌曲名不能为空")
    name = name[:120]

    s = db.get(Score, score_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")

    s.title = name
    if s.midi_data_json:
        j = dict(s.midi_data_json)
        j["title"] = name
        _write_json(db, s, j)
        s.status = "rendering"
        db.commit()
        background.add_task(render_only, score_id)
    else:
        db.commit()

    return {"score_id": score_id, "title": name, "status": s.status}


# ------------------------------------------------------------------ 改 BPM 整首重排
@app.patch("/scores/{score_id}/tempo")
def set_tempo(
    score_id: int,
    body: TempoIn,
    background: BackgroundTasks,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    if not 40 <= body.bpm <= 240:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "BPM 应在 40–240 之间")

    s = db.get(Score, score_id)
    if s is None or not s.midi_data_json:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在或尚无谱面数据")

    current = (s.midi_data_json.get("tempo") or {}).get("bpm")
    if current and abs(float(current) - body.bpm) < 0.01:
        return {"score_id": score_id, "status": s.status,
                "bars": len(s.midi_data_json.get("bars") or [])}

    try:
        rebuilt = requantize(s.midi_data_json, body.bpm)
    except ValueError as exc:
        # 没有 raw 时间戳（例如纯手动录入的谱），只改页眉速度，不重排
        _write_json(db, s, {**s.midi_data_json,
                             "tempo": {**(s.midi_data_json.get("tempo") or {}), "bpm": body.bpm}})
        s.status = "rendering"
        db.commit()
        background.add_task(render_only, score_id)
        return {"score_id": score_id, "status": "rendering",
                "bars": len(s.midi_data_json.get("bars") or []), "note": str(exc)}

    _write_json(db, s, rebuilt)
    s.status = "rendering"
    db.commit()

    background.add_task(render_only, score_id)
    return {"score_id": score_id, "status": "rendering", "bars": len(rebuilt["bars"])}


# ------------------------------------------------------------------ 下载
@app.get("/download/{score_id}")
def download(
    score_id: int,
    user: Annotated[User, Depends(current_user_from_header_or_query)],
    db: Annotated[Session, Depends(get_db)],
    name: str | None = Query(default=None, alias="name", max_length=120),
):
    """下载 PDF。

    `?name=` 可选：手机端（iOS「存储到文件」/ Android「下载」）会拿这个
    当保存后的文件名。前端本地出稿时用的是「歌名_D Kurd 9音.pdf」，
    这里也接受同名参数，保证「网页看到的名字」和「存到手机的名字」一致。
    """
    s = db.get(Score, score_id)
    if s is None or not s.pdf_url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF 尚未生成")

    name = _safe_pdf_name(name) or _pdf_name(s)

    # 同域代理：服务端拉取对象存储（S3/R2）字节后直接回传，浏览器只与后端通信。
    # 避免浏览器跨域直连 Supabase（国内链路不稳、跨域重定向在部分网络/浏览器下会被拦），
    # 也免去「后端 302 → 第三方存储 → 浏览器」的多跳。代理失败再回退老行为。
    try:
        data = download_bytes(s.pdf_url)
        if data:
            from urllib.parse import quote
            return Response(
                data,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}",
                    "Cache-Control": "public, max-age=3600",
                },
            )
    except Exception as exc:  # 代理偶发失败（网络抖动）→ 退回 302 预签名，最后退回本地盘
        logging.getLogger("download").warning("PDF 同域代理失败，回退 302：%s", exc)

    url = presign_get(s.pdf_url, expires=86400, download_name=name)
    if url:
        return RedirectResponse(url, status_code=status.HTTP_302_FOUND)

    path = local_path(s.pdf_url)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF 文件已丢失，请重新出稿")
    from urllib.parse import quote
    return FileResponse(
        path, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


def _pdf_name(s: Score) -> str:
    base = (s.title or f"score-{s.id}").replace("/", "_").replace("\\", "_")
    return f"{base}_D Kurd 9音_{s.created_at:%Y%m%d}.pdf"


def _safe_pdf_name(raw: str | None) -> str | None:
    """清洗前端传来的文件名。

    只保留安全字符，强制 .pdf 后缀，丢掉路径分隔符 —— 这个值会进
    Content-Disposition 头，不能让换行/引号之类的字符混进去造成头注入。
    """
    if not raw:
        return None
    base = os.path.basename(str(raw).replace("\\", "/")).strip()
    base = re.sub(r"[\r\n\t\"'<>|:*?/\\]", "", base)
    base = base.strip(". ")
    if not base:
        return None
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base[:120]


@app.get("/files/{score_id}/original")
@app.get("/files/{score_id}/original/{index}")
def original_file(
    score_id: int,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
    index: int = 0,
):
    """原文件预览：多图任务用 index 取第 N 张，供校对时对照原谱。"""
    s = db.get(Score, score_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")

    keys = parse_keys(s.original_file_url)
    if not keys:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "原文件已丢失")
    key = keys[max(0, min(index, len(keys) - 1))]

    path = local_path(key)
    if path is not None:
        return FileResponse(path)

    url = presign_get(key, expires=3600)
    if url:
        return RedirectResponse(url, status_code=status.HTTP_302_FOUND)
    raise HTTPException(status.HTTP_404_NOT_FOUND, "原文件已丢失")


@app.get("/healthz")
def healthz(deep: int = Query(default=0, ge=0, le=1)):
    """存活探针。deep=1 时额外做一次真实数据库查询。

    之前 /healthz 完全不碰数据库，所以数据库挂掉时它仍返回 200，
    表现为「服务活着但所有功能全挂」，很难第一时间发现问题。
    """
    # db_pool：连接池实时水位。checkedout 长期贴近 pool_size+max_overflow
    # 就说明有请求没归还连接 —— 这是「连接泄漏」最直接的观测点。
    out: dict = {"ok": True, "ts": int(time.time()), "db_backend": db_backend(),
                 "db_pool": pool_stats()}
    _err = db_last_error()
    if _err:
        # 最近一次 Postgres 故障原文：配额超限 / 出网额度用尽 / 连不上 / 认证失败。
        # 没有这一项就只能看到「已降级」，无法判断该怎么处置。
        out["db_error"] = _err
    if is_fallback():
        out["db_fallback"] = True
        out["db_file"] = str(SQLITE_PATH)
        # 落在容器临时盘上 → 容器重启后这份 SQLite 会丢（Render 免费版没有持久盘，
        # /var/data 不存在时会回落到代码目录）。必须让用户看得见。
        out["db_persistent"] = str(SQLITE_PATH).startswith("/var/data")
        out["note"] = (
            "Postgres 不可用，已降级到本地 SQLite（历史记录暂不可见）"
            if out["db_persistent"] else
            "Postgres 不可用，已降级到本地 SQLite；注意当前容器无持久盘，重启后本次降级数据会丢失"
        )
    if deep:
        try:
            db = next(get_db())
            try:
                db.execute(select(1)).scalar()
                out["db"] = "ok"
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001
            out["ok"] = False
            out["db"] = "unavailable"
            out["error"] = str(exc)[:200]
    return out


@app.get("/api-status")
def root():
    """API 状态页（同源托管前端时，/ 由静态挂载提供应用页面）。"""
    return {
        "service": "handpan-backend",
        "status": "running",
        "hint": "这是 API 服务，请使用前端页面 https://handpan-frontend.onrender.com",
        "docs": "/docs",
        "health": "/healthz",
    }


# ------------------------------------------------------------------ 前端静态托管
# 单服务部署：把 vite build 产物放进 backend/frontend-dist/（或用 FRONTEND_DIST 指向别处），
# 后端直接同源托管前端 —— 用户浏览器里不再有跨域/预检问题。
# 三服务部署（前端单独托管）时不放这个目录，自动跳过。
def _find_dist() -> Path | None:
    env = os.environ.get("FRONTEND_DIST")
    if env and Path(env).is_dir():
        return Path(env)
    local = BASE_DIR / "frontend-dist"
    if local.is_dir():
        return local
    sibling = BASE_DIR.parent / "handpan-frontend" / "dist"
    if sibling.is_dir():
        return sibling
    return None


DIST = _find_dist()


@app.get("/version")
def version():
    """当前前端构建号。前端轮询它，发现变了就提示刷新，避免一直跑旧代码。"""
    try:
        html = (DIST / "index.html").read_text(encoding="utf-8")
        m = re.search(r"assets/(index-[A-Za-z0-9_\-]+\.js)", html)
        return {"build": m.group(1) if m else ""}
    except OSError:
        return {"build": ""}


@app.get("/build-info")
def build_info():
    """后端部署版本指纹 —— 用来确认线上跑的是不是最新那次推送的代码。

    Render 每次 push 都会重新构建，但没有官方的「当前 commit」接口，
    于是把关键文件的内容哈希 + 源码里记录的版本号一起返回：
    线上验收脚本据此断言「新功能真的上线了」，而不是只看 healthz 200。
    """
    import hashlib

    def _fp(rel: str) -> str:
        try:
            b = (BASE_DIR / rel).read_bytes()
        except OSError:
            return ""
        return hashlib.sha256(b).hexdigest()[:12]

    files = {
        "extract/rl_score.py": _fp("extract/rl_score.py"),
        "extract/grid_ocr.py": _fp("extract/grid_ocr.py"),
        "extract/omr.py": _fp("extract/omr.py"),
        "score/note_mapper.py": _fp("score/note_mapper.py"),
        "score/layout.py": _fp("score/layout.py"),
        "score/render_pdf.py": _fp("score/render_pdf.py"),
        "database.py": _fp("database.py"),
        "main.py": _fp("main.py"),
    }
    # 能力探针：源码里出现这些符号 = 新版能力已上线
    caps = {}
    try:
        rl = (BASE_DIR / "extract" / "rl_score.py").read_text(encoding="utf-8")
        go = (BASE_DIR / "extract" / "grid_ocr.py").read_text(encoding="utf-8")
        ly = (BASE_DIR / "score" / "layout.py").read_text(encoding="utf-8")
        caps = {
            "rl_slot_anchors": "_slot_anchors" in rl,
            "rl_merge_split": "_merge_split_groups" in rl,
            "rl_dp_assign": "_dp_assign_slots" in rl,
            "tesseract_discovery": "_find_tesseract" in go,
            "three_lane_bass": "bass_drop_mm" in ly,
        }
    except OSError:
        pass
    return {"files": files, "caps": caps, "ts": int(time.time())}


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")
