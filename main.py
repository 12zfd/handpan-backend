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

import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import get_db, init_db
from models import Score, User
from pipeline import render_only, run_pipeline
from score.note_mapper import requantize
from security import decode_token, make_token, verify_password
from storage import dump_keys, local_path, parse_keys, presign_get, upload_bytes

BASE_DIR = Path(__file__).resolve().parent

ALLOWED_EXT = {
    ".mp4": "video", ".mov": "video", ".webm": "video",
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".mid": "midi", ".midi": "midi",
}
MAX_BYTES = 200 * 1024 * 1024
MAX_FILES = 5          # 图片一次最多 5 张（多页谱）

bearer = HTTPBearer(auto_error=False)

app = FastAPI(title="Handpan Sheet Forge", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    # 前端用 fetch + blob 下载，需要能读到文件名
    expose_headers=["Content-Disposition"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


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


@app.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Annotated[Session, Depends(get_db)]):
    user = db.scalar(select(User).where(User.username == body.username))
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
    return {"score_id": score.id, "status": score.status, "files": len(keys)}


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
    rows = db.scalars(select(Score).order_by(Score.created_at.desc()).limit(limit)).all()
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

    s.midi_data_json = payload
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
        s.midi_data_json = j
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
        s.midi_data_json = {**s.midi_data_json,
                            "tempo": {**(s.midi_data_json.get("tempo") or {}), "bpm": body.bpm}}
        s.status = "rendering"
        db.commit()
        background.add_task(render_only, score_id)
        return {"score_id": score_id, "status": "rendering",
                "bars": len(s.midi_data_json.get("bars") or []), "note": str(exc)}

    s.midi_data_json = rebuilt
    s.status = "rendering"
    db.commit()

    background.add_task(render_only, score_id)
    return {"score_id": score_id, "status": "rendering", "bars": len(rebuilt["bars"])}


# ------------------------------------------------------------------ 下载
@app.get("/download/{score_id}")
def download(
    score_id: int,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    s = db.get(Score, score_id)
    if s is None or not s.pdf_url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF 尚未生成")

    name = _pdf_name(s)
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
def healthz():
    return {"ok": True, "ts": int(time.time())}


@app.get("/")
def root():
    """根路径：浏览器直接打开后端地址时给出友好提示，而不是 404。"""
    return {
        "service": "handpan-backend",
        "status": "running",
        "hint": "这是 API 服务，请使用前端页面 https://handpan-frontend.onrender.com",
        "docs": "/docs",
        "health": "/healthz",
    }


# ------------------------------------------------------------------ 前端静态托管
# 本地/单服务部署时，把 vite build 产物挂在根路径；Render 三服务部署时该目录不存在，自动跳过。
DIST = BASE_DIR.parent / "handpan-frontend" / "dist"


@app.get("/version")
def version():
    """当前前端构建号。前端轮询它，发现变了就提示刷新，避免一直跑旧代码。"""
    try:
        html = (DIST / "index.html").read_text(encoding="utf-8")
        m = re.search(r"assets/(index-[A-Za-z0-9_\-]+\.js)", html)
        return {"build": m.group(1) if m else ""}
    except OSError:
        return {"build": ""}


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")
