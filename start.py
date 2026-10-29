"""启动服务并自动打开浏览器。

用法（在 handpan-backend 目录下）：
    .venv/Scripts/python.exe start.py

它会做三件事：读 .env → 起 uvicorn → 2 秒后打开 http://127.0.0.1:8000/
"""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_env() -> None:
    """.env 极简解析（不引第三方依赖，KEY=VALUE 一行一个）。"""
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if __name__ == "__main__":
    load_env()
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    url = f"http://{host}:{port}/"

    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    print(f"\n  手碟谱面统一转换 · {url}\n  按 Ctrl+C 停止\n")
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")
