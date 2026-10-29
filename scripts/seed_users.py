"""预置两个授权账号（本产品不开放注册）。

用法（在 handpan-backend 目录下）：
    .venv/Scripts/python.exe -m scripts.seed_users weijie handpan2026
    .venv/Scripts/python.exe -m scripts.seed_users partner 另一个密码
    .venv/Scripts/python.exe -m scripts.seed_users --list
已存在的账号会被跳过，不会覆盖密码；要改密码用 --reset。
"""
from __future__ import annotations

import sys

from database import SessionLocal, init_db
from models import User
from security import hash_password


def create(username: str, password: str, reset: bool = False) -> str:
    db = SessionLocal()
    try:
        row = db.query(User).filter_by(username=username).first()
        if row:
            if not reset:
                return f"已存在，跳过：{username}"
            row.password = hash_password(password)
            db.commit()
            return f"密码已重置：{username}"
        db.add(User(username=username, password=hash_password(password)))
        db.commit()
        return f"创建成功：{username}"
    finally:
        db.close()


def main() -> None:
    init_db()
    args = [a for a in sys.argv[1:]]

    if not args or args[0] == "--list":
        db = SessionLocal()
        users = db.query(User).all()
        print("现有账号：", ", ".join(u.username for u in users) or "（无）")
        db.close()
        return

    reset = "--reset" in args
    args = [a for a in args if a != "--reset"]
    if len(args) < 2:
        print("用法：python -m scripts.seed_users <用户名> <密码> [--reset]")
        return

    print(create(args[0], args[1], reset=reset))


if __name__ == "__main__":
    main()
