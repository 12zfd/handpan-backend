"""密码哈希与 JWT。

密码：PBKDF2-HMAC-SHA256（hashlib 标准库，无需编译依赖，260k 轮）。
      哈希串自带算法前缀，将来换 bcrypt/argon2 可以并存、逐步迁移。
JWT：PyJWT，HS256，7 天有效期。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY = os.environ.get("JWT_SECRET") or "dev-only-secret-change-before-deploy"
ALGORITHM = "HS256"
TOKEN_HOURS = 24 * 7
_ITERATIONS = 260_000

# 部署到 Render（会设置 RENDER=true）却忘了配 JWT_SECRET 时直接启动失败，
# 否则任何人都能伪造 token。
if os.environ.get("RENDER") and SECRET_KEY == "dev-only-secret-change-before-deploy":
    raise RuntimeError(
        "生产环境必须配置 JWT_SECRET 环境变量（不要用默认值），"
        "可用 python -c \"import secrets;print(secrets.token_urlsafe(48))\" 生成"
    )


def hash_password(raw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", raw.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(raw: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", raw.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def make_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),
        "u": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return None
