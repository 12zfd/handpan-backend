"""文件存储：本地磁盘（开发） / S3 兼容对象存储（生产）。

为什么要有这一层：Render 的容器磁盘是临时盘，重启即丢。
生产必须配 S3_BUCKET（Cloudflare R2 或 AWS S3）；本地跑则直接落 data/ 目录。
上层只认 key，不关心落在哪。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parent
# 生产环境把 DATA_DIR 指向 Render Disk 的挂载点（如 /var/data），
# 否则容器重启后上传件和 PDF 会全部丢失（Render 的容器盘是临时的）。
LOCAL_ROOT = Path(os.environ.get("DATA_DIR") or (BASE_DIR / "data"))
BUCKET = os.environ.get("S3_BUCKET", "")


def parse_keys(value: str | None) -> list[str]:
    """original_file_url 里可能是一个 key，也可能是 JSON 数组（多图任务）。
    老数据是纯路径字符串，天然兼容。"""
    s = (value or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(k) for k in arr if str(k).strip()]
        except (ValueError, TypeError):
            pass
    return [s]


def dump_keys(keys: list[str]) -> str:
    if len(keys) == 1:
        return keys[0]
    return json.dumps(keys, ensure_ascii=False)


def use_s3() -> bool:
    return bool(BUCKET)


def _client():
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT") or None,
        aws_access_key_id=os.environ.get("S3_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET"),
        region_name=os.environ.get("S3_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    if use_s3():
        _client().put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
        return
    path = LOCAL_ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def download_bytes(key: str) -> bytes:
    if use_s3():
        return _client().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return (LOCAL_ROOT / key).read_bytes()


def local_path(key: str) -> Path | None:
    """本地存储时返回物理路径，供 FileResponse 使用；S3 时返回 None。"""
    if use_s3():
        return None
    path = LOCAL_ROOT / key
    return path if path.exists() else None


def presign_get(key: str, expires: int = 86400, download_name: str | None = None) -> str | None:
    if not use_s3():
        return None
    params: dict = {"Bucket": BUCKET, "Key": key}
    if download_name:
        params["ResponseContentDisposition"] = (
            f"attachment; filename*=UTF-8''{quote(download_name)}"
        )
    return _client().generate_presigned_url("get_object", Params=params, ExpiresIn=expires)
