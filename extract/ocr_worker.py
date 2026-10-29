# -*- coding: utf-8 -*-
"""独立进程 OCR：子进程跑识别，主进程只收 JSON 结果。
子进程退出 → 全部内存（含 onnxruntime arena）立刻归还操作系统。
这从根本上消除「arena 累积」与「并发叠加」两个 OOM 根因。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent


def _worker_main() -> int:
    """子进程入口：stdin 收图片字节（base64），stdout 吐 JSON 结果。"""
    import base64
    import io

    import numpy as np
    from PIL import Image

    try:
        raw = sys.stdin.buffer.read()
        img_bytes = base64.b64decode(raw)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img)

        # 关键：子进程内必须强制 inproc，否则会无限递归开子进程。
        os.environ["OCR_IN_WORKER"] = "1"
        os.environ["OCR_RUN_MODE"] = "inproc"
        # 线程限制必须在建 onnx session 之前生效。
        # 注意：ORT 线程数由 CPU 配额决定 —— Render Free 只有 0.1 CPU，
        # 开满物理核会让线程自旋白烧 CPU（实测 44.1 CPU 秒 → 9.5 CPU 秒）。
        from extract.ort_tune import tune_ort

        tune_ort()
        from extract.ocr_provider import LocalRapidOcrProvider
        provider = LocalRapidOcrProvider()
        timing: dict = {}
        items = provider.recognize(bytes(img_bytes), timing=timing)
        out = {"ok": True, "items": items, "timing": timing}
    except Exception as exc:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()
    return 0


def recognize_in_subprocess(image_bytes: bytes, timeout: float = 120.0,
                            timing: dict | None = None) -> list[dict]:
    """在独立子进程里跑本地 OCR，返回 OcrItem 列表。

    - 子进程每次都是全新的 → onnxruntime arena 不会跨任务累积
    - 一个任务崩了也不影响主进程（OOM 只杀子进程）
    - 峰值内存只属于那一个子进程，并发任务由上层信号量串行化
    """
    import base64

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # OMP 线程数必须在**进程启动前**通过环境变量设定才生效
    # （ORT 读的是 OMP_NUM_THREADS，运行时改 os.environ 已经太晚）。
    try:
        from .ort_tune import ort_threads

        n = ort_threads()
        if n > 0:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[var] = str(n)
    except Exception:  # noqa: BLE001
        pass

    payload = base64.b64encode(image_bytes)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "extract.ocr_worker"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            cwd=str(_BACKEND),
        )
    except subprocess.TimeoutExpired as exc:
        from .ocr_provider import OcrProviderError
        raise OcrProviderError(
            "timeout", f"OCR 子进程超时（{timeout}s）", retryable=True
        ) from exc

    out = (proc.stdout or b"").decode("utf-8", "replace").strip()
    if not out:
        err = (proc.stderr or b"").decode("utf-8", "replace")[-300:]
        from .ocr_provider import OcrProviderError
        raise OcrProviderError(
            "unavailable", f"OCR 子进程无输出（rc={proc.returncode}）：{err}"
        )
    try:
        data = json.loads(out)
    except Exception as exc:  # noqa: BLE001
        from .ocr_provider import OcrProviderError
        raise OcrProviderError("bad_response", f"OCR 子进程返回无法解析：{exc}") from exc

    if not data.get("ok"):
        from .ocr_provider import OcrProviderError
        raise OcrProviderError("unavailable", f"OCR 子进程失败：{data.get('error')}")
    if timing is not None and isinstance(data.get("timing"), dict):
        timing.update(data["timing"])
    return data.get("items") or []


if __name__ == "__main__":
    sys.exit(_worker_main())
