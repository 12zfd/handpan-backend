# -*- coding: utf-8 -*-
"""onnxruntime 运行时调优（Render Free 的 0.1 CPU 必需）。

问题
----
rapidocr-onnxruntime 1.2.3 创建 SessionOptions 时**没有设**
`intra_op_num_threads`，onnxruntime 于是默认开满**物理核数**（本地 16）。
每个算子内部用自旋（spin-wait）等待线程同步 —— 这在多核机器上几乎免费，
但在 Render Free 的 0.1 CPU 上是灾难：

    threads=1   → 合计 CPU 10.7s
    threads=2   → 合计 CPU 15.5s
    threads=4   → 合计 CPU 26.6s
    threads=8   → 合计 CPU 50.4s
    threads=16  → 合计 CPU 50.7s   ← 默认

也就是说：**默认配置下 ~80% 的 CPU 都烧在自旋等待上**，真正的浮点运算
只占 20%。单线程反而 CPU 用量最少。

线上实测（默认线程）：墙钟 150s
线上实测（单线程预计）：CPU 10.7s → 墙钟 ≈ 32s，进入 60s 硬指标内。

用法
----
    from extract.ort_tune import tune_ort
    tune_ort()          # 在创建任何 InferenceSession 之前调用一次
"""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def ort_threads() -> int:
    """要使用的 intra_op 线程数。

    `OCR_ORT_THREADS=1`（默认）—— Render Free 只有 0.1 CPU，
    多线程自旋纯属浪费。设为 0 则不干预（用 onnxruntime 默认）。
    """
    return _env_int("OCR_ORT_THREADS", 1)


def tune_ort(threads: int | None = None) -> int:
    """给 rapidocr 用的 SessionOptions 注入线程限制。

    必须在 `RapidOCR()` 构造**之前**调用（构造时才会建 InferenceSession）。
    返回实际生效的线程数。
    """
    n = ort_threads() if threads is None else threads
    if n <= 0:
        return 0

    # 让 OpenMP / numpy 也别开太多线程（同样会自旋烧 CPU）
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))

    try:
        import rapidocr_onnxruntime.utils as U
    except ImportError:
        return n

    if getattr(U, "_workbuddy_tuned", False):
        return getattr(U, "_workbuddy_threads", n)

    orig = U.SessionOptions

    def factory(*args, **kwargs):
        so = orig()
        try:
            so.intra_op_num_threads = n
            so.inter_op_num_threads = 1
        except Exception:  # noqa: BLE001
            pass
        return so

    U.SessionOptions = factory
    U._workbuddy_tuned = True
    U._workbuddy_threads = n
    return n
