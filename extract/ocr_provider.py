"""OCR provider 适配层。

目标：**把「文字识别」这一步从业务逻辑里剥离出来**，让上层
`extract/omr.py` 的落格 / 符杠 / 排版逻辑完全不变，只消费统一的
`OcrItem` 结构。

统一输出契约（所有 provider 必须遵守）
------------------------------------
    OcrItem = {
        "text": str,        # 识别到的文本（可能是多个字符连在一起）
        "x0": float,        # 文本框左边界（图片像素坐标）
        "x1": float,        # 文本框右边界
        "yc": float,        # 文本框垂直中心
        "h":  float,        # 文本框高度
    }

Provider 选择（环境变量 `OCR_PROVIDER`）
---------------------------------------
    local  (默认)  → 本地 RapidOCR / onnxruntime，零外网依赖
    cloud          → 云端 OCR（Google Cloud Vision / 百度 / 腾讯 等），
                     通过 `CLOUD_OCR_*` 环境变量配置
    auto           → 配了云端凭据就用云端，否则回落本地

选 cloud 的收益：本地容器不需要再常驻 onnxruntime 的 arena（实测约
90MB 常驻 + 单次识别数百 MB 峰值），Render Free 的 512MB 才不会被
打爆导致实例重启、任务永远卡在「提取中」。

异常契约
--------
所有 provider 抛出的异常都是 `OcrProviderError`，带稳定的 `code`。
上层 `pipeline.run_pipeline` 会把它转成 `status="failed"` +
`error_message`，与既有接口的失败返回格式完全一致（不新增字段、
不改变结构），因此前端无需任何改动即可正确显示错误。
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Protocol


class OcrProviderError(RuntimeError):
    """OCR provider 统一异常。

    code 取值（稳定，前端/日志可依赖）：
        timeout       —— 网络超时
        quota         —— 额度耗尽 / 限流
        auth          —— 凭据无效
        unavailable   —— 服务不可用
        bad_response  —— 返回体无法解析
        unsupported   —— provider 未安装或未配置
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class OcrProvider(Protocol):
    """provider 接口。"""

    name: str

    def recognize(self, image_bytes: bytes,
                  timing: dict | None = None) -> list[dict]:
        """识别一张图片，返回 OcrItem 列表。"""
        ...


# ---------------------------------------------------------------- 工具
def _decode_items(raw_items: list[dict]) -> list[dict]:
    """把 provider 的原始框转成统一的 OcrItem，丢弃无效项。"""
    out: list[dict] = []
    for it in raw_items or []:
        try:
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            x0 = float(it["x0"])
            x1 = float(it["x1"])
            yc = float(it["yc"])
            h = float(it["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        out.append({"text": text, "x0": x0, "x1": x1, "yc": yc, "h": h})
    return out


# ---------------------------------------------------------------- 本地
class LocalRapidOcrProvider:
    """本地 RapidOCR（onnxruntime）。默认 provider，零外网依赖。

    两种运行模式（环境变量 `OCR_RUN_MODE`）：
        inproc   (默认) —— 在主进程内跑。省一次进程启动（约 0.4s），
                          但 onnxruntime 的 arena 会跨任务累积、不归还系统，
                          Render Free 的 512MB 会被打爆。
        subproc  —— 每次识别开一个独立子进程，子进程退出即释放全部内存，
                    arena 永不累积；且子进程被 OOM 杀掉也不影响主进程，
                    主进程能把失败状态如实写回数据库（而不是永远卡在「提取中」）。

    延迟加载：只有真正调用时才 import + 建模，避免进程启动就吃掉
    ~90MB 常驻内存。
    """

    name = "local"

    def __init__(self) -> None:
        self._engine: Any = None
        # 子进程（ocr_worker）内强制 inproc，避免无限递归开子进程
        if os.environ.get("OCR_IN_WORKER") == "1":
            self.mode = "inproc"
        else:
            self.mode = (os.environ.get("OCR_RUN_MODE") or "inproc").strip().lower()

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as exc:
            raise OcrProviderError(
                "unsupported",
                "未安装本地 OCR 引擎：pip install rapidocr-onnxruntime",
            ) from exc
        try:
            self._engine = RapidOCR()
        except Exception as exc:  # noqa: BLE001
            raise OcrProviderError(
                "unavailable", f"本地 OCR 引擎启动失败：{exc}"
            ) from exc
        return self._engine

    def recognize(self, image_bytes: bytes,
                  timing: dict | None = None) -> list[dict]:
        if self.mode == "subproc":
            from .ocr_worker import recognize_in_subprocess
            return _decode_items(recognize_in_subprocess(image_bytes, timing=timing))

        try:
            from .ocr_raster import recognize_raster
        except ImportError as exc:  # pragma: no cover
            raise OcrProviderError(
                "unsupported", f"本地 OCR 模块不可用：{exc}"
            ) from exc

        try:
            return _decode_items(recognize_raster(image_bytes, timing=timing))
        except OcrProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OcrProviderError(
                "unavailable", f"本地 OCR 识别失败：{exc}", retryable=True
            ) from exc

    def recognize_legacy(self, image_bytes: bytes) -> list[dict]:
        """旧通道（整图一次检测+识别），保留用于对照与回归验证。

        整图放大 2 倍后检测的峰值内存约 1.6GB，Render Free 跑不动，
        线上不要用；留在这里是为了能随时验证新通道没把结果改坏。
        """
        import io

        import numpy as np
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(img)
        except Exception as exc:  # noqa: BLE001
            raise OcrProviderError("bad_response", f"图片解码失败：{exc}") from exc

        engine = self._get_engine()
        try:
            out = engine(arr)
        except Exception as exc:  # noqa: BLE001
            raise OcrProviderError(
                "unavailable", f"本地 OCR 识别失败：{exc}", retryable=True
            ) from exc

        items: list[dict] = []

        def push(box, text):
            if not text:
                return
            try:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
            except Exception:
                return
            items.append({
                "text": str(text),
                "x0": min(xs), "x1": max(xs),
                "yc": sum(ys) / len(ys), "h": max(ys) - min(ys),
            })

        # rapidocr-onnxruntime: (result, elapse)；rapidocr v3: .boxes/.txts
        if isinstance(out, tuple):
            for row in out[0] or []:
                if len(row) >= 2:
                    push(row[0], row[1])
        else:
            boxes = getattr(out, "boxes", None)
            txts = getattr(out, "txts", None)
            if boxes is not None and txts is not None:
                for i, box in enumerate(boxes):
                    push(box, txts[i])
            elif isinstance(out, list):
                for row in out:
                    if len(row) >= 2:
                        push(row[0], row[1])
        return _decode_items(items)


# ---------------------------------------------------------------- 云端
class CloudOcrProvider:
    """云端 OCR provider（通用 REST 协议，兼容 Google Cloud Vision 风格）。

    配置（环境变量）：
        CLOUD_OCR_ENDPOINT   完整 URL，如
            https://vision.googleapis.com/v1/images:annotate
        CLOUD_OCR_API_KEY    凭据，作为 `X-API-Key` 头发送
        CLOUD_OCR_TIMEOUT    单次请求超时秒数，默认 20
        CLOUD_OCR_RETRIES    失败重试次数（仅对可重试错误），默认 2
        CLOUD_OCR_PROVIDER   `google`（默认）/ `generic`

    请求体（google 风格）：
        {"requests":[{"image":{"content":"<base64>"},
                      "features":[{"type":"DOCUMENT_TEXT_DETECTION"}]}]}

    返回体解析兼容三种常见形态：
        1. Google: responses[0].fullTextAnnotation.pages[].blocks[].paragraphs[].words[]
           每个 word 有 boundingBox（0~1 归一化的四点）与 text
        2. 通用:    {"words":[{"text":..,"box":[[x,y],...]}], "boxes":[...], "texts":[...]}
        3. 简写:    {"results":[{"text":..,"bbox":[x0,y0,x1,y1]}]}
    """

    name = "cloud"

    def __init__(self) -> None:
        self.endpoint = (os.environ.get("CLOUD_OCR_ENDPOINT") or "").strip()
        self.api_key = (os.environ.get("CLOUD_OCR_API_KEY") or "").strip()
        self.kind = (os.environ.get("CLOUD_OCR_PROVIDER") or "google").strip().lower()
        try:
            self.timeout = float(os.environ.get("CLOUD_OCR_TIMEOUT") or 20)
        except ValueError:
            self.timeout = 20.0
        try:
            self.retries = int(os.environ.get("CLOUD_OCR_RETRIES") or 2)
        except ValueError:
            self.retries = 2
        if not self.endpoint:
            raise OcrProviderError("unsupported", "未配置 CLOUD_OCR_ENDPOINT")

    # -- HTTP ---------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if exc.code in (401, 403):
                raise OcrProviderError(
                    "auth", f"云端 OCR 凭据无效（HTTP {exc.code}）：{detail}"
                ) from exc
            if exc.code in (408, 429):
                raise OcrProviderError(
                    "quota",
                    f"云端 OCR 额度或频率受限（HTTP {exc.code}）：{detail}",
                    retryable=True,
                ) from exc
            if exc.code >= 500:
                raise OcrProviderError(
                    "unavailable",
                    f"云端 OCR 服务异常（HTTP {exc.code}）：{detail}",
                    retryable=True,
                ) from exc
            raise OcrProviderError(
                "unavailable", f"云端 OCR 请求失败（HTTP {exc.code}）：{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if "timed out" in str(reason).lower():
                raise OcrProviderError(
                    "timeout", f"云端 OCR 请求超时（{self.timeout}s）", retryable=True
                ) from exc
            raise OcrProviderError(
                "unavailable", f"云端 OCR 网络错误：{reason}", retryable=True
            ) from exc
        except TimeoutError as exc:
            raise OcrProviderError(
                "timeout", f"云端 OCR 请求超时（{self.timeout}s）", retryable=True
            ) from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise OcrProviderError(
                "bad_response", f"云端 OCR 返回体无法解析：{exc}"
            ) from exc

    # -- 解析 ---------------------------------------------------------
    def _parse_google(self, data: dict, width: int, height: int) -> list[dict]:
        out: list[dict] = []
        responses = data.get("responses") or []
        for resp in responses:
            fa = resp.get("fullTextAnnotation") or {}
            for page in fa.get("pages") or []:
                for block in page.get("blocks") or []:
                    for para in block.get("paragraphs") or []:
                        for word in para.get("words") or []:
                            bb = word.get("boundingBox") or {}
                            vs = bb.get("vertices") or []
                            if not vs:
                                continue
                            xs = [float(v.get("x", 0)) * width for v in vs]
                            ys = [float(v.get("y", 0)) * height for v in vs]
                            out.append({
                                "text": word.get("text") or "",
                                "x0": min(xs), "x1": max(xs),
                                "yc": sum(ys) / len(ys), "h": max(ys) - min(ys),
                            })
        return out

    def _parse_generic(self, data: dict) -> list[dict]:
        out: list[dict] = []
        results = data.get("results") or data.get("words") or []
        for r in results:
            if not isinstance(r, dict):
                continue
            text = r.get("text") or r.get("word") or ""
            box = r.get("bbox") or r.get("box") or r.get("points")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x0, y0, x1, y1 = [float(v) for v in box]
                out.append({"text": text, "x0": x0, "x1": x1,
                            "yc": (y0 + y1) / 2, "h": max(1.0, y1 - y0)})
            elif isinstance(box, (list, tuple)) and box:
                try:
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) for p in box]
                    out.append({"text": text, "x0": min(xs), "x1": max(xs),
                                "yc": sum(ys) / len(ys), "h": max(ys) - min(ys)})
                except Exception:  # noqa: BLE001
                    continue
        return out

    # -- 主流程 -------------------------------------------------------
    def recognize(self, image_bytes: bytes,
                  timing: dict | None = None) -> list[dict]:
        """识别一张图片，返回 OcrItem 列表。

        `timing` 若传入 dict，会被填入分段耗时（decode_ms / infer_ms）。
        """
        from PIL import Image
        import io

        try:
            im = Image.open(io.BytesIO(image_bytes))
            width, height = im.size
        except Exception as exc:  # noqa: BLE001
            raise OcrProviderError("bad_response", f"图片解码失败：{exc}") from exc

        payload = {
            "requests": [{
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }]
        }

        last: OcrProviderError | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self._post(payload)
                break
            except OcrProviderError as exc:
                last = exc
                if not exc.retryable or attempt >= self.retries:
                    raise
                time.sleep(min(2 ** attempt, 5))
        else:  # pragma: no cover
            raise last or OcrProviderError("unavailable", "云端 OCR 调用失败")

        if self.kind == "google":
            items = self._parse_google(data, width, height)
        else:
            items = self._parse_generic(data)
        if not items:
            # google 若没找到 fullTextAnnotation，退回通用解析
            items = self._parse_generic(data)
        return _decode_items(items)


# ---------------------------------------------------------------- 工厂
_PROVIDERS: dict[str, OcrProvider] = {}


def get_provider() -> OcrProvider:
    """按环境变量选择 provider。"""
    mode = (os.environ.get("OCR_PROVIDER") or "auto").strip().lower()

    if mode in ("cloud", "google", "vision"):
        key = "cloud"
    elif mode == "local":
        key = "local"
    else:  # auto
        key = "cloud" if (os.environ.get("CLOUD_OCR_ENDPOINT") or "").strip() else "local"

    if key not in _PROVIDERS:
        _PROVIDERS[key] = CloudOcrProvider() if key == "cloud" else LocalRapidOcrProvider()
    return _PROVIDERS[key]


def provider_name() -> str:
    try:
        return get_provider().name
    except OcrProviderError:
        return "unconfigured"
