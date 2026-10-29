# 性能方案：512MB 下的图片转谱面（含云端 OCR 外包伪代码）

> 本文记录 Render Free（0.1 CPU / 512MB RAM）上「上传图片 → 音谱」的完整性能方案、
> 实测数据，以及当本地方案仍不够时的**云端 OCR 外包**接口封装与伪代码。
>
> 硬约束：**对外 API 路径、请求参数、返回 JSON 结构 100% 不变；前端一行不改。**

---

## 一、原始故障与根因链

用户现象：上传图片后前端一直显示「读取中」，页面卡死。

根因是一条完整的链条：

| # | 根因 | 证据 |
|---|---|---|
| 1 | `UPSCALE=2.0` 把整图放大一倍（1080×2412 → 2160×4824） | 峰值内存实测 **1682MB** |
| 2 | Render Free 只有 512MB，容器（父进程+子进程+运行时）必然 OOM | Render 实例被杀重启 |
| 3 | 任务跑在 `BackgroundTasks` 里，**重启即丢失** | 任务永远停在 `extracting` |
| 4 | onnxruntime 的 arena 跨任务累积、不归还 OS | 同进程第 2 张起稳定 582MB |
| 5 | 两个任务并发时峰值叠加 | 更容易触发 OOM |

结论：**不是"算法太慢"，是"内存爆了导致任务被重启后永远丢在 extracting"**。
所以方案的主线是「把峰值压到 512MB 以内」，而不是「等更久」。

---

## 二、本地方案（已落地，实测峰值 ~340MB / 9.5 CPU 秒）

### 2.1 核心洞察：det 与 rec 的内存维度不同

RapidOCR 的两个阶段可以分离调用：

```python
engine.text_detector(img)          # 检测：输出文字框
engine.get_crop_img_list(img, boxes)  # 按框裁剪
engine.text_recognizer(crops)      # 识别：只吃裁剪块
```

- **det 的内存 ∝ 输入图的面积**（要做整图特征图）；
- **rec 的内存 ∝ 裁剪块的数量与尺寸**（输入固定 `rec_img_shape=[3,48,320]`，48px 高，内存可忽略）；
- **rec 的精度取决于裁剪块的像素密度**（字符越大越准）。

于是有了「**det 跑原图分块 + rec 从放大图裁剪**」这条同时省内存又保召回的路：

```
原图 1080×2412
   ├─ 纵向切 4 块（带 overlap=120px），逐块 det   → det 内存 ÷ 4
   └─ rec 阶段从（按需放大的）整图按框裁剪          → 字符像素密度不降
```

### 2.2 关键调参发现：`limit_type=min` 是「放大」不是「缩小」

RapidOCR 的 `det_limit_side_len=736` + `limit_type=min` 语义是：
**当最短边 < 736 时把图放大到 736**。所以：

- 传更小的 `det_limit_side_len` **不会**让 det 降内存、也不会降精度；
- 实测传 736 / 1024 / 1536 三档，识别结果**一模一样**（310 字符 / 24 小节）；
- 切块反而会踩坑：逐行切出的 1080×40 扁条会被**放大 18 倍**（736/40）→ 峰值爆到 6GB。

所以正确做法是**切「块」而不是切「行」**，并保持 `det_limit_side_len` 默认。

### 2.3 符杠检测：从「猜」改成「独立检测」

原实现靠 OCR 框的位置去猜符杠在哪儿，误判严重。新方案改为**独立横线连通域检测**
（`extract/beams.py`）：

```python
# 形态学横向膨胀（9×1 kernel）把断续的横线接成一段
# → connectedComponentsWithStats
# → 按「宽 ≫ 高」筛出横线段（min_len=12, max_h=6）
# → 按 x 区间与字组的重叠归属，count_beams() 判定 0/1/2 条
```

实测在 1080×2412 原图上检出 **79 段横线，长度中位 24px**，与放大图人眼核对的
「每个字组下方一条短下划线」结构完全吻合。

### 2.4 子进程隔离 + 并发闸

```python
# extract/ocr_worker.py：每次识别开独立子进程
#   → 进程退出即释放全部内存（含 arena），永不累积
#   → 子进程被 OOM 杀掉不影响主进程，主进程能把 failed 状态如实写回 DB
#   （而不是永远卡在 extracting）

# pipeline.py
_OCR_LOCK = threading.Semaphore(1)   # 串行化 OCR，防止并发峰值叠加
```

### 2.5 输入归一化：结果与图片尺寸解耦（本轮最终方案）

原始问题的第二个表现是「同一张谱，不同分辨率上传出来的结果不一样」。
根因是 RapidOCR 的 det 内部 `limit_type=min` 语义：

- `min(h, w) < limit_side_len(默认 736)` 时 **ratio = limit_side_len / min(h, w)，
  是「放大」而不是缩小**；
- 切成 4 块后每块高约 600px < 736 → det 把每块**放大 1.2 倍** → 等效分辨率升高
  → 召回虚高 → 小节数漂移；
- 调 `det_limit_side_len` 对结果无影响，因为 ratio 由 `min(h, w)` 决定，
  而不是由 limit 决定。

**解法：把所有输入统一缩放到固定宽度。**

```python
# extract/omr.py
MAX_IMAGE_WIDTH = int(os.environ.get("OCR_NORM_WIDTH") or 1080)

def _normalize(raw, target_width):
    # 统一缩放到 target_width（既缩也放），而不是"只缩到上限"
    ...
```

于是 OCR 与符杠检测**共用同一张 1080 宽的缩放图**（坐标系天然一致，
不需要来回换算），det 看到的分辨率永远一样。

实测：1080 / 1600 / 2200 / 3000 宽的同一张图，输入归一化后
**输出结果完全一致**（24 小节 / 311 音符），峰值稳定在 270~350MB。

### 2.6 切块参数实测矩阵（subproc 真实口径 = 父进程 + 子进程合计 RSS）

| 方案 | 音符 | 小节 | 峰值内存 | 耗时 | 判定 |
|---|---|---|---|---|---|
| **固定高度切块 + overlap=120 + 文本去重** | **311** | **24** | **~290MB** | **7.7s** | ✅ **定案** |
| 固定高度切块 + overlap=120 + 几何去重 | 310 | 23 ❌ | ~290MB | 7.6s | 过度删除 |
| 固定高度切块 + overlap=0 | 300 | 24 | ~280MB | 7.2s | 掉行 |
| 行带切块（行投影找行间空白） | 309 | 23 ❌ | ~285MB | 7.9s | 半行截断 |
| 低谷 + DP 均衡切点 | 309 | 23 ❌ | ~290MB | 7.8s | 半行截断 |
| 整图 det（不切块） | 309 | 24 | **483MB** | 6.9s | 内存超限风险 |
| 整图 det 缩到 880 宽 | 240 | 18 ❌ | ~200MB | 5.1s | 召回崩塌 |

**基线（旧通道整图 2x 放大）**：309 字符 / 24 小节 / **峰值 1682MB** / 9.5s → 必爆

**定案：`OCR_TILES=4` + `OCR_TILE_OVERLAP=120` + `OCR_REC_SCALE=1.0` + `OCR_NORM_WIDTH=1080`**

**切块碎片的根因与取舍**（踩坑记录）：

1. det 缩图并不能省内存 —— 1080 宽 121 框 / 483MB，缩到 880 宽只剩 76 框（掉 37%），
   **召回直接崩塌**。所以只能分块，不能缩图。
2. 固定高度切块会把一整行**切断** → det 在半行上产出碎片框：
   - overlap=0 → 框变少（-5 ~ -21），有行整段丢失；
   - overlap=120 → 碎片变**重复框**，靠文本去重压掉即可。
3. 几何去重（按 x/y 区间重叠）会**过度删除** → 小节数 24→23，
   因为相邻小节的字组 x 区间本来就紧邻。最终改用**文本判据**：
   `text` 相同 + `yc` 差 < max(8, h*0.6) + `x0` 差 < 12。
4. 符杠检测的「合并竖直碎片」`_merge_vertical` 一旦启用 → 24→25 小节，
   默认关闭（函数保留，docstring 记录实验结论）。
5. `cv2.connectedComponentsWithStats` 会分配 **int32 标签图**
   （3000×6700 = 80MB）→ 符杠检测改为**分带处理**（band ≤ 1024，带间重叠 32px），
   format_ms 从 4448ms 降到 280ms。

**最终生产代码 3 轮复验**：311 音符 / 24 小节，3 轮结果完全一致，
耗时 7.6~7.9s，峰值 367~415MB。

> `OCR_TILES` 调到 5 以上音符数会虚高（重复计数），小节数漂到 29~34，
> 反而离真值更远 —— 4 是「召回与准确性同时最优」的点。

### 2.7 RapidOCR 版本必须锁死（本轮线上事故）

线上曾报 `'RapidOCR' object has no attribute 'text_detector'`。
原因是 `requirements.txt` 写了 `rapidocr-onnxruntime>=1.2`，pip 装到了 **1.4.4**，
而 1.4.x 把入口改名了：

| 版本 | 检测入口 | 识别入口 |
|---|---|---|
| 1.2.x | `engine.text_detector` | `engine.text_recognizer` |
| 1.4.x | `engine.text_det` | `engine.text_rec` |

双保险修复：

1. `requirements.txt` 锁 `rapidocr-onnxruntime==1.2.3`（本地完整验证过的版本）；
2. `extract/ocr_raster.py` 新增 `_EngineAdapter`，同时兼容两代属性名 ——
   以后即使被升版也不会直接挂。

```python
class _EngineAdapter:
    def __init__(self, engine):
        self._det = getattr(engine, "text_detector", None) or getattr(engine, "text_det", None)
        self._rec = getattr(engine, "text_recognizer", None) or getattr(engine, "text_rec", None)
        if self._det is None or self._rec is None:
            raise RuntimeError("RapidOCR API 不兼容：找不到检测/识别入口")
```

### 2.8 性能日志

`pipeline.py` 打印分段耗时，一眼看出瓶颈：

```
2026-10-30 05:34:10 INFO [handpan.pipeline] image extract: decode=21ms infer=6022ms format=269ms total=6927ms
```

- `decode_ms` —— 图片解码（Pillow）+ 预处理
- `infer_ms` —— **OCR 模型推理（瓶颈）**
- `format_ms` —— 结果落格 / 符杠判定 / 排版
- `ocr_ms` —— 图片分支总耗时

> 注意：FastAPI 只配了自己的 logger，root logger 默认 WARNING 会把
> `log.info` 吞掉。已在 `main.py` 加 `logging.basicConfig(level=INFO)`，
> 否则 Render 日志里看不到上面这行。

---

## 三、异步架构（防 502 / 防"一直读取中"）

```
POST /tasks            → 202 Accepted, {"task_id": 47, ...}   （113ms 返回）
GET  /tasks/{task_id}  → {status, stage, progress, error_message, has_pdf}
```

- 上传即落盘 + 立即返回 `task_id`（**113ms**，绝不会 502）；
- `BackgroundTasks` 跑转换，DB 里 `status` 推进：`extracting → mapping → rendering → done`；
- 前端每 3 秒轮询 `task_id`。

实测本地端到端：`POST /tasks` 113ms 返回 → 4 次轮询（9s）→ `done`，
谱面 24 小节 / 311 音符。

---

## 四、云端 OCR 外包（终极方案 / 兜底）

### 4.1 选型

**选定：Google Cloud Vision `images:annotate` + `DOCUMENT_TEXT_DETECTION`**

| 维度 | 说明 |
|---|---|
| 免费额度 | 每月 **1000 张**（`vision.googleapis.com` 首 1000 张/月免费，超出 $0.0015/张） |
| 识别类型 | `DOCUMENT_TEXT_DETECTION` —— 适合密集印刷体谱面，能给出**每个 word 的四点包围盒** |
| 延迟 | 单图 1~3s |
| 凭据 | Service Account JSON → 用 `google-auth` 取 access_token，或直接用 API Key（`X-API-Key`） |

备选：腾讯云 OCR / 百度 OCR（国内网络更稳，但**没有「密集文本 + 逐词包围盒」这一档**，
需要自己拼；Vision 的 `fullTextAnnotation` 开箱即用，与本地 `_ocr_items` 的框结构几乎一一对应）。

**什么时候才需要切云端**：见 4.4 —— 本地已优化到 9.5 CPU 秒（线上约 32s），
60s 硬指标可达，**云端作为兜底保留**：用户传超大图 / 手写谱 / 机器规格再降 /
免费额度用尽时，切过去只需改一个环境变量，**不改前端、不改接口**。

### 4.2 为什么外包不需要改前端

关键在于 `extract/ocr_provider.py` 定义的**适配层契约**：

```
所有 provider 统一输出 OcrItem = {text, x0, x1, yc, h}   ← 原图坐标系
```

上层 `extract/omr.py` 的落格 / 符杠 / 排版逻辑**一行都没改**，
它只认 `OcrItem`，不关心是谁识别的。云端结果经过 `_parse_google()` 映射成同样的结构即可。

### 4.3 本地优化的最终结论（为什么现在能进 60s）

线上验收第一轮（ORT 默认线程）功能全通过但**墙钟 150s**：

| 验收项 | 结果 |
|---|---|
| 24 小节 / 311 音符 | ✅ |
| PDF 生成 | ✅ |
| 连续两个任务都成功、结果一致 | ✅ |
| 无 OOM | ✅ |
| 墙钟 | ❌ 150s（目标 60s） |

排查发现**根因不是算力总量，而是 onnxruntime 的线程配置**：

`rapidocr-onnxruntime` 1.2.3 建 `SessionOptions` 时**没有设**
`intra_op_num_threads`，onnxruntime 于是开满**物理核数**，每个算子内部
用自旋（spin-wait）等待线程同步。在 16 核本机上这几乎免费，
在 Render Free 的 **0.1 CPU** 上就是灾难 —— 线程大部分时间在自旋烧 CPU。

实测（同一张谱，CPU 秒 = user + sys）：

| intra_op 线程数 | 合计 CPU 秒 |
|---|---|
| **1** | **10.7** ← 定案 |
| 2 | 15.5 |
| 4 | 26.6 |
| 8 | 50.4 |
| 16（默认） | 50.7 |

完整流程端到端：**44.1 CPU 秒 → 9.5 CPU 秒（-78%）**，谱面结果
**完全一致**（311 音符 / 24 小节）。

> 换算关系：线上 0.1 CPU，墙钟 ≈ CPU 秒 × 10。
> 44.1 CPU 秒 → 150s（实测吻合），9.5 CPU 秒 → 约 32s。

实现：`extract/ort_tune.py`（注入线程限制 + OMP 环境变量），
在 `ocr_raster._load_engine` / `ocr_provider._get_engine` 建 session 前调用；
worker 子进程在 **spawn 时**把 `OMP_NUM_THREADS` 注入环境
（ORT 读的是环境变量，运行时改 `os.environ` 已经太晚）。

> 附带发现：`sess_opt.enable_cpu_mem_arena = False` —— rapidocr 本来就关了
> arena，所以**不存在 arena 跨任务累积**，子进程隔离只是为了「OOM 不拖垮主进程」。

### 4.4 云端 provider 的离线验收结果

用 mock 拦截 `urlopen`（不起外网、不消耗额度）验证：

| 验收项 | 结果 |
|---|---|
| 请求体格式（`requests[0].image.content` + `features.DOCUMENT_TEXT_DETECTION`） | ✅ |
| 图片 base64 编码 + `X-API-Key` 头 | ✅ |
| 归一化坐标 → 像素坐标换算 | ✅ 精确还原 |
| 401 → `auth` | ✅ |
| 429 → `quota` | ✅ |
| 5xx → `unavailable` | ✅ |
| 坏 JSON → `bad_response` | ✅ |
| 超时 → `timeout` | ✅ |
| 连接失败 → `unavailable` | ✅ |
| 429 自动重试（retries=1 → 实际 2 次） | ✅ |
| **云端 vs 本地 OcrItem 结构一致**（`text/x0/x1/yc/h`） | ✅ 前端零改动 |

### 4.5 伪代码

#### (1) 接口封装 —— `CloudOcrProvider._post()`

```python
def _post(self, payload: dict) -> dict:
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
        detail = exc.read().decode("utf-8", "replace")[:300]
        if exc.code in (401, 403):                       # 凭据无效
            raise OcrProviderError("auth", f"云端 OCR 凭据无效（HTTP {exc.code}）：{detail}")
        if exc.code in (408, 429):                       # 额度耗尽 / 限流
            raise OcrProviderError("quota",
                f"云端 OCR 额度或频率受限（HTTP {exc.code}）：{detail}", retryable=True)
        if exc.code >= 500:                             # 服务端异常
            raise OcrProviderError("unavailable",
                f"云端 OCR 服务异常（HTTP {exc.code}）：{detail}", retryable=True)
        raise OcrProviderError("unavailable", f"云端 OCR 请求失败（HTTP {exc.code}）：{detail}")
    except urllib.error.URLError as exc:
        if "timed out" in str(getattr(exc, "reason", exc)).lower():
            raise OcrProviderError("timeout",
                f"云端 OCR 请求超时（{self.timeout}s）", retryable=True)
        raise OcrProviderError("unavailable",
            f"云端 OCR 网络错误：{exc.reason}", retryable=True)
    except TimeoutError as exc:
        raise OcrProviderError("timeout", f"云端 OCR 请求超时（{self.timeout}s）", retryable=True)

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise OcrProviderError("bad_response", f"云端 OCR 返回体无法解析：{exc}")
```

#### (2) 结果映射（第三方 → 我们的 OcrItem）

```python
def _parse_google(self, data: dict, width: int, height: int) -> list[dict]:
    """Google Vision → OcrItem（原图像素坐标）。

    关键点：Vision 的 boundingBox 顶点是 0~1 归一化坐标，
    必须乘回 width/height 才能和本地通道同一坐标系。
    """
    out = []
    for resp in data.get("responses") or []:
        fa = resp.get("fullTextAnnotation") or {}
        for page in fa.get("pages") or []:
            for block in page.get("blocks") or []:
                for para in block.get("paragraphs") or []:
                    for word in para.get("words") or []:
                        vs = (word.get("boundingBox") or {}).get("vertices") or []
                        if not vs:
                            continue
                        xs = [float(v.get("x", 0)) * width for v in vs]
                        ys = [float(v.get("y", 0)) * height for v in vs]
                        out.append({
                            "text": word.get("text") or "",
                            "x0": min(xs), "x1": max(xs),
                            "yc": sum(ys) / len(ys),
                            "h":  max(ys) - min(ys),
                        })
    return out
```

#### (3) 主流程（超时 / 重试 / 兜底）

```python
def recognize(self, image_bytes: bytes, timing: dict | None = None) -> list[dict]:
    im = Image.open(io.BytesIO(image_bytes))
    width, height = im.size                      # 记下原图尺寸，用于反归一化

    payload = {"requests": [{
        "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
    }]}

    last = None
    for attempt in range(self.retries + 1):      # 默认 retries=2
        try:
            data = self._post(payload)
            break
        except OcrProviderError as exc:
            last = exc
            if not exc.retryable or attempt >= self.retries:
                raise                        # 不可重试或次数用尽 → 直接上抛
            time.sleep(min(2 ** attempt, 5))  # 指数退避：1s → 2s

    items = self._parse_google(data, width, height)
    if not items:
        items = self._parse_generic(data)     # 兼容非 Google 形态的返回体
    return _decode_items(items)                # 统一做非法项剔除 + 坐标纠正
```

#### (4) 异常 → 接口契约（**JSON 结构与现有一致**）

`OcrProviderError` 在 `pipeline.run_pipeline` 里被捕获，落库为
`status="failed"` + `error_message`。对外接口返回的结构**完全不变**：

```json
{
  "task_id": 47,
  "score_id": 47,
  "status": "failed",
  "stage": "失败",
  "progress": 100,
  "error_message": "云端 OCR 额度或频率受限（HTTP 429）：Quota exceeded",
  "has_pdf": false,
  "created_at": "2026-10-30T05:34:01Z",
  "updated_at": "2026-10-30T05:34:12Z"
}
```

**前端无需任何改动** —— 它本来就是读 `status` / `error_message` 显示错误。

错误码契约（稳定，日志与前端可依赖）：

| code | 含义 | 触发场景 |
|---|---|---|
| `timeout` | 网络超时 | `urlopen` timeout / URLError timed out |
| `quota` | 额度耗尽 / 限流 | HTTP 408 / 429 |
| `auth` | 凭据无效 | HTTP 401 / 403 |
| `unavailable` | 服务不可用 | HTTP 5xx / 网络错误 |
| `bad_response` | 返回体无法解析 | JSON decode 失败 |
| `unsupported` | provider 未配置 | 缺 `CLOUD_OCR_ENDPOINT` / 依赖缺失 |

#### (5) 切换方式（只改环境变量）

```yaml
# render.yaml
envVars:
  - key: OCR_PROVIDER
    value: cloud
  - key: CLOUD_OCR_ENDPOINT
    value: https://vision.googleapis.com/v1/images:annotate
  - key: CLOUD_OCR_API_KEY
    sync: false        # 在 Render Dashboard 填 Google API Key
  - key: CLOUD_OCR_TIMEOUT
    value: "20"
  - key: CLOUD_OCR_RETRIES
    value: "2"
```

`OCR_PROVIDER=auto`（默认）时：配了 `CLOUD_OCR_ENDPOINT` 就用云端，否则回落本地。

---

## 五、生产参数（已写进 `render.yaml`）

```yaml
- key: OCR_RUN_MODE
  value: subproc         # 子进程隔离，OOM 不拖垮主进程
- key: OCR_TILES
  value: "4"
- key: OCR_TILE_OVERLAP
  value: "120"
- key: OCR_BLOCK_H
  value: "600"
- key: OCR_REC_SCALE
  value: "1.0"
- key: OCR_NORM_WIDTH
  value: "1080"          # 所有输入统一归一化 → 结果与输入尺寸无关
- key: OCR_ORT_THREADS
  value: "1"             # onnxruntime 单线程：44.1 CPU 秒 → 9.5 CPU 秒（关键！）
- key: OCR_PROVIDER
  value: local           # 配了 CLOUD_OCR_ENDPOINT 会自动切云端
- key: LOG_LEVEL
  value: INFO
```

代码默认值同样锁死为这组（`extract/ocr_raster.py`）：
`OCR_TILES=4`、`OCR_TILE_OVERLAP=120`、`OCR_BLOCK_H=600`、`OCR_REC_SCALE=1.0`、
`OCR_MIN_SCORE=0.5`、`OCR_DET_WIDTH=0`（det 不缩图，缩了就掉召回）。

---

## 六、改动清单

| 文件 | 改动 |
|---|---|
| `extract/ocr_raster.py` | **新增**：分块 det + 裁剪 rec 通道，原图坐标系输出 OcrItem，含 `_dedup` 与 timing 回传；`_EngineAdapter` 兼容 rapidocr 两代 API |
| `extract/beams.py` | **新增**：独立横线连通域检测 + `count_beams` 归属判定；分带处理避免 int32 标签图占满内存 |
| `extract/ocr_provider.py` | **新增**：provider 适配层（local / cloud / auto）+ `OcrProviderError` 错误码契约 |
| `extract/ocr_worker.py` | **新增**：子进程 OCR worker，进程退出即释放全部内存；spawn 时注入 OMP 线程数 |
| `extract/ort_tune.py` | **新增**：onnxruntime 线程数限制（`intra_op=1`）—— 0.1 CPU 上的决定性优化 |
| `extract/omr.py` | 接入新 provider + 独立符杠 + 输入归一化；**原有落格/排版业务逻辑一字未改** |
| `pipeline.py` | 接入 `_Timing` 分段计时 + 性能日志 + OCR 并发闸 |
| `main.py` | 新增 `POST /tasks`(202) / `GET /tasks/{task_id}`；`/upload` 追加 `task_id`；补 `logging.basicConfig` |
| `requirements.txt` | 锁 `rapidocr-onnxruntime==1.2.3`（1.4.x 改了 API 入口名） |
| `render.yaml` | plan 改 free；前端改纯静态；补全部 OCR 性能参数 |
