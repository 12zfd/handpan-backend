"""视频/音频 → note_events。

引擎选择（环境变量 TRANSCRIBE_ENGINE）：
  handpan      —— 默认。专门为手碟设计的轻量转录：onset 检测 + 九音模板匹配（纯 numpy）
  basic_pitch  —— 可选。装了 basic-pitch 才可用，精度更高但吃 1GB+ 内存

为什么默认不用 basic-pitch：它按"通用乐器"设计，手碟泛音强、延音长，
实测会被拆成一堆伪音符；而本产品只需要判断"9 个固定音位里响了哪几个"，
模板匹配反而更准、更快、512MB 机器也能跑。
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512

# 与 score/note_mapper.py 的 D_KURD_9 保持一致（这里只要频率）
_NOTE_MIDI = {1: 50, 2: 57, 3: 58, 4: 60, 5: 62, 6: 64, 7: 65, 8: 67, 9: 69}
FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)


def _ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return os.environ.get("FFMPEG_BIN", "ffmpeg")


def decode_audio(raw: bytes) -> np.ndarray:
    """任意容器 → 单声道 22050Hz float32 PCM（走 ffmpeg 管道，不落中间文件）。"""
    exe = _ffmpeg_bin()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".media") as f:
        f.write(raw)
        src = f.name
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
             "-f", "s16le", "-"],
            capture_output=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "ignore").strip()[:300]
            raise RuntimeError(f"音频解码失败：{err or '文件可能没有音轨'}")
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass

    y = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if y.size < SR // 4:
        raise RuntimeError("音频过短或为空，未识别到有效音符")
    return y


def _stft(y: np.ndarray) -> np.ndarray:
    win = np.hanning(N_FFT).astype(np.float32)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    return np.abs(np.fft.rfft(frames * win, axis=1))


def _flux(mag: np.ndarray) -> np.ndarray:
    d = np.diff(mag, axis=0)
    f = np.maximum(0.0, d).sum(axis=1)
    return np.concatenate([[0.0], f])


def _pick_onsets(flux: np.ndarray, min_gap_sec: float = 0.09, delta_ratio: float = 0.06) -> list[int]:
    if flux.size < 3:
        return []
    k = 17
    local = np.convolve(flux, np.ones(k) / k, mode="same")
    thr = local * 1.5 + float(flux.max()) * delta_ratio
    peak = np.zeros_like(flux, dtype=bool)
    peak[1:-1] = (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])
    cand = np.where(peak & (flux > thr))[0]

    gap = max(1, int(min_gap_sec * SR / HOP))
    out: list[int] = []
    for i in cand:
        if not out or i - out[-1] >= gap:
            out.append(int(i))
    return out


def _band_max(spec: np.ndarray, f: float, tol: float = 0.035) -> float:
    idx = np.where((FREQS >= f * (1 - tol)) & (FREQS <= f * (1 + tol)))[0]
    return float(spec[idx].max()) if idx.size else 0.0


def _note_score(spec: np.ndarray, midi: int) -> float:
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    total = 0.0
    for h, w in ((1, 1.0), (2, 0.45), (3, 0.2)):
        f = f0 * h
        if f < SR / 2:
            total += w * _band_max(spec, f)
    return total


def transcribe_handpan(y: np.ndarray) -> list[dict]:
    mag = _stft(y)
    flux = _flux(mag)
    onsets = _pick_onsets(flux)
    if not onsets:
        return []

    floor = float(mag.max()) * 0.02
    events: list[dict] = []

    for t in onsets:
        window = mag[t:t + 7]           # ~160ms，覆盖手碟的起振段
        if window.size == 0:
            continue
        spec = window.max(axis=0)
        scored = [(pos, _note_score(spec, midi)) for pos, midi in _NOTE_MIDI.items()]
        scored.sort(key=lambda x: -x[1])
        top = scored[0][1]
        if top < floor:
            continue

        picked = [(pos, s) for pos, s in scored[:2] if s >= top * 0.42]
        for pos, s in picked:
            events.append({
                "start": round(t * HOP / SR, 3),
                "pitch": _NOTE_MIDI[pos],
                "amp": round(s / top, 3),
                "pos_hint": pos,
            })

    # 同一时刻同一音位去重
    seen = set()
    uniq = []
    for e in events:
        key = (e["start"], e["pitch"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def _transcribe_basic_pitch(raw: bytes) -> list[dict]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(_to_wav(raw))
        path = f.name
    try:
        _, _, note_events = predict(
            path, ICASSP_2022_MODEL_PATH,
            onset_threshold=0.65, frame_threshold=0.35,
            minimum_note_length=80, minimum_frequency=100.0, maximum_frequency=1200.0,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return [{"start": float(s), "pitch": int(p), "amp": float(a)}
            for s, _e, p, a, _b in note_events if a >= 0.5]


def _to_wav(raw: bytes) -> bytes:
    exe = _ffmpeg_bin()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".media") as f:
        f.write(raw)
        src = f.name
    try:
        proc = subprocess.run([exe, "-v", "error", "-i", src, "-vn", "-ac", "1",
                               "-ar", "22050", "-f", "wav", "-"], capture_output=True)
        return proc.stdout
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass


def video_to_note_events(raw: bytes) -> list[dict]:
    engine = os.environ.get("TRANSCRIBE_ENGINE", "handpan").lower()
    if engine == "basic_pitch":
        return _transcribe_basic_pitch(raw)
    return transcribe_handpan(decode_audio(raw))


def bpm_from_audio(raw: bytes) -> float | None:
    """从 onset 包络的自相关估速度；估不准就让用户在前端手填（FR-11）。"""
    try:
        y = decode_audio(raw)
        flux = _flux(_stft(y))
        x = flux - flux.mean()
        if not np.any(x):
            return None
        ac = np.correlate(x, x, mode="full")[x.size - 1:]
        lo = int(60 / 240 * SR / HOP)
        hi = int(60 / 40 * SR / HOP)
        if hi >= ac.size:
            return None
        seg = ac[lo:hi]
        if seg.size == 0 or seg.max() <= 0:
            return None
        lag = lo + int(np.argmax(seg))
        bpm = 60.0 * (SR / HOP) / lag
        while bpm < 70:
            bpm *= 2
        while bpm > 200:
            bpm /= 2
        return float(round(bpm))
    except Exception:
        return None
