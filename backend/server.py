import asyncio
import json as _json
import numpy as np
from threading import Thread
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

import hashlib
from collections import OrderedDict

from .config import VOICE_DESCRIPTION, PLAY_STEPS_IN_S, CACHE_MAX_ENTRIES, TEMPERATURE
from .normalizer import normalize_text
from .chunker import split_into_sentences
from .tts_engine import engine

app = FastAPI()

_generation_lock = asyncio.Lock()

_audio_cache: OrderedDict[str, bytes] = OrderedDict()

_CONFIG_FINGERPRINT = hashlib.sha256(
    _json.dumps({
        "model": "ai4bharat/indic-parler-tts",
        "voice": VOICE_DESCRIPTION,
        "temperature": TEMPERATURE,
    }, sort_keys=True).encode()
).hexdigest()[:8]


def _cache_key(text: str) -> str:
    return hashlib.sha256((text + _CONFIG_FINGERPRINT).encode()).hexdigest()


def cache_get(text: str) -> bytes | None:
    key = _cache_key(text)
    if key in _audio_cache:
        _audio_cache.move_to_end(key)
        return _audio_cache[key]
    return None


def cache_put(text: str, pcm_bytes: bytes):
    key = _cache_key(text)
    _audio_cache[key] = pcm_bytes
    _audio_cache.move_to_end(key)
    while len(_audio_cache) > CACHE_MAX_ENTRIES:
        _audio_cache.popitem(last=False)

try:
    import pyloudnorm as pyln
    from scipy.signal import butter, sosfilt
    _POSTPROCESS_AVAILABLE = True
except ImportError:
    _POSTPROCESS_AVAILABLE = False

SILENCE_THRESHOLD = 0.005
FADE_SAMPLES = 480
TARGET_LUFS = -13.0
_HPF_SOS = None
_LUFS_METER = None


def _get_hpf_sos(sample_rate: int):
    global _HPF_SOS
    if _HPF_SOS is None:
        _HPF_SOS = butter(4, 60, btype='highpass', fs=sample_rate, output='sos')
    return _HPF_SOS


def _get_lufs_meter(sample_rate: int):
    global _LUFS_METER
    if _LUFS_METER is None:
        _LUFS_METER = pyln.Meter(sample_rate)
    return _LUFS_METER


def postprocess_chunk_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(samples) == 0:
        return samples

    window = min(256, len(samples))
    step = max(1, window // 4)
    end = len(samples)
    for i in range(len(samples) - window, -1, -step):
        rms = np.sqrt(np.mean(samples[i:i + window] ** 2))
        if rms > SILENCE_THRESHOLD:
            end = min(len(samples), i + window + int(0.04 * sample_rate))
            break
    else:
        end = 0

    samples = samples[:end]
    if len(samples) == 0:
        return samples

    fade_len = min(FADE_SAMPLES, len(samples) // 4)
    if fade_len > 0:
        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        samples[-fade_len:] *= fade

    return np.clip(samples, -1.0, 1.0)


def postprocess_full_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(samples) == 0:
        return samples
    if _POSTPROCESS_AVAILABLE:
        samples = sosfilt(_get_hpf_sos(sample_rate), samples).astype(np.float32)
    samples = postprocess_chunk_audio(samples, sample_rate)
    if len(samples) == 0:
        return samples
    if _POSTPROCESS_AVAILABLE:
        meter = _get_lufs_meter(sample_rate)
        loudness = meter.integrated_loudness(samples)
        if loudness > -70.0:
            samples = pyln.normalize.loudness(samples, loudness, TARGET_LUFS)
    return np.clip(samples, -1.0, 1.0)


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


async def async_stream_generate(prompt_text, description_text, play_steps_in_s=PLAY_STEPS_IN_S):
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    STOP = object()

    def producer():
        try:
            for audio_piece, t in engine.stream_generate(prompt_text, description_text, play_steps_in_s):
                loop.call_soon_threadsafe(q.put_nowait, (audio_piece, t))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, STOP)

    Thread(target=producer, daemon=True).start()

    while True:
        item = await q.get()
        if item is STOP:
            break
        yield item


@app.websocket("/ws")
async def ws_synthesize(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "config", "sample_rate": engine.sample_rate})
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") != "synthesize":
                continue

            text = msg.get("text", "")
            chunks_raw = split_into_sentences(text)
            chunks = [normalize_text(c) for c in chunks_raw]
            await websocket.send_json({
                "type": "chunks",
                "raw_chunks": chunks_raw,
                "chunks": chunks,
            })

            async with _generation_lock:
                for i, chunk_text in enumerate(chunks):
                    end_punct = chunk_text.strip()[-1] if chunk_text.strip() else '.'
                    cached_bytes = cache_get(chunk_text)
                    if cached_bytes is not None:
                        await websocket.send_json({"type": "chunk_start", "index": i, "text": chunk_text, "cached": True})
                        await websocket.send_bytes(cached_bytes)
                    else:
                        await websocket.send_json({"type": "chunk_start", "index": i, "text": chunk_text, "cached": False})
                        pieces = []
                        async for audio_piece, t in async_stream_generate(chunk_text, VOICE_DESCRIPTION):
                            clipped = np.clip(audio_piece, -1.0, 1.0)
                            await websocket.send_bytes((clipped * 32767).astype(np.int16).tobytes())
                            pieces.append(audio_piece)
                        if pieces:
                            full_audio = np.concatenate(pieces)
                            full_audio = postprocess_full_audio(full_audio, engine.sample_rate)
                            if len(full_audio) > 0:
                                cache_put(chunk_text, (full_audio * 32767).astype(np.int16).tobytes())
                    await websocket.send_json({"type": "chunk_end", "index": i, "end_punct": end_punct})

            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
