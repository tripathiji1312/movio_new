import asyncio
import numpy as np
from threading import Thread
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .config import VOICE_DESCRIPTION, PLAY_STEPS_IN_S
from .normalizer import normalize_text
from .chunker import split_into_sentences
from .tts_engine import engine

app = FastAPI()

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

            for i, chunk_text in enumerate(chunks):
                await websocket.send_json({"type": "chunk_start", "index": i, "text": chunk_text})
                async for audio_piece, t in async_stream_generate(chunk_text, VOICE_DESCRIPTION):
                    pcm16 = np.clip(audio_piece, -1.0, 1.0)
                    pcm16 = (pcm16 * 32767).astype(np.int16)
                    await websocket.send_bytes(pcm16.tobytes())
                await websocket.send_json({"type": "chunk_end", "index": i})

            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
