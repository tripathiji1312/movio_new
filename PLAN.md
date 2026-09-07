# PLAN: SOTA Audio for Tamil / English (Tamil Accent) / Tanglish

> **Date:** 2026-09-08
> **Project:** Movio-2 — Cab Service AI Voice
> **Research log:** `research/findings-2026-09-08.md`

---

## 1. Objective & Success Criteria

**Objective:** Transform Movio-2's TTS audio quality from "functional" to "state-of-the-art" for Tamil, English (Tamil accent), and Tanglish (code-mixed Tamil-English), while maintaining low-latency streaming.

**Done looks like:**
- Tamil speech sounds natural with correct prosody, not robotic or monotone
- Tamil numbers are spoken in Tamil words (e.g., "ஆயிரத்து இருநூற்று ஐம்பது" not "one thousand two hundred fifty")
- Tanglish code-switching is handled gracefully (model handles this internally)
- Audio has consistent loudness (-16 LUFS ± 1), no DC offset, no trailing silence
- Time-to-first-audio is < 400ms (true streaming, not buffered)
- Voice cloning is available as an offline/batch mode via IndicF5

**Verification:**
- A/B listening test: before/after config changes on the same Tamil sentences
- LUFS measurement of output audio (target: -16 LUFS)
- TTFA measurement via frontend metrics panel
- Automated test: Tamil number "₹1250" → spoken in Tamil words

---

## 2. Locked Scope & Constraints

**IN scope:**
- Config parameter tuning for indic-parler-tts (biggest single quality win)
- Fix streaming bug (server buffers all chunks before sending)
- Tamil text normalization improvements (numbers in Tamil, Unicode normalization)
- Audio post-processing pipeline (HPF, LUFS normalization, crossfade)
- IndicF5 integration for offline voice cloning (non-streaming, batch mode)
- Production hardening (concurrency safety, health checks, cache invalidation)

**OUT of scope:**
- Replacing indic-parler-tts entirely (it's the best open-source Tamil TTS at 44.1kHz)
- Sarvam AI Bulbul V3 API integration (costs money, API dependency — can be added later)
- Neural speech enhancement (Resemble-Enhance — too slow for real-time, can be Phase 2 project)
- Custom model training/fine-tuning (requires GPU cluster + weeks of training)
- Romanized Tamil (Latin-script Tanglish like "Vanakkam") — model handles internally, normalizer passes through

**Why not replace with Sarvam/ElevenLabs?** API dependency for a cab service voice is a reliability risk. Indic-parler-tts runs locally, is Apache 2.0, and at 44.1kHz has higher sample rate than most alternatives (IndicF5=24kHz, F5-TTS=24kHz, XTTS=24kHz). The quality gap is primarily due to wrong config parameters, not model limitations.

---

## 3. Research Findings (evidence-backed)

### 3.1 Config Parameters Are Severely Wrong
- **TEMPERATURE=0.3** is way too low. Official Space and INFERENCE.md both use 1.0. This causes flat, robotic prosody. Source: `github.com/huggingface/parler-tts/blob/main/INFERENCE.md` streaming section; `huggingface.co/spaces/ai4bharat/indic-parler-tts/raw/main/app.py`
- **TOP_K=50** is not used in ANY official example. Codebook has 1024 entries; top_k=50 cuts 95% of valid continuations. Source: All official code examples verified.
- **REPETITION_PENALTY=1.15** penalizes natural codec token repetition, causing micro-stutters and pitch shifts. DAC uses 9 parallel codebook streams that all repeat naturally. Source: `parler-tts/dac_44khZ_8kbps/config.json` (num_codebooks: 9).
- **float16** risks NaN cascades; official Space uses bfloat16. Source: `app.py` at `torch_dtype = torch.bfloat16`.

### 3.2 Streaming is Defeated by Server Buffering
- `server.py:121-131`: The `async for` loop collects ALL pieces into `pieces` list, concatenates, THEN sends. TTFA = entire sentence generation time, not 300ms. Source: Direct code analysis.

### 3.3 Tamil Numbers Spoken in English
- `normalizer.py` always passes `lang="en"` to `indic_num2words` at lines 100, 104, 106, 115, 119, 139, 181. `contains_tamil()` exists at line 69 but is NEVER CALLED. Source: Direct code analysis.
- `indic_numtowords` fully supports Tamil: `num2words(42, lang='ta')` → "நாற்பத்து இரண்டு". Source: Live PyPI verification.
- Bug: `num2words(2.5, lang='ta')` raises `KeyError: '.5'`. Source: Live test.

### 3.4 IndicF5 is the Best Voice Cloning Option for Tamil
- `ai4bharat/IndicF5`: 0.4B params, MIT license, reference-audio cloning, 11 Indian languages including Tamil, by the same AI4Bharat team. Source: `huggingface.co/ai4bharat/IndicF5`.
- Does NOT support streaming. Requires reference audio as file path on disk. Model is gated (requires HF auth). Source: HuggingFace model page.

### 3.5 Audio Post-Processing is Minimal
- Only silence trimming + fade-out exists (`server.py:45-69`). No HPF, no LUFS normalization, no crossfade between chunks. Source: Direct code analysis.
- `pyloudnorm` (MIT, 782 stars) provides ITU-R BS.1770-4 LUFS normalization, CPU-only, <1ms. Source: `github.com/csteinmetz1/pyloudnorm`.

---

## 4. Chosen Approach & Why

**Primary engine stays as indic-parler-tts** with fixed parameters + post-processing pipeline.

| Alternative | Why rejected |
|---|---|
| Replace with Sarvam Bulbul V3 | API-only, costs ₹30/10K chars, 8kHz benchmark sample rate, reliability dependency |
| Replace with Fish Speech S2 Pro | Tamil is Tier 3 (quality unverified), 5B params (too large), restrictive license |
| Replace with IndicF5 entirely | 24kHz output (vs 44.1kHz), no streaming support, breaks existing UX |
| Replace with ElevenLabs | Commercial API, Tamil quality unverified, per-character pricing |
| Keep current config as-is | Config is provably wrong vs official defaults; single biggest quality bottleneck |

**IndicF5 added as secondary engine** for offline/batch voice cloning use cases (e.g., pre-generate a branded voice catalog).

---

## 5. Implementation Plan

### Phase 1: Fix Config Parameters & Cache Invalidation
**Estimated impact: ~60% of total quality improvement. This is the single biggest win.**

#### Task 1.1: Update `backend/config.py`

```python
# BEFORE (current):
TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
TEMPERATURE = 0.3
TOP_K = 50
REPETITION_PENALTY = 1.15
VOICE_DESCRIPTION = (
    "Jaya speaks Tamil and English in a calm, steady, natural, and friendly tone, "
    "at a clear, consistent, composed pace that is not rushed and not too fast. ..."
)

# AFTER (fixed):
TORCH_DTYPE = (
    torch.bfloat16 if DEVICE.startswith("cuda") and torch.cuda.is_bf16_supported()
    else torch.float16 if DEVICE.startswith("cuda")
    else torch.float32
)
TEMPERATURE = 1.0
# TOP_K: removed (no official example uses it)
# REPETITION_PENALTY: removed (codec tokens repeat naturally)
VOICE_DESCRIPTION = (
    "Jaya speaks with a slightly low-pitched voice at a moderate pace. "
    "The recording is very high quality, clear and close-sounding, with no background noise."
)
```

**Exact changes:**
- `config.py:5` — Add `torch.cuda.is_bf16_supported()` check before bfloat16 (prevents crash on V100/T4)
- `config.py:19` — TEMPERATURE: 0.3 → 1.0
- `config.py:20` — Delete TOP_K line
- `config.py:21` — Delete REPETITION_PENALTY line
- `config.py:8-14` — Replace VOICE_DESCRIPTION with shorter official-style description
- Remove `TOP_K` and `REPETITION_PENALTY` from the imports in `tts_engine.py:12-14`

#### Task 1.2: Update `backend/tts_engine.py` generation kwargs

```python
# In stream_generate(), remove top_k and repetition_penalty from gen_kwargs (lines 176-187):
gen_kwargs = dict(
    encoder_outputs=encoder_outputs,
    prompt_input_ids=prompt_ids.input_ids,
    prompt_attention_mask=prompt_ids.attention_mask,
    streamer=streamer,
    do_sample=True,
    temperature=TEMPERATURE,
    min_new_tokens=min_new_tokens,
    max_new_tokens=estimated_max_tokens,
)
```

#### Task 1.3: Update warmup to include both Tamil and English

```python
# In _warmup() (tts_engine.py:124-137), add English warmup:
def _warmup(self):
    for text in ["வணக்கம்.", "Hello, welcome."]:
        desc = self.description_tokenizer(VOICE_DESCRIPTION, return_tensors="pt").to(DEVICE)
        prompt = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            _ = self.model.generate(
                input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
                max_new_tokens=50,
            )
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    print("Warm-up complete.")
```

#### Task 1.4: Flush audio cache on startup

```python
# In server.py, at module level after _audio_cache definition (line 19):
# Cache is automatically empty on startup (OrderedDict()).
# Add config fingerprint to cache keys to prevent stale entries:
import json as _json

_CONFIG_FINGERPRINT = hashlib.sha256(
    _json.dumps({
        "model": "ai4bharat/indic-parler-tts",
        "voice": VOICE_DESCRIPTION,
        "temperature": TEMPERATURE,
    }, sort_keys=True).encode()
).hexdigest()[:8]

def _cache_key(text: str) -> str:
    return hashlib.sha256((text + _CONFIG_FINGERPRINT).encode()).hexdigest()
```

#### Task 1.5: Update `requirements.txt`
Add `parler-tts` and `transformers` which are currently missing:
```
accelerate
sentencepiece
soundfile
indic-numtowords
fastapi
uvicorn[standard]
numpy
torch
transformers>=4.46.1
parler-tts
```

**Verification:** Generate "உங்கள் கேப் 5 நிமிடத்தில் வரும்" before and after. Listen for prosody improvement (should sound noticeably more natural and less robotic).

---

### Phase 2: Fix Streaming + Concurrency Safety
**Estimated impact: TTFA improvement from ~3-5s to ~300ms for first audio.**

#### Task 2.1: Fix server-side streaming (`backend/server.py`)

Replace the current buffered approach (lines 119-131) with true streaming:

```python
# BEFORE (buffers everything):
else:
    await websocket.send_json({"type": "chunk_start", "index": i, "text": chunk_text, "cached": False})
    pieces = []
    async for audio_piece, t in async_stream_generate(chunk_text, VOICE_DESCRIPTION):
        pieces.append(audio_piece)
    if pieces:
        full_audio = np.concatenate(pieces)
        full_audio = postprocess_chunk_audio(full_audio, engine.sample_rate)
        ...

# AFTER (sends each sub-chunk immediately):
else:
    await websocket.send_json({"type": "chunk_start", "index": i, "text": chunk_text, "cached": False})
    pieces = []
    async for audio_piece, t in async_stream_generate(chunk_text, VOICE_DESCRIPTION):
        clipped = np.clip(audio_piece, -1.0, 1.0)  # Keep quality gate
        pcm16 = (clipped * 32767).astype(np.int16)
        await websocket.send_bytes(pcm16.tobytes())
        pieces.append(audio_piece)
    # Post-generate: postprocess full audio for cache only
    if pieces:
        full_audio = np.concatenate(pieces)
        full_audio = postprocess_chunk_audio(full_audio, engine.sample_rate)
        if len(full_audio) > 0:
            cache_put(chunk_text, (full_audio * 32767).astype(np.int16).tobytes())
```

Key decisions:
- Every streamed sub-chunk gets `np.clip(-1.0, 1.0)` to prevent clipping artifacts
- Full postprocessing (silence trim + fade) runs AFTER generation, result goes to cache only
- Cached playback still sends the postprocessed version (existing path, line 117-118)

#### Task 2.2: Fix frontend to play sub-chunks incrementally (`frontend/index.html`)

The frontend must change `handleAudioFrame` and `scheduleSentence` to play audio progressively:

```javascript
// Replace handleAudioFrame to schedule each sub-chunk immediately:
function handleAudioFrame(buf) {
    if (!firstByteRecorded) {
        firstByteRecorded = true;
        valTtfa.textContent = ((performance.now() - synthStart) / 1000).toFixed(2) + 's';
        subTtfa.textContent = 'Confirmed';
        metricTtfa.classList.add('highlight');
    }
    const int16 = new Int16Array(buf);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const trimmed = trimAudioSilence(float32, sampleRate);
    if (trimmed.length === 0) return;

    // Store for full player later
    currentChunkPieces.push(trimmed);

    // Schedule immediate playback
    const buffer = audioCtx.createBuffer(1, trimmed.length, sampleRate);
    buffer.getChannelData(0).set(trimmed);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    const gainNode = audioCtx.createGain();
    source.connect(gainNode);
    gainNode.connect(analyser);

    const startAt = Math.max(nextBoundary, audioCtx.currentTime);
    gainNode.gain.setValueAtTime(1, startAt);
    source.start(startAt);
    isPlaying = true;

    nextBoundary = startAt + buffer.duration;
    lastGainNode = gainNode;
}
```

Update `scheduleSentence` to skip playback (already played) and only handle marking + storage:

```javascript
function scheduleSentence(index) {
    if (currentChunkPieces.length === 0) { markChunk(index, 'done'); return; }
    let totalLen = 0;
    for (const p of currentChunkPieces) totalLen += p.length;
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const p of currentChunkPieces) { merged.set(p, offset); offset += p.length; }
    currentChunkPieces = [];
    storedSentences[index] = merged;
    // Mark chunk done after a delay matching remaining playback
    const remaining = Math.max(0, (nextBoundary - audioCtx.currentTime) * 1000);
    scheduledTimeouts.push(setTimeout(() => {
        markChunk(index, chunkCachedFlags[index] ? 'cached' : 'done');
    }, remaining));
}
```

#### Task 2.3: Add concurrency guard (`backend/server.py`)

```python
# At module level, after app definition:
_generation_lock = asyncio.Lock()

# Wrap the generation loop in ws_synthesize:
async with _generation_lock:
    for i, chunk_text in enumerate(chunks):
        # ... existing generation code ...
```

This prevents two concurrent WebSocket handlers from corrupting GPU state.

#### Task 2.4: Add punctuation metadata to chunk_end message

```python
# In the chunk loop, detect ending punctuation:
for i, chunk_text in enumerate(chunks):
    end_punct = chunk_text.strip()[-1] if chunk_text.strip() else '.'
    # ... generation code ...
    await websocket.send_json({
        "type": "chunk_end",
        "index": i,
        "end_punct": end_punct,  # NEW: for frontend crossfade timing
    })
```

**Verification:** Measure TTFA before/after using the frontend metrics panel. Before: ~3-5s. After: ~300-500ms.

---

### Phase 3: Fix Tamil Text Normalization
**Estimated impact: Correct Tamil number pronunciation, proper Unicode handling.**

#### Task 3.1: Add indic-nlp-library dependency

```bash
pip install indic-nlp-library
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git /path/to/indic_nlp_resources
export INDIC_RESOURCES_PATH=/path/to/indic_nlp_resources
```

Add to `requirements.txt`:
```
indic-nlp-library>=0.92
```

**Note:** The `INDIC_RESOURCES_PATH` env var must be set before importing the library. Add to `backend/config.py`:
```python
import os
os.environ.setdefault(
    "INDIC_RESOURCES_PATH",
    os.path.join(os.path.dirname(__file__), "..", "indic_nlp_resources")
)
```

The `indic_nlp_resources` directory should be cloned into the project root (add to `.gitignore` or include as a git submodule).

#### Task 3.2: Wire `contains_tamil()` into normalization pipeline (`backend/normalizer.py`)

The existing `contains_tamil()` at line 69 is dead code. Wire it into every function that produces spoken words:

```python
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

_tamil_normalizer = IndicNormalizerFactory().get_normalizer('ta')

def normalize_text(text: str) -> str:
    is_tamil = contains_tamil(text)

    # Step 0: Tamil Unicode normalization (ZWJ/ZWNJ, compound chars)
    if is_tamil:
        text = _tamil_normalizer.normalize(text)

    # Step 1: Tamil numeral conversion (௧-௯ → 1-9)
    if is_tamil:
        text = convert_tamil_numerals(text)

    text = apply_pronunciation_overrides(text)
    text = normalize_tagged_text(text)
    text = heuristic_normalize(text, lang='ta' if is_tamil else 'en')
    return text
```

#### Task 3.3: Add Tamil numeral conversion

```python
import re

TAMIL_NUMERAL_RE = re.compile(r'[௦-௯]+')

def convert_tamil_numerals(text: str) -> str:
    """Convert Tamil digit characters (௦-௯) to Arabic digits (0-9).
    NOTE: Only handles simple digit sequences, not multiplicative Tamil numerals (௰=10, ௱=100, ௲=1000).
    Multiplicative numerals are extremely rare in modern Tamil text."""
    def _replace(m):
        return ''.join(str(ord(c) - 0x0BE6) for c in m.group(0))
    return TAMIL_NUMERAL_RE.sub(_replace, text)
```

#### Task 3.4: Fix number-to-words language selection

Modify `spell_amount()`, `spell_time()`, `spell_digits()` and `heuristic_normalize()` to accept a `lang` parameter:

```python
def spell_amount(num_str: str, lang: str = "en") -> str:
    cleaned = re.sub(r'[^\d.]', '', str(num_str)).strip()
    if not cleaned:
        return num_str
    parts = cleaned.split('.')
    whole = parts[0] or '0'
    result = indic_num2words(int(whole), lang=lang)
    if len(parts) > 1 and parts[1]:
        decimal = parts[1]
        point_word = "புள்ளி" if lang == 'ta' else "point"
        result += f" {point_word} " + " ".join(
            indic_num2words(int(d), lang=lang) for d in decimal
        )
    return result

def heuristic_normalize(text: str, lang: str = "en") -> str:
    # ... existing logic but pass lang to spell_amount:
    text = SHORT_NUM_RE.sub(lambda m: spell_amount(m.group(0), lang=lang), text)
    # ... etc
```

#### Task 3.5: Add Tamil abbreviation overrides

```python
TAMIL_PRONUNCIATION_OVERRIDES = {
    "செ.மீ": "சென்டிமீட்டர்",
    "கி.மீ": "கிலோமீட்டர்",
    "ரூ": "ரூபாய்",
    "திரு": "திருவாளர்",
    "திருமதி": "திருமதி",
    "முனைவர்": "முனைவர்",
}
```

**Verification:** Test: `normalize_text("உங்கள் கட்டணம் ₹1250")` → should contain Tamil words for 1250, not English. Test: `normalize_text("5 கி.மீ தூரத்தில்")` → should expand "கி.மீ".

---

### Phase 4: Audio Post-Processing Pipeline
**Estimated impact: Consistent loudness, cleaner audio, smoother transitions.**

#### Task 4.1: Add pyloudnorm dependency

```bash
pip install pyloudnorm
```

Add to `requirements.txt`:
```
pyloudnorm>=0.1.0
```

This brings `scipy` as a transitive dependency.

#### Task 4.2: Add post-processing pipeline (`backend/server.py`)

Add new functions after existing `postprocess_chunk_audio`:

```python
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

# High-pass filter to remove DC offset and sub-bass rumble
_HPF_SOS = None  # Lazy init after sample_rate is known

def _get_hpf_sos(sample_rate: int):
    global _HPF_SOS
    if _HPF_SOS is None:
        _HPF_SOS = butter(4, 60, btype='highpass', fs=sample_rate, output='sos')
    return _HPF_SOS

_LUFS_METER = None

def _get_lufs_meter(sample_rate: int):
    global _LUFS_METER
    if _LUFS_METER is None:
        _LUFS_METER = pyln.Meter(sample_rate)
    return _LUFS_METER

TARGET_LUFS = -16.0

def postprocess_full_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Full post-processing pipeline for cached/final audio."""
    if len(samples) == 0:
        return samples

    # 1. High-pass filter at 60Hz (remove DC offset, sub-bass rumble)
    samples = sosfilt(_get_hpf_sos(sample_rate), samples).astype(np.float32)

    # 2. Existing silence trim + fade-out
    samples = postprocess_chunk_audio(samples, sample_rate)
    if len(samples) == 0:
        return samples

    # 3. LUFS normalization to -16 LUFS
    meter = _get_lufs_meter(sample_rate)
    loudness = meter.integrated_loudness(samples)
    if loudness > -70.0:  # Only normalize if signal is present
        samples = pyln.normalize.loudness(samples, loudness, TARGET_LUFS)

    # 4. Final clip to prevent any overshoot from normalization
    return np.clip(samples, -1.0, 1.0)
```

Update the generation loop to use `postprocess_full_audio` for cached audio:
```python
# Replace: full_audio = postprocess_chunk_audio(full_audio, engine.sample_rate)
# With:    full_audio = postprocess_full_audio(full_audio, engine.sample_rate)
```

#### Task 4.3: Frontend crossfade with variable pauses (`frontend/index.html`)

Update the sentence gap between chunks based on punctuation:

```javascript
// In handleControl, when receiving chunk_end:
} else if (msg.type === 'chunk_end') {
    const punct = msg.end_punct || '.';
    const pauseMs = punct === ',' ? 80 : punct === '?' ? 200 : punct === '!' ? 180 : 150;
    // Add pause to nextBoundary
    nextBoundary += pauseMs / 1000;
    scheduleSentence(msg.index);
}
```

Replace the hard-coded 100ms pause in `buildFullAudioPlayer`:
```javascript
// Instead of: const pauseSamples = Math.floor(0.10 * sampleRate);
// Use variable pause stored per-chunk (would need chunkEndPuncts array)
```

**Verification:** Measure LUFS of output WAV file using: `pyloudnorm.Meter(44100).integrated_loudness(audio)`. Should be -16 ± 1 LUFS.

---

### Phase 5: IndicF5 Voice Cloning (Offline/Batch Mode)
**This is a secondary engine for offline voice cloning, NOT a streaming replacement.**

#### Task 5.1: Pre-requisites

IndicF5 is a **gated model** on HuggingFace. Before installing:
1. Accept the model terms at `https://huggingface.co/ai4bharat/IndicF5`
2. Set HuggingFace auth token: `huggingface-cli login`

```bash
pip install git+https://github.com/ai4bharat/IndicF5.git
```

**WARNING:** IndicF5 pins `numpy<=1.26.4`. This may conflict with other dependencies. Install in a separate venv or use dependency resolution:
```bash
pip install git+https://github.com/ai4bharat/IndicF5.git --no-deps
pip install -r requirements.txt  # Then resolve conflicts manually
```

Add to `requirements.txt` with a comment:
```
# Optional: voice cloning engine (gated model, requires HF auth)
# pip install git+https://github.com/ai4bharat/IndicF5.git
```

#### Task 5.2: Create `backend/tts_engine_f5.py`

```python
"""IndicF5 voice cloning engine — offline/batch mode only.
NOT streaming-capable. Use for pre-generating audio with cloned voices.
"""
import numpy as np
from pathlib import Path

class F5Engine:
    def __init__(self):
        self.model = None
        self.sample_rate = 24000  # IndicF5 outputs 24kHz

    def load(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            "ai4bharat/IndicF5", trust_remote_code=True
        )
        print("IndicF5 loaded (24kHz, offline voice cloning)")

    def generate(
        self,
        text: str,
        ref_audio_path: str,  # Path to .wav file on disk
        ref_text: str,        # Transcript of the reference audio
    ) -> np.ndarray:
        """Generate speech with cloned voice. Returns float32 numpy array at 24kHz."""
        if self.model is None:
            raise RuntimeError("F5Engine not loaded. Call load() first.")
        audio = self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)
        if hasattr(audio, 'numpy'):
            audio = audio.numpy()
        return audio.astype(np.float32)
```

#### Task 5.3: Add REST endpoint for batch voice cloning (`backend/server.py`)

IndicF5 takes a **file path**, not binary data. Design:
- Pre-store reference voices in `voices/` directory
- REST endpoint accepts voice name + text, returns WAV

```python
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import io, struct

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

@app.post("/api/clone")
async def clone_voice(request: dict):
    """Batch voice cloning via IndicF5. NOT streaming.
    Body: {"text": "...", "voice": "voice_name", "ref_text": "..."}
    Response: WAV audio file
    """
    text = request.get("text", "")
    voice_name = request.get("voice", "")
    ref_text = request.get("ref_text", "")

    ref_path = VOICES_DIR / f"{voice_name}.wav"
    if not ref_path.exists():
        raise HTTPException(404, f"Voice '{voice_name}' not found")

    # Import lazily to avoid crash if IndicF5 not installed
    from .tts_engine_f5 import f5_engine
    audio = f5_engine.generate(text, str(ref_path), ref_text)

    # Encode as WAV
    buf = io.BytesIO()
    # ... WAV encoding ...
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")
```

#### Task 5.4: Create `voices/` directory structure

```
voices/
  README.md  # Instructions for adding reference voices
  # Place .wav files here: <name>.wav + <name>.txt (transcript)
```

**Verification:** Place a 10-30 second Tamil reference audio in `voices/test_voice.wav` with transcript in `voices/test_voice.txt`. POST to `/api/clone` with test text. Compare voice similarity to reference.

---

### Phase 6: Production Hardening

#### Task 6.1: Health check endpoint

```python
@app.get("/health")
async def health():
    return {
        "status": "ok" if engine.model is not None else "loading",
        "model": "ai4bharat/indic-parler-tts",
        "sample_rate": engine.sample_rate,
        "device": str(engine.model.device) if engine.model else None,
    }
```

#### Task 6.2: Make seed configurable

```python
# config.py:
GENERATION_SEED = int(os.environ.get("GENERATION_SEED", "0"))  # 0 = random
```

```python
# tts_engine.py, in _run():
def _run(**kwargs):
    seed = GENERATION_SEED if GENERATION_SEED > 0 else int(time.time() * 1000) % 2**31
    torch.manual_seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed(seed)
    with torch.inference_mode():
        self.model.generate(**kwargs)
```

#### Task 6.3: GPU memory monitoring

```python
# In server.py, add to health endpoint or as middleware:
import torch

def gpu_stats():
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_mem / 1024**2,
    }
```

#### Task 6.4: Static file serving note

Add comment to `server.py:144`:
```python
# PRODUCTION NOTE: In production, serve static files via nginx/Caddy reverse proxy
# instead of FastAPI. This frees the Python process for WebSocket/API work.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
```

**Verification:** Hit `/health` endpoint. Check GPU memory after 10 generations. Verify random seed produces varied audio for same text.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **bfloat16 not supported on target GPU** | Medium | Build fails | `torch.cuda.is_bf16_supported()` check with float16 fallback (Task 1.1) |
| **Temperature=1.0 too variable for cab service** | Low | Inconsistent voice | Tune down to 0.85-0.9 if A/B test shows issues; fixed seed provides consistency |
| **`indic_numtowords` decimal bug in Tamil** | High for decimal amounts | Crash | Digit-by-digit approach with "புள்ளி" for decimal point (Task 3.4) |
| **`indic-nlp-library` INDIC_RESOURCES_PATH not set** | High if skipped | Import crash | Default path in config.py + setup instructions (Task 3.1) |
| **IndicF5 numpy version conflict** | High | Dependency hell | Install with `--no-deps`, document in requirements.txt as optional (Task 5.1) |
| **Concurrent WebSocket requests corrupt GPU state** | High without fix | Wrong audio / crash | `asyncio.Lock` in Phase 2 (Task 2.3) |
| **Stale cache entries after config change** | Certain | Old bad audio served | Config fingerprint in cache key (Task 1.4) |
| **Streaming sub-chunks have no silence trim** | Medium | Trailing noise | `np.clip` on every sub-chunk + full postprocess for cache (Task 2.1) |

---

## 7. Verification Checklist

Run these top-to-bottom after all phases:

- [ ] **Config sanity:** `config.py` has TEMPERATURE=1.0, no TOP_K, no REPETITION_PENALTY
- [ ] **bfloat16 check:** Model loads without error on target GPU
- [ ] **A/B test:** Generate "உங்கள் கேப் 5 நிமிடத்தில் வரும்" — sounds natural, not robotic
- [ ] **Tamil numbers:** "₹1250" spoken as Tamil words ("ஆயிரத்து இருநூற்று ஐம்பது")
- [ ] **Tamil numeral chars:** "௫ நிமிடம்" correctly converted and spoken
- [ ] **Tanglish:** "Unga order dispatch aayiduchu" generates audio without crash
- [ ] **TTFA metric:** First audio plays within 500ms of clicking Speak (not 3-5s)
- [ ] **Concurrency:** Two browser tabs sending simultaneously — second queues, no crash
- [ ] **LUFS:** Download WAV, measure loudness — should be -16 LUFS ± 1
- [ ] **Health endpoint:** `GET /health` returns status=ok with model details
- [ ] **Cache invalidation:** Change VOICE_DESCRIPTION → old cache entries not served
- [ ] **Crossfade:** Listen to multi-sentence text — no jarring gaps between sentences
- [ ] **IndicF5 (optional):** `POST /api/clone` with reference audio → returns cloned voice

---

## 8. Open Questions

1. **Target deployment GPU?** The bfloat16 optimization and overall latency budget depend on whether this runs on T4, A10G, A100, or H100. If T4/V100, bfloat16 will fall back to float16 automatically, but latency expectations should be adjusted.

2. **Desired voice variety vs consistency?** Currently `GENERATION_SEED=42` (fixed) makes every generation deterministic. Phase 6 makes it configurable. For a cab service, should every call sound exactly the same (seed=42) or slightly different (seed=random)? This is a product decision.

3. **Sarvam AI Bulbul V3 as future upgrade?** If budget allows ₹30/10K chars, Sarvam's API has the best verified Tanglish quality. This could be a Phase 2 project as a premium tier alongside the local model.

4. **`indic_nlp_resources` — bundle or git submodule?** The ~200MB resources repository needs to be available at runtime. Bundle in Docker image? Git submodule? Download at startup? This is a deployment decision.

5. **IndicF5 reference voice recording requirements?** For the voice cloning feature, who records the reference audio? What quality level (studio mic vs phone recording)? The Rasa dataset (training data) was recorded at 48kHz with studio equipment. Minimum recommended: 10-30 seconds, clean audio, close-mic.
