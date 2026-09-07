import os
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = (
    torch.bfloat16 if DEVICE.startswith("cuda") and torch.cuda.is_bf16_supported()
    else torch.float16 if DEVICE.startswith("cuda")
    else torch.float32
)
MODEL_ID = "ai4bharat/indic-parler-tts"

VOICE_DESCRIPTION = (
    "Jaya speaks with a slightly low-pitched voice at a moderate pace. "
    "The recording is very high quality, clear and close-sounding, with no background noise."
)

CHARS_PER_SEC = 13.0
MAX_CHUNK_CHARS = 100
PLAY_STEPS_IN_S = 0.3
GENERATION_SEED = 42
TEMPERATURE = 1.0
CACHE_MAX_ENTRIES = 200
PORT = int(os.environ.get("PORT", "8000"))
