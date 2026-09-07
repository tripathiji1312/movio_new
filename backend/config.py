import os
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
MODEL_ID = "ai4bharat/indic-parler-tts"

VOICE_DESCRIPTION = (
    "Jaya speaks Tamil and English in a calm, steady, natural, and friendly tone, "
    "at a clear, consistent, composed pace that is not rushed and not too fast. "
    "Her volume is even and steady throughout, with pleasant, relaxed intonation. "
    "Her pronunciation is very clear and easy to understand. "
    "The recording is very high quality, close-up, and completely free of background noise."
)

CHARS_PER_SEC = 13.0
MAX_CHUNK_CHARS = 100
PLAY_STEPS_IN_S = 0.3
PORT = int(os.environ.get("PORT", "8000"))
