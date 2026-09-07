import time
import queue
import numpy as np
import torch
from threading import Thread
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import BaseModelOutput
from parler_tts import ParlerTTSForConditionalGeneration

from .config import DEVICE, TORCH_DTYPE, MODEL_ID, VOICE_DESCRIPTION, CHARS_PER_SEC


class ParlerTTSStreamerLocal(BaseStreamer):
    def __init__(self, model, device=None, play_steps=10, stride=None, timeout=None):
        self.model = model
        self.audio_encoder = model.audio_encoder
        self.generation_config = model.generation_config
        self.device = device if device is not None else model.device
        self.play_steps = play_steps

        hop_length = np.prod(self.audio_encoder.config.upsampling_ratios)
        self.stride = stride if stride is not None else hop_length * (play_steps - 1)

        self.token_cache = None
        self.to_yield = 0
        self.audio_queue = queue.Queue()
        self.stop_signal = None
        self.timeout = timeout

    def apply_delay_pattern_mask(self, input_ids):
        _, delay_pattern_mask = self.model.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        input_ids = self.model.decoder.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        input_ids = input_ids[:, 1:]
        first_valid = (input_ids[0, :] != self.generation_config.pad_token_id).nonzero()[0, 0]
        input_ids = input_ids[..., first_valid:]
        output_values = self.audio_encoder.decode(
            input_ids.unsqueeze(0).unsqueeze(0),
            audio_scales=[None],
        )
        audio_values = output_values.audio_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0]
        if batch_size > 1:
            raise ValueError("ParlerTTSStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0, :, None]

        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            self.on_finalized_audio(audio_values[self.to_yield:-self.stride])
            self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
        else:
            audio_values = np.zeros(self.to_yield)
        self.on_finalized_audio(audio_values[self.to_yield:], stream_end=True)

    def on_finalized_audio(self, audio, stream_end=False):
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        return value


def _get_streamer_class():
    try:
        from parler_tts import ParlerTTSStreamer
        return ParlerTTSStreamer
    except ImportError:
        return ParlerTTSStreamerLocal


class TTSEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.sample_rate = None
        self._description_cache = {}
        self._streamer_cls = _get_streamer_class()

    def load(self):
        print(f"Loading model {MODEL_ID} on {DEVICE} ({TORCH_DTYPE})...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            attn_implementation="eager",
        ).to(DEVICE)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self.sample_rate = self.model.config.sampling_rate
        print(f"Model loaded. Sample rate: {self.sample_rate}")
        self._warmup()

    def _warmup(self):
        desc = self.description_tokenizer(VOICE_DESCRIPTION, return_tensors="pt").to(DEVICE)
        prompt = self.tokenizer("வணக்கம்.", return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            _ = self.model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
                max_new_tokens=50,
            )
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        print("Warm-up complete.")

    def _get_cached_encoder_outputs(self, description_text: str):
        if description_text not in self._description_cache:
            desc_ids = self.description_tokenizer(description_text, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                hidden = self.model.get_text_encoder()(
                    input_ids=desc_ids.input_ids,
                    attention_mask=desc_ids.attention_mask,
                    return_dict=True,
                ).last_hidden_state
                if (
                    self.model.text_encoder.config.hidden_size != self.model.decoder.config.hidden_size
                    and self.model.decoder.config.cross_attention_hidden_size is None
                ):
                    hidden = self.model.enc_to_dec_proj(hidden)
                hidden = hidden * desc_ids.attention_mask[..., None]
            self._description_cache[description_text] = hidden
        return self._description_cache[description_text]

    def stream_generate(self, prompt_text: str, description_text: str, play_steps_in_s: float = 0.5):
        frame_rate = self.model.audio_encoder.config.frame_rate
        play_steps = max(1, int(frame_rate * play_steps_in_s))

        streamer = self._streamer_cls(self.model, device=DEVICE, play_steps=play_steps)

        prompt_clean = prompt_text.strip()
        if not prompt_clean.endswith(('.', '!', '?', '।')):
            prompt_clean += '.'

        prompt_ids = self.tokenizer(prompt_clean, return_tensors="pt").to(DEVICE)

        estimated_seconds = len(prompt_clean) / CHARS_PER_SEC
        min_new_tokens = max(10, int(frame_rate * estimated_seconds * 0.85))
        estimated_max_tokens = int(frame_rate * estimated_seconds * 1.6)

        cached_desc = self._get_cached_encoder_outputs(description_text)
        encoder_outputs = BaseModelOutput(last_hidden_state=cached_desc)

        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.25,
            min_new_tokens=min_new_tokens,
            max_new_tokens=estimated_max_tokens,
        )

        def _run(**kwargs):
            with torch.inference_mode():
                self.model.generate(**kwargs)

        thread = Thread(target=_run, kwargs=gen_kwargs)
        start = time.time()
        thread.start()

        for chunk in streamer:
            yield chunk, time.time() - start

        thread.join()


engine = TTSEngine()
