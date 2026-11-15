import os
import time

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration


def audio_stats(audio, rate):
    """Basic measurements of a mono float waveform."""
    finite = bool(np.isfinite(audio).all())
    clean = np.nan_to_num(audio)
    frame = int(0.02 * rate)  # 20 ms frames
    n = len(clean) // frame
    frames = clean[: n * frame].reshape(n, frame)
    frame_rms = np.sqrt((frames ** 2).mean(axis=1))
    return {
        "peak": float(np.abs(clean).max()),
        "rms": float(np.sqrt((clean ** 2).mean())),
        "silence_fraction": float((frame_rms < 0.01).mean()),
        "all_finite": finite,
    }


class MusicPredictor:
    MODEL = "facebook/musicgen-small"
    DEFAULT_REVISION = "4c8334b02c6ec4e8664a91979669a501ec497792"
    TOKENS_PER_SECOND = 50
    MIN_SECONDS = 1
    MAX_SECONDS = 10
    MAX_PROMPT_CHARS = 300

    def __init__(self, revision=None):
        self.revision = revision or self.DEFAULT_REVISION
        self.processor = None
        self.model = None

    def load(self):
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(self.MODEL, revision=self.revision)
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.MODEL, revision=self.revision
            )
            self.model.eval()

    @classmethod
    def validate(cls, prompt, seconds, seed):
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        prompt = prompt.strip()
        if len(prompt) > cls.MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt has {len(prompt)} characters; limit is {cls.MAX_PROMPT_CHARS}.")
        if type(seconds) not in (int, float) or not (cls.MIN_SECONDS <= seconds <= cls.MAX_SECONDS):
            raise ValueError(f"Duration must be a number from {cls.MIN_SECONDS} to {cls.MAX_SECONDS} seconds.")
        if type(seed) is not int:
            raise ValueError("Seed must be an integer.")
        return prompt

    def generate(self, prompt, seconds=5, seed=0, guidance_scale=3.0, output_path=None):
        prompt = self.validate(prompt, seconds, seed)
        self.load()
        torch.manual_seed(seed)
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
        max_new_tokens = int(round(seconds * self.TOKENS_PER_SECOND))

        start = time.perf_counter()
        out = self.model.generate(
            **inputs, do_sample=True, guidance_scale=guidance_scale, max_new_tokens=max_new_tokens
        )
        gen_seconds = time.perf_counter() - start

        rate = int(self.model.config.audio_encoder.sampling_rate)
        audio = out[0, 0].cpu().numpy().astype(np.float32)
        duration = len(audio) / rate

        if output_path:
            folder = os.path.dirname(output_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            pcm = (np.clip(np.nan_to_num(audio), -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(output_path, rate, pcm)

        record = {
            "prompt": prompt,
            "model": self.MODEL,
            "revision": self.revision,
            "seconds_requested": seconds,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "sample_rate": rate,
            "num_samples": int(len(audio)),
            "duration_seconds": duration,
            "generation_seconds": gen_seconds,
            "real_time_factor": gen_seconds / duration,
            "output_path": output_path,
            **audio_stats(audio, rate),
        }
        return audio, record