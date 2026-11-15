import time
import torch
from scipy.io import wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

torch.manual_seed(0)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(text=["calm piano melody"], padding=True, return_tensors="pt")
start = time.perf_counter()
audio = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
seconds = time.perf_counter() - start

rate = model.config.audio_encoder.sampling_rate
wavfile.write("smoke.wav", rate, audio[0, 0].numpy())
print("sample rate:", rate, "| samples:", audio.shape[-1], "| generation seconds:", round(seconds, 1))