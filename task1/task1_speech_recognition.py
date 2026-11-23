import torch
from transformers import pipeline

transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    dtype=torch.float16,
)

audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

print(transcriber(audio_url)["text"])
