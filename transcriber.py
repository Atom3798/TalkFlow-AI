import torch
from faster_whisper import WhisperModel
import numpy as np
from typing import Tuple

class Transcriber:
    def __init__(self, model_size: str = "medium", device: str = "auto"):
        # Prefer CUDA if available when device='auto'
        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            device_arg = "cuda"
            compute_type = "float16"
        else:
            device_arg = "cpu"
            compute_type = "int8"

        self.model = WhisperModel(model_size, device=device_arg, compute_type=compute_type)

    def transcribe(self, pcm16: np.ndarray) -> Tuple[str, str]:
        # pcm16: int16 mono 16kHz
        audio = pcm16.astype("float32") / 32768.0
        segments, info = self.model.transcribe(audio, beam_size=5, vad_filter=False)
        text = "".join(seg.text for seg in segments).strip()
        lang = info.language or "auto"
        return text, lang
