from dataclasses import dataclass

@dataclass
class AppConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    vad_aggressiveness: int = 2  # 0..3
    target_lang: str = "en"
    model_size: str = "medium"  # tiny, base, small, medium, large-v3
    device: str = "auto"        # auto|cpu|cuda
    port: int = 7860
    input_device: int | None = None  # sounddevice device index
