import queue
import numpy as np
import sounddevice as sd
import webrtcvad
from rich.console import Console

console = Console()

class AudioSegmenter:
    """
    Captures audio from the default (or specified) input device, applies WebRTC VAD,
    and yields utterance chunks as raw int16 PCM arrays at 16kHz mono.
    """
    def __init__(self, sample_rate=16000, frame_ms=30, vad_aggressiveness=2, input_device=None):
        assert frame_ms in (10, 20, 30), "frame_ms must be 10, 20, or 30"
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.input_device = input_device
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.q = queue.Queue()
        self.buffer = []
        self.silence_frames = 0
        self.max_silence_frames = int(0.6 * 1000 / frame_ms)  # ~600ms of silence ends an utterance

        self.stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            dtype="int16",
            blocksize=self.frame_len,
            device=input_device,
            callback=self._callback
        )

    def _callback(self, indata, frames, time, status):
        if status:
            console.log(f"[yellow]Audio status: {status}[/yellow]")
        self.q.put(indata.copy())

    def frames(self):
        """Generator of fixed-length frames."""
        with self.stream:
            while True:
                chunk = self.q.get()
                if chunk is None:
                    break
                # Ensure exact frame size
                data = chunk.reshape(-1)
                i = 0
                while i + self.frame_len <= len(data):
                    yield data[i:i+self.frame_len]
                    i += self.frame_len

    def utterances(self):
        """Generator of utterance chunks (np.int16 arrays)."""
        for frame in self.frames():
            is_speech = self.vad.is_speech(frame.tobytes(), sample_rate=self.sample_rate)
            self.buffer.append(frame)
            if is_speech:
                self.silence_frames = 0
            else:
                self.silence_frames += 1

            if self.silence_frames >= self.max_silence_frames and self.buffer:
                utt = np.concatenate(self.buffer).astype(np.int16)
                self.buffer = []
                self.silence_frames = 0
                yield utt

    def stop(self):
        self.q.put(None)
