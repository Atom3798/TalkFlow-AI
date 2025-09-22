# AI Live Translator (Python)

A lightweight, local-first AI agent that provides live speech **transcription** and **translation** during online meetings (Zoom, Google Meet, etc.).
It listens to your microphone, detects speech with VAD, transcribes locally with [faster-whisper], and translates with Meta's **M2M100** model via 🤗 Transformers.
It serves a minimal caption web UI that you can share on-screen or window-capture in any meeting app.

> No keys required. Works fully offline after models download (first run will fetch models).

## Features
- Low-latency VAD-based chunking (WebRTC VAD)
- Accurate transcription (faster-whisper)
- Multilingual translation (facebook/m2m100_418M)
- Local web caption page via Flask + Socket.IO
- Keyboard-free: auto-detect source language
- Configurable target language (default: English)

## Quickstart

### 0) Python & Audio setup
- Python 3.10+ recommended.
- macOS: grant microphone permission to your terminal/IDE.
- Windows: install **Microsoft C++ Build Tools** if needed for `webrtcvad`.
- Linux: ensure PortAudio libs (e.g., `sudo apt install portaudio19-dev`).

### 1) Create and activate a virtual env
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
