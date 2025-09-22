import argparse
import queue
from threading import Thread
from rich.console import Console

from app.config import AppConfig
from app.audio_stream import AudioSegmenter
from app.transcriber import Transcriber
from app.translator import M2MTranslator
from app.server import create_app

console = Console()

def run_server(port, out_q: queue.Queue):
    app, socketio = create_app()

    @socketio.on("connect")
    def _connect():
        console.log("[green]Client connected to captions page[/green]")

    def broadcaster():
        while True:
            data = out_q.get()
            if data is None:
                break
            socketio.emit("caption", data)

    t = Thread(target=broadcaster, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=port)

def main():
    parser = argparse.ArgumentParser(description="AI Live Translator")
    parser.add_argument("--target-lang", default="en", help="Target language (e.g., en, es, fr, de, ja, zh)")
    parser.add_argument("--model-size", default="medium", help="faster-whisper size: tiny, base, small, medium, large-v3")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, help="0..3 (higher=more aggressive)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=30, choices=[10,20,30])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index")

    args = parser.parse_args()
    cfg = AppConfig(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        vad_aggressiveness=args.vad_aggressiveness,
        target_lang=args.target_lang,
        model_size=args.model_size,
        device=args.device,
        port=args.port,
        input_device=args.input_device
    )

    out_q: queue.Queue = queue.Queue()

    # Start web server
    server_thread = Thread(target=run_server, args=(cfg.port, out_q), daemon=True)
    server_thread.start()
    console.log(f"[cyan]Caption page: http://localhost:{cfg.port}[/cyan]")

    # Init models
    console.log("[magenta]Loading models (whisper + m2m100)...[/magenta]")
    transcriber = Transcriber(model_size=cfg.model_size, device=cfg.device)
    translator = M2MTranslator(device=cfg.device)

    # Start audio stream
    segmenter = AudioSegmenter(
        sample_rate=cfg.sample_rate,
        frame_ms=cfg.frame_ms,
        vad_aggressiveness=cfg.vad_aggressiveness,
        input_device=cfg.input_device
    )

    console.log("[green]Listening... speak into your mic.[/green]")
    try:
        for utt in segmenter.utterances():
            try:
                text, src_lang = transcriber.transcribe(utt)
                if not text:
                    continue
                translated = translator.translate(text, src_lang=src_lang, tgt_lang=cfg.target_lang)
                console.print(f"[bold yellow]{src_lang}[/bold yellow] → [bold cyan]{cfg.target_lang}[/bold cyan]: {translated}")
                out_q.put({"original": text, "translated": translated})
            except Exception as e:
                console.log(f"[red]Error processing segment:[/red] {e}")
    except KeyboardInterrupt:
        console.log("[red]Shutting down...[/red]")
        out_q.put(None)

if __name__ == "__main__":
    main()
