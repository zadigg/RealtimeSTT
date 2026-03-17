"""
Dual-source transcription using macloop (no virtual driver needed).

Separates YOUR voice (microphone) from PC audio (YouTube, Spotify, etc.)
using macloop's native ScreenCaptureKit capture — no BlackHole, no Stereo Mix.

- "You:" = microphone input (your speech)
- "PC:"  = system audio output (YouTube, music, calls, etc.)

Usage:
    python tests/realtimestt_dual_source.py

First run will prompt for Screen Recording permission (required by macOS).
"""
import os
import sys
import time
import threading
import numpy as np
from collections import deque
from difflib import SequenceMatcher

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(__file__))
    from install_packages import check_and_install_packages
    check_and_install_packages([
        {'import_name': 'rich'},
        {'import_name': 'colorama'},
        {'import_name': 'macloop'},
    ])

    import macloop
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from RealtimeSTT import AudioToTextRecorder
    import colorama
    colorama.init()

    console = Console()
    console.print("[bold]Dual-source transcription[/bold] (macloop — no virtual driver)")
    console.print("Initializing...")

    # Show available devices
    console.print("\n[dim]Microphones:[/dim]")
    for mic in macloop.MicrophoneSource.list_devices():
        marker = " [green]← default[/green]" if mic["is_default"] else ""
        console.print(f"  [dim]{mic['id']}[/dim]: {mic['name']}{marker}")

    console.print("[dim]System audio: captured via ScreenCaptureKit (all PC output)[/dim]\n")

    # Shared state
    you_sentences = deque(maxlen=50)
    pc_sentences = deque(maxlen=50)
    you_realtime = ""
    pc_realtime = ""
    lock = threading.Lock()
    stop_event = threading.Event()

    ECHO_SIMILARITY_THRESHOLD = 0.65

    def is_echo_from_pc(text):
        """Check if mic text is actually echo from PC speakers."""
        if not text:
            return False
        text_clean = text.strip().lower()
        if len(text_clean) < 12:
            return False
        with lock:
            pc_texts = list(pc_sentences) + ([pc_realtime] if pc_realtime else [])
            for pt in pc_texts:
                if not pt or len(pt.strip()) < 10:
                    continue
                pt_clean = pt.strip().lower()
                ratio = SequenceMatcher(None, text_clean, pt_clean).ratio()
                if ratio >= ECHO_SIMILARITY_THRESHOLD:
                    return True
                if text_clean in pt_clean or pt_clean in text_clean:
                    return True
        return False

    # --- Callbacks ---

    def on_you_realtime(text):
        global you_realtime
        t = (text or "").strip().lstrip("...").lstrip()
        if not t:
            with lock:
                you_realtime = ""
            return
        t = t[0].upper() + t[1:]
        if is_echo_from_pc(t):
            with lock:
                you_realtime = ""
            return
        with lock:
            you_realtime = t

    def on_pc_realtime(text):
        global pc_realtime
        t = (text or "").strip().lstrip("...").lstrip()
        if t:
            t = t[0].upper() + t[1:]
        with lock:
            pc_realtime = t

    def on_you_final(text):
        text = (text or "").strip().rstrip("...").strip()
        if not text:
            return
        if is_echo_from_pc(text):
            return
        with lock:
            you_sentences.append(text)

    def on_pc_final(text):
        text = (text or "").strip().rstrip("...").strip()
        if not text:
            return
        with lock:
            pc_sentences.append(text)
            text_lower = text.lower()
            keep = []
            for y in you_sentences:
                if len(text) < 10 or len(y) < 10:
                    keep.append(y)
                    continue
                y_lower = y.lower()
                r = SequenceMatcher(None, y_lower, text_lower).ratio()
                if r >= ECHO_SIMILARITY_THRESHOLD or y_lower in text_lower or text_lower in y_lower:
                    continue
                keep.append(y)
            you_sentences.clear()
            you_sentences.extend(keep[-50:])

    # --- Recorders (both use feed_audio, no built-in mic) ---

    base_config = {
        'use_microphone': False,
        'spinner': False,
        'model': 'tiny.en',
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        'device': 'cpu',
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'silero_deactivity_detection': False,
        'beam_size': 2,
        'beam_size_realtime': 2,
        'no_log_file': True,
        'min_length_of_recording': 0.8,
        'min_gap_between_recordings': 0.3,
    }

    console.print("[dim]Loading STT models...[/dim]")
    recorder_you = AudioToTextRecorder(
        **base_config,
        on_realtime_transcription_update=on_you_realtime,
    )
    recorder_pc = AudioToTextRecorder(
        **base_config,
        on_realtime_transcription_update=on_pc_realtime,
    )

    # --- macloop audio pipeline ---

    console.print("[dim]Starting macloop audio engine...[/dim]")
    engine = macloop.AudioEngine()
    engine.__enter__()

    mic_stream = engine.create_stream(
        macloop.MicrophoneSource,
        device_id=None,       # default mic
        vpio_enabled=True,    # Apple voice processing (echo cancellation + noise reduction)
    )
    sys_stream = engine.create_stream(macloop.SystemAudioSource)

    mic_route = engine.route("mic", stream=mic_stream)
    sys_route = engine.route("sys", stream=sys_stream)

    mic_sink = macloop.AsrSink(
        routes=[mic_route],
        chunk_frames=512,
        sample_rate=16_000,
        channels=1,
        sample_format="i16",
    )
    sys_sink = macloop.AsrSink(
        routes=[sys_route],
        chunk_frames=512,
        sample_rate=16_000,
        channels=1,
        sample_format="i16",
    )

    # Feed macloop chunks into RealtimeSTT recorders
    def mic_feeder():
        try:
            for chunk in mic_sink.chunks():
                if stop_event.is_set():
                    break
                recorder_you.feed_audio(chunk.samples.tobytes())
        except Exception as e:
            if not stop_event.is_set():
                console.print(f"[red]Mic feeder error: {e}[/red]")

    def sys_feeder():
        try:
            for chunk in sys_sink.chunks():
                if stop_event.is_set():
                    break
                recorder_pc.feed_audio(chunk.samples.tobytes())
        except Exception as e:
            if not stop_event.is_set():
                console.print(f"[red]System audio feeder error: {e}[/red]")

    def run_recorder(recorder, callback):
        try:
            while not stop_event.is_set():
                recorder.text(callback)
        except Exception as e:
            if not stop_event.is_set():
                console.print(f"[red]Recorder error: {e}[/red]")

    # --- Display ---

    def build_panel():
        with lock:
            rt_you = you_realtime
            rt_pc = pc_realtime
            sent_you = list(you_sentences)
            sent_pc = list(pc_sentences)

        lines = []
        for s in sent_you:
            lines.append(Text("You: ", style="bold green") + Text(s, style="green"))
        if rt_you:
            lines.append(Text("You: ", style="bold green") + Text(rt_you, style="bold green"))
        for s in sent_pc:
            lines.append(Text("PC:  ", style="bold cyan") + Text(s, style="cyan"))
        if rt_pc:
            lines.append(Text("PC:  ", style="bold cyan") + Text(rt_pc, style="bold cyan"))

        if not lines:
            return Panel(
                Text("Listening... Speak or play audio on your PC.", style="dim"),
                title="[bold green]You[/bold green] (mic) vs [bold cyan]PC[/bold cyan] (system audio)",
                border_style="blue",
            )

        content = Text()
        for i, line in enumerate(lines):
            content += line
            if i < len(lines) - 1:
                content += Text("\n")
        return Panel(
            content,
            title="[bold green]You[/bold green] (mic) vs [bold cyan]PC[/bold cyan] (system audio)",
            border_style="blue",
        )

    # --- Start everything ---

    live = Live(console=console, refresh_per_second=20, screen=False)
    live.start()

    threads = [
        threading.Thread(target=mic_feeder, daemon=True),
        threading.Thread(target=sys_feeder, daemon=True),
        threading.Thread(target=run_recorder, args=(recorder_you, on_you_final), daemon=True),
        threading.Thread(target=run_recorder, args=(recorder_pc, on_pc_final), daemon=True),
    ]
    for t in threads:
        t.start()

    console.print("[bold green]Ready![/bold green] Speak into mic or play audio on your PC.\n")

    try:
        while True:
            live.update(build_panel())
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        live.stop()

        mic_sink.close()
        sys_sink.close()
        engine.__exit__(None, None, None)

        recorder_you.shutdown()
        recorder_pc.shutdown()

        for t in threads:
            t.join(timeout=2)

        console.print("[yellow]Stopped.[/yellow]")
