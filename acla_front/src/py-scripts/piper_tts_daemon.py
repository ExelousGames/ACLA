"""Piper TTS daemon for on-track guidance cues (Phase 4).

Long-lived Python subprocess spawned by Electron's main process. Reads
phrases from stdin (one JSON object per line), synthesizes them with
Piper, and writes length-prefixed WAV bytes to stdout.

Why a persistent daemon instead of one-shot CLI calls?
    - Piper model load is ~1s. On-track cues need <150ms latency.
    - Keeping the voice resident makes subsequent synths take ~30ms.

Protocol
--------
The daemon emits these JSON lines on STDERR (status channel):
    {"status": "starting"}                              -- before model load
    {"status": "ready", "voice": "..."}                 -- model loaded
    {"status": "error", "error": "...", "fatal": bool}  -- recoverable or fatal

Each command from STDIN is one JSON object per line:
    {"cmd": "synth", "id": "...", "text": "Brake now"}
    {"cmd": "cancel"}                                   -- discard any queued/inflight
    {"cmd": "shutdown"}                                 -- exit cleanly

For each "synth" command, STDOUT emits a length-prefixed binary frame:
    [4 bytes big-endian total length][N bytes payload]

The payload is a JSON header followed by binary WAV bytes:
    [4 bytes big-endian header length][header JSON][WAV bytes]

Header JSON: {"id": "...", "ok": true, "bytes": <wav-size>}
On synthesis failure: {"id": "...", "ok": false, "error": "..."} + empty WAV body.

This double-framing keeps the binary channel parseable from Node.js even
when WAVs contain incidental newline bytes that would confuse a line-based
protocol.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Status logging (stderr, JSON-line)
# ---------------------------------------------------------------------------


def _emit_status(payload: dict) -> None:
    try:
        sys.stderr.write(json.dumps(payload) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_model_paths() -> tuple[Path, Path]:
    """Locate the Piper model + config. Download from HF on first run.

    Cache lives under the user's app-data dir if available, falling back to
    a temp dir. Path can be overridden via PIPER_MODEL_DIR env var.
    """
    cache_root = os.environ.get("PIPER_MODEL_DIR")
    if not cache_root:
        if sys.platform == "win32":
            cache_root = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "acla", "piper")
        elif sys.platform == "darwin":
            cache_root = os.path.expanduser("~/Library/Application Support/acla/piper")
        else:
            cache_root = os.path.expanduser("~/.local/share/acla/piper")

    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)

    # Default voice — small, low-latency, clear American English female.
    voice_repo = os.environ.get("PIPER_VOICE_REPO", "rhasspy/piper-voices")
    voice_subdir = os.environ.get("PIPER_VOICE_SUBDIR", "en/en_US/amy/medium")
    voice_basename = os.environ.get("PIPER_VOICE_NAME", "en_US-amy-medium")

    model_path = cache / f"{voice_basename}.onnx"
    config_path = cache / f"{voice_basename}.onnx.json"

    if not model_path.exists() or not config_path.exists():
        _emit_status({"status": "downloading", "voice": voice_basename})
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download Piper voices on first run"
            ) from exc

        for suffix in (".onnx", ".onnx.json"):
            filename = f"{voice_subdir}/{voice_basename}{suffix}"
            downloaded = hf_hub_download(
                repo_id=voice_repo,
                filename=filename,
                local_dir=str(cache),
            )
            # hf_hub_download nests the file by `filename` path — copy/rename
            # to the flat layout Piper expects.
            target = model_path if suffix == ".onnx" else config_path
            if Path(downloaded).resolve() != target.resolve():
                import shutil
                shutil.copyfile(downloaded, target)

    return model_path, config_path


def _load_voice() -> "object":
    """Instantiate piper.PiperVoice. Raises on failure."""
    from piper import PiperVoice  # type: ignore

    model_path, config_path = _resolve_model_paths()
    return PiperVoice.load(str(model_path), config_path=str(config_path))


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _synthesize_wav(voice, text: str) -> bytes:
    """Synthesize `text` to WAV bytes using Piper.

    Piper's `synthesize_wav` writes to a file-like; we use a BytesIO.
    """
    buffer = BytesIO()
    # piper-tts 1.2+ exposes `synthesize_wav(text, wav_file_like)`.
    voice.synthesize_wav(text, buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Framed I/O on stdout (length-prefixed binary)
# ---------------------------------------------------------------------------


def _send_frame(header: dict, body: bytes) -> None:
    """Write one framed message to stdout's binary buffer.

    Format: [4B total_len BE][4B header_len BE][header_json][body bytes]
    """
    header_bytes = json.dumps(header).encode("utf-8")
    total_len = 4 + len(header_bytes) + len(body)
    out = sys.stdout.buffer
    out.write(struct.pack(">I", total_len))
    out.write(struct.pack(">I", len(header_bytes)))
    out.write(header_bytes)
    out.write(body)
    out.flush()


# ---------------------------------------------------------------------------
# Cancellation state
# ---------------------------------------------------------------------------

_cancel_event = threading.Event()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    _emit_status({"status": "starting"})
    try:
        voice = _load_voice()
    except Exception as exc:
        _emit_status({"status": "error", "error": str(exc), "fatal": True})
        sys.exit(1)

    _emit_status({"status": "ready", "voice": os.environ.get("PIPER_VOICE_NAME", "en_US-amy-medium")})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            cmd_msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit_status({"status": "error", "error": f"bad-json: {exc}", "fatal": False})
            continue

        cmd = cmd_msg.get("cmd")

        if cmd == "shutdown":
            _emit_status({"status": "shutdown"})
            return

        if cmd == "cancel":
            _cancel_event.set()
            # No frame response — cancel is fire-and-forget.
            continue

        if cmd == "synth":
            req_id = cmd_msg.get("id", "")
            text = (cmd_msg.get("text") or "").strip()
            if not text:
                _send_frame({"id": req_id, "ok": False, "error": "empty-text"}, b"")
                continue

            _cancel_event.clear()
            try:
                wav_bytes = _synthesize_wav(voice, text)
            except Exception as exc:
                _emit_status({"status": "error", "error": f"synth-failed: {exc}", "fatal": False})
                _send_frame({"id": req_id, "ok": False, "error": str(exc)}, b"")
                continue

            if _cancel_event.is_set():
                # Cancel arrived while synthesizing — drop the result silently.
                _send_frame({"id": req_id, "ok": False, "error": "cancelled"}, b"")
                continue

            _send_frame({"id": req_id, "ok": True, "bytes": len(wav_bytes)}, wav_bytes)
            continue

        _emit_status({"status": "error", "error": f"unknown-cmd: {cmd}", "fatal": False})


if __name__ == "__main__":
    main()
