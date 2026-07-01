"""Kokoro neural TTS service.

Phase 2 of the voice plan. Replaces the browser's `window.speechSynthesis`
(which falls back to robotic SAPI voices on Windows) with the open-source
Kokoro-82M model, served from this AI service over HTTP.

The model and voice pack are downloaded from kokoro-onnx's GitHub Releases
on first use (per upstream's documented workflow) and cached at
`settings.kokoro_model_dir` — mounted as a Docker volume so the ~330MB
doesn't redownload on every container restart.

Usage:
    service = await get_kokoro_service()
    wav_bytes = await service.synthesize("Hello, racer.", voice="af_bella")

Notes:
    - Inference is CPU-friendly (~300ms for a short sentence) and faster on GPU
      via onnxruntime-gpu. The runtime picks the available provider automatically.
    - Model load happens lazily on the first call. Startup stays fast; the
      first request pays a one-time ~5s warmup.
    - This service intentionally does not stream — Phase 2 ships a simple
      "fetch full WAV, play it" UX. Streaming is Phase 2.5.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)


class KokoroService:
    """Lazy-loaded Kokoro TTS engine producing WAV bytes."""

    def __init__(self) -> None:
        self._kokoro = None
        self._load_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        """Download model files (if missing) and instantiate the engine."""
        if self._kokoro is not None:
            return

        async with self._load_lock:
            # Double-check inside the lock: another concurrent caller may have
            # finished loading while we were waiting.
            if self._kokoro is not None:
                return

            # Deferred so the rest of the AI service still starts cleanly
            # even if kokoro-onnx isn't installed in some envs.
            from kokoro_onnx import Kokoro

            model_dir = Path(settings.kokoro_model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / Path(settings.kokoro_model_url).name
            voices_path = model_dir / Path(settings.kokoro_voices_url).name

            LOGGER.info(
                "Loading Kokoro TTS — model=%s voices=%s",
                model_path.name,
                voices_path.name,
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, _download_if_missing, settings.kokoro_model_url, model_path
            )
            await loop.run_in_executor(
                None, _download_if_missing, settings.kokoro_voices_url, voices_path
            )

            # Engine construction itself is fast; building the inference
            # session may not be — still run in a thread to be safe.
            def _build() -> "Kokoro":
                return Kokoro(str(model_path), str(voices_path))

            self._kokoro = await loop.run_in_executor(None, _build)
            LOGGER.info("Kokoro TTS ready (model=%s)", model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        language: str = "en-us",
    ) -> bytes:
        """Synthesize `text` to WAV bytes.

        Returns a complete RFC-1521 WAV file as bytes — suitable for
        `Response(content=..., media_type="audio/wav")`.
        """
        if not text or not text.strip():
            raise ValueError("synthesize: text must be non-empty")

        await self._ensure_loaded()
        chosen_voice = voice or settings.kokoro_default_voice

        # kokoro-onnx returns (samples: np.ndarray[float32, shape=(N,)], sample_rate: int)
        # The call is synchronous — push to a thread so request handling
        # remains async-friendly under concurrency.
        loop = asyncio.get_event_loop()

        def _synth() -> bytes:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=chosen_voice,
                speed=speed,
                lang=language,
            )
            return _samples_to_wav_bytes(samples, sample_rate)

        return await loop.run_in_executor(None, _synth)

    async def list_voices(self) -> list[str]:
        """Return the available voice names (e.g. ["af_bella", "am_michael", ...])."""
        await self._ensure_loaded()
        try:
            return sorted(self._kokoro.get_voices())
        except AttributeError:
            # Older kokoro-onnx versions expose voices via .voices mapping
            return sorted(getattr(self._kokoro, "voices", {}).keys())

    async def is_ready(self) -> bool:
        """Cheap check used by the health endpoint — does not force a load."""
        return self._kokoro is not None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _samples_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """Encode a numpy float32 sample array as WAV bytes."""
    import soundfile as sf  # local import keeps cold-start fast

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _download_if_missing(url: str, dest: Path) -> None:
    """Stream `url` to `dest` if not already cached. Atomic via .tmp + rename."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    LOGGER.info("Kokoro: downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


# ----------------------------------------------------------------------
# Module-level singleton accessor
# ----------------------------------------------------------------------

_service_singleton: Optional[KokoroService] = None
_singleton_lock = asyncio.Lock()


async def get_kokoro_service() -> KokoroService:
    """Return the process-wide KokoroService instance.

    Creates it on first call. The underlying model load is deferred until
    the first synthesize() invocation.
    """
    global _service_singleton
    if _service_singleton is not None:
        return _service_singleton
    async with _singleton_lock:
        if _service_singleton is None:
            _service_singleton = KokoroService()
        return _service_singleton
