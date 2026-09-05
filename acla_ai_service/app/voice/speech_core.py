"""Application-owned TTS and STT pools, independent of chat sessions.

Each pool lazily loads ten real engines on first use and grows on contention.
Only inference engines are shared; Pipecat processors and their audio/text
buffers remain local to each connection. Pools are process-local, so every
ASGI worker has its own model population and memory cost.
"""

from __future__ import annotations

import asyncio
import io

from app.voice.instance_pool import InstancePool
from app.voice.kokoro_service import KokoroService

MIN_SPEECH_INSTANCES = 10


async def _create_tts() -> KokoroService:
    engine = KokoroService()
    await engine._ensure_loaded()
    return engine


async def _create_stt():
    def load():
        from faster_whisper import WhisperModel

        return WhisperModel(
            "large-v3-turbo", device="auto", compute_type="default",
        )

    return await asyncio.to_thread(load)


class SpeechCore:
    """Facade routing speech operations through independent elastic pools."""

    def __init__(self) -> None:
        self.tts = InstancePool(_create_tts, min_size=MIN_SPEECH_INSTANCES)
        self.stt = InstancePool(_create_stt, min_size=MIN_SPEECH_INSTANCES)

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        language: str = "en-us",
    ) -> bytes:
        if not text or not text.strip():
            raise ValueError("synthesize: text must be non-empty")
        return await self.tts.run(
            lambda engine: engine.synthesize(text, voice, speed, language),
        )

    async def list_voices(self) -> list[str]:
        return await self.tts.run(lambda engine: engine.list_voices())

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe a complete WAV segment supplied by SegmentedSTTService."""
        def transcribe(engine):
            segments, _ = engine.transcribe(io.BytesIO(audio), language="en")
            # faster-whisper returns a lazy generator. Consume it in the
            # worker thread, while the engine is still exclusively leased.
            return " ".join(
                segment.text.strip()
                for segment in segments
                if segment.no_speech_prob < 0.4
            ).strip()

        return await self.stt.run(
            lambda engine: asyncio.to_thread(transcribe, engine),
        )

    async def aclose(self) -> None:
        await asyncio.gather(self.tts.aclose(), self.stt.aclose())


_speech_core: SpeechCore | None = None


def get_speech_core() -> SpeechCore:
    """Return the application core without importing or loading native models."""
    global _speech_core
    if _speech_core is None:
        _speech_core = SpeechCore()
    return _speech_core


async def close_speech_core() -> None:
    global _speech_core
    if _speech_core is not None:
        await _speech_core.aclose()
        _speech_core = None
