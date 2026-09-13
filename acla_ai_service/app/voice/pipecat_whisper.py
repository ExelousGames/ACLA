"""Session-local speech segmentation backed by the application STT pool."""

from __future__ import annotations

from app.voice.speech_core import SpeechCore


def build_whisper_processor(speech_core: SpeechCore):
    """Keep Pipecat's VAD buffering while delegating inference to the core."""
    from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
    from pipecat.services.stt_service import SegmentedSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.utils.time import time_now_iso8601

    class PooledWhisperSTTProcessor(SegmentedSTTService):
        # SegmentedSTTService supplies WAV segments by default. Passing that
        # container to faster-whisper also preserves the input sample rate.
        async def run_stt(self, audio: bytes):
            await self.start_processing_metrics()
            try:
                text = await speech_core.transcribe(audio)
            except Exception as exc:
                yield ErrorFrame(f"Whisper transcription failed: {exc}")
                return
            finally:
                await self.stop_processing_metrics()

            if text:
                yield TranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), Language.EN,
                )

    return PooledWhisperSTTProcessor
