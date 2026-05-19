"""Pipecat TTSService wrapping our existing KokoroService (Phase 3).

Pipecat's pipeline expects each component to be a `FrameProcessor` that
consumes some frames and emits others. For TTS, the standard interface is
`TTSService`, which inherits from `FrameProcessor` and provides:
    async def run_tts(text: str) -> AsyncGenerator[Frame, None]

We don't subclass Pipecat's `TTSService` directly here — instead we hand-roll
the minimal FrameProcessor that consumes `TextFrame`/`LLMFullResponseEndFrame`
and emits `AudioRawFrame`s. This keeps us insulated from Pipecat's evolving
TTS base class while reusing the proven Kokoro synthesis path from Phase 2.

The audio frames produced are PCM16 mono at Kokoro's native sample rate
(24kHz by default). The pipeline's transport handles resampling to the
WebSocket's target rate as needed.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

# Pipecat imports are deferred to where they're used so the AI service can
# still boot even when pipecat-ai isn't fully installed (e.g. AMD env where
# pip resolution is tricky).

from app.voice import get_kokoro_service
from app.voice.sentence_streamer import SentenceStreamer

LOGGER = logging.getLogger(__name__)


def build_kokoro_processor():
    """Construct a Pipecat FrameProcessor that synthesizes incoming text.

    Imported lazily so module import doesn't require pipecat to be installed.
    """
    import numpy as np
    from pipecat.frames.frames import (
        AudioRawFrame,
        EndFrame,
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class KokoroTTSProcessor(FrameProcessor):
        """Sentence-chunked TTS that emits PCM16 audio frames.

        Strategy:
          - Consumes `TextFrame`s as they come from the LLM service.
          - Buffers via SentenceStreamer until a complete sentence is ready
            (same logic the Phase 2.5 SSE path uses).
          - On `LLMFullResponseEndFrame`, flushes remaining buffer.
          - For each ready sentence, calls Kokoro and emits an AudioRawFrame.
        """

        def __init__(self, sample_rate: int = 24000) -> None:
            super().__init__()
            self._streamer = SentenceStreamer(min_words=6)
            self._sample_rate = sample_rate

        async def process_frame(self, frame: "Frame", direction: "FrameDirection") -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, TextFrame):
                # LLM token / partial answer chunk.
                self._streamer.feed(frame.text)
                async for sentence in self._drain():
                    await self._synth_and_push(sentence)
                # Don't forward the TextFrame downstream — the transport
                # doesn't need to see raw text; only the audio output matters.
                return

            if isinstance(frame, LLMFullResponseStartFrame):
                # Mark the start of a new spoken response.
                await self.push_frame(TTSStartedFrame(), direction)
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, LLMFullResponseEndFrame):
                # Flush trailing partial sentence at end of LLM stream.
                async for sentence in self._flush():
                    await self._synth_and_push(sentence)
                await self.push_frame(TTSStoppedFrame(), direction)
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, EndFrame):
                # Pipeline shutting down — flush whatever's left then propagate.
                async for sentence in self._flush():
                    await self._synth_and_push(sentence)
                await self.push_frame(frame, direction)
                return

            # All other frames pass through untouched (control, interruption, etc).
            await self.push_frame(frame, direction)

        async def _drain(self) -> AsyncGenerator[str, None]:
            for sentence in list(self._streamer.drain_sentences()):
                yield sentence

        async def _flush(self) -> AsyncGenerator[str, None]:
            for sentence in list(self._streamer.flush()):
                yield sentence

        async def _synth_and_push(self, sentence: str) -> None:
            try:
                kokoro = await get_kokoro_service()
                # KokoroService.synthesize returns WAV bytes. For Pipecat we
                # need raw PCM16 samples. Strip the WAV header by re-reading
                # via soundfile.
                wav_bytes = await kokoro.synthesize(sentence)
                pcm16 = _wav_bytes_to_pcm16(wav_bytes, target_sample_rate=self._sample_rate)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Pipecat Kokoro synth failed: %s", exc)
                return

            await self.push_frame(
                AudioRawFrame(
                    audio=pcm16,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                ),
            )

    return KokoroTTSProcessor


def _wav_bytes_to_pcm16(wav_bytes: bytes, target_sample_rate: int = 24000) -> bytes:
    """Decode a WAV blob to PCM16 mono bytes at `target_sample_rate`.

    Kokoro already produces 24kHz mono PCM16 in our service, so this is
    usually a no-op decode. The resample branch is the safety net for if
    we swap voices/models in the future and the rate changes.
    """
    import io
    import numpy as np
    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16", always_2d=False)
    if samples.ndim > 1:
        # Down-mix to mono if Kokoro ever produces stereo (it doesn't today).
        samples = samples.mean(axis=1).astype("int16")

    if sr != target_sample_rate:
        # Simple linear resample. Good enough — Kokoro's native rate is
        # 24kHz and we configure the pipeline at 24kHz, so this branch
        # should never execute in normal operation.
        ratio = target_sample_rate / float(sr)
        new_len = int(len(samples) * ratio)
        x_old = np.linspace(0, 1, num=len(samples), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype("int16")

    return samples.tobytes()
