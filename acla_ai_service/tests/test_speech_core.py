from __future__ import annotations

import asyncio
import io
import threading
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api import voice
from app.voice import speech_core
from app.voice.pipecat_kokoro import build_kokoro_processor
from app.voice.pipecat_whisper import build_whisper_processor


def _wav_audio():
    content = io.BytesIO()
    with wave.open(content, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    return content.getvalue()


@pytest.mark.asyncio
async def test_core_loads_independent_pools_and_consumes_whisper_generator_in_worker(monkeypatch):
    main_thread = threading.get_ident()
    wav_audio = _wav_audio()
    received = []

    class Whisper:
        def transcribe(self, audio, *, language):
            assert threading.get_ident() != main_thread
            assert audio.read() == wav_audio
            assert language == "en"

            def segments():
                assert threading.get_ident() != main_thread
                received.append(self)
                yield SimpleNamespace(text=" Brake earlier.", no_speech_prob=0.1)
                yield SimpleNamespace(text="noise", no_speech_prob=0.7)

            return segments(), None

    tts_factory = AsyncMock(side_effect=lambda: SimpleNamespace(
        synthesize=AsyncMock(return_value=b"speech"),
        list_voices=AsyncMock(return_value=["af_bella"]),
    ))
    stt_factory = AsyncMock(side_effect=Whisper)
    monkeypatch.setattr(speech_core, "_create_tts", tts_factory)
    monkeypatch.setattr(speech_core, "_create_stt", stt_factory)
    core = speech_core.SpeechCore()
    try:
        assert await core.synthesize("Hello", "af_bella", 1.2, "en-us") == b"speech"
        assert tts_factory.await_count == 10
        stt_factory.assert_not_awaited()
        assert await core.list_voices() == ["af_bella"]

        assert await core.transcribe(wav_audio) == "Brake earlier."
        assert await core.transcribe(wav_audio) == "Brake earlier."
        assert stt_factory.await_count == 10
        assert received[0] is received[1]
        assert core.tts.stats["available"] == core.stt.stats["available"] == 10
    finally:
        await core.aclose()


@pytest.mark.asyncio
async def test_http_health_does_not_load_models_and_shutdown_resets_core(monkeypatch):
    tts_factory = AsyncMock()
    stt_factory = AsyncMock()
    monkeypatch.setattr(speech_core, "_create_tts", tts_factory)
    monkeypatch.setattr(speech_core, "_create_stt", stt_factory)
    monkeypatch.setattr(speech_core, "_speech_core", None)

    core = speech_core.get_speech_core()
    assert speech_core.get_speech_core() is core
    health = await voice.voice_health()
    assert health["loaded"] is False
    assert health["pools"]["tts"]["minimum"] == 10
    assert health["pools"]["stt"]["minimum"] == 10
    tts_factory.assert_not_awaited()
    stt_factory.assert_not_awaited()

    with pytest.raises(ValueError, match="non-empty"):
        await core.synthesize(" ")
    tts_factory.assert_not_awaited()

    await speech_core.close_speech_core()
    assert speech_core._speech_core is None
    with pytest.raises(RuntimeError, match="closed"):
        await core.synthesize("closed")


@pytest.mark.asyncio
async def test_pipecat_whisper_uses_session_buffers_and_pooled_inference():
    from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    core = SimpleNamespace(transcribe=AsyncMock(return_value="Turn in later."))
    Processor = build_whisper_processor(core)
    first, second = Processor(sample_rate=16000), Processor(sample_rate=16000)
    first._audio_buffer_size_1s = second._audio_buffer_size_1s = 32000
    try:
        await first.process_audio_frame(
            InputAudioRawFrame(audio=b"\x01\x00", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        assert first._audio_buffer == b"\x01\x00"
        assert second._audio_buffer == b""
        assert first.wants_wav_segments is True

        wav_audio = _wav_audio()
        first._user_id = "driver-1"
        second._user_id = "driver-2"
        for processor, user_id in [(first, "driver-1"), (second, "driver-2")]:
            frames = [frame async for frame in processor.run_stt(wav_audio)]
            assert len(frames) == 1
            assert isinstance(frames[0], TranscriptionFrame)
            assert frames[0].text == "Turn in later."
            assert frames[0].user_id == user_id
        assert core.transcribe.await_count == 2
        core.transcribe.assert_awaited_with(wav_audio)
    finally:
        await first.cleanup()
        await second.cleanup()


@pytest.mark.asyncio
async def test_pipecat_whisper_emits_error_frame_and_propagates_cancellation():
    from pipecat.frames.frames import ErrorFrame

    core = SimpleNamespace(transcribe=AsyncMock(side_effect=RuntimeError("offline")))
    processor = build_whisper_processor(core)()
    try:
        frames = [frame async for frame in processor.run_stt(_wav_audio())]
        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert "offline" in frames[0].error

        core.transcribe.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            _ = [frame async for frame in processor.run_stt(_wav_audio())]
    finally:
        await processor.cleanup()


@pytest.mark.asyncio
async def test_http_and_pipecat_tts_use_same_core_without_sharing_sentence_buffers(monkeypatch):
    from pipecat.frames.frames import OutputAudioRawFrame

    core = SimpleNamespace(synthesize=AsyncMock(return_value=_wav_audio()))
    monkeypatch.setattr(voice, "get_speech_core", lambda: core)
    response = await voice.synthesize(voice.SynthesizeRequest(
        text="Hello", voice="af_bella", speed=1.2, language="en-us",
    ))
    assert response.body == _wav_audio()
    core.synthesize.assert_awaited_with(
        "Hello", voice="af_bella", speed=1.2, language="en-us",
    )

    Processor = build_kokoro_processor(core)
    first, second = Processor(), Processor()
    first.push_frame = AsyncMock()
    try:
        first._streamer.feed("Private unfinished sentence")
        assert list(second._streamer.flush()) == []
        await first._synth_and_push("Brake smoothly.")
        core.synthesize.assert_awaited_with("Brake smoothly.")
        audio_frame = first.push_frame.call_args.args[0]
        assert isinstance(audio_frame, OutputAudioRawFrame)
        assert audio_frame.sample_rate == 24000
        assert audio_frame.num_channels == 1
        assert len(audio_frame.audio) == 480
    finally:
        await first.cleanup()
        await second.cleanup()
