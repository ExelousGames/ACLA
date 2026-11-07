from __future__ import annotations

import io
import math
import os
import threading
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch
from fastapi import FastAPI
from pydantic import ValidationError

from app.api import voice
from app.voice import speech_core
from app.voice.kokoro_prosody import (
    KokoroProsodyService,
    controlled_acoustics,
    controlled_durations,
    synthesize_with_curves,
)
from app.voice.kokoro_service import _samples_to_wav_bytes
from app.voice.prosody import PhonemeCurve, ProsodyControls


def _points(first, last=None):
    return [{"position": 0, "value": first}, {"position": 1, "value": first if last is None else last}]


def _controls(**kwargs):
    return ProsodyControls(expected_phonemes="ab", curves=[{"start": 0, "end": 1, **kwargs}])


@pytest.mark.parametrize("change", [
    {"start": -1}, {"start": 0.5}, {"start": True}, {"end": 0}, {"end": 3},
    {"pitch": []}, {"pitch": _points(float("nan"))}, {"pitch": _points(float("inf"))},
    {"pitch": _points(13)}, {"energy": _points(-13)}, {"timing": _points(0)},
    {"timing": _points(4.1)}, {"pitch": [{"position": 0.5, "value": 0}, {"position": 1, "value": 1}]},
    {"pitch": [{"position": 0, "value": 0}, {"position": 0, "value": 1}, {"position": 1, "value": 0}]},
    {"pitch": _points(0), "unknown": 1},
])
def test_invalid_curve_data_is_rejected(change):
    with pytest.raises(ValidationError):
        _controls(**{"pitch": _points(0), **change})


def test_empty_and_overlapping_ranges_are_rejected():
    with pytest.raises(ValidationError, match="At least one"):
        _controls()
    with pytest.raises(ValidationError, match="overlap"):
        ProsodyControls(expected_phonemes="ab", curves=[
            {"start": 0, "end": 2, "pitch": _points(1)},
            {"start": 1, "end": 2, "energy": _points(1)},
        ])


def test_timing_interpolates_before_rounding_and_keeps_boundary_tokens():
    predicted = torch.tensor([[2., 4., 4., 4., 3.]])
    curve = PhonemeCurve(start=0, end=3, timing=_points(0.5, 3.5))
    actual = controlled_durations(predicted, [curve])
    assert actual.tolist() == [2, 4, 8, 12, 3]
    assert predicted.tolist() == [[2, 4, 4, 4, 3]]


def test_acoustic_curves_follow_stretched_phoneme_and_preserve_unvoiced_frames():
    curves = _controls(timing=_points(2), pitch=_points(12), energy=_points(-6, 6)).curves
    durations = controlled_durations(torch.tensor([[2., 3., 5., 4.]]), curves)
    assert durations.tolist() == [2, 6, 5, 4]
    f0 = torch.full((1, 34), 100.)
    f0[0, 7] = 0
    energy = torch.ones_like(f0)
    changed_f0, changed_energy = controlled_acoustics(f0, energy, durations, curves)
    assert changed_f0[0, 4:16].tolist() == [200, 200, 200, 0, 200, 200, 200, 200, 200, 200, 200, 200]
    assert torch.equal(changed_f0[:, :4], f0[:, :4])
    assert torch.equal(changed_f0[:, 16:], f0[:, 16:])
    assert torch.equal(changed_energy[:, :4], energy[:, :4])
    assert torch.equal(changed_energy[:, 16:], energy[:, 16:])
    expected_db = torch.arange(-5.5, 6, 1)
    torch.testing.assert_close(changed_energy[0, 4:16], 1 + expected_db * math.log(10) / 20)
    assert f0[0, 4] == 100
    assert torch.all(energy == 1)


def test_neutral_curves_and_empty_controls_preserve_predictions():
    predicted = torch.tensor([[2.2, 3.1, 4.7, 4.2]])
    neutral = _controls(timing=_points(1), pitch=_points(0), energy=_points(0)).curves
    baseline = controlled_durations(predicted, [])
    assert torch.equal(controlled_durations(predicted, neutral), baseline)
    f0 = torch.arange(int(baseline.sum()) * 2, dtype=torch.float32).unsqueeze(0)
    for curves in ([], neutral):
        changed_f0, energy = controlled_acoustics(f0, f0, baseline, curves)
        assert torch.equal(changed_f0, f0)
        assert torch.equal(energy, f0)


def test_separate_curves_target_last_phoneme_without_touching_trailing_boundary():
    curves = [PhonemeCurve(start=1, end=2, pitch=_points(-12))]
    f0 = torch.full((1, 16), 100.)
    actual, _ = controlled_acoustics(f0, f0, torch.tensor([2, 2, 2, 2]), curves)
    assert actual.flatten().tolist() == [100.] * 8 + [50.] * 4 + [100.] * 4


def test_duration_budget_and_incompatible_acoustic_shapes_fail_before_decoder():
    with pytest.raises(ValueError, match="150 seconds"):
        controlled_durations(torch.full((1, 512), 50.), [])
    with pytest.raises(RuntimeError, match="alignment"):
        controlled_acoustics(torch.zeros(1, 10), torch.zeros(1, 10), torch.tensor([2, 2, 2]), [])


def test_native_inference_passes_modified_predictions_into_decoder():
    # Two phonemes plus boundary tokens, two duration frames each at speed 1.
    decoder_inputs = []

    def decode(text, f0, energy, style):
        decoder_inputs.append((text, f0, energy))
        return torch.zeros(1, 1, f0.shape[-1] * 300)

    model = SimpleNamespace(
        vocab={"a": 1, "b": 2}, device=torch.device("cpu"),
        bert=lambda ids, **kw: ids.float().unsqueeze(-1),
        bert_encoder=lambda value: value,
        predictor=SimpleNamespace(
            text_encoder=lambda encoded, *args: encoded.transpose(-1, -2),
            lstm=lambda value: (value, None),
            duration_proj=lambda value: torch.zeros(1, 4, 4),
            F0Ntrain=lambda features, style: (
                torch.full((1, features.shape[-1] * 2), 100.),
                torch.zeros(1, features.shape[-1] * 2),
            ),
        ),
        text_encoder=lambda ids, *args: ids.float().unsqueeze(1),
        decoder=decode,
    )
    curves = _controls(timing=_points(2), pitch=_points(12), energy=_points(6)).curves
    audio, durations = synthesize_with_curves(model, "ab", torch.zeros(1, 256), 1., curves)
    assert durations.tolist() == [2, 4, 2, 2]
    text, f0, energy = decoder_inputs[0]
    assert text.flatten().tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 0, 0]
    assert torch.all(f0[:, 4:12] == 200)
    torch.testing.assert_close(energy[:, 4:12], torch.full((1, 8), 6 * math.log(10) / 20))
    assert len(audio) == 6000
    _, fast_durations = synthesize_with_curves(model, "ab", torch.zeros(1, 256), 2., [])
    assert fast_durations.tolist() == [1, 1, 1, 1]


def _engine(phonemes="ab"):
    engine = KokoroProsodyService()
    engine._config = {"vocab": {p: i + 1 for i, p in enumerate("ab ˈ!ɪ")}}
    engine._tokenizer = SimpleNamespace(phonemize=lambda *args, **kwargs: phonemes)
    engine._artifact = Mock(side_effect=AssertionError("Unexpected model download"))
    return engine


def test_preview_uses_the_synthesis_vocabulary_and_indices():
    preview = _engine("aˈɪ b!?")._phonemize("hello", "en-us")
    assert preview.phonemes == "aˈɪ b!"
    assert [token.index for token in preview.tokens] == list(range(6))
    assert "".join(token.symbol for token in preview.tokens) == preview.phonemes
    assert preview.tokens[2].token_id == 6


@pytest.mark.parametrize(("text", "phonemes", "error"), [
    (" ", "ab", "non-empty"), ("hi", "?", "no supported"),
    ("long", "a" * 511, "at most 510"),
])
def test_preview_rejects_empty_or_overlong_input(text, phonemes, error):
    with pytest.raises(ValueError, match=error):
        _engine(phonemes)._phonemize(text, "en-us")


def test_stale_phonemes_and_invalid_voice_fail_before_loading_weights():
    engine = _engine()
    with pytest.raises(ValueError, match="sequence changed"):
        engine._synthesize("Hello", ProsodyControls(expected_phonemes="ba"), None, 1., "en-us")
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        engine._synthesize("Hello", ProsodyControls(expected_phonemes="ab"), "../bad", 1., "en-us")
    engine._artifact.assert_not_called()


@pytest.mark.asyncio
async def test_prosody_pool_is_independent_lazy_and_uses_worker_threads(monkeypatch):
    thread = threading.get_ident()
    engine = _engine()
    original = engine._phonemize

    def preview(*args):
        assert threading.get_ident() != thread
        return original(*args)

    engine._phonemize = preview
    factory = AsyncMock(return_value=engine)
    tts_factory, stt_factory = AsyncMock(), AsyncMock()
    monkeypatch.setattr(speech_core, "_create_prosody_tts", factory)
    monkeypatch.setattr(speech_core, "_create_tts", tts_factory)
    monkeypatch.setattr(speech_core, "_create_stt", stt_factory)
    core = speech_core.SpeechCore()
    try:
        assert core.prosody_tts.stats["total"] == 0
        assert (await core.phonemize("Hello")).phonemes == "ab"
        controls = ProsodyControls(expected_phonemes="ab")
        engine.synthesize = AsyncMock(return_value=b"wav")
        assert await core.synthesize_prosody("Hello", controls, "af_bella", 1.2, "en-us") == b"wav"
        engine.synthesize.assert_awaited_once_with("Hello", controls, "af_bella", 1.2, "en-us")
        factory.assert_awaited_once()
        tts_factory.assert_not_awaited()
        stt_factory.assert_not_awaited()
    finally:
        await core.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await core.phonemize("Hello")


@pytest.mark.asyncio
async def test_http_preview_synthesis_validation_and_error_responses(monkeypatch):
    wav = _samples_to_wav_bytes(torch.zeros(240).numpy(), 24000)
    core = SimpleNamespace(
        phonemize=AsyncMock(return_value=_engine()._phonemize("Hello", "en-us")),
        synthesize_prosody=AsyncMock(return_value=wav),
    )
    monkeypatch.setattr(voice, "get_speech_core", lambda: core)
    app = FastAPI()
    app.include_router(voice.router)
    body = {"text": "Hello", "prosody": _controls(pitch=_points(2)).model_dump()}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        preview = await client.post("/voice/phonemes", json={"text": "Hello"})
        assert preview.status_code == 200
        assert preview.json()["phonemes"] == "ab"
        response = await client.post("/voice/synthesize/prosody", json=body)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["cache-control"] == "no-store"
        with wave.open(io.BytesIO(response.content)) as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (24000, 1, 2)
        core.synthesize_prosody.assert_awaited_once_with(
            "Hello", _controls(pitch=_points(2)), voice=None, speed=1., language="en-us",
        )
        invalid = {**body, "prosody": {"expected_phonemes": "ab", "curves": [{"start": 0, "end": 3}]}}
        assert (await client.post("/voice/synthesize/prosody", json=invalid)).status_code == 422
        assert core.synthesize_prosody.await_count == 1
        for error, status in ((ValueError("Phoneme sequence changed"), 400), (RuntimeError("internal"), 500)):
            core.synthesize_prosody.side_effect = error
            assert (await client.post("/voice/synthesize/prosody", json=body)).status_code == status
        core.phonemize.side_effect = ValueError("text must be non-empty")
        assert (await client.post("/voice/phonemes", json={"text": " "})).status_code == 400


@pytest.mark.skipif(os.environ.get("ACLA_KOKORO_INTEGRATION") != "1", reason="Opt-in model download/inference")
def test_real_kokoro_neutral_parity_and_controlled_wav():
    engine = KokoroProsodyService()
    preview = engine._phonemize("Brake smoothly.", "en-us")
    baseline_controls = ProsodyControls(expected_phonemes=preview.phonemes)
    baseline_wav = engine._synthesize("Brake smoothly.", baseline_controls, "af_bella", 1., "en-us")
    model = engine._model
    style = engine._voices["af_bella"][len(preview.phonemes) - 1].to(model.device)
    neutral = PhonemeCurve(start=0, end=len(preview.phonemes),
                          timing=_points(1), pitch=_points(0), energy=_points(0))
    with torch.inference_mode():
        torch.manual_seed(123)
        upstream = model(preview.phonemes, style, return_output=True)
        torch.manual_seed(123)
        audio, durations = synthesize_with_curves(model, preview.phonemes, style, 1., [neutral])
    assert torch.equal(durations.cpu(), upstream.pred_dur)
    torch.testing.assert_close(audio.cpu(), upstream.audio, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(audio).all()
    # Stretch only the vowel used by the documented request example.
    target = next(token.index for token in preview.tokens if token.symbol == "u")
    changed = ProsodyControls(expected_phonemes=preview.phonemes, curves=[{
        "start": target, "end": target + 1,
        "timing": _points(2), "pitch": _points(0, 4), "energy": _points(0, 3),
    }])
    changed_wav = engine._synthesize("Brake smoothly.", changed, "af_bella", 1., "en-us")
    with wave.open(io.BytesIO(baseline_wav)) as baseline, wave.open(io.BytesIO(changed_wav)) as modified:
        assert modified.getframerate() == 24000
        assert modified.getnchannels() == 1
        assert modified.getnframes() > baseline.getnframes()
