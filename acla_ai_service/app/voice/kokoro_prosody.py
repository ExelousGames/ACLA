"""Native Kokoro inference with controls applied before the audio decoder.

The ONNX service remains responsible for ordinary TTS. This engine is leased
exclusively by SpeechCore; no global hooks or model patches are installed.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path

from app.infra.config import settings
from app.voice.kokoro_service import _samples_to_wav_bytes
from app.voice.prosody import PhonemeCurve, PhonemePreview, PhonemeToken, ProsodyControls

MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SAMPLE_RATE = 24000
MAX_PHONEMES = 510
# Bound alignment length and decoder allocation (150 seconds at 40 Hz).
MAX_DURATION_FRAMES = 6000


class KokoroProsodyService:
    def __init__(self) -> None:
        self._config = None
        self._tokenizer = None
        self._model = None
        self._voices = {}

    def _artifact(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=MODEL_REPO, revision=MODEL_REVISION, filename=filename,
            cache_dir=str(Path(settings.kokoro_model_dir) / "native"),
        )

    def _phonemize(self, text: str, language: str) -> PhonemePreview:
        if not text.strip():
            raise ValueError("text must be non-empty")
        if self._tokenizer is None:
            from kokoro_onnx.tokenizer import Tokenizer

            with open(self._artifact("config.json"), encoding="utf-8") as source:
                self._config = json.load(source)
            self._tokenizer = Tokenizer(vocab=self._config["vocab"])

        # Use the native model vocabulary, including stress, spaces and punctuation.
        # These are model symbols, not spelling offsets or grapheme clusters.
        vocab = self._config["vocab"]
        raw = self._tokenizer.phonemize(text, lang=language)
        phonemes = "".join(symbol for symbol in raw if symbol in vocab)
        if not phonemes.strip():
            raise ValueError("Text produced no supported phonemes")
        if len(phonemes) > MAX_PHONEMES:
            raise ValueError(f"Curve synthesis supports at most {MAX_PHONEMES} phonemes; split the text")
        return PhonemePreview(
            phonemes=phonemes,
            tokens=[PhonemeToken(index=i, symbol=p, token_id=vocab[p]) for i, p in enumerate(phonemes)],
        )

    async def phonemize(self, text: str, language: str) -> PhonemePreview:
        return await asyncio.to_thread(self._phonemize, text, language)

    async def synthesize(
        self, text: str, controls: ProsodyControls, voice: str | None = None,
        speed: float = 1.0, language: str = "en-us",
    ) -> bytes:
        return await asyncio.to_thread(self._synthesize, text, controls, voice, speed, language)

    def _synthesize(self, text, controls, voice, speed, language) -> bytes:
        import torch

        preview = self._phonemize(text, language)
        if preview.phonemes != controls.expected_phonemes:
            raise ValueError("Phoneme sequence changed; call /voice/phonemes again for this text/language")
        if not math.isfinite(speed) or not 0.5 <= speed <= 2:
            raise ValueError("speed must be between 0.5 and 2")
        chosen_voice = voice or settings.kokoro_default_voice
        if not re.fullmatch(r"[a-z]{2}_[a-z0-9_]+", chosen_voice):
            raise ValueError(f"Unknown Kokoro voice: {chosen_voice}")
        if chosen_voice not in self._voices:
            from huggingface_hub.errors import EntryNotFoundError

            try:
                voice_path = self._artifact(f"voices/{chosen_voice}.pt")
            except EntryNotFoundError as exc:
                raise ValueError(f"Unknown Kokoro voice: {chosen_voice}") from exc
            self._voices[chosen_voice] = torch.load(voice_path, map_location="cpu", weights_only=True)
        if self._model is None:
            from kokoro import KModel

            self._model = KModel(
                repo_id=MODEL_REPO, config=self._config,
                model=self._artifact("kokoro-v1_0.pth"),
            ).eval().to("cuda" if torch.cuda.is_available() else "cpu")
        style = self._voices[chosen_voice][len(preview.phonemes) - 1].to(self._model.device)
        with torch.inference_mode():
            audio, _ = synthesize_with_curves(
                self._model, preview.phonemes, style, speed, controls.curves,
            )
        return _samples_to_wav_bytes(audio.detach().cpu().numpy(), SAMPLE_RATE)


def _interpolate(points, positions, reference):
    import numpy as np
    import torch

    values = np.interp(positions, [p.position for p in points], [p.value for p in points])
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def controlled_durations(duration, curves: list[PhonemeCurve]):
    """Sample timing at token centers before rounding and rebuilding alignment."""
    import numpy as np

    duration = duration.clone()
    for curve in curves:
        if curve.timing is not None:
            count = curve.end - curve.start
            positions = (np.arange(count) + 0.5) / count
            # Index zero is Kokoro's leading boundary token, never user-targeted.
            duration[0, curve.start + 1:curve.end + 1] *= _interpolate(curve.timing, positions, duration)
    durations = duration.round().clamp(min=1).long().squeeze(0)
    if durations.sum().item() > MAX_DURATION_FRAMES:
        raise ValueError("Requested duration exceeds 150 seconds; shorten the text or timing curves")
    return durations


def controlled_acoustics(f0, energy, durations, curves: list[PhonemeCurve]):
    """Map curves onto acoustic frames using the modified duration alignment."""
    import numpy as np
    import torch

    # Kokoro v1 predicts two acoustic frames for each duration/alignment frame.
    if f0.shape != energy.shape or f0.shape[-1] != 2 * int(durations.sum().item()):
        raise RuntimeError("Unsupported Kokoro acoustic/duration alignment")
    edges = np.concatenate(([0], durations.detach().cpu().numpy().cumsum())) * 2
    f0, energy = f0.clone(), energy.clone()
    for curve in curves:
        start, end = int(edges[curve.start + 1]), int(edges[curve.end + 1])
        positions = (np.arange(end - start) + 0.5) / (end - start)
        if curve.pitch is not None:
            shift = _interpolate(curve.pitch, positions, f0)
            original = f0[..., start:end]
            f0[..., start:end] = torch.where(original > 0, original * 2 ** (shift / 12), original)
        if curve.energy is not None:
            # N predicts natural-log spectral norm; a dB gain is additive here.
            energy[..., start:end] += _interpolate(curve.energy, positions, energy) * (math.log(10) / 20)
    return f0, energy


def synthesize_with_curves(model, phonemes, style, speed, curves):
    """Kokoro 0.9.4 inference with duration and F0/N control points.

    Follows the stages in https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py.
    Explicit intermediates keep edits request-local and precede waveform generation.
    """
    import torch

    tokens = [0, *(model.vocab[p] for p in phonemes), 0]
    ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
    lengths = torch.tensor([len(tokens)], device=model.device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    encoded = model.bert_encoder(model.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
    prosody_style = style[:, 128:]
    prediction = model.predictor.text_encoder(encoded, prosody_style, lengths, mask)
    recurrent, _ = model.predictor.lstm(prediction)
    duration = model.predictor.duration_proj(recurrent).sigmoid().sum(dim=-1) / speed
    durations = controlled_durations(duration, curves)

    # Index selection is equivalent to a one-hot alignment matrix multiplication,
    # without allocating a tokens-by-frames dense matrix.
    alignment = torch.repeat_interleave(torch.arange(len(tokens), device=model.device), durations)
    features = prediction.transpose(-1, -2).index_select(-1, alignment)
    f0, energy = model.predictor.F0Ntrain(features, prosody_style)
    f0, energy = controlled_acoustics(f0, energy, durations, curves)
    text_features = model.text_encoder(ids, lengths, mask).index_select(-1, alignment)
    audio = model.decoder(text_features, f0, energy, style[:, :128]).squeeze()
    return audio, durations
