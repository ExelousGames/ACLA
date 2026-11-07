"""Validated phoneme-local controls for Kokoro's intermediate predictions."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    position: float = Field(ge=0, le=1, description="Position within the selected range.")
    value: float


Curve = Annotated[list[CurvePoint], Field(min_length=2, max_length=64)]


class PhonemeCurve(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0, strict=True, description="First phoneme index, inclusive.")
    end: int = Field(gt=0, strict=True, description="Last phoneme index, exclusive.")
    timing: Curve | None = Field(
        None, description="Duration multiplier, 0.25–4; 1 leaves timing unchanged.",
    )
    energy: Curve | None = Field(
        None, description="Log-energy adjustment in dB, -12–12; 0 is neutral.",
    )
    pitch: Curve | None = Field(
        None, description="Pitch shift in semitones, -12–12; 0 is neutral.",
    )

    @model_validator(mode="after")
    def validate_curves(self) -> PhonemeCurve:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if all(getattr(self, name) is None for name in ("timing", "energy", "pitch")):
            raise ValueError("At least one of timing, energy, or pitch is required")
        for name, lower, upper in (("timing", 0.25, 4), ("energy", -12, 12), ("pitch", -12, 12)):
            points = getattr(self, name)
            if points is None:
                continue
            if points[0].position != 0 or points[-1].position != 1:
                raise ValueError(f"{name} must have endpoints at positions 0 and 1")
            if any(a.position >= b.position for a, b in zip(points, points[1:])):
                raise ValueError(f"{name} positions must be strictly increasing")
            if any(not lower <= point.value <= upper for point in points):
                raise ValueError(f"{name} values must be between {lower} and {upper}")
        return self


class ProsodyControls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_phonemes: str = Field(
        min_length=1, max_length=510,
        description="Exact phoneme string returned by POST /voice/phonemes for this text/language.",
    )
    curves: list[PhonemeCurve] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def validate_ranges(self) -> ProsodyControls:
        previous_end = 0
        for curve in sorted(self.curves, key=lambda item: item.start):
            if curve.end > len(self.expected_phonemes):
                raise ValueError("Curve range exceeds the expected phoneme sequence")
            if curve.start < previous_end:
                raise ValueError("Curve ranges must not overlap; combine controls in one range")
            previous_end = curve.end
        return self


class PhonemeToken(BaseModel):
    index: int
    symbol: str
    token_id: int


class PhonemePreview(BaseModel):
    phonemes: str
    tokens: list[PhonemeToken]
    sample_rate: int = 24000
    max_phonemes: int = 510
