"""Complete-session parsing, target construction, and padded batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from app.ml.segment_cropper.data import prepare_feature_frame


@dataclass(frozen=True)
class CropperSession:
    session_id: str
    telemetry_data: list[dict]
    annotations: list[tuple[int, int]]


@dataclass(frozen=True)
class CropperSequence:
    session_id: str
    features: np.ndarray
    targets: np.ndarray


def parse_session_chunk(chunk: Any, session_id: Any) -> CropperSession:
    """Validate the cropper's single-dataset, complete-session contract."""
    if not isinstance(chunk, dict):
        raise ValueError("segment_cropper session chunk must be an object")
    telemetry = chunk.get("telemetry_data")
    annotations = chunk.get("annotations")
    if not isinstance(telemetry, list) or not all(
        isinstance(row, dict) for row in telemetry
    ):
        raise ValueError("segment_cropper session telemetry_data must be a list of rows")
    if not telemetry:
        raise ValueError("segment_cropper session telemetry_data must not be empty")
    if not isinstance(annotations, list):
        raise ValueError("segment_cropper session annotations must be a list")

    valid_ranges: list[tuple[int, int]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("parent_id"):
            continue
        start = annotation.get("start_index")
        end = annotation.get("end_index")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(telemetry)
        ):
            continue
        valid_ranges.append((start, end))

    return CropperSession(
        session_id=str(session_id),
        telemetry_data=[dict(row) for row in telemetry],
        annotations=valid_ranges,
    )


def build_boundary_targets(
    sequence_length: int,
    annotations: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Build unioned start/end/inside targets using exclusive end ranges."""
    targets = np.zeros((sequence_length, 3), dtype=np.float32)
    for start, end in annotations:
        if start < 0 or end <= start or end > sequence_length:
            continue
        targets[start, 0] = 1.0
        targets[end - 1, 1] = 1.0
        targets[start:end, 2] = 1.0
    return targets


def build_sequence(session: CropperSession, feature_names: Sequence[str]) -> CropperSequence:
    frame = prepare_feature_frame(session.telemetry_data, feature_names)
    return CropperSequence(
        session_id=session.session_id,
        features=frame.to_numpy(dtype=np.float32),
        targets=build_boundary_targets(len(frame), session.annotations),
    )


def pad_cropper_batch(batch):
    features, targets = zip(*batch)
    masks = [torch.ones(len(item), dtype=torch.float32) for item in features]
    return (
        pad_sequence(features, batch_first=True),
        pad_sequence(targets, batch_first=True),
        pad_sequence(masks, batch_first=True),
    )


__all__ = [
    "CropperSequence",
    "CropperSession",
    "build_boundary_targets",
    "build_sequence",
    "pad_cropper_batch",
    "parse_session_chunk",
]
