"""Temporal sequence construction and streaming classifier dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from app.shared.labels import BEHAVIOR_LABELS, normalize_label_ids
from app.shared.segment import AnnotatedSegment


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append first-order differences while preserving row alignment."""
    return pd.concat([df, df.diff().fillna(0).add_suffix("_diff")], axis=1)


@dataclass(frozen=True)
class TemporalSequence:
    features: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    start_index: int


def _chunk_records(chunk: Any) -> list[dict]:
    if isinstance(chunk, list):
        return [record for record in chunk if isinstance(record, dict)]
    if isinstance(chunk, dict) and isinstance(chunk.get("data"), list):
        return [record for record in chunk["data"] if isinstance(record, dict)]
    if isinstance(chunk, dict) and isinstance(chunk.get("payload"), dict):
        return [chunk["payload"]]
    return [chunk] if isinstance(chunk, dict) else []


def _contiguous_index_runs(indices: Sequence[int]) -> Iterator[list[int]]:
    current: list[int] = []
    for index in indices:
        if current and index != current[-1] + 1:
            yield current
            current = []
        current.append(index)
    if current:
        yield current


def build_temporal_sequences(
    chunk: Any,
    *,
    expected_features: Sequence[str],
    label_ids: Sequence[str],
    child_parent: Mapping[str, str],
) -> list[TemporalSequence]:
    """Rebuild contiguous session runs and their per-timestep targets."""
    annotations = []
    for record in _chunk_records(chunk):
        try:
            annotation = AnnotatedSegment.from_dict(record)
        except Exception:
            continue
        if annotation.start_index is None or annotation.end_index is None:
            continue
        annotations.append(annotation)

    parents = [
        annotation
        for annotation in annotations
        if not annotation.parent_id and annotation.telemetry_data
    ]
    if not parents:
        return []

    rows_by_index: Dict[int, dict] = {}
    for parent in sorted(parents, key=lambda item: (int(item.start_index), int(item.end_index))):
        start = int(parent.start_index)
        end = int(parent.end_index)
        for offset, row in enumerate(parent.telemetry_data):
            index = start + offset
            if index >= end:
                break
            if isinstance(row, dict):
                rows_by_index.setdefault(index, row)

    label_index = {label_id: index for index, label_id in enumerate(label_ids)}
    behavior_ids = tuple(label_id for label_id in BEHAVIOR_LABELS if label_id in label_index)
    sequences: list[TemporalSequence] = []

    for run_indices in _contiguous_index_runs(sorted(rows_by_index)):
        run_start = run_indices[0]
        run_end = run_indices[-1] + 1
        frame = pd.DataFrame([rows_by_index[index] for index in run_indices])
        frame = frame.reindex(columns=list(expected_features), fill_value=0)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
        frame = compute_derived_features(frame)
        if frame.empty:
            continue

        targets = np.zeros((len(frame), len(label_ids)), dtype=np.float32)
        loss_mask = np.ones_like(targets)

        for annotation in parents:
            start = max(run_start, int(annotation.start_index))
            end = min(run_end, int(annotation.end_index))
            if end <= start:
                continue
            local_start = start - run_start
            local_end = end - run_start
            for label_id in normalize_label_ids(annotation.labels):
                if label_id in behavior_ids:
                    targets[local_start:local_end, label_index[label_id]] = 1.0

        for annotation in annotations:
            if not annotation.parent_id:
                continue
            start = max(run_start, int(annotation.start_index))
            end = min(run_end, int(annotation.end_index))
            if end <= start:
                continue
            local_start = start - run_start
            local_end = end - run_start
            for label_id in normalize_label_ids(annotation.labels):
                if label_id in child_parent:
                    targets[local_start:local_end, label_index[label_id]] = 1.0

        sequences.append(TemporalSequence(
            features=frame.to_numpy(dtype=np.float32),
            targets=targets,
            loss_mask=loss_mask,
            start_index=run_start,
        ))

    return sequences


class TemporalStreamingDataset(IterableDataset):
    def __init__(
        self,
        store,
        cache_key: str,
        scaler,
        expected_features: Sequence[str],
        label_ids: Sequence[str],
        child_parent: Mapping[str, str],
    ) -> None:
        self.store = store
        self.cache_key = cache_key
        self.scaler = scaler
        self.expected_features = tuple(expected_features)
        self.label_ids = tuple(label_ids)
        self.child_parent = dict(child_parent)

    def __iter__(self):
        for chunk in self.store.get_cached_data_chunks(self.cache_key):
            for sequence in build_temporal_sequences(
                chunk,
                expected_features=self.expected_features,
                label_ids=self.label_ids,
                child_parent=self.child_parent,
            ):
                scaled = self.scaler.transform(sequence.features)
                yield (
                    torch.tensor(scaled, dtype=torch.float32),
                    torch.tensor(sequence.targets, dtype=torch.float32),
                    torch.tensor(sequence.loss_mask, dtype=torch.float32),
                )


def pad_temporal_batch(batch):
    features, targets, masks = zip(*batch)
    return (
        pad_sequence(features, batch_first=True),
        pad_sequence(targets, batch_first=True),
        pad_sequence(masks, batch_first=True),
    )


__all__ = [
    "TemporalSequence",
    "TemporalStreamingDataset",
    "build_temporal_sequences",
    "compute_derived_features",
    "pad_temporal_batch",
]
