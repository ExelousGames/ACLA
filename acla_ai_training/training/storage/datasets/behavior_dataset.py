"""Streaming dataset and derived features for behavior classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from app.ml.behavior_classifier.label_heads import (
    PARENT_HEAD_NAME,
    behavior_parent_labels,
    head_is_active,
    labels_for_head,
)
from app.shared.labels import normalize_label_ids
from app.shared.segment import AnnotatedSegment


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append first-order differences for every telemetry feature."""
    return pd.concat([df, df.diff().fillna(0).add_suffix("_diff")], axis=1)


class BehaviorStreamingDataset(IterableDataset):
    def __init__(self, store, cache_key, head_mlbs, head_specs, scaler, max_length, expected_features):
        self.store = store
        self.cache_key = cache_key
        self.head_mlbs = head_mlbs
        self.head_specs = head_specs
        self.scaler = scaler
        self.max_length = max_length
        self.expected_features = expected_features

    def __iter__(self):
        for chunk in self.store.get_cached_data_chunks(self.cache_key):
            if isinstance(chunk, list):
                chunk_data = chunk
            elif isinstance(chunk, dict) and "data" in chunk:
                chunk_data = chunk["data"]
            elif isinstance(chunk, dict) and "payload" in chunk:
                chunk_data = [chunk["payload"]]
            else:
                chunk_data = [chunk]

            for raw_segment in chunk_data:
                if not isinstance(raw_segment, dict):
                    continue
                try:
                    segment = AnnotatedSegment.from_dict(raw_segment)
                except Exception:
                    continue
                if not segment.telemetry_data:
                    continue

                labels = normalize_label_ids(segment.labels)
                parent_labels = behavior_parent_labels(labels)
                if len(parent_labels) != 1:
                    continue

                df = pd.DataFrame(segment.telemetry_data)
                if df.columns.tolist() != self.expected_features:
                    df = df.reindex(columns=self.expected_features, fill_value=0)
                df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
                df = compute_derived_features(df)
                if df.empty:
                    continue

                scaled_X = self.scaler.transform(df.values)
                sequence_length = min(len(scaled_X), self.max_length)
                base_mask = np.ones((sequence_length, 1), dtype=np.float32)

                parent_mlb = self.head_mlbs[PARENT_HEAD_NAME]
                parent_index = list(parent_mlb.classes_).index(parent_labels[0])
                targets = {
                    PARENT_HEAD_NAME: np.full(sequence_length, parent_index, dtype=np.int64)
                }
                masks = {PARENT_HEAD_NAME: base_mask.copy()}

                for spec in self.head_specs:
                    if spec.is_parent:
                        continue
                    mlb = self.head_mlbs[spec.name]
                    label_vec = mlb.transform([labels_for_head(labels, spec)])[0]
                    targets[spec.name] = np.tile(label_vec, (sequence_length, 1)).astype(np.float32)
                    masks[spec.name] = (
                        base_mask.copy()
                        if head_is_active(labels, spec)
                        else np.zeros_like(base_mask)
                    )

                scaled_X = scaled_X[:self.max_length]
                pad_len = self.max_length - sequence_length
                if pad_len > 0:
                    scaled_X = np.pad(scaled_X, ((0, pad_len), (0, 0)), "constant")
                    targets[PARENT_HEAD_NAME] = np.pad(
                        targets[PARENT_HEAD_NAME], (0, pad_len), "constant"
                    )
                    for head_name in masks:
                        masks[head_name] = np.pad(masks[head_name], ((0, pad_len), (0, 0)), "constant")
                    for spec in self.head_specs:
                        if not spec.is_parent:
                            targets[spec.name] = np.pad(
                                targets[spec.name], ((0, pad_len), (0, 0)), "constant"
                            )

                yield (
                    torch.FloatTensor(scaled_X),
                    {
                        name: torch.LongTensor(value) if name == PARENT_HEAD_NAME else torch.FloatTensor(value)
                        for name, value in targets.items()
                    },
                    {name: torch.FloatTensor(value) for name, value in masks.items()},
                )


__all__ = ["BehaviorStreamingDataset", "compute_derived_features"]
