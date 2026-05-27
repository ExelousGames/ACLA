"""Streaming dataset + derived-feature helper for the segment classifier.

Two pieces sit here:

  - ``compute_derived_features``: appends first-order deltas (`*_diff`)
    to every column of a telemetry DataFrame. Pure function over a
    DataFrame; not specific to the classifier and could grow more
    callers later.

  - ``StreamingSegmentDataset``: PyTorch ``IterableDataset`` that
    streams ``AnnotatedSegment`` records from the shared telemetry
    store (Lance-backed), applies scaling + label binarisation +
    padding, and yields ``(X, y, mask)`` tensors. Used by training
    (full pass) and by inference flows that need lazy iteration over
    a large dataset.

Lives in ``app/storage/datasets/`` because it owns I/O (reading from
the telemetry store) — not pure ML model code. Extracted from
``app/ml/segment_classifier/service.py`` in refactor/hexagonal-v4
(Page 5 of acla-ai-service-architecture.drawio).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from app.domain.segment import AnnotatedSegment


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived features for telemetry data.
    Adds first-order differences (deltas) for all columns.
    """
    # Calculate difference
    # We use fillna(0) for the first element
    df_diff = df.diff().fillna(0).add_suffix('_diff')

    # Concatenate
    df_combined = pd.concat([df, df_diff], axis=1)
    return df_combined


class StreamingSegmentDataset(IterableDataset):
    def __init__(self, store, cache_key, mlb, scaler, max_length, expected_features):
        self.store = store
        self.cache_key = cache_key
        self.mlb = mlb
        self.scaler = scaler
        self.max_length = max_length
        self.expected_features = expected_features

    def __iter__(self):
        chunks = self.store.get_cached_data_chunks(self.cache_key)

        for chunk in chunks:
            chunk_data = []
            if isinstance(chunk, list):
                chunk_data = chunk
            elif isinstance(chunk, dict) and "data" in chunk:
                 chunk_data = chunk["data"]
            elif isinstance(chunk, dict) and "payload" in chunk:
                 chunk_data = [chunk["payload"]]
            else:
                 chunk_data = [chunk]

            for d in chunk_data:
                if not isinstance(d, dict):
                    continue

                try:
                    ann = AnnotatedSegment.from_dict(d)
                except Exception:
                    continue

                if not ann.telemetry_data:
                    continue

                df = pd.DataFrame(ann.telemetry_data)

                # Fast path if columns match
                if df.columns.tolist() != self.expected_features:
                     df = df.reindex(columns=self.expected_features, fill_value=0)

                # Ensure numeric
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

                # Compute derived features (deltas)
                df = compute_derived_features(df)

                if df.empty:
                    continue

                seg_X = df.values

                # Use labels directly (IDs)
                mapped_labels = ann.labels

                # Create target
                label_vec = self.mlb.transform([mapped_labels])[0]
                seg_y = np.tile(label_vec, (len(seg_X), 1))

                # Scale
                scaled_X = self.scaler.transform(seg_X)

                # Create mask
                mask = np.ones((len(scaled_X), 1))

                # Pad
                pad_len = self.max_length - len(scaled_X)
                if pad_len > 0:
                    scaled_X = np.pad(scaled_X, ((0, pad_len), (0, 0)), 'constant')
                    seg_y = np.pad(seg_y, ((0, pad_len), (0, 0)), 'constant')
                    mask = np.pad(mask, ((0, pad_len), (0, 0)), 'constant')
                elif pad_len < 0:
                    # Truncate
                    scaled_X = scaled_X[:self.max_length]
                    seg_y = seg_y[:self.max_length]
                    mask = mask[:self.max_length]

                yield torch.FloatTensor(scaled_X), torch.FloatTensor(seg_y), torch.FloatTensor(mask)


__all__ = ["compute_derived_features", "StreamingSegmentDataset"]
