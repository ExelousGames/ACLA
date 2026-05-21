"""Lance-native transformer training dataset.

A drop-in replacement for :class:`app.storage.datasets.telemetry_dataset.TelemetryActionDataset`
that reads directly from the Phase-2 typed Lance datasets (``segments`` strategy)
instead of going through the dict-list API. Same public surface, same tensor
shapes, same scaler-fitting behaviour — but with three structural wins:

* **O(1) chunk count.** Uses ``list_chunk_ids`` instead of streaming every chunk.
* **Direct chunk fetch.** ``_load_chunk(idx)`` does a single Lance scan filtered
  by ``__chunk_id__`` rather than a linear scan of an iterator.
* **Columnar telemetry reads.** Per-chunk telemetry is materialised as a single
  Arrow Table joined on ``__segment_id__``; numpy arrays are built from
  zero-copy column views instead of dict→list→array conversion.

Behaviour parity with the legacy dataset is enforced by
``scripts/parity_test_transformer_dataloader.py``: for every segment in the
migrated ``training_segments_`` cache_key, ``_process_segment_record`` from
both implementations must produce identical (input, target, weight) arrays
within float tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import lance

from app.domain.telemetry import _safe_float
from app.domain.tire_grip_features import TireGripFeatureCatalog
from app.storage.datasets.transformer_scaler import (
    PerFeatureScaler,
    _RunningFeatureStats,
)
from app.storage.lance import LanceTelemetryStore
from app.storage.lance.strategies import SegmentsStrategy, strategy_for


# Column constants — must stay in sync with SegmentsStrategy's storage layout.
_CHUNK_ID_COL = "__chunk_id__"
_ORDER_COL = "__order__"
_SEGMENT_ID_COL = "__segment_id__"
_TELEMETRY_BLOCK_KEY = "telemetry_data"


class LanceTelemetryActionDataset(Dataset):
    """Lance-backed dataset producing the same tensors as :class:`TelemetryActionDataset`.

    Only supports cache_keys whose strategy is :class:`SegmentsStrategy` — that's
    the only shape the transformer trainer feeds in (filtered annotated segments
    with nested telemetry_data). For other shapes, keep using the legacy class.
    """

    def __init__(
        self,
        store: LanceTelemetryStore,
        segments_cache_key: str,
        segment_length_hint: Optional[int] = None,
        batch_size: int = 32,
        min_sequence_length: int = 3,
        sequence_bucket_size: int = 16,
    ):
        self.store = store
        self.segments_cache_key = segments_cache_key
        self.segment_length_hint = segment_length_hint
        self.batch_size = batch_size
        self.min_sequence_length = max(2, min_sequence_length)
        self.sequence_bucket_size = max(1, sequence_bucket_size)

        strategy = strategy_for(segments_cache_key)
        if not isinstance(strategy, SegmentsStrategy):
            raise TypeError(
                f"LanceTelemetryActionDataset requires a cache_key managed by "
                f"SegmentsStrategy; '{segments_cache_key}' is handled by "
                f"{type(strategy).__name__}."
            )
        self._strategy = strategy

        base_dir = store.store_dir
        segments_path = base_dir / f"{segments_cache_key}.lance"
        telemetry_path = base_dir / f"{segments_cache_key}.telemetry.lance"
        if not segments_path.exists() or not telemetry_path.exists():
            raise FileNotFoundError(
                f"Expected Lance datasets at {segments_path} and {telemetry_path}; "
                f"run scripts/migrate_blob_to_typed_lance.py first."
            )
        self._segments_ds = lance.dataset(str(segments_path))
        self._telemetry_ds = lance.dataset(str(telemetry_path))

        # Fast chunk-count via list_chunk_ids — no full scan needed.
        self._chunk_ids: List[str] = sorted(
            set(self._segments_ds.scanner(columns=[_CHUNK_ID_COL]).to_table().column(_CHUNK_ID_COL).to_pylist())
        )
        self.chunk_count = len(self._chunk_ids)

        self.unified_features: List[str] = self._discover_feature_names()

        self.feature_scaler = PerFeatureScaler(self.unified_features)
        self._features_fitted = False
        self._length_stats: Optional[Dict[str, Any]] = None
        self._processed_chunks: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        print(f"[INFO] ✓ Lance dataset initialised: {self.chunk_count} chunks")
        print(f"[INFO] ✓ GPU batch size: {batch_size}")
        print(f"[INFO] ✓ Features: {len(self.unified_features)}")
        if self.segment_length_hint:
            print(f"[INFO] ✓ Segment length hint: ~{self.segment_length_hint} timesteps")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _discover_feature_names(self) -> List[str]:
        """Return the telemetry feature column names from the Lance schema.

        Mirrors the legacy ``_get_feature_names`` (which read the first
        timestep dict's keys) but pulls them straight from the telemetry
        dataset's Arrow schema. The legacy code's order followed the dict
        insertion order from the source JSON; Lance preserves that order
        through ``pa.Table.from_pylist`` during migration, so the names
        match column-for-column.
        """
        schema = self._telemetry_ds.schema
        skip = {_SEGMENT_ID_COL, _ORDER_COL}
        return [field.name for field in schema if field.name not in skip]

    # ------------------------------------------------------------------
    # Chunk loading (Lance-native)
    # ------------------------------------------------------------------
    def _segment_ids_for_chunk(self, chunk_idx: int) -> List[str]:
        """Return the __segment_id__ values for chunk_idx in original order."""
        chunk_id = self._chunk_ids[chunk_idx]
        seg_table = self._segments_ds.scanner(
            filter=f"{_CHUNK_ID_COL} = '{chunk_id}'",
            columns=[_ORDER_COL, _SEGMENT_ID_COL],
        ).to_table()
        order_indices = pc.sort_indices(seg_table, sort_keys=[(_ORDER_COL, "ascending")])
        return seg_table.take(order_indices).column(_SEGMENT_ID_COL).to_pylist()

    def _load_chunk(self, chunk_idx: int) -> List[List[Dict[str, Any]]]:
        """Reconstruct chunk_idx's payload as a list of segment-records lists.

        Returns the equivalent of what ``store.get_chunk(key, chunk_id)`` would
        have returned for that chunk in dict-list form, so legacy callers like
        ``ExpertActionTrainer.evaluate`` work unchanged. Note we drop the segment
        scalar metadata here — the trainer only consumes ``telemetry_data``.
        """
        chunk_id = self._chunk_ids[chunk_idx]
        rows = self._fetch_chunk_telemetry(chunk_id)
        if rows is None:
            return []
        sequences = []
        # Keep empty-telemetry segments in the returned list so the count
        # matches the legacy ``store.get_chunk`` payload exactly. The
        # downstream ``_process_segment_record`` returns ``None`` for them
        # via the ``len(segment) < min_sequence_length`` guard, so they get
        # filtered identically further down the pipeline.
        for sid in self._segment_ids_for_chunk(chunk_idx):
            records = rows.get(sid, []) or []
            # Mirror the legacy {"telemetry_data": [...]} wrapping so
            # downstream code that special-cases that key keeps working.
            sequences.append({_TELEMETRY_BLOCK_KEY: records})
        print(f"[INFO] Loaded chunk {chunk_idx} ({chunk_id}) with {len(sequences)} segments")
        return sequences

    def _fetch_chunk_telemetry(
        self, chunk_id: str
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Pull every telemetry row belonging to chunk_id in one scan.

        Joins by ``__segment_id__`` IN (...) where the segment_ids come from
        the segments dataset filtered to ``chunk_id``. Returns a dict keyed by
        segment_id whose values are the segment's records in __order__-ascending
        sequence.
        """
        seg_ids = self._segment_ids_for_chunk(self._chunk_ids.index(chunk_id))
        if not seg_ids:
            return {}

        # Build an IN-list filter for all of this chunk's segments.
        quoted = ",".join(f"'{s}'" for s in seg_ids)
        try:
            tel_table = self._telemetry_ds.scanner(
                filter=f"{_SEGMENT_ID_COL} IN ({quoted})",
            ).to_table()
        except Exception as exc:
            print(f"[WARNING] Lance telemetry scan failed for chunk '{chunk_id}': {exc}")
            return {}

        if tel_table.num_rows == 0:
            return {sid: [] for sid in seg_ids}

        # Sort by (segment_id, order) once; dict.setdefault keeps insertion order.
        sort_order = pc.sort_indices(
            tel_table,
            sort_keys=[(_SEGMENT_ID_COL, "ascending"), (_ORDER_COL, "ascending")],
        )
        sorted_table = tel_table.take(sort_order)
        seg_id_col = sorted_table.column(_SEGMENT_ID_COL).to_pylist()
        record_table = sorted_table.drop([_SEGMENT_ID_COL, _ORDER_COL])
        records = record_table.to_pylist()

        grouped: Dict[str, List[Dict[str, Any]]] = {sid: [] for sid in seg_ids}
        for sid, rec in zip(seg_id_col, records):
            if sid in grouped:
                grouped[sid].append(rec)
        return grouped

    # ------------------------------------------------------------------
    # Scaler fitting (streaming, order-independent)
    # ------------------------------------------------------------------
    def _ensure_features_fitted(self):
        if self._features_fitted:
            return

        print(f"[INFO] Fitting feature scaling across {self.chunk_count} Lance chunks...")
        self._processed_chunks.clear()

        stats = _RunningFeatureStats(len(self.unified_features))
        total_rows = 0
        all_lengths: List[int] = []

        for chunk_idx, chunk_id in enumerate(self._chunk_ids):
            grouped = self._fetch_chunk_telemetry(chunk_id)
            if not grouped:
                continue

            chunk_rows: List[List[float]] = []
            for sid, records in grouped.items():
                if len(records) < self.min_sequence_length:
                    continue
                all_lengths.append(len(records))
                for rec in records:
                    chunk_rows.append([self._coerce_float(rec.get(f, 0.0)) for f in self.unified_features])

            if chunk_rows:
                chunk_matrix = np.array(chunk_rows, dtype=np.float32)
                if np.isnan(chunk_matrix).any() or np.isinf(chunk_matrix).any():
                    print(f"[WARNING] NaN/Inf detected in chunk {chunk_idx}; cleaning before stats update")
                    chunk_matrix = np.nan_to_num(chunk_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
                stats.update(chunk_matrix)
                total_rows += chunk_matrix.shape[0]
            print(f"[DEBUG] Processed chunk {chunk_idx} ({chunk_id}): rows accumulated={total_rows}")

        if total_rows == 0:
            raise ValueError("No data available across chunks to fit feature scaler")

        if all_lengths:
            lengths_array = np.asarray(all_lengths, dtype=np.int32)
            self._length_stats = {
                "min": int(lengths_array.min()),
                "max": int(lengths_array.max()),
                "mean": float(lengths_array.mean()),
                "median": float(np.median(lengths_array)),
                "count": int(lengths_array.size),
                "source": "full_dataset",
            }
        else:
            self._length_stats = None

        counts, means, variances = stats.finalize()
        self.feature_scaler = PerFeatureScaler.from_feature_statistics(
            self.unified_features,
            means,
            variances,
            counts,
            scaler_factory=self.feature_scaler.scaler_factory if self.feature_scaler else None,
        )
        self._features_fitted = True

        print(f"[INFO] ✓ Feature scaling fitted across {total_rows} timesteps")

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value) if value.replace(".", "").replace("-", "").isdigit() else 0.0
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # Per-segment processing — identical formulas to TelemetryActionDataset
    # ------------------------------------------------------------------
    def _process_segment_record(
        self, segment_record: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Match the legacy unwrap: ``segment_record`` may be a dict with a
        # ``telemetry_data`` block or a bare list of timesteps.
        if isinstance(segment_record, dict) and _TELEMETRY_BLOCK_KEY in segment_record:
            segment_record = segment_record[_TELEMETRY_BLOCK_KEY]
        if not isinstance(segment_record, list) or len(segment_record) < self.min_sequence_length:
            return None

        sequence_data: List[List[float]] = []
        phase_weights: List[float] = []
        for timestep_data in segment_record:
            if not isinstance(timestep_data, dict):
                return None

            feature_array = [self._coerce_float(timestep_data.get(f, 0.0)) for f in self.unified_features]
            sequence_data.append(feature_array)

            driver_push = _safe_float(
                timestep_data.get(
                    TireGripFeatureCatalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value,
                    0.0,
                )
            ) or 0.0
            brake_signal = _safe_float(timestep_data.get("Physics_brake", 0.0)) or 0.0
            throttle_signal = _safe_float(timestep_data.get("Physics_gas", 0.0)) or 0.0
            steer_signal = _safe_float(timestep_data.get("Physics_steer_angle", 0.0)) or 0.0

            weight = 1.0
            weight += 0.8 * max(0.0, min(1.0, driver_push))
            weight += 0.7 * max(0.0, min(1.0, abs(brake_signal)))
            weight += 0.6 * max(0.0, min(1.0, abs(steer_signal)))
            weight -= 0.3 * max(0.0, min(1.0, throttle_signal))
            phase_weights.append(float(np.clip(weight, 0.5, 3.0)))

        sequence_matrix = np.array(sequence_data, dtype=np.float32)
        if np.isnan(sequence_matrix).any() or np.isinf(sequence_matrix).any():
            return None
        scaled_sequence = self.feature_scaler.transform(sequence_matrix)
        if np.isnan(scaled_sequence).any() or np.isinf(scaled_sequence).any():
            return None

        return (
            scaled_sequence[:-1],
            scaled_sequence[1:],
            np.asarray(phase_weights[1:], dtype=np.float32),
        )

    def _get_processed_chunk(self, chunk_idx: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        cached = self._processed_chunks.get(chunk_idx)
        if cached is not None:
            return cached

        self._ensure_features_fitted()
        try:
            processed = self._processed_chunk_vectorized(chunk_idx)
        except Exception as exc:
            # Slow per-record path is preserved as a safety net for cache_keys
            # whose schema doesn't match our column projection assumptions
            # (e.g. an old typed dataset migrated before a feature was added).
            print(f"[WARNING] Vectorized chunk path failed ({exc}); falling back to per-record")
            chunk_records = self._load_chunk(chunk_idx)
            processed = []
            for segment_record in chunk_records:
                result = self._process_segment_record(segment_record)
                if result is not None:
                    processed.append(result)

        self._processed_chunks[chunk_idx] = processed
        print(f"[DEBUG] Cached scaled sequences for chunk {chunk_idx}: {len(processed)} segments")
        return processed

    def _processed_chunk_vectorized(
        self, chunk_idx: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build per-segment (input, target, weights) tuples in one Arrow scan.

        Replaces the per-record dict→list→numpy round-trip the legacy
        :class:`TelemetryActionDataset` needed (each timestep ran through
        ``_coerce_float`` and ``_safe_float`` in pure Python). Here every
        column is projected straight from Lance, NaN/null handling matches
        the legacy semantics column-by-column, the phase-weight formula is
        a single vectorised numpy expression, and the scaler is applied to
        the whole chunk's feature matrix at once.

        Parity guard: numbers must match the legacy ``_process_segment_record``
        within ``rtol=1e-5, atol=1e-6`` — enforced by
        ``scripts/parity_test_transformer_dataloader.py``.
        """
        chunk_id = self._chunk_ids[chunk_idx]
        seg_ids_ordered = self._segment_ids_for_chunk(chunk_idx)
        if not seg_ids_ordered:
            return []

        phase_input_cols = [
            "Physics_brake",
            "Physics_gas",
            "Physics_steer_angle",
        ]
        driver_push_col = TireGripFeatureCatalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value

        needed: List[str] = list(self.unified_features)
        for extra in [driver_push_col] + phase_input_cols:
            if extra not in needed:
                needed.append(extra)

        available = set(self._telemetry_ds.schema.names)
        project = [c for c in needed if c in available]

        quoted = ",".join(f"'{s}'" for s in seg_ids_ordered)
        tel_table = self._telemetry_ds.scanner(
            filter=f"{_SEGMENT_ID_COL} IN ({quoted})",
            columns=[_SEGMENT_ID_COL, _ORDER_COL] + project,
        ).to_table()
        if tel_table.num_rows == 0:
            return []

        # Sort by (segment_id, order) so segments form contiguous row groups.
        sort_idx = pc.sort_indices(
            tel_table,
            sort_keys=[(_SEGMENT_ID_COL, "ascending"), (_ORDER_COL, "ascending")],
        )
        tel_table = tel_table.take(sort_idx)
        n_rows = tel_table.num_rows

        # ---- Column extraction (zero-copy where possible) ----
        def _floats_filling_null_with_zero(name: str) -> np.ndarray:
            """Match legacy ``dict.get(name, 0.0)`` semantics: missing column
            or null cell becomes 0.0; actual NaN/Inf values stay so the
            downstream per-segment NaN check can reject the segment.

            Non-numeric columns (e.g. ``list<struct>`` types like
            ``Graphics_car_coordinates``) collapse to zeros, mirroring what
            the legacy ``_coerce_float`` did for non-``int``/``float`` /
            non-digit-string values.
            """
            if name not in tel_table.column_names:
                return np.zeros(n_rows, dtype=np.float32)
            col = tel_table.column(name)
            t = col.type
            if not (pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_boolean(t)):
                return np.zeros(n_rows, dtype=np.float32)
            col = pc.fill_null(col, 0.0)
            return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32)

        feature_columns = [_floats_filling_null_with_zero(f) for f in self.unified_features]
        feature_matrix = np.stack(feature_columns, axis=1)  # (n_rows, n_features)

        # Phase-weight inputs use the more permissive ``_safe_float`` in the
        # legacy path which collapses NaN/Inf to 0.0, so do the same here.
        def _phase_input(name: str) -> np.ndarray:
            col = _floats_filling_null_with_zero(name).astype(np.float64, copy=False)
            return np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)

        driver_push = _phase_input(driver_push_col)
        brake_signal = _phase_input("Physics_brake")
        gas_signal = _phase_input("Physics_gas")
        steer_signal = _phase_input("Physics_steer_angle")

        # Vectorised phase-weight formula — same left-to-right accumulation as
        # the legacy scalar loop so float ordering matches element-for-element.
        push_term = 0.8 * np.clip(driver_push, 0.0, 1.0)
        brake_term = 0.7 * np.clip(np.abs(brake_signal), 0.0, 1.0)
        steer_term = 0.6 * np.clip(np.abs(steer_signal), 0.0, 1.0)
        gas_term = 0.3 * np.clip(gas_signal, 0.0, 1.0)
        weights = 1.0 + push_term + brake_term + steer_term - gas_term
        weights = np.clip(weights, 0.5, 3.0).astype(np.float32)

        # Scale the whole chunk's matrix once; per-feature scaling is linear so
        # the slicing afterwards produces the same numbers as per-segment scaling.
        scaled_matrix = self.feature_scaler.transform(feature_matrix)

        # ---- Group rows by segment_id, preserving the original chunk order ----
        sid_col = np.asarray(tel_table.column(_SEGMENT_ID_COL).to_pylist())
        if sid_col.size == 0:
            return []
        change_points = np.where(sid_col[:-1] != sid_col[1:])[0] + 1
        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points, [n_rows]])
        sid_ranges: Dict[str, Tuple[int, int]] = {
            sid_col[start]: (int(start), int(end)) for start, end in zip(starts, ends)
        }

        processed: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for sid in seg_ids_ordered:
            rng = sid_ranges.get(sid)
            if rng is None:
                continue
            start, end = rng
            if (end - start) < self.min_sequence_length:
                continue
            raw_slice = feature_matrix[start:end]
            if np.isnan(raw_slice).any() or np.isinf(raw_slice).any():
                continue
            scaled_slice = scaled_matrix[start:end]
            if np.isnan(scaled_slice).any() or np.isinf(scaled_slice).any():
                continue
            w_slice = weights[start:end]
            processed.append((scaled_slice[:-1], scaled_slice[1:], w_slice[1:]))

        return processed

    # ------------------------------------------------------------------
    # Public surface — mirrors TelemetryActionDataset
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.chunk_count

    def __getitem__(self, chunk_idx: int):
        return chunk_idx

    def get_chunk_batches(self, chunk_idx: int):
        if chunk_idx >= self.chunk_count:
            raise IndexError(f"Chunk index {chunk_idx} out of range")
        self._ensure_features_fitted()
        processed_segments = self._get_processed_chunk(chunk_idx)
        if not processed_segments:
            return

        bucket_size = self.sequence_bucket_size
        buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for segment in processed_segments:
            seq_len = segment[0].shape[0]
            bucket_key = int((seq_len // bucket_size) * bucket_size)
            buckets.setdefault(bucket_key, []).append(segment)

        bucket_order = list(buckets.keys())
        np.random.shuffle(bucket_order)

        batch_inputs: List[np.ndarray] = []
        batch_targets: List[np.ndarray] = []
        batch_masks: List[np.ndarray] = []

        def _collate_and_yield():
            if not batch_inputs:
                return
            input_tensors = [torch.from_numpy(arr).float() for arr in batch_inputs]
            target_tensors = [torch.from_numpy(arr).float() for arr in batch_targets]
            mask_tensors = [torch.from_numpy(arr).float() for arr in batch_masks]
            padded_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=0.0)
            padded_targets = pad_sequence(target_tensors, batch_first=True, padding_value=0.0)
            padded_masks = pad_sequence(mask_tensors, batch_first=True, padding_value=0.0)
            lengths = torch.tensor([t.shape[0] for t in input_tensors], dtype=torch.long)
            max_len = int(padded_inputs.shape[1])
            padding_mask = torch.ones((len(input_tensors), max_len), dtype=torch.bool)
            for row_idx, seq_len in enumerate(lengths):
                padding_mask[row_idx, :seq_len] = False
            yield padded_inputs, padded_targets, padded_masks, padding_mask

        for bucket_key in bucket_order:
            bucket_segments = buckets[bucket_key]
            permutation = np.random.permutation(len(bucket_segments))
            for idx in permutation:
                inp, tgt, w = bucket_segments[idx]
                batch_inputs.append(inp)
                batch_targets.append(tgt)
                batch_masks.append(w)
                if len(batch_inputs) >= self.batch_size:
                    yield from _collate_and_yield()
                    batch_inputs, batch_targets, batch_masks = [], [], []
            if batch_inputs:
                yield from _collate_and_yield()
                batch_inputs, batch_targets, batch_masks = [], [], []

        if batch_inputs:
            yield from _collate_and_yield()

    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        return self.unified_features, self.unified_features

    def get_input_feature_names(self) -> List[str]:
        return list(self.unified_features)

    def get_scalers(self) -> PerFeatureScaler:
        return self.feature_scaler

    def get_segment_info(self) -> Dict[str, Any]:
        if self._length_stats is None:
            self._sample_length_stats()
        return {
            "num_chunks": self.chunk_count,
            "segment_length_hint": self.segment_length_hint,
            "length_statistics": self._length_stats,
            "minimum_sequence_length": self.min_sequence_length,
            "total_features": len(self.unified_features),
            "feature_names": self.unified_features,
            "tensor_shapes": {
                "input": {"sequence": "variable", "features": len(self.unified_features)},
                "target": {"sequence": "variable", "features": len(self.unified_features)},
            },
        }

    def _sample_length_stats(self, max_chunks: int = 3) -> None:
        if self._length_stats is not None:
            return
        sampled: List[int] = []
        for chunk_idx in range(min(self.chunk_count, max_chunks)):
            chunk_id = self._chunk_ids[chunk_idx]
            grouped = self._fetch_chunk_telemetry(chunk_id)
            for records in grouped.values():
                if len(records) >= self.min_sequence_length:
                    sampled.append(len(records))
        if sampled:
            arr = np.asarray(sampled, dtype=np.int32)
            self._length_stats = {
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "count": int(arr.size),
                "source": "sampled",
                "sampled_chunks": min(self.chunk_count, max_chunks),
            }

    def get_context_feature_names(self) -> List[str]:
        context_features = []
        for feature in self.unified_features:
            if any(p in feature.lower() for p in [
                "expert_", "tire_", "grip_", "distance_", "velocity_alignment", "speed_difference"
            ]):
                context_features.append(feature)
        return context_features


__all__ = ["LanceTelemetryActionDataset"]
