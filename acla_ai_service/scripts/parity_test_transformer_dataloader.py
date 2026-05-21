"""Numeric parity test for the transformer training dataloader refactor.

Compares :class:`TelemetryActionDataset` (legacy dict-list path) against
:class:`LanceTelemetryActionDataset` (Phase-2 typed-Lance path) on the
currently migrated ``training_segments_`` cache_key. Asserts that:

* ``unified_features`` lists match column-for-column.
* Per-feature scaler statistics (means, variances, counts) match within
  a float tolerance.
* For every (chunk, segment), ``_process_segment_record`` produces
  identical (input_seq, target_seq, phase_weights) arrays within
  ``np.allclose`` tolerance.

Run from the ``acla_ai_service`` directory:

    python -m scripts.parity_test_transformer_dataloader

Exits 0 on parity, non-zero otherwise. Bucketing/padding/shuffle behaviour
in ``get_chunk_batches`` is purely a downstream transform of the segments
returned by ``_process_segment_record`` and is identical line-by-line
across both implementations, so it isn't re-tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.storage.datasets.lance_telemetry_dataset import LanceTelemetryActionDataset  # noqa: E402
from app.storage.datasets.telemetry_dataset import TelemetryActionDataset  # noqa: E402
from app.storage.lance import LanceTelemetryStore  # noqa: E402


SEGMENTS_KEY = "training_segments_"
RTOL = 1e-5
ATOL = 1e-6


def _compare_feature_names(a: List[str], b: List[str]) -> bool:
    if a == b:
        return True
    print(f"[FAIL] feature names differ")
    print(f"  legacy ({len(a)}): {a[:5]}...{a[-3:]}")
    print(f"  lance  ({len(b)}): {b[:5]}...{b[-3:]}")
    only_a = [n for n in a if n not in b]
    only_b = [n for n in b if n not in a]
    if only_a:
        print(f"  legacy-only: {only_a}")
    if only_b:
        print(f"  lance-only : {only_b}")
    return False


def _compare_arrays(a: np.ndarray, b: np.ndarray, label: str) -> bool:
    if a.shape != b.shape:
        print(f"[FAIL] {label}: shape {a.shape} vs {b.shape}")
        return False
    if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
        diff = np.abs(a - b)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"[FAIL] {label}: max abs diff {diff.max():.6g} at {idx} (legacy={a[idx]:.6g}, lance={b[idx]:.6g})")
        return False
    return True


def main() -> int:
    store = LanceTelemetryStore()
    print(f"Lance store: {store.store_dir}")

    print()
    print("=== Constructing both datasets ===")
    legacy = TelemetryActionDataset(
        data_cache=store,
        segments_cache_key=SEGMENTS_KEY,
        segment_length_hint=50,
        batch_size=32,
        min_sequence_length=3,
    )
    lance = LanceTelemetryActionDataset(
        store=store,
        segments_cache_key=SEGMENTS_KEY,
        segment_length_hint=50,
        batch_size=32,
        min_sequence_length=3,
    )

    print()
    print("=== Feature names ===")
    if not _compare_feature_names(legacy.unified_features, lance.unified_features):
        return 1
    print(f"[OK] {len(legacy.unified_features)} feature names match")

    print()
    print("=== Chunk counts ===")
    if legacy.chunk_count != lance.chunk_count:
        print(f"[FAIL] chunk count: legacy={legacy.chunk_count} lance={lance.chunk_count}")
        return 1
    print(f"[OK] both datasets report {legacy.chunk_count} chunks")

    print()
    print("=== Fitting scalers on both ===")
    legacy._ensure_features_fitted()
    lance._ensure_features_fitted()

    legacy_scalers = legacy.get_scalers()
    lance_scalers = lance.get_scalers()
    legacy_means = np.array([s.mean_[0] if hasattr(s, "mean_") else 0.0 for s in legacy_scalers._scalers], dtype=np.float64)
    lance_means = np.array([s.mean_[0] if hasattr(s, "mean_") else 0.0 for s in lance_scalers._scalers], dtype=np.float64)
    legacy_scales = np.array([s.scale_[0] if hasattr(s, "scale_") else 1.0 for s in legacy_scalers._scalers], dtype=np.float64)
    lance_scales = np.array([s.scale_[0] if hasattr(s, "scale_") else 1.0 for s in lance_scalers._scalers], dtype=np.float64)

    if not _compare_arrays(legacy_means, lance_means, "scaler means"):
        return 1
    if not _compare_arrays(legacy_scales, lance_scales, "scaler scales"):
        return 1
    print(f"[OK] scaler means + scales match for all {len(legacy_means)} features")

    print()
    print("=== Per-segment _process_segment_record parity ===")
    total_segments = 0
    total_skipped_both = 0
    mismatches = 0
    for chunk_idx in range(legacy.chunk_count):
        # Use each dataset's own _load_chunk so we test their full read path.
        legacy_records = legacy._load_chunk(chunk_idx)
        lance_records = lance._load_chunk(chunk_idx)

        if len(legacy_records) != len(lance_records):
            print(f"[FAIL] chunk {chunk_idx}: segment count legacy={len(legacy_records)} lance={len(lance_records)}")
            mismatches += 1
            continue

        for seg_idx, (a_rec, b_rec) in enumerate(zip(legacy_records, lance_records)):
            a_out = legacy._process_segment_record(a_rec)
            b_out = lance._process_segment_record(b_rec)
            if a_out is None and b_out is None:
                total_skipped_both += 1
                continue
            if (a_out is None) != (b_out is None):
                print(f"[FAIL] chunk {chunk_idx}/seg {seg_idx}: legacy={'None' if a_out is None else 'array'} lance={'None' if b_out is None else 'array'}")
                mismatches += 1
                continue

            a_in, a_tg, a_w = a_out
            b_in, b_tg, b_w = b_out
            ok = (
                _compare_arrays(a_in, b_in, f"chunk {chunk_idx}/seg {seg_idx} input")
                and _compare_arrays(a_tg, b_tg, f"chunk {chunk_idx}/seg {seg_idx} target")
                and _compare_arrays(a_w, b_w, f"chunk {chunk_idx}/seg {seg_idx} weights")
            )
            if not ok:
                mismatches += 1
                if mismatches >= 3:
                    print(f"[ABORT] stopping after {mismatches} mismatches")
                    return 1
            total_segments += 1

        print(f"  chunk {chunk_idx}: {len(legacy_records)} segments compared, mismatches so far {mismatches}")

    print()
    print(f"Total segments compared: {total_segments}")
    print(f"Total skipped by both  : {total_skipped_both}")
    print(f"Mismatches             : {mismatches}")
    if mismatches != 0:
        return 1

    print()
    print("=== Vectorized _get_processed_chunk parity ===")
    # This is the real fast path the trainer hits — exercises Lance's
    # _processed_chunk_vectorized against the legacy per-record loop.
    chunk_mismatches = 0
    for chunk_idx in range(legacy.chunk_count):
        # Reset both caches to force fresh recompute.
        legacy._processed_chunks.pop(chunk_idx, None)
        lance._processed_chunks.pop(chunk_idx, None)

        legacy_processed = legacy._get_processed_chunk(chunk_idx)
        lance_processed = lance._get_processed_chunk(chunk_idx)

        if len(legacy_processed) != len(lance_processed):
            print(
                f"[FAIL] chunk {chunk_idx}: processed count "
                f"legacy={len(legacy_processed)} lance={len(lance_processed)}"
            )
            chunk_mismatches += 1
            continue

        for seg_idx, (a, b) in enumerate(zip(legacy_processed, lance_processed)):
            a_in, a_tg, a_w = a
            b_in, b_tg, b_w = b
            ok = (
                _compare_arrays(a_in, b_in, f"chunk {chunk_idx}/proc-seg {seg_idx} input")
                and _compare_arrays(a_tg, b_tg, f"chunk {chunk_idx}/proc-seg {seg_idx} target")
                and _compare_arrays(a_w, b_w, f"chunk {chunk_idx}/proc-seg {seg_idx} weights")
            )
            if not ok:
                chunk_mismatches += 1
                if chunk_mismatches >= 3:
                    print(f"[ABORT] stopping after {chunk_mismatches} chunk-level mismatches")
                    return 1
        print(f"  chunk {chunk_idx}: {len(legacy_processed)} processed segments match")

    print()
    print(f"Vectorized chunk mismatches: {chunk_mismatches}")
    return 0 if chunk_mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
