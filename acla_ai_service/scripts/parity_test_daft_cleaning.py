"""Parity test for the Daft cleaning module.

Pulls a few real chunks from ``racing_sessions_.lance``, runs both the
legacy :class:`FeatureProcessor` cleaning (general clean → flip → filter)
and the new :func:`clean_session_table_with_daft`, then compares the
resulting tables column-for-column within float tolerance.

Run from the ``acla_ai_service`` directory:

    python -m scripts.parity_test_daft_cleaning

Does not exercise the legacy ``_handle_complex_fields`` or
``strip_dataframe_by_time_gap`` paths — those aren't covered by this
slice. The comparison filters down to the union of columns both
implementations produce.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.domain.telemetry import FeatureProcessor, TelemetryFeatures  # noqa: E402
from app.pipelines.training.daft_cleaning import clean_session_table_with_daft  # noqa: E402
from app.storage.lance import LanceTelemetryStore  # noqa: E402


SESSIONS_KEY = "racing_sessions_"
SAMPLE_CHUNKS = 2  # 2 chunks of ~20k rows each is enough to catch any divergence
RTOL = 1e-5
ATOL = 1e-6


def _legacy_pipeline(table: pa.Table, feature_list: List[str]) -> pa.Table:
    """Mirror ``clean_session_table_with_daft`` via the legacy FeatureProcessor.

    Skips ``_handle_complex_fields`` so we're comparing apples-to-apples
    with the Daft module — that step is explicitly out of scope here.
    """
    df = table.to_pandas()
    processor = FeatureProcessor(df)
    # Inline the legacy general_cleaning_for_analysis body WITHOUT
    # _handle_complex_fields, to isolate the simple-ops behaviour.
    if any(not isinstance(col, str) for col in df.columns):
        df.columns = [str(col) for col in df.columns]
    pd.set_option("future.no_silent_downcasting", True)
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0).infer_objects(copy=False)
    boolean_features = [
        col for col in df.columns
        if isinstance(col, str)
        and any(k in col.lower() for k in ["on", "enabled", "valid", "running", "controlled"])
    ]
    for col in boolean_features:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map({
                "True": True, "False": False, "true": True, "false": False,
                "1": True, "0": False, 1: True, 0: False
            }).fillna(False).infer_objects(copy=False)
    float_columns = df.select_dtypes(include=["float"]).columns
    df[float_columns] = df[float_columns].round(6)
    processor.df = df
    flipped = processor.flip_y_z_features()
    filtered = processor.filter_features_by_list(flipped, feature_list)
    return pa.Table.from_pandas(filtered, preserve_index=False)


def _compare_tables(a: pa.Table, b: pa.Table) -> int:
    if a.num_rows != b.num_rows:
        print(f"[FAIL] row count: legacy={a.num_rows} daft={b.num_rows}")
        return 1
    if a.column_names != b.column_names:
        only_a = [n for n in a.column_names if n not in b.column_names]
        only_b = [n for n in b.column_names if n not in a.column_names]
        print(f"[FAIL] column sets differ: legacy_only={only_a} daft_only={only_b}")
        return 1

    mismatches = 0
    for name in a.column_names:
        ca, cb = a.column(name), b.column(name)
        if pa.types.is_floating(ca.type) and pa.types.is_floating(cb.type):
            xa = np.asarray(ca.to_pylist(), dtype=np.float64)
            xb = np.asarray(cb.to_pylist(), dtype=np.float64)
            if not np.allclose(xa, xb, rtol=RTOL, atol=ATOL, equal_nan=True):
                idx = int(np.argmax(np.abs(xa - xb)))
                print(f"[FAIL] {name}: max diff at row {idx}: legacy={xa[idx]:.6g} daft={xb[idx]:.6g}")
                mismatches += 1
        else:
            la = ca.to_pylist()
            lb = cb.to_pylist()
            if la != lb:
                # Find first diff for diagnostics.
                for i, (va, vb) in enumerate(zip(la, lb)):
                    if va != vb:
                        print(f"[FAIL] {name}: first diff at row {i}: legacy={va!r} daft={vb!r}")
                        break
                mismatches += 1
        if mismatches >= 3:
            print("[ABORT] too many mismatches")
            return 1
    return mismatches


def main() -> int:
    store = LanceTelemetryStore()
    print(f"Lance store: {store.store_dir}")

    chunk_ids = store.list_chunk_ids(SESSIONS_KEY)
    if not chunk_ids:
        print(f"[ERROR] no chunks for {SESSIONS_KEY}")
        return 2

    sample_ids = chunk_ids[:SAMPLE_CHUNKS]
    print(f"Sampling {len(sample_ids)} chunks: {sample_ids}")

    expert_features = TelemetryFeatures().get_features_for_learning_expert()
    print(f"Expert feature list size: {len(expert_features)}")

    total_mismatches = 0
    for chunk_id in sample_ids:
        payload = store.get_chunk(SESSIONS_KEY, chunk_id)
        if not isinstance(payload, list) or not payload:
            print(f"[SKIP] chunk {chunk_id}: empty / unexpected shape")
            continue
        table = pa.Table.from_pylist(payload)
        # Only test on the subset of features that both pipelines can produce —
        # _handle_complex_fields isn't applied, so synthesized columns like
        # Graphics_player_pos_x can't be in the feature list for this test.
        available_features = [f for f in expert_features if f in table.column_names]
        if not available_features:
            print(f"[SKIP] chunk {chunk_id}: no expert features available")
            continue
        print(f"  chunk {chunk_id}: {table.num_rows} rows, {len(available_features)} testable features")

        daft_out = clean_session_table_with_daft(table, feature_list=available_features)
        legacy_out = _legacy_pipeline(table, available_features)

        mm = _compare_tables(legacy_out, daft_out)
        if mm == 0:
            print(f"  chunk {chunk_id}: OK ({legacy_out.num_rows} rows × {legacy_out.num_columns} cols)")
        else:
            print(f"  chunk {chunk_id}: {mm} mismatches")
        total_mismatches += mm

    print()
    print(f"Total mismatches: {total_mismatches}")
    return 0 if total_mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
