"""Backfill ``telemetry_data`` on existing lap-annotation segments.

The earlier lap-excerpter save paths (staged-review + batch auto-save)
built ``AnnotatedSegment``s without slicing the underlying telemetry,
so saved segments had indices but no rows. The save side is fixed
going forward; this script repairs already-saved chunks.

For each lap annotation node that has both an ``input_key`` (forked
telemetry source) and an ``output_key`` (segment dataset) wired in the
pipeline JSON, every segment in every chunk gets its ``telemetry_data``
filled from ``input_key.iloc[start:end]``. Idempotent — segments that
already carry telemetry are left alone.

    python scripts/backfill_lap_telemetry.py [pipeline_id]

Defaults to ``migrated_2026_05_21``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_paths()


import pandas as pd

from app.pipelines.manifest.models import AnnotationNode, Pipeline
from app.pipelines.manifest.registry import load as load_pipeline
from app.storage.lance import get_shared_lance_store


_LAP_KINDS = {"lap", "batch_lap"}


def _load_input_session(store, input_key: str, session_id: str) -> pd.DataFrame:
    chunk = store.get_chunk(input_key, session_id)
    if chunk is None:
        return pd.DataFrame()
    if isinstance(chunk, list):
        return pd.DataFrame(chunk)
    if isinstance(chunk, dict):
        if "data" in chunk and isinstance(chunk["data"], list):
            return pd.DataFrame(chunk["data"])
        return pd.DataFrame([chunk])
    return pd.DataFrame()


def _slice_segment(df: pd.DataFrame, seg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    start = seg.get("start_index")
    end = seg.get("end_index")
    if start is None or end is None:
        return None
    s = max(0, int(start))
    e = min(len(df), int(end))
    if s >= e:
        return None
    return df.iloc[s:e].to_dict(orient="records")


def _backfill_chunk(
    segments: List[Dict[str, Any]],
    df: pd.DataFrame,
) -> tuple[List[Dict[str, Any]], int, int]:
    out: List[Dict[str, Any]] = []
    filled = 0
    already = 0
    for seg in segments:
        if not isinstance(seg, dict):
            out.append(seg)
            continue
        existing = seg.get("telemetry_data")
        if isinstance(existing, list) and existing:
            already += 1
            out.append(seg)
            continue
        sliced = _slice_segment(df, seg)
        if sliced is None:
            out.append(seg)
            continue
        new_seg = dict(seg)
        new_seg["telemetry_data"] = sliced
        new_seg["segment_length"] = len(sliced)
        out.append(new_seg)
        filled += 1
    return out, filled, already


def _backfill_node(store, node: AnnotationNode) -> None:
    print(f"\n── {node.id} ({node.kind}) ──")
    input_key = node.input_key
    output_key = node.output_key

    if not input_key:
        print(f"  · no input_key in manifest — nothing to slice from; skipping")
        return
    if not output_key:
        print(f"  · no output_key in manifest — skipping")
        return
    if not store.has_cached_data(input_key):
        print(f"  ✗ input `{input_key}` not in store — skipping")
        return
    if not store.has_cached_data(output_key):
        print(f"  · output `{output_key}` is empty — nothing to backfill")
        return

    session_ids = store.list_chunk_ids(output_key)
    print(f"  input:  {input_key}")
    print(f"  output: {output_key}")
    print(f"  chunks: {len(session_ids)}")

    total_filled = 0
    total_already = 0
    total_segments = 0
    chunks_written = 0

    for sid in session_ids:
        segments = store.get_chunk(output_key, sid)
        if not isinstance(segments, list) or not segments:
            continue
        df = _load_input_session(store, input_key, sid)
        if df.empty:
            print(f"  · {sid}: input session not found — skipped "
                  f"({len(segments)} segments untouched)")
            continue

        updated, filled, already = _backfill_chunk(segments, df)
        total_segments += len(segments)
        total_already += already
        total_filled += filled

        if filled > 0:
            store.save_chunk(output_key, sid, updated)
            chunks_written += 1
            print(f"  · {sid}: filled {filled}/{len(segments)} "
                  f"({already} already had telemetry)")
        else:
            print(f"  · {sid}: nothing to fill "
                  f"({already}/{len(segments)} already had telemetry)")

    print(f"  ✓ done — filled {total_filled} segment(s) across "
          f"{chunks_written} chunk(s); {total_already}/{total_segments} "
          f"already had telemetry.")


def main(pipeline_id: str = "migrated_2026_05_21") -> None:
    pipeline = load_pipeline(pipeline_id)
    if pipeline is None:
        raise SystemExit(f"Pipeline `{pipeline_id}` not found.")

    store = get_shared_lance_store()

    lap_nodes = [n for n in pipeline.annotations if n.kind in _LAP_KINDS]
    if not lap_nodes:
        print(f"No lap-style annotation nodes in `{pipeline_id}`.")
        return

    print(f"Backfilling telemetry_data for `{pipeline_id}` — "
          f"{len(lap_nodes)} lap node(s):")
    for n in lap_nodes:
        print(f"  • {n.id} ({n.kind}) "
              f"input_key={n.input_key!r} output_key={n.output_key!r}")

    for node in lap_nodes:
        _backfill_node(store, node)

    print("\nAll done.")


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "migrated_2026_05_21"
    main(pid)
