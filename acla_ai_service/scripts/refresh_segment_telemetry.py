"""Refresh ``telemetry_data`` on one annotation node's saved segments.

Run this after data preparation refreshes source data. Saved segments
in ``output_key`` carry their old ``telemetry_data`` snapshot; this
script re-slices every segment's ``telemetry_data`` from the resolved
input source so the new columns land on the saved segment dicts.

    python scripts/refresh_segment_telemetry.py <pipeline_id> <node_id>
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_paths()


from app.pipelines.manifest.registry import load as load_pipeline
from app.pipelines.manifest.segment_refresh import refresh_node_segments
from app.storage.lance import get_shared_lance_store


def main(pipeline_id: str, node_id: str) -> None:
    pipeline = load_pipeline(pipeline_id)
    if pipeline is None:
        raise SystemExit(f"Pipeline `{pipeline_id}` not found.")

    node = next((n for n in pipeline.annotations if n.id == node_id), None)
    if node is None:
        available = ", ".join(n.id for n in pipeline.annotations) or "(none)"
        raise SystemExit(
            f"Node `{node_id}` not found in `{pipeline_id}`. "
            f"Available: {available}"
        )

    store = get_shared_lance_store()
    input_key = pipeline.effective_input_key(node)
    print(f"Refreshing `{node.id}` ({node.kind}) — "
          f"input={input_key} → output={node.output_key}")

    try:
        summary = refresh_node_segments(store, node, input_key=input_key)
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(f"  chunks:   refreshed {summary.chunks_written}/{summary.chunks_total}")
    print(f"  segments: refreshed {summary.segments_refreshed}/{summary.segments_total} "
          f"({summary.segments_skipped} skipped — missing indices)")
    if summary.missing_input_sessions:
        print(f"  missing input sessions: "
              f"{', '.join(summary.missing_input_sessions)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python scripts/refresh_segment_telemetry.py "
            "<pipeline_id> <node_id>"
        )
    main(sys.argv[1], sys.argv[2])
