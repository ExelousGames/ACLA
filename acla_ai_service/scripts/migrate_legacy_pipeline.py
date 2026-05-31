"""Migrate the legacy single-key annotation workflow into the new
per-annotation pipeline schema.

Old workflow had:
    racing_sessions_enriched_                                  (telemetry)
    manual_segment_annotations_FirstBatchOfSegmentAnnotation   (one shared
        annotation key written to by parent + children + llm steps)

New workflow gives each annotation its own input fork and its own output
dataset. This script copies the existing data into those new keys so the
new UI sees the legacy work as a normal pipeline:

    parent_v1     input: copy of racing_sessions_enriched_
                  output: copy of the shared annotation key
    children_v1   input: copy of parent_v1's output
                  output: copy of the shared annotation key
    llm_v1        input: copy of children_v1's output
                  output: copy of the shared annotation key

All copies use the store's chunk API (same plumbing as fork_dataset) and
are idempotent — re-running is a no-op.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def _ensure_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_paths()


from app.pipelines.manifest.forking import fork_dataset, derive_annotation_input_key
from app.pipelines.manifest.models import AnnotationNode, Pipeline, TrainingNode
from app.pipelines.manifest.registry import save as save_pipeline
from app.storage.lance import get_shared_lance_store


PIPELINE_ID = "migrated_2026_05_21"
TELEMETRY_SOURCE = "racing_sessions_enriched_"
LEGACY_ANNOTATION_KEY = "manual_segment_annotations_FirstBatchOfSegmentAnnotation"
ANNOTATION_PREFIX = LEGACY_ANNOTATION_KEY  # mirrors TrainingPipelineConfig._annotation_prefix


def _out_key(node_id: str) -> str:
    return f"{ANNOTATION_PREFIX}_{PIPELINE_ID}__{node_id}"


def _input_key(node_id: str) -> str:
    return derive_annotation_input_key(
        pipeline_id=PIPELINE_ID, node_id=node_id, version=1,
    )


def _human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _copy(store, src: str, dst: str) -> None:
    if not store.has_cached_data(src):
        raise SystemExit(f"  ✗ source `{src}` not found in store")
    if store.has_cached_data(dst):
        meta = store.get_cache_metadata(dst)
        bytes_ = meta.total_bytes if meta else 0
        print(f"  · `{dst}` already exists ({_human(bytes_)}) — skipping")
        return
    src_meta = store.get_cache_metadata(src)
    src_bytes = src_meta.total_bytes if src_meta else 0
    print(f"  → copy `{src}` → `{dst}` ({_human(src_bytes)})")

    def _progress(done: int, total: int) -> None:
        print(f"    chunk {done}/{total}", end="\r", flush=True)

    fork_dataset(source_key=src, target_key=dst, store=store, progress=_progress)
    print()  # newline after the \r progress


def main() -> None:
    store = get_shared_lance_store()

    # 1) Telemetry fork for parent_v1's input.
    print("[1/6] parent_v1 input fork (telemetry)")
    _copy(store, TELEMETRY_SOURCE, _input_key("parent_v1"))

    # 2-4) Per-node output datasets (each is a copy of the legacy shared key).
    print("[2/6] parent_v1 output")
    _copy(store, LEGACY_ANNOTATION_KEY, _out_key("parent_v1"))
    print("[3/6] children_v1 output")
    _copy(store, LEGACY_ANNOTATION_KEY, _out_key("children_v1"))
    print("[4/6] llm_v1 output")
    _copy(store, LEGACY_ANNOTATION_KEY, _out_key("llm_v1"))

    # 5-6) Input forks for children/llm — copies of their respective sources.
    # Source semantics in the new model: children's source = parent_v1.output,
    # llm's source = children_v1.output. After step 2-4 those resolve to the
    # per-node output keys we just created.
    print("[5/6] children_v1 input fork (copy of parent_v1.output)")
    _copy(store, _out_key("parent_v1"), _input_key("children_v1"))
    print("[6/6] llm_v1 input fork (copy of children_v1.output)")
    _copy(store, _out_key("children_v1"), _input_key("llm_v1"))

    # ── Build the manifest ───────────────────────────────────────────────
    now = datetime.now().isoformat()
    src_meta = store.get_cache_metadata(TELEMETRY_SOURCE)
    src_updated_at = src_meta.updated_at if src_meta else now

    def _node(node_id: str, kind: str, source_ref: str, source_updated_at: str) -> AnnotationNode:
        return AnnotationNode(
            id=node_id,
            kind=kind,
            output_key=_out_key(node_id),
            source_ref=source_ref,
            input_key=_input_key(node_id),
            copied_at=now,
            source_updated_at_on_copy=source_updated_at,
        )

    parent_out_meta = store.get_cache_metadata(_out_key("parent_v1"))
    children_out_meta = store.get_cache_metadata(_out_key("children_v1"))

    pipeline = Pipeline(
        id=PIPELINE_ID,
        version=1,
        created_at=now,
        annotations=[
            _node("parent_v1",   "parent",   TELEMETRY_SOURCE, src_updated_at),
            _node("children_v1", "children", "parent_v1.output",
                  parent_out_meta.updated_at if parent_out_meta else now),
            _node("llm_v1",      "llm",      "children_v1.output",
                  children_out_meta.updated_at if children_out_meta else now),
        ],
        trainings=[
            TrainingNode(id="classifier_v1",  kind="classifier",   input_ref="children_v1.output"),
            TrainingNode(id="transformer_v1", kind="transformer",  input_ref="children_v1.output"),
            TrainingNode(id="llm_train_v1",   kind="llm_training", input_ref="llm_v1.output"),
        ],
    )
    save_pipeline(pipeline)
    print(f"\n✓ Wrote manifest `{PIPELINE_ID}.json`")
    print(json.dumps(pipeline.to_dict(), indent=2))


if __name__ == "__main__":
    main()
