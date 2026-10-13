"""Remove track and named-section labels from an existing annotation dataset."""

from __future__ import annotations

import argparse
import json
from typing import Any

from training.pipelines.manifest.migrate_labels import _extract_segments
from training.shared.labels import TRACK_LABEL_IDS


def remove_dataset_track_labels(store: Any, dataset_key: str, *, dry_run: bool = False) -> dict:
    if not dataset_key or not store.has_cached_data(dataset_key):
        raise ValueError(f"Dataset {dataset_key!r} not in store.")

    summary = dict(sessions_processed=0, sessions_updated=0, segments_updated=0, labels_removed=0)
    for session_id in store.list_chunk_ids(dataset_key):
        chunk = store.get_chunk(dataset_key, session_id)
        segments, wrapped_key = _extract_segments(chunk)
        summary["sessions_processed"] += 1
        updated = []
        removed_here = 0
        for segment in segments:
            if not isinstance(segment, dict) or not isinstance(segment.get("labels"), list):
                updated.append(segment)
                continue
            labels = [label for label in segment["labels"] if str(label) not in TRACK_LABEL_IDS]
            removed = len(segment["labels"]) - len(labels)
            if removed:
                updated.append({**segment, "labels": labels})
                summary["segments_updated"] += 1
                removed_here += removed
            else:
                updated.append(segment)

        if not removed_here:
            continue
        summary["sessions_updated"] += 1
        summary["labels_removed"] += removed_here
        if not dry_run:
            payload = {**chunk, wrapped_key: updated} if wrapped_key else updated
            if not store.save_chunk(dataset_key, session_id, payload):
                raise RuntimeError(f"Failed to save {dataset_key!r} session {session_id!r}.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_key")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from training.storage import get_shared_telemetry_store

    print(json.dumps(remove_dataset_track_labels(
        get_shared_telemetry_store(), args.dataset_key, dry_run=args.dry_run,
    ), indent=2))


if __name__ == "__main__":
    main()
