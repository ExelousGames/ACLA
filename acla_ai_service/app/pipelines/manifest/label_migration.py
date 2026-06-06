"""Migrate legacy annotation labels in a saved segment dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.shared.labels import LEGACY_LABEL_MAP


@dataclass
class LabelMigrationSummary:
    sessions_processed: int = 0
    sessions_updated: int = 0
    segments_updated: int = 0
    labels_replaced: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "sessions_processed": self.sessions_processed,
            "sessions_updated": self.sessions_updated,
            "segments_updated": self.segments_updated,
            "labels_replaced": self.labels_replaced,
        }


def _extract_segments(chunk: Any) -> Tuple[List[Any], Optional[str]]:
    if isinstance(chunk, list):
        return list(chunk), None
    if isinstance(chunk, dict) and isinstance(chunk.get("data"), list):
        return list(chunk["data"]), "data"
    return [], None


def _replace_labels(
    labels: Any, migration_map: Dict[Any, str],
) -> Tuple[Any, int]:
    if not isinstance(labels, list):
        return labels, 0

    replaced = 0
    next_labels: List[Any] = []
    for label in labels:
        if label in migration_map:
            next_labels.append(migration_map[label])
            replaced += 1
        else:
            next_labels.append(label)
    return next_labels, replaced


def migrate_dataset_labels(
    store: Any,
    dataset_key: str,
    migration_map: Dict[Any, str] = LEGACY_LABEL_MAP,
    *,
    dry_run: bool = False,
) -> LabelMigrationSummary:
    """Replace legacy labels in every segment chunk for ``dataset_key``."""
    if not dataset_key:
        raise ValueError("No dataset key provided.")
    if not store.has_cached_data(dataset_key):
        raise ValueError(f"Dataset {dataset_key!r} not in store.")

    summary = LabelMigrationSummary()

    for session_id in store.list_chunk_ids(dataset_key):
        summary.sessions_processed += 1
        chunk = store.get_chunk(dataset_key, session_id)
        segments, wrapped_key = _extract_segments(chunk)
        if not segments:
            continue

        updated_segments: List[Any] = []
        session_modified = False
        segments_updated_here = 0

        for segment in segments:
            if not isinstance(segment, dict):
                updated_segments.append(segment)
                continue

            next_labels, replaced = _replace_labels(
                segment.get("labels"), migration_map
            )
            if replaced:
                next_segment = dict(segment)
                next_segment["labels"] = next_labels
                updated_segments.append(next_segment)
                session_modified = True
                segments_updated_here += 1
                summary.labels_replaced += replaced
            else:
                updated_segments.append(segment)

        if session_modified:
            summary.sessions_updated += 1
            summary.segments_updated += segments_updated_here
            if not dry_run:
                if wrapped_key:
                    updated_chunk = dict(chunk)
                    updated_chunk[wrapped_key] = updated_segments
                    store.save_chunk(dataset_key, session_id, updated_chunk)
                else:
                    store.save_chunk(dataset_key, session_id, updated_segments)

    return summary


__all__ = [
    "LEGACY_LABEL_MAP",
    "LabelMigrationSummary",
    "migrate_dataset_labels",
]
