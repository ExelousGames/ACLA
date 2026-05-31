"""Migrate legacy labels in an annotation dataset to the current labels.

This is the CLI wrapper for the same migration used by the Telemetry
Annotation Pipeline maintenance dropdown.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_paths() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    path_str = root_dir.as_posix()
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_paths()

from app.pipelines.manifest.label_migration import (  # noqa: E402
    LEGACY_LABEL_MAP,
    migrate_dataset_labels,
)
from app.storage import get_shared_telemetry_store  # noqa: E402


def update_legacy_labels(
    dataset_key: str,
    migration_map: Dict[Any, str] = LEGACY_LABEL_MAP,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Update labels in ``dataset_key`` and return migration statistics."""
    store = get_shared_telemetry_store()
    summary = migrate_dataset_labels(
        store,
        dataset_key,
        migration_map,
        dry_run=dry_run,
    )
    return summary.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy labels in an annotation dataset."
    )
    parser.add_argument(
        "dataset_key",
        help="The cache key of the annotation dataset to update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without saving changes.",
    )
    args = parser.parse_args()

    print(f"Starting migration for dataset: {args.dataset_key}")
    print(f"Mapping: {LEGACY_LABEL_MAP}")

    results = update_legacy_labels(
        args.dataset_key,
        LEGACY_LABEL_MAP,
        dry_run=args.dry_run,
    )

    print("\nMigration Complete.")
    print(f"Sessions Processed: {results['sessions_processed']}")
    print(f"Sessions Updated: {results['sessions_updated']}")
    print(f"Segments Updated: {results['segments_updated']}")
    print(f"Labels Replaced: {results['labels_replaced']}")


if __name__ == "__main__":
    main()
