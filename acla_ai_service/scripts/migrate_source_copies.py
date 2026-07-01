"""Populate private source-copy datasets for existing source nodes.

Run this after upgrading manifests that previously read source datasets
directly. It does not change the manifest schema; it registers each
source node's copy dataset and copies/updates source chunks while
preserving input rows referenced by the annotation output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from app.pipelines.manifest.models import MODE_SOURCE
from app.pipelines.manifest.registry import list_pipelines, load
from app.pipelines.manifest.source_copy import (
    SourceCopySummary,
    source_copy_key,
    sync_source_copy,
)
from app.storage import get_shared_telemetry_store


@dataclass
class MigratedNode:
    pipeline_id: str
    node_id: str
    source_key: str
    copy_key: str
    summary: SourceCopySummary | None = None
    skipped_reason: str | None = None


@dataclass
class MigrationSummary:
    nodes: list[MigratedNode] = field(default_factory=list)

    @property
    def copied_chunks(self) -> int:
        return sum(n.summary.chunks_copied for n in self.nodes if n.summary)

    @property
    def updated_chunks(self) -> int:
        return sum(n.summary.chunks_updated for n in self.nodes if n.summary)

    @property
    def preserved_rows(self) -> int:
        return sum(n.summary.rows_preserved for n in self.nodes if n.summary)


def migrate_source_copies(store: Any, *, dry_run: bool = False) -> MigrationSummary:
    summary = MigrationSummary()
    for pipeline_id in list_pipelines():
        pipeline = load(pipeline_id)
        if pipeline is None:
            continue
        for node in pipeline.annotations:
            if node.mode != MODE_SOURCE or not node.output_key:
                continue
            source_key = pipeline.resolve_source_key(node.source_ref)
            copy_key = source_copy_key(node.output_key)
            item = MigratedNode(
                pipeline_id=pipeline.id,
                node_id=node.id,
                source_key=source_key or "",
                copy_key=copy_key,
            )
            summary.nodes.append(item)
            if not source_key:
                item.skipped_reason = "source does not resolve"
                continue
            if hasattr(store, "register_directory"):
                store.register_directory(node.output_key, node.output_dir)
                store.register_directory(copy_key, node.output_dir)
            if not store.has_cached_data(source_key):
                item.skipped_reason = "source has no cached data"
                continue
            if not node.protection_reference:
                item.skipped_reason = "source-copy protection is not configured"
                continue
            if dry_run:
                item.skipped_reason = "dry run"
                continue
            item.summary = sync_source_copy(
                store,
                source_key=source_key,
                copy_key=copy_key,
                output_key=pipeline.effective_output_key(node),
                protection_reference=node.protection_reference,
            )
    return summary


def _format_node(item: MigratedNode) -> str:
    prefix = (
        f"{item.pipeline_id}/{item.node_id}: "
        f"{item.source_key} -> {item.copy_key}"
    )
    if item.summary is None:
        return f"{prefix} ({item.skipped_reason or 'skipped'})"
    return (
        f"{prefix} copied={item.summary.chunks_copied} "
        f"updated={item.summary.chunks_updated} "
        f"preserved_rows={item.summary.rows_preserved} "
        f"failures={len(item.summary.read_failures) + len(item.summary.write_failures)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate source-copy datasets for source annotation nodes."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = migrate_source_copies(get_shared_telemetry_store(), dry_run=args.dry_run)
    for item in summary.nodes:
        print(_format_node(item))
    print(
        "Totals: "
        f"nodes={len(summary.nodes)} "
        f"copied={summary.copied_chunks} "
        f"updated={summary.updated_chunks} "
        f"preserved_rows={summary.preserved_rows}"
    )


if __name__ == "__main__":
    main()
