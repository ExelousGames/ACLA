"""Phase-2 migration: re-write Phase-1 blob Lance datasets in typed form.

Run from the ``acla_ai_service`` directory:

    python -m scripts.migrate_blob_to_typed_lance               # all 5 keys
    python -m scripts.migrate_blob_to_typed_lance --dry-run     # report only
    python -m scripts.migrate_blob_to_typed_lance --keys foo bar  # subset

The script never touches ``app/storage/telemetry_zarr_store/``; the Phase-1
Lance datasets remain on disk as ``<key>.lance.phase1.bak`` after each key is
migrated, so a rollback is a single rename per key. Annotation keys (blob
strategy) are untouched.

Pipeline per cache_key
----------------------
1. **Pass 1 — schema inference.** Stream every chunk through the strategy's
   ``precompute_schema`` to derive a unified Arrow record schema. For
   :class:`SegmentsStrategy` this yields both a segment scalar schema and a
   telemetry record schema; the strategy stashes them internally.
2. **Pass 2 — write.** Stream the same chunks again, calling the strategy's
   ``write_chunk`` for each. The strategy serialises records using the
   inferred schema, casting missing fields to null and rejecting writes that
   would introduce unseen fields.
3. **Atomic swap.** Once writes complete, rename the temp dataset over the
   original location and persist the :class:`CacheMetadata` to the sidecar
   JSON next to the new dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple

# Make `app` importable when invoked as a script from the project root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.storage.lance import LanceTelemetryStore  # noqa: E402
from app.storage.lance.strategies import (  # noqa: E402
    BlobStrategy,
    SchemaStrategy,
    strategy_for,
    is_typed,
    write_sidecar_metadata,
)
from app.storage.cache_metadata import CacheMetadata  # noqa: E402


def _stream_phase1_from_backup(
    backup_path: Path,
) -> Iterator[Tuple[str, Any]]:
    """Iterate (chunk_id, payload) from a renamed Phase-1 blob dataset.

    Pulls one chunk's payload at a time rather than loading the whole
    payload column as a single Arrow Table — for racing_sessions_ at
    20 GB the cumulative payload overflows pyarrow's 32-bit binary
    offsets.
    """
    import lance as _lance

    dataset = _lance.dataset(str(backup_path))
    ids_table = dataset.scanner(
        filter="chunk_id != '__metadata__'",
        columns=["chunk_id"],
    ).to_table()
    chunk_ids = sorted(ids_table.column("chunk_id").to_pylist())
    for cid in chunk_ids:
        try:
            payload_table = dataset.scanner(
                filter=f"chunk_id = '{cid}'",
                columns=["payload"],
            ).to_table()
        except Exception as exc:
            print(f"    [WARN] failed to scan chunk '{cid}': {exc}")
            continue
        if payload_table.num_rows == 0:
            continue
        try:
            payload_bytes = bytes(payload_table.column("payload")[0].as_py())
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            print(f"    [WARN] failed to decode chunk '{cid}': {exc}")
            continue
        yield (cid, payload)


def _backup_path(base_dir: Path, cache_key: str) -> Path:
    return base_dir / f"{cache_key}.lance.phase1.bak"


def _migrate_one(
    *,
    cache_key: str,
    store: LanceTelemetryStore,
    overwrite: bool,
) -> dict:
    start = time.time()
    base_dir = store.store_dir
    primary_path = base_dir / f"{cache_key}.lance"
    backup_path = _backup_path(base_dir, cache_key)

    if not primary_path.exists():
        return {
            "cache_key": cache_key,
            "status": "missing",
            "reason": f"{primary_path.name} not found",
            "chunks": 0,
            "elapsed_s": 0.0,
        }

    if backup_path.exists() and not overwrite:
        return {
            "cache_key": cache_key,
            "status": "skipped",
            "reason": "Phase-1 backup already exists; pass --overwrite to redo",
            "chunks": 0,
            "elapsed_s": 0.0,
        }

    strategy = strategy_for(cache_key)
    if isinstance(strategy, BlobStrategy):
        return {
            "cache_key": cache_key,
            "status": "skipped",
            "reason": "cache_key is registered as blob (annotation); no Phase-2 migration",
            "chunks": 0,
            "elapsed_s": 0.0,
        }

    # Pass 1 must read from the still-existing primary path, so it has to
    # happen BEFORE the rename. The chunks iterator is fed through the
    # strategy's precompute_schema so it can build a unified record schema.
    print(f"  - {cache_key}: pass 1 (schema inference)...")
    blob = BlobStrategy()

    def _pass1_iter() -> Iterator[Tuple[str, Any]]:
        for payload, chunk_id in blob.iter_chunks(base_dir, cache_key, include_ids=True):
            yield (chunk_id, payload)

    strategy.precompute_schema(_pass1_iter())

    # Move Phase-1 dataset out of the canonical path; the strategy writes
    # the new typed dataset there. On partial failure we roll back by
    # renaming the backup back into place.
    if backup_path.exists():
        shutil.rmtree(backup_path)
    primary_path.rename(backup_path)

    print(f"  - {cache_key}: pass 2 (writing typed dataset)...")
    metadata = CacheMetadata(cache_key=cache_key)
    chunks_written = 0
    try:
        for chunk_id, payload in _stream_phase1_from_backup(backup_path):
            size_estimate = strategy.write_chunk(base_dir, cache_key, chunk_id, payload)
            metadata.register_chunk(chunk_bytes=size_estimate)
            chunks_written += 1
    except Exception as exc:
        # Roll back: discard the partial typed dataset, restore Phase-1.
        for path in strategy.dataset_paths(base_dir, cache_key):
            if path.exists():
                shutil.rmtree(path)
        backup_path.rename(primary_path)
        raise RuntimeError(
            f"Failed during Phase-2 write for {cache_key}: {exc}"
        ) from exc

    write_sidecar_metadata(base_dir, metadata)

    return {
        "cache_key": cache_key,
        "status": "migrated",
        "chunks": chunks_written,
        "strategy": strategy.name,
        "elapsed_s": round(time.time() - start, 2),
    }


async def _run(
    *,
    keys: Optional[List[str]],
    overwrite: bool,
    dry_run: bool,
) -> int:
    store = LanceTelemetryStore()
    base_dir = store.store_dir

    # Eligible keys are those that exist on disk AND map to a typed strategy.
    available_keys: List[str] = []
    for entry in base_dir.glob("*.lance"):
        cache_key = entry.stem
        if "." in cache_key:  # skip auxiliary datasets
            continue
        if is_typed(cache_key):
            available_keys.append(cache_key)
    available_keys.sort()

    selected = keys if keys else available_keys
    if keys:
        missing = [k for k in keys if k not in available_keys]
        if missing:
            print(f"[ERROR] cache_keys not eligible for Phase-2 migration: {missing}", file=sys.stderr)
            print(f"        (eligible: {available_keys})", file=sys.stderr)
            return 2

    print(f"Lance store : {base_dir}")
    print(f"Migrating {len(selected)} typed cache_keys (overwrite={overwrite}, dry_run={dry_run})")
    for cache_key in selected:
        print(f"  - {cache_key} -> {strategy_for(cache_key).name}")
    print()

    if dry_run:
        return 0

    failed = 0
    for cache_key in selected:
        try:
            result = _migrate_one(cache_key=cache_key, store=store, overwrite=overwrite)
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {cache_key}: {exc}")
            continue

        tag = f"[{result['status'].upper()}]"
        extras = ""
        if "chunks" in result and result["status"] == "migrated":
            extras = f" ({result['chunks']} chunks, {result['elapsed_s']}s, strategy={result['strategy']})"
        elif "reason" in result:
            extras = f" — {result['reason']}"
        print(f"  {tag} {cache_key}{extras}")

    print()
    print(f"Done. {failed} failed. Phase-1 datasets backed up as <key>.lance.phase1.bak.")
    return 0 if failed == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Subset of cache_keys to migrate (defaults to all typed keys).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-migrate even if a Phase-1 backup already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without writing anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = asyncio.run(_run(keys=args.keys, overwrite=args.overwrite, dry_run=args.dry_run))
    sys.exit(exit_code)
