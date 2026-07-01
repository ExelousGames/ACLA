"""Backup and restore helpers for the Lance telemetry store."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from io import BytesIO
from hashlib import sha1
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.storage.lance import LanceTelemetryStore, get_shared_lance_store


ARCHIVE_PREFIX = "telemetry_lance"
ARCHIVE_SUFFIX = ".tar.gz"
STORE_ARCHIVE_ROOT = "telemetry_lance_store"
CUSTOM_ARCHIVE_ROOT = "telemetry_lance_custom_dirs"


def get_lance_backup_dir(store: Optional[LanceTelemetryStore] = None) -> Path:
    store = store or get_shared_lance_store()
    configured = os.environ.get("LANCE_BACKUP_DIR")
    if configured:
        return Path(configured)
    return store.store_dir.parent / "telemetry_lance_backups"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _paths_size(paths: List[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_file():
            total += path.stat().st_size
        elif path.exists():
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total


def _paths_file_count(paths: List[Path]) -> int:
    count = 0
    for path in paths:
        if path.is_file():
            count += 1
        elif path.exists():
            count += sum(1 for f in path.rglob("*") if f.is_file())
    return count


def _archive_info(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _backup_glob() -> str:
    return f"{ARCHIVE_PREFIX}*{ARCHIVE_SUFFIX}"


def _resolve_backup_path(backup_name: str, backup_dir: Path) -> Path:
    if not backup_name:
        raise ValueError("backup_name is required")

    backup_dir = backup_dir.resolve()
    path = (backup_dir / backup_name).resolve()
    if not _is_relative_to(path, backup_dir):
        raise ValueError("backup_name must resolve inside the Lance backup directory")
    if path.name != backup_name:
        raise ValueError("backup_name must be a filename, not a path")
    if not path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_name}")
    if not path.is_file():
        raise ValueError(f"Backup is not a file: {backup_name}")
    return path


def _add_manifest(archive: tarfile.TarFile, manifest: Dict[str, Any]) -> None:
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_info = tarfile.TarInfo("manifest.json")
    manifest_info.size = len(manifest_bytes)
    manifest_info.mtime = int(datetime.now(timezone.utc).timestamp())
    archive.addfile(manifest_info, BytesIO(manifest_bytes))


def _add_archive_root(archive: tarfile.TarFile, root: str) -> None:
    store_info = tarfile.TarInfo(root)
    store_info.type = tarfile.DIRTYPE
    store_info.mtime = int(datetime.now(timezone.utc).timestamp())
    archive.addfile(store_info)


def _custom_archive_root(directory: Path) -> str:
    digest = sha1(str(directory.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{CUSTOM_ARCHIVE_ROOT}/{digest}"


def _sidecar_path(directory: Path, cache_key: str) -> Path:
    return directory / f"{cache_key}.lance.meta.json"


def _dataset_paths(store: LanceTelemetryStore, directory: Path, cache_key: str) -> List[Path]:
    paths = [
        path
        for path in store._strategy(cache_key).dataset_paths(directory, cache_key)
        if path.exists()
    ]
    sidecar = _sidecar_path(directory, cache_key)
    if sidecar.exists():
        paths.append(sidecar)
    return paths


def _storage_specs(store: LanceTelemetryStore) -> List[Dict[str, Any]]:
    default_dir = store.store_dir.resolve()
    specs: List[Dict[str, Any]] = [{
        "archive_root": STORE_ARCHIVE_ROOT,
        "path": store.store_dir,
        "cache_keys": None,
        "paths": list(store.store_dir.iterdir()) if store.store_dir.exists() else [],
    }]

    custom_dirs: Dict[Path, set[str]] = {}
    for cache_key, directory in store.registered_directories().items():
        resolved = directory.resolve()
        if resolved == default_dir:
            continue
        custom_dirs.setdefault(resolved, set()).add(cache_key)

    for resolved, cache_keys in sorted(custom_dirs.items(), key=lambda item: str(item[0])):
        paths: List[Path] = []
        for cache_key in sorted(cache_keys):
            paths.extend(_dataset_paths(store, resolved, cache_key))
        specs.append({
            "archive_root": _custom_archive_root(resolved),
            "path": resolved,
            "cache_keys": sorted(cache_keys),
            "paths": paths,
        })

    return specs


def _manifest_storage_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "archive_root": spec["archive_root"],
            "path": str(spec["path"]),
            "cache_keys": spec["cache_keys"],
        }
        for spec in specs
    ]


def _read_manifest(archive_path: Path) -> Dict[str, Any]:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            manifest_file = archive.extractfile("manifest.json")
            if manifest_file is None:
                return {}
            return json.loads(manifest_file.read().decode("utf-8"))
    except KeyError:
        return {}


def _restore_directory(
    extracted_root: Path,
    target_dir: Path,
    store: LanceTelemetryStore,
    cache_keys: Optional[List[str]],
) -> None:
    if not extracted_root.exists() or not extracted_root.is_dir():
        raise ValueError(f"Backup archive does not contain {extracted_root.name}/")

    target_dir.mkdir(parents=True, exist_ok=True)
    if cache_keys is None:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        for cache_key in cache_keys:
            store.register_directory(cache_key, str(target_dir))
            store.clear_cache(cache_key)

    for child in extracted_root.iterdir():
        destination = target_dir / child.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(child), str(destination))


def create_lance_backup(
    store: Optional[LanceTelemetryStore] = None,
    *,
    backup_dir: Optional[Path] = None,
    prefix: str = ARCHIVE_PREFIX,
) -> Dict[str, Any]:
    store = store or get_shared_lance_store()
    backup_dir = backup_dir or get_lance_backup_dir(store)
    backup_dir.mkdir(parents=True, exist_ok=True)

    archive_path = backup_dir / f"{prefix}_{_utc_stamp()}{ARCHIVE_SUFFIX}"
    if archive_path.exists():
        archive_path = backup_dir / f"{prefix}_{_utc_stamp()}_{datetime.now(timezone.utc).microsecond}{ARCHIVE_SUFFIX}"

    storage_specs = _storage_specs(store)
    backup_dir_resolved = backup_dir.resolve()
    archive_path_resolved = archive_path.resolve()
    archived_paths = [
        path
        for spec in storage_specs
        for path in spec["paths"]
    ]
    store_info = store.get_cache_info()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "store_directory": str(store.store_dir),
        "entry_count": store_info["entry_count"],
        "size_bytes": _paths_size(archived_paths),
        "file_count": _paths_file_count(archived_paths),
        "entries": store_info["entries"],
        "storage_directories": _manifest_storage_specs(storage_specs),
    }

    with tarfile.open(archive_path, "w:gz") as archive:
        _add_manifest(archive, manifest)
        for spec in storage_specs:
            _add_archive_root(archive, spec["archive_root"])

            for child in spec["paths"]:
                child_resolved = child.resolve()
                if child_resolved == archive_path_resolved:
                    continue
                if child_resolved == backup_dir_resolved:
                    continue
                if _is_relative_to(backup_dir_resolved, child_resolved):
                    continue
                archive.add(child, arcname=f"{spec['archive_root']}/{child.name}")

    return {
        "status": "success",
        "backup": _archive_info(archive_path),
        "manifest": manifest,
    }


def list_lance_backups(
    store: Optional[LanceTelemetryStore] = None,
    *,
    backup_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    store = store or get_shared_lance_store()
    backup_dir = backup_dir or get_lance_backup_dir(store)
    backup_dir.mkdir(parents=True, exist_ok=True)

    backups = sorted(
        (_archive_info(path) for path in backup_dir.glob(_backup_glob())),
        key=lambda item: item["created_at"],
        reverse=True,
    )
    return {
        "backup_directory": str(backup_dir),
        "backup_count": len(backups),
        "backups": backups,
    }


def get_latest_lance_backup(
    store: Optional[LanceTelemetryStore] = None,
    *,
    backup_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    backups = list_lance_backups(store, backup_dir=backup_dir)["backups"]
    if not backups:
        return None
    return backups[0]


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not _is_relative_to(target, destination):
                raise ValueError(f"Unsafe backup archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Backup archive links are not supported: {member.name}")
        archive.extractall(destination)


def restore_lance_backup(
    backup_name: str,
    store: Optional[LanceTelemetryStore] = None,
    *,
    backup_dir: Optional[Path] = None,
    create_safety_backup: bool = True,
) -> Dict[str, Any]:
    store = store or get_shared_lance_store()
    backup_dir = backup_dir or get_lance_backup_dir(store)
    archive_path = _resolve_backup_path(backup_name, backup_dir)
    manifest = _read_manifest(archive_path)

    safety_backup = None
    if create_safety_backup and store.list_cache_keys():
        safety_backup = create_lance_backup(
            store,
            backup_dir=backup_dir,
            prefix=f"{ARCHIVE_PREFIX}_prerestore",
        )["backup"]

    temp_dir = Path(tempfile.mkdtemp(prefix="lance_restore_", dir=str(backup_dir)))
    try:
        _safe_extract(archive_path, temp_dir)
        storage_specs = manifest.get("storage_directories") or [{
            "archive_root": STORE_ARCHIVE_ROOT,
            "path": str(store.store_dir),
            "cache_keys": None,
        }]
        for spec in storage_specs:
            archive_root = spec.get("archive_root") or STORE_ARCHIVE_ROOT
            target_dir = (
                store.store_dir
                if archive_root == STORE_ARCHIVE_ROOT
                else Path(spec["path"])
            )
            _restore_directory(
                temp_dir / archive_root,
                target_dir,
                store,
                spec.get("cache_keys"),
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "status": "success",
        "restored_backup": _archive_info(archive_path),
        "safety_backup": safety_backup,
        "store_info": store.get_cache_info(),
    }
