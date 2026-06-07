from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("lance")

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.storage.lance import LanceTelemetryStore
from app.storage.lance.backup import (
    create_lance_backup,
    list_lance_backups,
    restore_lance_backup,
)
from scripts import lance_storage_backup as cli


_CACHE_KEY = "manual_segment_annotations_backup_test"
_CUSTOM_CACHE_KEY = "manual_segment_annotations_custom_backup_test"


def _store(tmp_path: Path) -> LanceTelemetryStore:
    return LanceTelemetryStore(str(tmp_path / "telemetry_lance_store"))


def test_create_backup_includes_manifest_and_store_contents(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup_dir = tmp_path / "backups"
    store.save_chunk(_CACHE_KEY, "session_a", {"value": "original"})

    result = create_lance_backup(store, backup_dir=backup_dir)

    backup_path = Path(result["backup"]["path"])
    assert backup_path.exists()
    assert result["manifest"]["entry_count"] == 1

    with tarfile.open(backup_path, "r:gz") as archive:
        names = archive.getnames()

    assert "manifest.json" in names
    assert any(name.startswith("telemetry_lance_store/") for name in names)


def test_backup_includes_registered_custom_output_directory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup_dir = tmp_path / "backups"
    custom_dir = tmp_path / "custom_outputs"
    store.register_directory(_CUSTOM_CACHE_KEY, str(custom_dir))
    store.save_chunk(_CUSTOM_CACHE_KEY, "session_a", {"value": "custom"})

    result = create_lance_backup(store, backup_dir=backup_dir)

    assert result["manifest"]["entry_count"] == 1
    assert result["manifest"]["entries"][0]["cache_key"] == _CUSTOM_CACHE_KEY
    assert result["manifest"]["entries"][0]["directory"] == str(custom_dir)
    assert any(
        spec["path"] == str(custom_dir)
        for spec in result["manifest"]["storage_directories"]
    )

    with tarfile.open(result["backup"]["path"], "r:gz") as archive:
        names = archive.getnames()

    assert any(
        name.endswith(f"/{_CUSTOM_CACHE_KEY}.lance")
        for name in names
    )


def test_list_backups_newest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup_dir = tmp_path / "backups"
    store.save_chunk(_CACHE_KEY, "session_a", {"value": "first"})
    first = create_lance_backup(store, backup_dir=backup_dir)["backup"]
    os.utime(first["path"], (1, 1))

    store.save_chunk(_CACHE_KEY, "session_a", {"value": "second"})
    second = create_lance_backup(store, backup_dir=backup_dir)["backup"]

    backups = list_lance_backups(store, backup_dir=backup_dir)["backups"]

    assert backups[0]["filename"] == second["filename"]
    assert backups[1]["filename"] == first["filename"]


def test_restore_named_backup_replaces_whole_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup_dir = tmp_path / "backups"
    store.save_chunk(_CACHE_KEY, "session_a", {"value": "original"})
    backup = create_lance_backup(store, backup_dir=backup_dir)["backup"]

    store.clear_cache()
    store.save_chunk(_CACHE_KEY, "session_a", {"value": "changed"})

    restore_lance_backup(
        backup["filename"],
        store,
        backup_dir=backup_dir,
        create_safety_backup=False,
    )

    assert store.get_chunk(_CACHE_KEY, "session_a") == {"value": "original"}


def test_restore_registered_custom_output_directory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup_dir = tmp_path / "backups"
    custom_dir = tmp_path / "custom_outputs"
    store.register_directory(_CUSTOM_CACHE_KEY, str(custom_dir))
    store.save_chunk(_CUSTOM_CACHE_KEY, "session_a", {"value": "original"})
    backup = create_lance_backup(store, backup_dir=backup_dir)["backup"]

    store.clear_cache(_CUSTOM_CACHE_KEY)
    store.save_chunk(_CUSTOM_CACHE_KEY, "session_a", {"value": "changed"})

    restore_lance_backup(
        backup["filename"],
        store,
        backup_dir=backup_dir,
        create_safety_backup=False,
    )

    assert store.get_chunk(_CUSTOM_CACHE_KEY, "session_a") == {"value": "original"}


def test_restore_rejects_path_traversal_backup_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    with pytest.raises(ValueError, match="inside the Lance backup directory"):
        restore_lance_backup("../x.tar.gz", store, backup_dir=backup_dir)


def test_restore_creates_pre_restore_safety_backup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backup_dir = tmp_path / "backups"
    store.save_chunk(_CACHE_KEY, "session_a", {"value": "original"})
    backup = create_lance_backup(store, backup_dir=backup_dir)["backup"]

    store.save_chunk(_CACHE_KEY, "session_a", {"value": "current"})
    result = restore_lance_backup(backup["filename"], store, backup_dir=backup_dir)

    assert result["safety_backup"] is not None
    assert result["safety_backup"]["filename"].startswith("telemetry_lance_prerestore_")
    assert Path(result["safety_backup"]["path"]).exists()


def test_cli_restore_latest_uses_newest_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    restored = {}

    monkeypatch.setattr(
        cli,
        "get_latest_lance_backup",
        lambda: {"filename": "telemetry_lance_latest.tar.gz"},
    )

    def fake_restore(backup_name: str, *, create_safety_backup: bool):
        restored["backup_name"] = backup_name
        restored["create_safety_backup"] = create_safety_backup
        return {
            "restored_backup": {"filename": backup_name},
            "safety_backup": None,
            "store_info": {"entry_count": 3},
        }

    monkeypatch.setattr(cli, "restore_lance_backup", fake_restore)

    assert cli.main(["restore", "--latest", "--no-safety-backup"]) == 0
    assert restored == {
        "backup_name": "telemetry_lance_latest.tar.gz",
        "create_safety_backup": False,
    }
