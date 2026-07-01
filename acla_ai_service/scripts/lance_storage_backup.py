"""Create, list, and restore Lance telemetry store backups.

Examples:
    python scripts/lance_storage_backup.py create
    python scripts/lance_storage_backup.py list
    python scripts/lance_storage_backup.py restore telemetry_lance_20260606_120000.tar.gz
    python scripts/lance_storage_backup.py restore --latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_paths()


from app.storage.lance.backup import (  # noqa: E402
    create_lance_backup,
    get_latest_lance_backup,
    list_lance_backups,
    restore_lance_backup,
)


def _format_size(item: Dict[str, Any]) -> str:
    size_mb = item.get("size_mb")
    if size_mb is None:
        return "unknown size"
    return f"{size_mb} MB"


def _print_backup(item: Dict[str, Any]) -> None:
    print(f"{item['filename']}  {_format_size(item)}  {item['created_at']}")


def create_backup(_args: argparse.Namespace) -> int:
    result = create_lance_backup()
    backup = result["backup"]
    manifest = result["manifest"]
    print(f"Created Lance backup: {backup['path']}")
    print(f"Entries: {manifest['entry_count']}")
    print(f"Files: {manifest['file_count']}")
    print(f"Size: {round(manifest['size_bytes'] / (1024 * 1024), 2)} MB")
    return 0


def list_backups(_args: argparse.Namespace) -> int:
    result = list_lance_backups()
    print(f"Backup directory: {result['backup_directory']}")
    if result["backup_count"] == 0:
        print("No Lance backups found.")
        return 0

    for backup in result["backups"]:
        _print_backup(backup)
    return 0


def restore_backup(args: argparse.Namespace) -> int:
    backup_name = args.backup_name
    if args.latest:
        latest = get_latest_lance_backup()
        if latest is None:
            raise SystemExit("No Lance backups found.")
        backup_name = latest["filename"]

    if not backup_name:
        raise SystemExit("restore requires a backup filename or --latest")

    result = restore_lance_backup(
        backup_name,
        create_safety_backup=not args.no_safety_backup,
    )
    print(f"Restored Lance backup: {result['restored_backup']['filename']}")
    if result["safety_backup"]:
        print(f"Pre-restore safety backup: {result['safety_backup']['filename']}")
    print(f"Store entries: {result['store_info']['entry_count']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, list, and restore AI service Lance telemetry store backups.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a Lance store backup.")
    create_parser.set_defaults(func=create_backup)

    list_parser = subparsers.add_parser("list", help="List available Lance store backups.")
    list_parser.set_defaults(func=list_backups)

    restore_parser = subparsers.add_parser("restore", help="Restore a Lance store backup.")
    restore_parser.add_argument("backup_name", nargs="?", help="Backup filename to restore.")
    restore_parser.add_argument(
        "--latest",
        action="store_true",
        help="Restore the newest available Lance backup.",
    )
    restore_parser.add_argument(
        "--no-safety-backup",
        action="store_true",
        help="Do not create a pre-restore safety backup of the current store.",
    )
    restore_parser.set_defaults(func=restore_backup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
