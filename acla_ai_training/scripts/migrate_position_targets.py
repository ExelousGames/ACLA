"""Convert ACC left-positive local targets to right/forward/up, once.

For datasets prepared with negative ACC heading/pitch/roll. Older rotation or
world-space datasets must be regenerated first. Images and annotations are untouched.
Without --apply this only validates and counts files. Applying requires a new
backup archive; all original JSON files are backed up before any replacement.

python scripts/migrate_position_targets.py storage/session_downloads
python scripts/migrate_position_targets.py storage/session_downloads --apply --backup storage/backups/position-targets-left-positive.tar.gz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tarfile
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.position_data import COORDINATE_FRAME, TARGET_DEFINITION


def _validate(data, path):
    if data.get("game") != "acc" or data.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError(f"Migration requires ACC car-local targets: {path}")
    if data.get("target_definition") not in (
        None, "inverse_world_rotation @ (reference_position - current_position)", TARGET_DEFINITION,
    ):
        raise ValueError(f"Unknown target definition: {path}")
    target = data.get("target")
    if not (
        isinstance(target, list) and len(target) in (1, 2)
        and all(isinstance(row, list) and len(row) == 3 for row in target)
        and all(type(v) in (int, float) and math.isfinite(v) for row in target for v in row)
    ):
        raise ValueError(f"Targets must contain one or two finite XYZ triples: {path}")


def migrate_position_targets(directory: Path, *, backup: Path | None = None):
    directory = directory.resolve()
    files = sorted(directory.rglob("*.position.json"))
    if not files:
        raise ValueError(f"No position targets found: {directory}")
    pending = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        _validate(data, path)
        if data.get("target_definition") != TARGET_DEFINITION:
            pending.append(path)
    result = {"files": len(files), "pending": len(pending), "unchanged": len(files) - len(pending), "migrated": 0}
    if backup is None or not pending:
        return result

    backup = backup.resolve()
    backup.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(backup, "x:gz") as archive:
        for path in pending:
            archive.add(path, arcname=str(path.relative_to(directory)))

    # Apply the backed-up snapshot, with target and convention replaced together.
    with tarfile.open(backup, "r:gz") as archive:
        for member in archive:
            path = directory / member.name
            with archive.extractfile(member) as original:
                data = json.load(original)
            for target in data["target"]:
                target[0] = -target[0]
            data["target_definition"] = TARGET_DEFINITION
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    json.dump(data, temporary, indent=2, allow_nan=False)
                    temporary.write("\n")
                    temporary.flush()
                    temporary_path.chmod(member.mode)
                    temporary_path.replace(path)
                finally:
                    temporary_path.unlink(missing_ok=True)
            result["migrated"] += 1
    result["backup"] = str(backup)
    result["pending"] = 0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path, help="New .tar.gz archive for original target files")
    args = parser.parse_args(argv)
    if args.apply != (args.backup is not None):
        parser.error("Use --apply and --backup together; omit both for a dry run.")
    print(json.dumps(migrate_position_targets(args.directory, backup=args.backup), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
