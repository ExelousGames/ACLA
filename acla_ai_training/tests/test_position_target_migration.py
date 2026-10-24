import json
import tarfile

import pytest

from scripts.migrate_position_targets import main, migrate_position_targets
from training.image_segmentation.position_data import TARGET_DEFINITION


@pytest.fixture
def targets(tmp_path):
    directory = tmp_path / "dataset" / "session" / "frames"
    directory.mkdir(parents=True)
    for index, values in enumerate(([[1.25, -2, 3]], [[-4, 5, -6], [7, 8, 9]])):
        (directory / f"frame_{index}.position.json").write_text(json.dumps({
            "game": "acc", "coordinate_frame": "car_local", "target": values,
            "image": f"frame_{index}.jpg", "references": [[10, 20, 30]],
            "driver_position": [11, 22, 33], "normalized_position": 0.25,
        }))
    return tmp_path / "dataset"


def test_migration_backs_up_exact_originals_changes_only_x_and_is_repeatable(targets, tmp_path):
    originals = {path: path.read_bytes() for path in targets.rglob("*.position.json")}
    annotation = targets / "session" / "frames" / "frame_0.json"
    annotation.write_text('{"shapes": []}')
    image = annotation.with_suffix(".jpg")
    image.write_bytes(b"original image")
    assert migrate_position_targets(targets) == {"files": 2, "pending": 2, "unchanged": 0, "migrated": 0}
    assert all(path.read_bytes() == original for path, original in originals.items())

    backup = tmp_path / "backup.tar.gz"
    result = migrate_position_targets(targets, backup=backup)
    assert result == {"files": 2, "pending": 0, "unchanged": 0, "migrated": 2, "backup": str(backup)}
    with tarfile.open(backup) as archive:
        for path, original in originals.items():
            assert archive.extractfile(str(path.relative_to(targets))).read() == original
            expected = json.loads(original)
            expected["target"] = [[-x, y, z] for x, y, z in expected["target"]]
            expected["target_definition"] = TARGET_DEFINITION
            assert json.loads(path.read_text()) == expected
    corrected = {path: path.read_bytes() for path in originals}
    assert migrate_position_targets(targets, backup=backup) == {
        "files": 2, "pending": 0, "unchanged": 2, "migrated": 0,
    }
    assert all(path.read_bytes() == content for path, content in corrected.items())
    assert annotation.read_text() == '{"shapes": []}'
    assert image.read_bytes() == b"original image"


@pytest.mark.parametrize("change", [
    {"game": "iracing"}, {"coordinate_frame": "world"}, {"target_definition": "unknown"},
    {"target": [[1, 2]]}, {"target": [[1, 2, float("nan")]]}, {"target": [[True, 2, 3]]},
])
def test_preflight_rejects_invalid_file_before_any_write(targets, tmp_path, change):
    path = sorted(targets.rglob("*.position.json"))[-1]
    data = json.loads(path.read_text())
    data.update(change)
    path.write_text(json.dumps(data))
    originals = {p: p.read_bytes() for p in targets.rglob("*.position.json")}
    backup = tmp_path / "backup.tar.gz"
    with pytest.raises(ValueError):
        migrate_position_targets(targets, backup=backup)
    assert not backup.exists()
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_existing_backup_is_not_overwritten(targets, tmp_path):
    backup = tmp_path / "backup.tar.gz"
    backup.write_bytes(b"previous backup")
    originals = {p: p.read_bytes() for p in targets.rglob("*.position.json")}
    with pytest.raises(FileExistsError):
        migrate_position_targets(targets, backup=backup)
    assert backup.read_bytes() == b"previous backup"
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_cli_requires_apply_and_backup_together(targets, tmp_path, capsys):
    assert main([str(targets)]) == 0
    assert json.loads(capsys.readouterr().out)["migrated"] == 0
    for flags in (["--apply"], ["--backup", str(tmp_path / "backup.tar.gz")]):
        with pytest.raises(SystemExit):
            main([str(targets), *flags])
