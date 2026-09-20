"""Labelme polygon/polyline annotation and Ultralytics training for racing images."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PACKAGE_DIR.parents[1]
DEFAULT_LABELS = PACKAGE_DIR / "labels.txt"


def read_labels(path: Path = DEFAULT_LABELS) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("The labels file must contain unique, nonempty class names.")
    return labels
