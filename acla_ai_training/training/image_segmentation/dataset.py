"""Convert Labelme JSON polygons into Ultralytics segmentation labels."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import yaml
from PIL import Image

from . import DEFAULT_LABELS, read_labels


def _polygon_rows(annotation: dict, labels: list[str]) -> str:
    width, height = annotation["imageWidth"], annotation["imageHeight"]
    rows = []
    for shape in annotation["shapes"]:
        label = shape["label"]
        if label not in labels:
            raise ValueError(f"Unknown label {label!r}; add it to the labels file before exporting.")
        if shape.get("shape_type") != "polygon":
            raise ValueError("Only polygon shapes are supported; redraw other shapes as polygons.")
        points = shape["points"]
        if len(points) < 3 or any(len(point) != 2 for point in points):
            raise ValueError("Each polygon needs at least three (x, y) points.")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for point in points for value in point
        ):
            raise ValueError("Polygon coordinates must be finite numbers.")
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
            raise ValueError("Polygon coordinates must lie inside the image.")
        area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        if area == 0:
            raise ValueError("Polygon must have nonzero area.")
        coordinates = " ".join(f"{x / width:.8f} {y / height:.8f}" for x, y in points)
        rows.append(f"{labels.index(label)} {coordinates}\n")
    return "".join(rows)


def prepare_dataset(
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    *,
    labels_file: Path = DEFAULT_LABELS,
) -> Path:
    """Validate all input first, then write a new dataset without changing sources.

    Splits are explicit so adjacent frames from a session can stay together.
    Each polygon becomes one instance; Labelme group IDs are not merged.
    """
    labels = read_labels(labels_file)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new dataset directory: {output_dir}")
    records = []
    seen_images: set[Path] = set()
    for split, source in (("train", train_dir), ("val", val_dir)):
        source = source.resolve()
        annotations = sorted(source.rglob("*.json"))
        if not annotations:
            raise ValueError(f"No Labelme JSON annotations in {split} directory: {source}")
        for annotation_path in annotations:
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                image_path = (annotation_path.parent / annotation["imagePath"]).resolve()
                if image_path in seen_images:
                    raise ValueError(f"Image occurs more than once in the dataset: {image_path}")
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                    raise ValueError(f"Unsupported image type: {image_path.suffix}")
                with Image.open(image_path) as image:
                    image.load()
                    if image.size != (annotation["imageWidth"], annotation["imageHeight"]):
                        raise ValueError("Annotation dimensions do not match the original image.")
                rows = _polygon_rows(annotation, labels)
            except (KeyError, TypeError, OSError, ValueError) as exc:
                raise ValueError(f"{annotation_path}: {exc}") from exc
            seen_images.add(image_path)
            relative = annotation_path.relative_to(source)
            records.append((split, relative, image_path, rows))

    for split, relative, image_path, rows in records:
        target_image = output_dir / "images" / split / relative.with_suffix(image_path.suffix.lower())
        target_label = output_dir / "labels" / split / relative.with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        target_label.write_text(rows, encoding="utf-8")
    data_path = output_dir / "data.yaml"
    # Omitting `path` makes Ultralytics resolve paths relative to this YAML,
    # so a dataset prepared on the host also works through the Docker mount.
    data_path.write_text(yaml.safe_dump({
        "train": "images/train", "val": "images/val", "names": labels,
    }, sort_keys=False), encoding="utf-8")
    return data_path
