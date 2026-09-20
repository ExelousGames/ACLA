"""Convert Labelme polygons and stroked polylines into YOLO segmentation labels."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import yaml
from PIL import Image

from . import DEFAULT_LABELS, read_labels


def _polyline_polygon(points: list, width: int, height: int, stroke_width: int) -> list:
    import cv2
    import numpy as np

    pixels = np.rint(points).astype(np.int32)
    if len(np.unique(pixels, axis=0)) < 2:
        raise ValueError("Polyline needs at least two distinct pixel points.")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, [pixels], isClosed=False, color=255, thickness=stroke_width)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1 or cv2.contourArea(contours[0]) == 0:
        raise ValueError("Polyline stroke must form one nonzero-area region without holes inside the image.")
    return contours[0].reshape(-1, 2).tolist()


def _segmentation_rows(annotation: dict, labels: list[str], polyline_width: int) -> str:
    width, height = annotation["imageWidth"], annotation["imageHeight"]
    rows = []
    for shape in annotation["shapes"]:
        label = shape["label"]
        if label not in labels:
            raise ValueError(f"Unknown label {label!r}; add it to the labels file before exporting.")
        shape_type = shape.get("shape_type")
        if shape_type not in {"polygon", "linestrip", "line"}:
            raise ValueError("Only polygon, linestrip (polyline), and line shapes are supported.")
        is_polyline = shape_type != "polygon"
        kind = "Polyline" if is_polyline else "Polygon"
        points = shape["points"]
        minimum = 2 if is_polyline else 3
        if len(points) < minimum or any(len(point) != 2 for point in points):
            count = "two" if is_polyline else "three"
            raise ValueError(f"Each {kind.lower()} needs at least {count} (x, y) points.")
        if shape_type == "line" and len(points) != 2:
            raise ValueError("A line needs exactly two points; use linestrip for a polyline.")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for point in points for value in point
        ):
            raise ValueError(f"{kind} coordinates must be finite numbers.")
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
            raise ValueError(f"{kind} coordinates must lie inside the image.")
        if is_polyline:
            # YOLO segments are closed polygons. Stroke the open path rather
            # than joining its endpoints and filling the enclosed track area.
            points = _polyline_polygon(points, width, height, polyline_width)
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
    polyline_width: int = 8,
) -> Path:
    """Validate all input first, then write a new dataset without changing sources.

    Splits are explicit so adjacent frames from a session can stay together.
    Each polygon or polyline becomes one instance; group IDs are not merged.
    Polyline width is the stroke thickness in original-image pixels.
    """
    if not isinstance(polyline_width, int) or polyline_width < 2:
        raise ValueError("Polyline width must be an integer of at least 2 pixels.")
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
                rows = _segmentation_rows(annotation, labels, polyline_width)
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
