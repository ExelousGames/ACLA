"""Persist Labelme sample assignments and prepare exactly those samples for training."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

from . import DEFAULT_LABELS, WORKSPACE_DIR
from .dataset import prepare_dataset


DEFAULT_SOURCE = WORKSPACE_DIR / "storage/annotation_images"
DEFAULT_SPLIT_FILE = WORKSPACE_DIR / "storage/image_segmentation/split.json"


def _split_samples(source: Path, val_fraction: float, seed: int) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1 (exclusive).")
    samples = []
    for path in sorted(source.rglob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        # Frame folders may also contain metadata unrelated to Labelme.
        if not isinstance(annotation, dict) or not {"imagePath", "shapes"} <= annotation.keys():
            continue
        samples.append(path.relative_to(source).as_posix())
    if len(samples) < 2:
        raise ValueError(f"Need at least two annotated images in {source} to split training and validation.")
    random.Random(seed).shuffle(samples)
    count = max(1, min(len(samples) - 1, round(len(samples) * val_fraction)))
    return {
        "seed": seed, "val_fraction": val_fraction,
        "train": sorted(samples[count:]), "val": sorted(samples[:count]),
    }


def prepare_training_dataset(
    source: Path | None = None,
    *,
    split_file: Path = DEFAULT_SPLIT_FILE,
    val_fraction: float = 0.2,
    seed: int = 42,
    rebuild_split: bool = False,
    labels_file: Path = DEFAULT_LABELS,
    polyline_width: int = 8,
) -> Path:
    """Reuse a saved split, or discover samples once and save a new split.

    Paths are relative so the workspace can move between the host and Docker.
    Each invocation exports a fresh dataset to pick up annotation edits without
    including stale files or moving samples between training and validation.
    """
    split_file = split_file.expanduser().resolve()
    source = source.expanduser().resolve() if source is not None else None
    reuse = split_file.exists() and not rebuild_split
    if reuse or (split_file.exists() and source is None):
        manifest = json.loads(split_file.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("source"), str):
            raise ValueError(f"Invalid split file: {split_file}")
        saved_source = (split_file.parent / manifest["source"]).resolve()
        if reuse and source is not None and source != saved_source:
            raise ValueError(
                f"Saved split belongs to {saved_source}; use --rebuild-split or another --split-file."
            )
        source = saved_source
    if not reuse:
        source = source or DEFAULT_SOURCE
        if not source.is_dir():
            raise ValueError(f"Annotation directory does not exist: {source}")
        manifest = _split_samples(source, val_fraction, seed)
        manifest["source"] = os.path.relpath(source, split_file.parent)

    annotations = {}
    for split in ("train", "val"):
        samples = manifest.get(split)
        if not isinstance(samples, list) or not samples or any(
            not isinstance(sample, str) or not sample
            or Path(sample).is_absolute() or ".." in Path(sample).parts
            for sample in samples
        ):
            raise ValueError(f"Split {split!r} must be a nonempty list of relative annotation paths.")
        annotations[split] = [source / sample for sample in samples]

    split_file.parent.mkdir(parents=True, exist_ok=True)
    # A new export also prevents Ultralytics from reusing a stale label cache.
    run_dir = Path(tempfile.mkdtemp(prefix="dataset_", dir=split_file.parent))
    try:
        data = prepare_dataset(
            source, source, run_dir / "yolo", annotation_splits=annotations,
            labels_file=labels_file, polyline_width=polyline_width,
        )
    except (OSError, ValueError):
        if not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise
    if not reuse:
        temporary = split_file.with_suffix(split_file.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(split_file)
    print(
        f"{'Reusing' if reuse else 'Saved'} split: {split_file} "
        f"({len(annotations['train'])} training, {len(annotations['val'])} validation)",
        flush=True,
    )
    return data
