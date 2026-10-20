"""Train a standalone position CNN from scratch with split frames and raw XYZ errors."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random

import httpx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .position_data import OUTPUT_NAMES, PositionDataset, load_position_manifest
from .position_export import export_position_checkpoint
from .position_model import PositionModel


OFFSET_LENGTH_RANGES = (
    ("<0.1", 0, 0.1),
    ("0.1-0.25", 0.1, 0.25),
    ("0.25-0.5", 0.25, 0.5),
    ("0.5-1", 0.5, 1),
    ("1-2", 1, 2),
    ("2-5", 2, 5),
    (">=5", 5, float("inf")),
)


def split_position_frames(records: list[dict], *, val_fraction=0.2, seed=42) -> dict[str, list[dict]]:
    """Shuffle individual frames reproducibly, regardless of their source sessions."""
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1 (exclusive).")
    if len(records) < 2:
        raise ValueError("Position training requires at least two frames for training and validation.")
    frames = sorted(records, key=lambda row: str(row["image"]))
    random.Random(seed).shuffle(frames)
    count = max(1, min(len(frames) - 1, round(len(frames) * val_fraction)))
    return {"train": frames[count:], "val": frames[:count]}


def train_position_model(
    manifest: Path,
    *,
    output: Path,
    epochs=100,
    imgsz=768,
    batch=8,
    device="0",
    workers=0,
    val_fraction=0.2,
    seed=42,
    upload=True,
    upload_name="track-position",
):
    if epochs < 1 or batch < 1 or workers < 0:
        raise ValueError("Epochs and batch must be positive; workers must be nonnegative.")
    if output.exists():
        raise FileExistsError(f"Choose a new position training output directory: {output}")
    sessions = load_position_manifest(manifest)
    records = [frame for frames in sessions.values() for frame in frames]
    splits = split_position_frames(records, val_fraction=val_fraction, seed=seed)
    torch.manual_seed(seed)
    device = f"cuda:{device}" if str(device).isdigit() else device
    categories = {field: sorted({row[field] for row in records}) for field in ("game", "map_name")}
    network = PositionModel(imgsz, categories=categories).to(device)
    positions = torch.from_numpy(np.stack([row["references"][0] for row in splits["train"]])).to(device)
    # Standardize XYZ using training references alone; context stays in [0, 1].
    network.position_mean.copy_(positions.mean(0))
    network.position_std.copy_(positions.std(0, unbiased=False).clamp_min(1.0))
    loaders = {
        split: DataLoader(PositionDataset(records, imgsz, categories), batch_size=batch, shuffle=split == "train", num_workers=workers)
        for split, records in splits.items()
    }
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    loss_function = nn.SmoothL1Loss()
    output.mkdir(parents=True)
    assignments = {split: [str(row["image"]) for row in frames] for split, frames in splits.items()}
    (output / "split.json").write_text(
        json.dumps({"seed": seed, "val_fraction": val_fraction, **assignments}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved split: {output / 'split.json'} "
        f"({len(splits['train'])} training frames, {len(splits['val'])} validation frames)",
        flush=True,
    )
    best, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        network.train()
        loss_total = 0.0
        for sample in loaders["train"]:
            optimizer.zero_grad()
            prediction = network(sample["image"].to(device), sample["references"].to(device), sample["context"].to(device))
            loss = loss_function(prediction, sample["target"].to(device))
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(prediction)
        network.eval()
        errors = torch.zeros(1, 3, dtype=torch.float64, device=device)
        distance_errors = torch.zeros(len(OFFSET_LENGTH_RANGES), dtype=torch.float64, device=device)
        distance_counts = torch.zeros(len(OFFSET_LENGTH_RANGES), dtype=torch.int64, device=device)
        direction_errors = torch.zeros(len(OFFSET_LENGTH_RANGES), dtype=torch.float64, device=device)
        direction_counts = torch.zeros(len(OFFSET_LENGTH_RANGES), dtype=torch.int64, device=device)
        with torch.no_grad():
            for sample in loaders["val"]:
                # Compute vector geometry in double precision, including tiny offsets.
                prediction = network(sample["image"].to(device), sample["references"].to(device), sample["context"].to(device)).double()
                target = sample["target"].to(device=device, dtype=torch.float64)
                difference = prediction - target
                errors += difference.abs().sum(0)
                distances = torch.linalg.vector_norm(difference, dim=-1)
                actual_lengths = torch.linalg.vector_norm(target, dim=-1)
                valid_direction = prediction.ne(0).any(-1) & target.ne(0).any(-1)
                angles = torch.rad2deg(torch.atan2(
                    torch.linalg.vector_norm(torch.cross(prediction, target, dim=-1), dim=-1),
                    (prediction * target).sum(-1),
                ))
                for index, (_, lower, upper) in enumerate(OFFSET_LENGTH_RANGES):
                    in_range = (actual_lengths >= lower) & (actual_lengths < upper)
                    distance_errors[index] += distances[in_range].sum()
                    distance_counts[index] += in_range.sum()
                    direction_errors[index] += angles[in_range & valid_direction].sum()
                    direction_counts[index] += (in_range & valid_direction).sum()
        per_axis = (errors / len(splits["val"])).flatten().cpu().tolist()
        offset_metrics = {
            name: {
                "count": count,
                "mean_distance_error": distance / count if count else None,
                "mean_direction_error_deg": direction / valid_count if valid_count else None,
                "direction_valid_count": valid_count,
            }
            for (name, _, _), distance, count, direction, valid_count in zip(
                OFFSET_LENGTH_RANGES, distance_errors.cpu().tolist(), distance_counts.cpu().tolist(),
                direction_errors.cpu().tolist(), direction_counts.cpu().tolist(),
            )
        }
        metrics = {
            "epoch": epoch,
            "train_loss": loss_total / len(splits["train"]),
            "val_mae": sum(per_axis) / len(OUTPUT_NAMES),
            "val_mae_per_axis": dict(zip(OUTPUT_NAMES, per_axis)),
            "val_errors_by_offset_length": offset_metrics,
        }
        if not np.isfinite([
            metrics["train_loss"], *per_axis,
            *(value for bucket in offset_metrics.values() for value in bucket.values() if value is not None),
        ]).all():
            raise RuntimeError("Position training produced non-finite loss or errors.")
        history.append(metrics)
        network.save(output / "last.pt", metrics=metrics)
        if metrics["val_mae"] < best:
            best = metrics["val_mae"]
            network.save(output / "best.pt", metrics=metrics)
        (output / "results.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"Epoch {epoch}/{epochs}: loss={metrics['train_loss']:.6g}, val MAE={metrics['val_mae']:.6g} (raw coordinates)", flush=True)
        for name, bucket in offset_metrics.items():
            distance = bucket["mean_distance_error"]
            distance_text = "n/a" if distance is None else f"{distance:.6g} (raw coordinates)"
            direction = bucket["mean_direction_error_deg"]
            angle = "n/a" if direction is None else f"{direction:.6g} deg"
            print(
                f"  actual offset length {name}: mean distance error={distance_text}, "
                f"mean direction error={angle} ({bucket['direction_valid_count']}/{bucket['count']} valid)",
                flush=True,
            )
    checkpoint = output / "best.pt"
    print(f"Exporting Track Vision ONNX: {checkpoint}", flush=True)
    try:
        artifact = export_position_checkpoint(checkpoint)
    except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Training completed, but ONNX export failed: {exc}. "
            f"The local checkpoint is preserved at {checkpoint}. "
            "Use the upload-position command to export and publish it without retraining."
        ) from exc
    print(f"Track Vision ONNX ready: {artifact}", flush=True)
    if upload:
        from .position_publication import upload_position_checkpoint

        print(f"Uploading Track Vision ONNX: {artifact}", flush=True)
        try:
            record = asyncio.run(upload_position_checkpoint(artifact, name=upload_name))
        except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as exc:
            raise RuntimeError(
                f"Training completed, but backend upload failed: {exc}. "
                f"The local checkpoint and ONNX artifact are preserved at {checkpoint} and {artifact}. "
                "Use the upload-position command to publish it without retraining."
            ) from exc
        print(f"Model uploaded to backend: {record['_id']}", flush=True)
    return checkpoint
