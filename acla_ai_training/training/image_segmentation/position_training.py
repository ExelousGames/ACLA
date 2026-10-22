"""Train a position CNN with per-map session splits and car-local XYZ errors."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random

import httpx
import numpy as np
import torch
from torch.nn import functional as F
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
NORMALIZED_POSITION_BINS = 20


def position_regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Smooth L1 over the predicted and target XYZ offsets."""
    return F.smooth_l1_loss(prediction, target)


def split_position_frames(records: list[dict], *, val_fraction=0.2, seed=42) -> dict[str, list[dict]]:
    """Split whole sessions per map, choosing the closest validation frame count."""
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1 (exclusive).")
    if not records:
        raise ValueError("Position training requires prepared frames.")
    maps, session_maps = {}, {}
    for row in sorted(records, key=lambda row: str(row["image"])):
        key = (row["game"], row["map_name"])
        session_id = row["session_id"]
        if session_maps.setdefault(session_id, key) != key:
            raise ValueError(f"Session {session_id!r} contains multiple game/map pairs.")
        maps.setdefault(key, {}).setdefault(session_id, []).append(row)
    rng = random.Random(seed)
    splits = {"train": [], "val": []}
    for (game, map_name), sessions in sorted(maps.items()):
        if len(sessions) < 2:
            raise ValueError(
                f"Position training requires at least two sessions per map; {game}/{map_name} has {len(sessions)}."
            )
        groups = [sessions[session_id] for session_id in sorted(sessions)]
        rng.shuffle(groups)
        total = sum(len(frames) for frames in groups)
        target = total * val_fraction
        # Bit n records whether n frames can be formed from whole sessions.
        # Keep each prefix so the chosen count can be traced back to sessions.
        reachable = [1]
        for frames in groups:
            reachable.append(reachable[-1] | (reachable[-1] << len(frames)))
        possible = reachable[-1] & ((1 << total) - 2)  # Exclude empty train/val sets.
        boundary = int(target) + 1
        below = possible & ((1 << boundary) - 1)
        above = possible >> boundary
        candidates = []
        if below:
            candidates.append(below.bit_length() - 1)
        if above:
            candidates.append(boundary + (above & -above).bit_length() - 1)
        rng.shuffle(candidates)
        count = min(candidates, key=lambda value: abs(value - target))
        # Prefer earlier shuffled sessions when several subsets achieve the count.
        for index in reversed(range(len(groups))):
            frames = groups[index]
            if reachable[index] & (1 << count):
                splits["train"].extend(frames)
            else:
                splits["val"].extend(frames)
                count -= len(frames)
    for frames in splits.values():
        rng.shuffle(frames)
    return splits


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
    output.mkdir(parents=True)
    assignments = {split: [str(row["image"]) for row in frames] for split, frames in splits.items()}
    maps = {
        split: [{"game": game, "map_name": map_name}
                for game, map_name in sorted({(row["game"], row["map_name"]) for row in frames})]
        for split, frames in splits.items()
    }
    (output / "split.json").write_text(
        json.dumps({"seed": seed, "val_fraction": val_fraction, "maps": maps, **assignments}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved split: {output / 'split.json'} "
        f"({len(splits['train'])} training frames, {len(splits['val'])} validation frames)",
        flush=True,
    )
    for split, label in (("train", "Training"), ("val", "Validation")):
        names = ", ".join(f"{row['game']}/{row['map_name']}" for row in maps[split])
        print(f"{label} maps: {names}", flush=True)
    # Row zero holds all validation frames; subsequent rows hold individual maps.
    map_indices = {(row["game"], row["map_name"]): index for index, row in enumerate(maps["val"], 1)}
    group_count = 1 + len(map_indices)
    section_boundaries = torch.tensor(
        [index / NORMALIZED_POSITION_BINS for index in range(1, NORMALIZED_POSITION_BINS)],
        dtype=torch.float64, device=device,
    )
    section_count = len(map_indices) * NORMALIZED_POSITION_BINS
    best, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        network.train()
        loss_total = 0.0
        train_error_total = torch.zeros((), dtype=torch.float64, device=device)
        for sample in loaders["train"]:
            optimizer.zero_grad()
            prediction = network(sample["image"].to(device), sample["references"].to(device), sample["context"].to(device))
            target = sample["target"].to(device)
            loss = position_regression_loss(prediction, target)
            train_error_total += (prediction.detach().double() - target.double()).abs().sum()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(prediction)
        network.eval()
        errors = torch.zeros(group_count, 3, dtype=torch.float64, device=device)
        distance_errors = torch.zeros(group_count, len(OFFSET_LENGTH_RANGES), dtype=torch.float64, device=device)
        distance_counts = torch.zeros_like(distance_errors, dtype=torch.int64)
        direction_errors = torch.zeros_like(distance_errors)
        direction_counts = torch.zeros_like(distance_counts)
        section_errors = torch.zeros(section_count, 3, dtype=torch.float64, device=device)
        section_distances = torch.zeros(section_count, dtype=torch.float64, device=device)
        section_counts = torch.zeros(section_count, dtype=torch.int64, device=device)
        with torch.no_grad():
            for sample in loaders["val"]:
                # Compute vector geometry in double precision, including tiny offsets.
                prediction = network(sample["image"].to(device), sample["references"].to(device), sample["context"].to(device)).double()
                target = sample["target"].to(device=device, dtype=torch.float64)
                difference = prediction - target
                distances = torch.linalg.vector_norm(difference, dim=-1)
                actual_lengths = torch.linalg.vector_norm(target, dim=-1)
                valid_direction = prediction.ne(0).any(-1) & target.ne(0).any(-1)
                angles = torch.rad2deg(torch.atan2(
                    torch.linalg.vector_norm(torch.cross(prediction, target, dim=-1), dim=-1),
                    (prediction * target).sum(-1),
                ))
                sample_maps = [map_indices[key] for key in zip(sample["game"], sample["map_name"])]
                # Use the original lap position, before float32 context encoding.
                # Internal boundaries belong to the next section; 1.0 stays in the last.
                section_indices = torch.bucketize(sample["normalized_position"].to(device), section_boundaries, right=True)
                section_indices += (torch.tensor(sample_maps, device=device) - 1) * NORMALIZED_POSITION_BINS
                section_errors.index_add_(0, section_indices, difference.abs().sum(1))
                section_distances.index_add_(0, section_indices, distances.squeeze(1))
                section_counts.index_add_(0, section_indices, torch.ones_like(section_indices))
                for group in {0, *sample_maps}:
                    in_map = torch.tensor([group == 0 or index == group for index in sample_maps], device=device)
                    errors[group] += difference[in_map].abs().sum((0, 1))
                    for index, (_, lower, upper) in enumerate(OFFSET_LENGTH_RANGES):
                        in_range = in_map[:, None] & (actual_lengths >= lower) & (actual_lengths < upper)
                        distance_errors[group, index] += distances[in_range].sum()
                        distance_counts[group, index] += in_range.sum()
                        direction_errors[group, index] += angles[in_range & valid_direction].sum()
                        direction_counts[group, index] += (in_range & valid_direction).sum()
        validation_metrics = []
        for axes, distances, counts, directions, valid_counts in zip(
            errors.cpu().tolist(), distance_errors.cpu().tolist(), distance_counts.cpu().tolist(),
            direction_errors.cpu().tolist(), direction_counts.cpu().tolist(),
        ):
            count = sum(counts)
            per_axis = [error / count for error in axes]
            validation_metrics.append({
                "count": count,
                "val_mae": sum(per_axis) / len(OUTPUT_NAMES),
                "val_mae_per_axis": dict(zip(OUTPUT_NAMES, per_axis)),
                "val_errors_by_offset_length": {
                    name: {
                        "count": bucket_count,
                        "mean_distance_error": distance / bucket_count if bucket_count else None,
                        "mean_direction_error_deg": direction / valid_count if valid_count else None,
                        "direction_valid_count": valid_count,
                    }
                    for (name, _, _), distance, bucket_count, direction, valid_count in zip(
                        OFFSET_LENGTH_RANGES, distances, counts, directions, valid_counts,
                    )
                },
            })
        for map_metrics, axes_by_section, distances, counts in zip(
            validation_metrics[1:],
            section_errors.view(-1, NORMALIZED_POSITION_BINS, 3).cpu().tolist(),
            section_distances.view(-1, NORMALIZED_POSITION_BINS).cpu().tolist(),
            section_counts.view(-1, NORMALIZED_POSITION_BINS).cpu().tolist(),
        ):
            sections = []
            for index, (axes, distance, count) in enumerate(zip(axes_by_section, distances, counts)):
                sections.append({
                    "normalized_position_start": index / NORMALIZED_POSITION_BINS,
                    "normalized_position_end": (index + 1) / NORMALIZED_POSITION_BINS,
                    "count": count,
                    "val_mae": sum(axes) / (count * len(OUTPUT_NAMES)) if count else None,
                    "val_mae_per_axis": {name: error / count if count else None for name, error in zip(OUTPUT_NAMES, axes)},
                    "mean_distance_error": distance / count if count else None,
                    "weakness_rank": None,
                })
            ranked = sorted((section for section in sections if section["count"]),
                            key=lambda section: section["mean_distance_error"], reverse=True)
            for rank, section in enumerate(ranked, 1):
                section["weakness_rank"] = rank
            map_metrics["val_errors_by_normalized_position"] = sections
        overall = validation_metrics[0]
        metrics = {
            "epoch": epoch,
            "train_loss": loss_total / len(splits["train"]),
            "train_mae": train_error_total.item() / (len(splits["train"]) * len(OUTPUT_NAMES)),
            "val_mae": overall["val_mae"],
            "val_mae_per_axis": overall["val_mae_per_axis"],
            "val_errors_by_offset_length": overall["val_errors_by_offset_length"],
            "val_by_map": [{**row, **values} for row, values in zip(maps["val"], validation_metrics[1:])],
        }
        if not np.isfinite([
            metrics["train_loss"],
            metrics["train_mae"],
            *(value for group in validation_metrics for value in group["val_mae_per_axis"].values()),
            *(value for group in validation_metrics for bucket in group["val_errors_by_offset_length"].values()
              for value in bucket.values() if value is not None),
        ]).all():
            raise RuntimeError("Position training produced non-finite loss or errors.")
        history.append(metrics)
        network.save(output / "last.pt", metrics=metrics)
        if metrics["val_mae"] < best:
            best = metrics["val_mae"]
            network.save(output / "best.pt", metrics=metrics)
        (output / "results.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch}/{epochs}: loss={metrics['train_loss']:.6g}, "
            f"train MAE={metrics['train_mae']:.6g}, val MAE={metrics['val_mae']:.6g} (local coordinates)",
            flush=True,
        )
        for row in metrics["val_by_map"]:
            print(
                f"  Validation map {row['game']}/{row['map_name']}: {row['count']} frames, "
                f"val MAE={row['val_mae']:.6g} (local coordinates)",
                flush=True,
            )
            for name, bucket in row["val_errors_by_offset_length"].items():
                distance = bucket["mean_distance_error"]
                distance_text = "n/a" if distance is None else f"{distance:.6g} (local coordinates)"
                direction = bucket["mean_direction_error_deg"]
                angle = "n/a" if direction is None else f"{direction:.6g} deg"
                print(
                    f"    actual offset length {name}: mean distance error={distance_text}, "
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
            record = asyncio.run(upload_position_checkpoint(artifact))
        except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as exc:
            raise RuntimeError(
                f"Training completed, but backend upload failed: {exc}. "
                f"The local checkpoint and ONNX artifact are preserved at {checkpoint} and {artifact}. "
                "Use the upload-position command to publish it without retraining."
            ) from exc
        print(f"Model uploaded to backend: {record['_id']}", flush=True)
    return checkpoint
