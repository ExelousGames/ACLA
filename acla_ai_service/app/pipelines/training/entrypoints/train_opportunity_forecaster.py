#!/usr/bin/env python3
"""Train the opportunity forecaster from an annotation dataset.

The Segment Annotation App launches this script from the Model Components
section. Each annotated O/OD segment becomes a positive example using the early
part of that segment as the "recent telemetry" window. Non-O/OD segments become
NO_OPPORTUNITY negatives, capped by --max-negatives.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from app.integrations.backend.client import backend_service
from app.ml.opportunity_forecaster.service import (
    FORECAST_LABELS,
    NO_OPPORTUNITY,
    opportunity_forecaster,
)
from app.storage import get_shared_telemetry_store


def _iter_annotation_segments(annotation_key: str) -> Iterable[Dict[str, Any]]:
    store = get_shared_telemetry_store()
    if not store.has_cached_data(annotation_key):
        raise ValueError(f"Annotation dataset not found: {annotation_key}")

    for chunk in store.get_cached_data_chunks(annotation_key):
        if isinstance(chunk, list):
            items = chunk
        elif isinstance(chunk, dict) and isinstance(chunk.get("data"), list):
            items = chunk["data"]
        elif isinstance(chunk, dict) and isinstance(chunk.get("payload"), dict):
            items = [chunk["payload"]]
        else:
            items = [chunk]

        for item in items:
            if isinstance(item, dict):
                yield item


def _target_label(labels: List[Any]) -> str:
    label_set = {str(label) for label in labels}
    for label in FORECAST_LABELS:
        if label in label_set:
            return label
    return NO_OPPORTUNITY


def _window_rows(segment: Dict[str, Any], input_fraction: float) -> List[Dict[str, Any]]:
    rows = segment.get("forecast_telemetry_data") or segment.get("telemetry_data") or []
    if not isinstance(rows, list) or not rows:
        return []
    keep = max(1, int(round(len(rows) * max(0.1, min(1.0, input_fraction)))))
    return rows[:keep]


def build_training_examples(
    annotation_key: str,
    *,
    input_fraction: float,
    max_negatives: int,
) -> List[Dict[str, Any]]:
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    skipped_empty = 0

    for segment in _iter_annotation_segments(annotation_key):
        rows = _window_rows(segment, input_fraction)
        if not rows:
            skipped_empty += 1
            continue

        label = _target_label(segment.get("labels") or [])
        example = {
            "telemetry_data": rows,
            "target_label": label,
            "source_segment_id": segment.get("id"),
            "source_labels": segment.get("labels") or [],
        }
        if label == NO_OPPORTUNITY:
            negatives.append(example)
        else:
            positives.append(example)

    if max_negatives >= 0 and len(negatives) > max_negatives:
        random.Random(42).shuffle(negatives)
        negatives = negatives[:max_negatives]

    examples = positives + negatives
    print(
        "[INFO] Built opportunity examples: "
        f"positives={len(positives)} negatives={len(negatives)} "
        f"skipped_empty={skipped_empty}"
    )
    by_label: Dict[str, int] = {}
    for example in examples:
        label = str(example["target_label"])
        by_label[label] = by_label.get(label, 0) + 1
    print(f"[INFO] Label distribution: {by_label}")

    if not positives:
        raise ValueError(
            "No positive O/OD opportunity examples found. Add annotations with "
            f"one of: {', '.join(FORECAST_LABELS)}"
        )
    if not examples:
        raise ValueError("No usable examples found in annotation dataset")
    return examples


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the future overtake/defense opportunity forecaster.",
    )
    parser.add_argument("--annotation-key", required=True)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    parser.add_argument("--max-negatives", type=int, default=5000)
    args = parser.parse_args()

    print(
        "[INFO] Starting opportunity forecaster training: "
        f"annotation_key={args.annotation_key} "
        f"input_fraction={args.input_fraction} "
        f"max_negatives={args.max_negatives}"
    )
    examples = build_training_examples(
        args.annotation_key,
        input_fraction=args.input_fraction,
        max_negatives=args.max_negatives,
    )
    result = opportunity_forecaster.train(examples)
    print(f"[INFO] Training complete: {result}")

    try:
        await backend_service.save_ai_model(
            model_type="opportunity_forecaster",
            model_data=opportunity_forecaster.serialize_artifacts(),
            metadata={
                "annotation_key": args.annotation_key,
                "input_fraction": args.input_fraction,
                "max_negatives": args.max_negatives,
                "classes": result.get("classes", []),
                "feature_count": result.get("feature_count", 0),
                "samples": result.get("samples", 0),
            },
            is_active=True,
        )
        print("[INFO] opportunity_forecaster uploaded to backend")
    except Exception as exc:
        print(f"[WARN] Backend upload failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
