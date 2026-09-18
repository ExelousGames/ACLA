"""Validation metrics and threshold calibration for cropper training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.ml.segment_cropper.decoding import CropperThresholds, decode_probabilities


@dataclass(frozen=True)
class ValidationProbabilities:
    start: np.ndarray
    end: np.ndarray
    inside: np.ndarray
    annotations: tuple[tuple[int, int], ...]


def interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return 0.0 if union <= 0 else intersection / union


def _maximum_matches(
    proposals: Sequence[tuple[int, int]],
    annotations: Sequence[tuple[int, int]],
    minimum_iou: float,
) -> int:
    edges = [
        [
            annotation_index
            for annotation_index, annotation in enumerate(annotations)
            if interval_iou(proposal, annotation) >= minimum_iou
        ]
        for proposal in proposals
    ]
    matched_proposal: dict[int, int] = {}

    def augment(proposal_index: int, visited: set[int]) -> bool:
        for annotation_index in edges[proposal_index]:
            if annotation_index in visited:
                continue
            visited.add(annotation_index)
            previous = matched_proposal.get(annotation_index)
            if previous is None or augment(previous, visited):
                matched_proposal[annotation_index] = proposal_index
                return True
        return False

    matches = 0
    for proposal_index in range(len(proposals)):
        if augment(proposal_index, set()):
            matches += 1
    return matches


def evaluate_thresholds(
    validation: Sequence[ValidationProbabilities],
    thresholds: CropperThresholds,
    minimum_iou: float = 0.5,
) -> dict[str, float | int]:
    true_positives = 0
    proposal_count = 0
    annotation_count = 0
    for session in validation:
        selected = decode_probabilities(
            session.start,
            session.end,
            session.inside,
            thresholds,
        )
        proposal_ranges = [
            (candidate.start_index, candidate.end_index)
            for candidate in selected
        ]
        true_positives += _maximum_matches(
            proposal_ranges,
            session.annotations,
            minimum_iou,
        )
        proposal_count += len(proposal_ranges)
        annotation_count += len(session.annotations)

    precision = true_positives / proposal_count if proposal_count else 0.0
    recall = true_positives / annotation_count if annotation_count else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "proposal_count": proposal_count,
        "annotation_count": annotation_count,
        "iou_threshold": float(minimum_iou),
    }


def calibrate_thresholds(
    validation: Sequence[ValidationProbabilities],
    threshold_values: Sequence[float] | None = None,
    target_recall: float = 0.95,
) -> tuple[CropperThresholds, dict[str, float | int]]:
    if not validation or not any(session.annotations for session in validation):
        raise ValueError("segment_cropper calibration requires validation annotations")
    source_values = (
        tuple(np.linspace(0.1, 0.9, 9))
        if threshold_values is None
        else threshold_values
    )
    values = tuple(
        float(value)
        for value in source_values
    )
    if not values or any(not 0 <= value <= 1 for value in values):
        raise ValueError("Calibration threshold values must be between zero and one")

    results: list[tuple[CropperThresholds, dict[str, float | int]]] = []
    for boundary in values:
        for inside in values:
            for proposal in values:
                thresholds = CropperThresholds(boundary, inside, proposal)
                results.append((thresholds, evaluate_thresholds(validation, thresholds)))

    eligible = [item for item in results if float(item[1]["recall"]) >= target_recall]
    pool = eligible or results

    def key(item):
        thresholds, metrics = item
        objective = (
            (float(metrics["precision"]), float(metrics["recall"]))
            if eligible
            else (float(metrics["recall"]), float(metrics["precision"]))
        )
        return (*objective, thresholds.boundary, thresholds.inside, thresholds.proposal)

    thresholds, metrics = max(pool, key=key)
    return thresholds, {
        **metrics,
        "target_recall": float(target_recall),
        "target_recall_attained": bool(float(metrics["recall"]) >= target_recall),
    }


__all__ = ["ValidationProbabilities", "calibrate_thresholds", "evaluate_thresholds", "interval_iou"]
