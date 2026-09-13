"""Boundary proposal decoding, hard non-overlap selection, and calibration."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CropCandidate:
    start_index: int
    end_index: int
    confidence: float
    start_probability: float
    end_probability: float
    inside_probability: float


@dataclass(frozen=True)
class CropperThresholds:
    boundary: float
    inside: float
    proposal: float

    def to_dict(self) -> dict[str, float]:
        return {
            "boundary": float(self.boundary),
            "inside": float(self.inside),
            "proposal": float(self.proposal),
        }

    @classmethod
    def from_dict(cls, payload) -> "CropperThresholds":
        if not isinstance(payload, dict):
            raise ValueError("segment_cropper thresholds must be an object")
        values = tuple(float(payload[name]) for name in ("boundary", "inside", "proposal"))
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("segment_cropper thresholds must be between zero and one")
        return cls(*values)


@dataclass(frozen=True)
class ValidationProbabilities:
    start: np.ndarray
    end: np.ndarray
    inside: np.ndarray
    annotations: tuple[tuple[int, int], ...]


def _peak_indices(
    probabilities: Sequence[float],
    minimum_probability: float = 0.0,
) -> list[int]:
    values = np.asarray(probabilities, dtype=float)
    peaks: list[int] = []
    for index, value in enumerate(values):
        previous = values[index - 1] if index else -math.inf
        following = values[index + 1] if index + 1 < len(values) else -math.inf
        # Pick the first row of a flat maximum so plateau handling is stable.
        if value >= minimum_probability and value > previous and value >= following:
            peaks.append(index)
    return peaks


def form_candidates(
    start_probabilities: Sequence[float],
    end_probabilities: Sequence[float],
    inside_probabilities: Sequence[float],
    minimum_boundary_probability: float = 0.0,
) -> list[CropCandidate]:
    """Pair learned peaks and score every valid half-open proposal."""
    start_values = np.asarray(start_probabilities, dtype=float)
    end_values = np.asarray(end_probabilities, dtype=float)
    inside_values = np.asarray(inside_probabilities, dtype=float)
    if not (len(start_values) == len(end_values) == len(inside_values)):
        raise ValueError("Boundary probability heads must have equal lengths")

    inside_prefix = np.concatenate(([0.0], np.cumsum(inside_values)))
    candidates: list[CropCandidate] = []
    for start in _peak_indices(start_values, minimum_boundary_probability):
        for end_peak in _peak_indices(end_values, minimum_boundary_probability):
            end = end_peak + 1
            if end <= start:
                continue
            inside = float((inside_prefix[end] - inside_prefix[start]) / (end - start))
            start_probability = float(start_values[start])
            end_probability = float(end_values[end_peak])
            confidence = (start_probability + end_probability + inside) / 3.0
            candidates.append(CropCandidate(
                start_index=start,
                end_index=end,
                confidence=confidence,
                start_probability=start_probability,
                end_probability=end_probability,
                inside_probability=inside,
            ))
    return candidates


def filter_candidates(
    candidates: Iterable[CropCandidate],
    thresholds: CropperThresholds,
) -> list[CropCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.start_probability >= thresholds.boundary
        and candidate.end_probability >= thresholds.boundary
        and candidate.inside_probability >= thresholds.inside
        and candidate.confidence >= thresholds.proposal
    ]


@dataclass(frozen=True)
class _ScheduleNode:
    candidate: CropCandidate
    previous: "_ScheduleNode | None"


def _schedule_candidates(node: _ScheduleNode | None) -> tuple[CropCandidate, ...]:
    reversed_candidates: list[CropCandidate] = []
    while node is not None:
        reversed_candidates.append(node.candidate)
        node = node.previous
    return tuple(reversed(reversed_candidates))


def _lexicographically_earlier(
    left: _ScheduleNode | None,
    right: _ScheduleNode | None,
) -> _ScheduleNode | None:
    left_ranges = tuple(
        (item.start_index, item.end_index)
        for item in _schedule_candidates(left)
    )
    right_ranges = tuple(
        (item.start_index, item.end_index)
        for item in _schedule_candidates(right)
    )
    return left if left_ranges <= right_ranges else right


def select_non_overlapping(candidates: Iterable[CropCandidate]) -> list[CropCandidate]:
    """Weighted interval scheduling with deterministic lexicographic ties."""
    ordered = sorted(
        candidates,
        key=lambda item: (item.end_index, item.start_index, -item.confidence),
    )
    if not ordered:
        return []

    end_indices = [candidate.end_index for candidate in ordered]
    predecessors = [
        bisect_right(end_indices, candidate.start_index, hi=index) - 1
        for index, candidate in enumerate(ordered)
    ]
    scores = [0.0]
    schedules: list[_ScheduleNode | None] = [None]
    for index, candidate in enumerate(ordered):
        predecessor_state = predecessors[index] + 1
        included_score = scores[predecessor_state] + candidate.confidence
        excluded_score = scores[index]
        included_schedule = _ScheduleNode(candidate, schedules[predecessor_state])
        excluded_schedule = schedules[index]
        if not math.isclose(
            included_score,
            excluded_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            if included_score > excluded_score:
                scores.append(included_score)
                schedules.append(included_schedule)
            else:
                scores.append(excluded_score)
                schedules.append(excluded_schedule)
        else:
            scores.append(max(included_score, excluded_score))
            schedules.append(_lexicographically_earlier(
                included_schedule,
                excluded_schedule,
            ))

    return sorted(
        _schedule_candidates(schedules[-1]),
        key=lambda item: (item.start_index, item.end_index),
    )


def decode_probabilities(
    start_probabilities: Sequence[float],
    end_probabilities: Sequence[float],
    inside_probabilities: Sequence[float],
    thresholds: CropperThresholds,
) -> list[CropCandidate]:
    candidates = form_candidates(
        start_probabilities,
        end_probabilities,
        inside_probabilities,
        thresholds.boundary,
    )
    selected = select_non_overlapping(filter_candidates(candidates, thresholds))
    if any(left.end_index > right.start_index for left, right in zip(selected, selected[1:])):
        raise RuntimeError("segment_cropper produced overlapping final crops")
    return selected


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


__all__ = [
    "CropCandidate",
    "CropperThresholds",
    "ValidationProbabilities",
    "calibrate_thresholds",
    "decode_probabilities",
    "evaluate_thresholds",
    "filter_candidates",
    "form_candidates",
    "interval_iou",
    "select_non_overlapping",
]
