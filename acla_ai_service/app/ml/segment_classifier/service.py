"""Artifact handling and inference for temporal behavior detection."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from app.ml.segment_classifier.model import TemporalDetectionModel
from app.shared.labels import BEHAVIOR_LABELS, LABEL_CATEGORIES, LABEL_MAPPING
from app.shared.segment import PredictedSegment
from app.shared.segment_classifier_features import SEGMENT_CLASSIFIER_FEATURES
from app.storage.datasets.segment_dataset import compute_derived_features


LOGGER = logging.getLogger(__name__)
MODEL_FORMAT = "segment_classifier/temporal-v2"
DEFAULT_THRESHOLD = 0.5
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DILATIONS = (1, 2, 4, 8)
DEFAULT_DROPOUT = 0.2


def _behavior_and_child_labels() -> tuple[list[str], dict[str, str]]:
    behaviors = [label_id for label_id in BEHAVIOR_LABELS if label_id in LABEL_MAPPING]
    child_parent: dict[str, str] = {}
    for behavior_id in behaviors:
        for child_id in LABEL_CATEGORIES.get(behavior_id, []):
            if child_id in LABEL_MAPPING:
                child_parent[child_id] = behavior_id
    return behaviors, child_parent


class SegmentClassifierService:
    def __init__(self, models_directory: str = "models") -> None:
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_directory / "segment_classifier.pth"
        self.scaler_path = self.models_directory / "segment_scaler.joblib"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TemporalDetectionModel] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_weights: dict[str, float] = {}
        self.feature_names = list(SEGMENT_CLASSIFIER_FEATURES)
        behavior_label_ids, child_parent = _behavior_and_child_labels()
        self.configure_labels([*behavior_label_ids, *child_parent])
        self.threshold = DEFAULT_THRESHOLD
        self.hidden_dim = DEFAULT_HIDDEN_DIM
        self.dilations = DEFAULT_DILATIONS
        self.dropout = DEFAULT_DROPOUT

    def configure_labels(self, label_ids: Sequence[str]) -> None:
        self.label_ids = list(dict.fromkeys(str(label_id) for label_id in label_ids))
        known_behavior_ids, known_child_parent = _behavior_and_child_labels()
        active_labels = set(self.label_ids)
        self.behavior_label_ids = [
            label_id for label_id in known_behavior_ids if label_id in active_labels
        ]
        self.child_parent = {
            child_id: parent_id
            for child_id, parent_id in known_child_parent.items()
            if child_id in active_labels
        }

    def _save_artifacts(self) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "label_weights": self.label_weights,
            },
            self.model_path,
        )
        joblib.dump(self.scaler, self.scaler_path)

    def load_model(self) -> bool:
        if not self.has_local_artifacts():
            return False
        self.scaler = joblib.load(self.scaler_path)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.label_weights = checkpoint["label_weights"]
        self.configure_labels(self.label_weights)
        self.model = TemporalDetectionModel(
            input_dim=int(self.scaler.mean_.shape[0]),
            output_dim=len(self.label_ids),
            hidden_dim=self.hidden_dim,
            dilations=self.dilations,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        return True

    _ARTIFACT_FILES = (
        "segment_classifier.pth",
        "segment_scaler.joblib",
    )

    def has_local_artifacts(self) -> bool:
        return all(
            (self.models_directory / name).is_file()
            for name in self._ARTIFACT_FILES
        )

    def serialize_artifacts(self) -> Dict[str, Any]:
        if not self.has_local_artifacts():
            raise FileNotFoundError("Temporal segment classifier artifacts are incomplete.")
        return {
            "format": MODEL_FORMAT,
            "files": {
                name: base64.b64encode((self.models_directory / name).read_bytes()).decode("ascii")
                for name in self._ARTIFACT_FILES
            },
        }

    def deserialize_artifacts(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"segment_classifier payload must use {MODEL_FORMAT}")
        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("segment_classifier payload missing files")
        missing = [name for name in self._ARTIFACT_FILES if name not in files]
        if missing:
            raise ValueError(f"segment_classifier payload missing: {', '.join(missing)}")
        for name in self._ARTIFACT_FILES:
            (self.models_directory / name).write_bytes(base64.b64decode(files[name]))

    def _prepare_numeric_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in dataframe.columns]
        if missing:
            LOGGER.warning(
                "segment_classifier input missing %d/%d features; filling with zero",
                len(missing),
                len(self.feature_names),
            )
        frame = dataframe.reindex(columns=self.feature_names, fill_value=0)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
        return compute_derived_features(frame) if not frame.empty else frame

    def score_sequence(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return per-timestep probabilities for the complete input sequence."""
        if self.model is None and not self.load_model():
            raise ValueError("Temporal segment classifier model not trained or found.")
        numeric = self._prepare_numeric_features(dataframe.reset_index(drop=True))
        if numeric.empty:
            return pd.DataFrame(columns=self.label_ids, dtype=float)
        scaled = self.scaler.transform(numeric.to_numpy())
        inputs = torch.tensor(scaled, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)
            pos_weight = logits.new_tensor([
                self.label_weights[label_id]
                for label_id in self.label_ids
            ])
            corrected_logits = logits - torch.log(pos_weight)
            scores = torch.sigmoid(corrected_logits)[0].cpu().numpy()
        return pd.DataFrame(scores, columns=self.label_ids)

    @staticmethod
    def _merge_score_runs(scores: Sequence[float], threshold: float):
        sequence_length = len(scores)
        core_start = None
        core_values: list[float] = []
        merged_start = None
        merged_end = None
        merged_values: list[float] = []

        for index in range(sequence_length + 1):
            score = float(scores[index]) if index < sequence_length else None
            if score is not None and score >= threshold:
                if core_start is None:
                    core_start = index
                core_values.append(score)
                continue
            if core_start is None:
                continue

            expanded_start = max(0, core_start - 1)
            expanded_end = min(sequence_length, index + 1)
            if merged_start is not None and expanded_start < merged_end:
                merged_end = max(merged_end, expanded_end)
                merged_values.extend(core_values)
            else:
                if merged_start is not None:
                    yield merged_start, merged_end, float(np.mean(merged_values))
                merged_start = expanded_start
                merged_end = expanded_end
                merged_values = list(core_values)

            core_start = None
            core_values = []

        if merged_start is not None:
            yield merged_start, merged_end, float(np.mean(merged_values))

    def detect_segments(
        self,
        dataframe: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> list[PredictedSegment]:
        """Detect every trained label, nesting known children when possible."""
        source = dataframe.reset_index(drop=True)
        if source.empty:
            return []
        active_threshold = self.threshold if threshold is None else float(threshold)
        sequence_scores = self.score_sequence(source)
        detections_by_label: dict[str, list[PredictedSegment]] = {}

        for label_id in self.label_ids:
            if label_id not in sequence_scores:
                continue
            detections_by_label[label_id] = [
                PredictedSegment(
                    label=label_id,
                    score=score,
                    start_index=start,
                    end_index=end,
                    telemetry_data=source.iloc[start:end].to_dict("records"),
                )
                for start, end, score in self._merge_score_runs(
                    sequence_scores[label_id].to_numpy(),
                    active_threshold,
                )
            ]

        children_by_parent: dict[str, list[str]] = {
            behavior_id: [] for behavior_id in self.behavior_label_ids
        }
        for child_id, parent_id in self.child_parent.items():
            children_by_parent.setdefault(parent_id, []).append(child_id)

        nested_child_ranges: dict[str, list[tuple[int, int]]] = {}
        for behavior_id in self.behavior_label_ids:
            for detection in detections_by_label.get(behavior_id, []):
                start = int(detection.start_index or 0)
                end = int(detection.end_index or start)
                crop = source.iloc[start:end].reset_index(drop=True)
                child_ids = children_by_parent.get(behavior_id, [])
                if child_ids:
                    child_scores = self.score_sequence(crop)
                    for child_id in child_ids:
                        if child_id not in child_scores:
                            continue
                        for child_start, child_end, child_score in self._merge_score_runs(
                            child_scores[child_id].to_numpy(),
                            active_threshold,
                        ):
                            global_start = start + child_start
                            global_end = start + child_end
                            detection.subsegments.append(PredictedSegment(
                                label=child_id,
                                score=child_score,
                                start_index=global_start,
                                end_index=global_end,
                                telemetry_data=source.iloc[global_start:global_end].to_dict("records"),
                            ))

                detection.subsegments.sort(
                    key=lambda item: (item.start_index, item.end_index, item.label)
                )
                for child in detection.subsegments:
                    nested_child_ranges.setdefault(child.label, []).append((
                        int(child.start_index or 0),
                        int(child.end_index or child.start_index or 0),
                    ))

        detections: list[PredictedSegment] = []
        for label_id in self.label_ids:
            for detection in detections_by_label.get(label_id, []):
                start = int(detection.start_index or 0)
                end = int(detection.end_index or start)
                represented_as_child = any(
                    start < nested_end and nested_start < end
                    for nested_start, nested_end in nested_child_ranges.get(label_id, [])
                )
                if not represented_as_child:
                    detections.append(detection)

        detections.sort(key=lambda item: (item.start_index, item.end_index, item.label))
        return detections

    def classify_ranges(
        self,
        dataframe: pd.DataFrame,
        ranges: Sequence[Dict[str, Any]],
    ) -> list[PredictedSegment]:
        """Classify each valid splitter range and retain every detection."""
        source = dataframe.reset_index(drop=True)
        accepted: list[PredictedSegment] = []

        for range_data in ranges:
            try:
                range_start = int(range_data["start_index"])
                range_end = int(range_data["end_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if range_start < 0 or range_end <= range_start or range_end > len(source):
                continue

            range_frame = source.iloc[range_start:range_end].reset_index(drop=True)
            detections = self.detect_segments(range_frame)
            for detection in detections:
                detection_start = range_start + int(detection.start_index or 0)
                detection_end = range_start + int(
                    detection.end_index
                    if detection.end_index is not None
                    else detection.start_index or 0
                )
                if (
                    detection_start < range_start
                    or detection_end <= detection_start
                    or detection_end > range_end
                ):
                    continue

                remapped_children: list[PredictedSegment] = []
                for child in detection.subsegments:
                    child_start = range_start + int(child.start_index or 0)
                    child_end = range_start + int(
                        child.end_index
                        if child.end_index is not None
                        else child.start_index or 0
                    )
                    if (
                        child_start < range_start
                        or child_end <= child_start
                        or child_end > range_end
                    ):
                        continue
                    remapped_children.append(PredictedSegment(
                        id=child.id,
                        label=child.label,
                        score=child.score,
                        start_index=child_start,
                        end_index=child_end,
                        telemetry_data=source.iloc[child_start:child_end].to_dict("records"),
                    ))

                accepted.append(PredictedSegment(
                    id=detection.id,
                    label=detection.label,
                    score=detection.score,
                    start_index=detection_start,
                    end_index=detection_end,
                    telemetry_data=source.iloc[detection_start:detection_end].to_dict("records"),
                    subsegments=remapped_children,
                ))

        return accepted


segment_classifier = SegmentClassifierService()


__all__ = [
    "DEFAULT_THRESHOLD",
    "MODEL_FORMAT",
    "SegmentClassifierService",
    "segment_classifier",
]
