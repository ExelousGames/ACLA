"""Training, artifact handling, and inference for temporal behavior detection."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from app.ml.segment_classifier.model import TemporalDetectionModel
from app.shared.labels import BEHAVIOR_LABELS, LABEL_CATEGORIES, LABEL_MAPPING
from app.shared.segment import PredictedSegment
from app.shared.segment_classifier_features import SEGMENT_CLASSIFIER_FEATURES
from app.storage import get_shared_telemetry_store
from app.storage.datasets.segment_dataset import (
    TemporalStreamingDataset,
    build_temporal_sequences,
    compute_derived_features,
    pad_temporal_batch,
)


LOGGER = logging.getLogger(__name__)
MODEL_FORMAT = "segment_classifier/temporal-v1"
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


def _chunk_records(chunk: Any) -> list[dict]:
    if isinstance(chunk, list):
        return [record for record in chunk if isinstance(record, dict)]
    if isinstance(chunk, dict) and isinstance(chunk.get("data"), list):
        return [record for record in chunk["data"] if isinstance(record, dict)]
    if isinstance(chunk, dict) and isinstance(chunk.get("payload"), dict):
        return [chunk["payload"]]
    return [chunk] if isinstance(chunk, dict) else []


def _annotation_samples(records: Sequence[dict]) -> list[tuple[str, list[dict]]]:
    """Keep each behavior annotation and its child annotations as one sample."""
    children_by_parent: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        parent_id = record.get("parent_id")
        if parent_id:
            children_by_parent[str(parent_id)].append(record)

    samples = []
    for index, record in enumerate(records):
        if record.get("parent_id"):
            continue
        if (
            record.get("start_index") is None
            or record.get("end_index") is None
            or not record.get("telemetry_data")
        ):
            continue
        fallback_id = f"{record['start_index']}:{record['end_index']}:{index}"
        sample_id = str(record.get("id") or fallback_id)
        samples.append((sample_id, [record, *children_by_parent.get(sample_id, [])]))
    return samples


class SegmentClassifierService:
    def __init__(self, models_directory: str = "models") -> None:
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_directory / "segment_classifier.pth"
        self.scaler_path = self.models_directory / "segment_scaler.joblib"
        self.config_path = self.models_directory / "segment_config.json"
        self.store = get_shared_telemetry_store()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TemporalDetectionModel] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = list(SEGMENT_CLASSIFIER_FEATURES)
        self.behavior_label_ids, self.child_parent = _behavior_and_child_labels()
        self.label_ids = [*self.behavior_label_ids, *self.child_parent]
        self.threshold = DEFAULT_THRESHOLD
        self.hidden_dim = DEFAULT_HIDDEN_DIM
        self.dilations = DEFAULT_DILATIONS
        self.dropout = DEFAULT_DROPOUT

    def _configure_training_backend(self) -> None:
        if self.device.type != "cuda" or not getattr(torch.version, "hip", None):
            return
        torch.backends.miopen.immediate = True
        print("[INFO] Enabled MIOpen Immediate Mode for ROCm classifier training.")

    @staticmethod
    def _split_order(session_id: str, sample_id: str) -> str:
        return hashlib.sha256(f"{session_id}\0{sample_id}".encode("utf-8")).hexdigest()

    async def prepare_training_data(
        self,
        source_cache_key: str,
        train_cache_key: str,
        val_cache_key: str,
        val_split: float = 0.2,
        session_ids: Optional[Collection[str]] = None,
    ) -> None:
        """Split annotated behavior samples while keeping child ranges attached."""
        if not 0 <= val_split < 1:
            raise ValueError(
                "Validation split must be greater than or equal to 0 and less than 1."
            )
        selected = None if session_ids is None else {str(value) for value in session_ids}
        if selected is not None and not selected:
            raise ValueError("At least one session must be selected for classifier training.")

        sessions_found: set[str] = set()
        samples: list[tuple[str, str, list[dict]]] = []
        for chunk, raw_session_id in self.store.get_cached_data_chunks(
            source_cache_key,
            include_ids=True,
        ):
            session_id = str(raw_session_id)
            if selected is not None and session_id not in selected:
                continue
            records = _chunk_records(chunk)
            if records:
                sessions_found.add(session_id)
                for sample_id, sample_records in _annotation_samples(records):
                    samples.append((session_id, sample_id, sample_records))

        if selected is not None and not sessions_found:
            names = ", ".join(sorted(selected))
            raise ValueError(f"No annotation sessions found for selection: {names}")
        if not samples:
            raise ValueError("No annotated behavior samples found for classifier training.")
        if val_split > 0 and len(samples) < 2:
            raise ValueError(
                "Validation requires at least two annotated behavior samples when "
                "Val split is greater than 0."
            )

        ordered_samples = sorted(
            samples,
            key=lambda item: self._split_order(item[0], item[1]),
        )
        validation_count = 0
        if val_split > 0:
            validation_count = min(len(samples) - 1, math.ceil(len(samples) * val_split))
        validation_samples = ordered_samples[:validation_count]
        training_samples = ordered_samples[validation_count:]

        training_records: dict[str, list[dict]] = defaultdict(list)
        validation_records: dict[str, list[dict]] = defaultdict(list)
        for session_id, _, records in training_samples:
            training_records[session_id].extend(records)
        for session_id, _, records in validation_samples:
            validation_records[session_id].extend(records)

        self.store.clear_cache(train_cache_key)
        self.store.clear_cache(val_cache_key)
        for session_id, records in training_records.items():
            self.store.save_chunk(train_cache_key, session_id, records)
        for session_id, records in validation_records.items():
            self.store.save_chunk(val_cache_key, session_id, records)
        print(
            f"[INFO] Classifier sample split: train={len(training_samples)} "
            f"validation={len(validation_samples)} requested_val_split={val_split:.1%}"
        )

    def _iter_sequences(self, cache_key: str):
        for chunk in self.store.get_cached_data_chunks(cache_key):
            yield from build_temporal_sequences(
                chunk,
                expected_features=self.feature_names,
                label_ids=self.label_ids,
                child_parent=self.child_parent,
            )

    async def fit_preprocessors(self, cache_key: str) -> None:
        scaler = StandardScaler()
        positives = np.zeros(len(self.label_ids), dtype=np.float64)
        sequence_count = 0

        for sequence in self._iter_sequences(cache_key):
            scaler.partial_fit(sequence.features)
            positives += (sequence.targets * sequence.loss_mask).sum(axis=0)
            sequence_count += 1

        if sequence_count == 0:
            raise ValueError("No contiguous annotated telemetry sequences found in cache.")
        behavior_count = len(self.behavior_label_ids)
        if positives[:behavior_count].sum() == 0:
            raise ValueError("No behavior annotations found in classifier training data.")
        if positives[behavior_count:].sum() == 0:
            LOGGER.warning(
                "No behavior sub-label annotations found in classifier training data; "
                "training will continue with parent behavior labels only, and child-label "
                "predictions from this model will not be reliable."
            )

        self.scaler = scaler

    def _dataset(self, cache_key: str) -> TemporalStreamingDataset:
        return TemporalStreamingDataset(
            self.store,
            cache_key,
            self.scaler,
            self.feature_names,
            self.label_ids,
            self.child_parent,
        )

    @staticmethod
    def _masked_loss(logits, targets, mask):
        raw_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        denominator = mask.sum()
        if denominator.item() == 0:
            return None
        return (raw_loss * mask).sum() / denominator

    async def train_model(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        val_split: float = 0.1,
        annotation_cache_key: Optional[str] = None,
        session_ids: Optional[Collection[str]] = None,
    ) -> None:
        self._configure_training_backend()

        from app.pipelines.training.config import TrainingPipelineConfig

        source_key = annotation_cache_key or TrainingPipelineConfig().annotation_cache_key
        train_key = f"{source_key}_train"
        val_key = f"{source_key}_val"
        await self.prepare_training_data(
            source_key,
            train_key,
            val_key,
            val_split,
            session_ids=session_ids,
        )
        await self.fit_preprocessors(train_key)

        train_loader = DataLoader(
            self._dataset(train_key),
            batch_size=batch_size,
            collate_fn=pad_temporal_batch,
            num_workers=0,
        )
        val_loader = DataLoader(
            self._dataset(val_key),
            batch_size=batch_size,
            collate_fn=pad_temporal_batch,
            num_workers=0,
        )
        input_dim = int(self.scaler.mean_.shape[0])
        self.model = TemporalDetectionModel(
            input_dim=input_dim,
            output_dim=len(self.label_ids),
            hidden_dim=self.hidden_dim,
            dilations=self.dilations,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        best_loss = float("inf")
        best_state = None
        best_epoch = None
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for features, targets, mask in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                optimizer.zero_grad()
                loss = self._masked_loss(
                    self.model(features),
                    targets,
                    mask,
                )
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            self.model.eval()
            val_losses = []
            val_sequence_count = 0
            with torch.no_grad():
                for features, targets, mask in val_loader:
                    logits = self.model(features.to(self.device))
                    device_targets = targets.to(self.device)
                    device_mask = mask.to(self.device)
                    loss = self._masked_loss(
                        logits,
                        device_targets,
                        device_mask,
                    )
                    if loss is not None:
                        val_losses.append(float(loss.item()))
                        val_sequence_count += int(features.shape[0])

            if not train_losses:
                raise ValueError("No valid temporal training samples were produced.")
            if val_split > 0 and not val_losses:
                raise ValueError("No valid temporal validation samples were produced by Val split.")

            train_loss = float(np.mean(train_losses))
            if val_losses:
                monitored_loss = float(np.mean(val_losses))
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {monitored_loss:.4f}, Val Samples: {val_sequence_count}"
                )
            else:
                monitored_loss = train_loss
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    "Validation: disabled (Val split is 0)"
                )
            scheduler.step(monitored_loss)
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if best_epoch is not None and val_split > 0:
            print(
                f"[INFO] Best validation result: epoch={best_epoch} loss={best_loss:.4f}"
            )
        self.model.eval()
        self._save_artifacts()

        try:
            from app.integrations.backend.client import backend_service

            await backend_service.save_ai_model(
                model_type="segment_classifier",
                model_data=self.serialize_artifacts(),
                metadata={
                    "format": MODEL_FORMAT,
                    "num_labels": len(self.label_ids),
                    "feature_count": input_dim,
                },
                is_active=True,
            )
        except Exception as exc:
            LOGGER.warning("segment_classifier backend upload failed: %s", exc)

    def _config(self) -> dict:
        return {
            "format": MODEL_FORMAT,
            "feature_names": self.feature_names,
            "label_ids": self.label_ids,
            "hidden_dim": self.hidden_dim,
            "dilations": list(self.dilations),
            "dropout": self.dropout,
            "threshold": self.threshold,
        }

    def _save_artifacts(self) -> None:
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(self._config(), handle)

    def load_model(self) -> bool:
        if not self.has_local_artifacts():
            return False
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("format") != MODEL_FORMAT:
            return False

        self.feature_names = list(config["feature_names"])
        self.label_ids = list(config["label_ids"])
        self.behavior_label_ids = [
            label_id for label_id in BEHAVIOR_LABELS if label_id in self.label_ids
        ]
        _, catalog_children = _behavior_and_child_labels()
        self.child_parent = {
            child_id: parent_id
            for child_id, parent_id in catalog_children.items()
            if child_id in self.label_ids and parent_id in self.label_ids
        }
        self.hidden_dim = int(config["hidden_dim"])
        self.dilations = tuple(int(value) for value in config["dilations"])
        self.dropout = float(config["dropout"])
        self.threshold = float(config.get("threshold", DEFAULT_THRESHOLD))
        self.scaler = joblib.load(self.scaler_path)
        self.model = TemporalDetectionModel(
            input_dim=int(self.scaler.mean_.shape[0]),
            output_dim=len(self.label_ids),
            hidden_dim=self.hidden_dim,
            dilations=self.dilations,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        return True

    _ARTIFACT_FILES = (
        "segment_classifier.pth",
        "segment_scaler.joblib",
        "segment_config.json",
    )

    def has_local_artifacts(self) -> bool:
        if not all((self.models_directory / name).is_file() for name in self._ARTIFACT_FILES):
            return False
        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle).get("format") == MODEL_FORMAT
        except (OSError, ValueError, TypeError):
            return False

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
            scores = torch.sigmoid(self.model(inputs))[0].cpu().numpy()
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
        """Detect behavior ranges, then rerun each crop for its sub-labels."""
        source = dataframe.reset_index(drop=True)
        if source.empty:
            return []
        active_threshold = self.threshold if threshold is None else float(threshold)
        sequence_scores = self.score_sequence(source)
        detections: list[PredictedSegment] = []

        children_by_parent: dict[str, list[str]] = {
            behavior_id: [] for behavior_id in self.behavior_label_ids
        }
        for child_id, parent_id in self.child_parent.items():
            children_by_parent.setdefault(parent_id, []).append(child_id)

        for behavior_id in self.behavior_label_ids:
            if behavior_id not in sequence_scores:
                continue
            for start, end, score in self._merge_score_runs(
                sequence_scores[behavior_id].to_numpy(),
                active_threshold,
            ):
                crop = source.iloc[start:end].reset_index(drop=True)
                child_detections: list[PredictedSegment] = []
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
                            child_detections.append(PredictedSegment(
                                label=child_id,
                                score=child_score,
                                start_index=global_start,
                                end_index=global_end,
                                telemetry_data=source.iloc[global_start:global_end].to_dict("records"),
                            ))
                    child_detections.sort(
                        key=lambda item: (item.start_index, item.end_index, item.label)
                    )

                detections.append(PredictedSegment(
                    label=behavior_id,
                    score=score,
                    start_index=start,
                    end_index=end,
                    telemetry_data=source.iloc[start:end].to_dict("records"),
                    subsegments=child_detections,
                ))

        detections.sort(key=lambda item: (item.start_index, item.end_index, item.label))
        return detections


segment_classifier = SegmentClassifierService()


__all__ = [
    "DEFAULT_THRESHOLD",
    "MODEL_FORMAT",
    "SegmentClassifierService",
    "segment_classifier",
]
