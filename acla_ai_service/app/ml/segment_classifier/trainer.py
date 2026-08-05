"""Training orchestration for the temporal segment classifier."""

from __future__ import annotations

import copy
import hashlib
import logging
import math
from collections import defaultdict
from typing import Any, Collection, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from app.ml.segment_classifier.model import TemporalDetectionModel
from app.ml.segment_classifier.service import (
    MODEL_FORMAT,
    SegmentClassifierService,
    segment_classifier,
)
from app.storage import get_shared_telemetry_store
from app.storage.datasets.segment_dataset import (
    TemporalStreamingDataset,
    build_temporal_sequences,
    pad_temporal_batch,
)


LOGGER = logging.getLogger(__name__)


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


class SegmentClassifierTrainer:
    def __init__(
        self,
        classifier_service: SegmentClassifierService = segment_classifier,
    ) -> None:
        self.classifier_service = classifier_service
        self.store = get_shared_telemetry_store()
        self.device = classifier_service.device
        self.model: Optional[TemporalDetectionModel] = None
        self.scaler: Optional[StandardScaler] = None
        self.pos_weight: Optional[torch.Tensor] = None

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
        classifier = self.classifier_service
        for chunk in self.store.get_cached_data_chunks(cache_key):
            yield from build_temporal_sequences(
                chunk,
                expected_features=classifier.feature_names,
                label_ids=classifier.label_ids,
                child_parent=classifier.child_parent,
            )

    async def fit_preprocessors(self, cache_key: str) -> None:
        classifier = self.classifier_service
        scaler = StandardScaler()
        positives = np.zeros(len(classifier.label_ids), dtype=np.float64)
        evaluated = np.zeros(len(classifier.label_ids), dtype=np.float64)
        sequence_count = 0

        for sequence in self._iter_sequences(cache_key):
            scaler.partial_fit(sequence.features)
            positives += (sequence.targets * sequence.loss_mask).sum(axis=0)
            evaluated += sequence.loss_mask.sum(axis=0)
            sequence_count += 1

        if sequence_count == 0:
            raise ValueError("No contiguous annotated telemetry sequences found in cache.")
        behavior_count = len(classifier.behavior_label_ids)
        if positives[:behavior_count].sum() == 0:
            raise ValueError("No behavior annotations found in classifier training data.")
        if positives[behavior_count:].sum() == 0:
            LOGGER.warning(
                "No behavior sub-label annotations found in classifier training data; "
                "training will continue with parent behavior labels only, and child-label "
                "predictions from this model will not be reliable."
            )

        self.scaler = scaler
        negatives = evaluated - positives
        self.pos_weight = torch.tensor(
            np.minimum(
                np.divide(
                    negatives,
                    positives,
                    out=np.ones_like(negatives),
                    where=positives > 0,
                ),
                20.0,
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def _dataset(self, cache_key: str) -> TemporalStreamingDataset:
        classifier = self.classifier_service
        return TemporalStreamingDataset(
            self.store,
            cache_key,
            self.scaler,
            classifier.feature_names,
            classifier.label_ids,
            classifier.child_parent,
        )

    @staticmethod
    def _masked_loss(logits, targets, mask, pos_weight):
        raw_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        denominator = mask.sum()
        if denominator.item() == 0:
            return None
        return (raw_loss * mask).sum() / denominator

    @staticmethod
    def _masked_class_accuracy_counts(
        logits,
        targets,
        mask,
        pos_weight,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        evaluated = mask > 0
        corrected_logits = logits - torch.log(pos_weight)
        predicted_positive = corrected_logits >= 0
        expected_positive = targets >= 0.5
        positives = expected_positive & evaluated
        negatives = ~expected_positive & evaluated
        return (
            (
                int((predicted_positive & positives).sum().item()),
                int(positives.sum().item()),
            ),
            (
                int((~predicted_positive & negatives).sum().item()),
                int(negatives.sum().item()),
            ),
        )

    @staticmethod
    def _accuracy_percentage(correct: int, evaluated: int) -> str:
        return "N/A" if evaluated == 0 else f"{correct / evaluated:.2%}"

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
        if self.pos_weight is None:
            raise RuntimeError("Classifier positive class weights were not fitted.")

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
        classifier = self.classifier_service
        input_dim = int(self.scaler.mean_.shape[0])
        self.model = TemporalDetectionModel(
            input_dim=input_dim,
            output_dim=len(classifier.label_ids),
            hidden_dim=classifier.hidden_dim,
            dilations=classifier.dilations,
            dropout=classifier.dropout,
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
        best_accuracy_counts = ((0, 0), (0, 0))
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
                    self.pos_weight,
                )
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            self.model.eval()
            val_losses = []
            val_sequence_count = 0
            val_positive_correct = 0
            val_positive_evaluated = 0
            val_negative_correct = 0
            val_negative_evaluated = 0
            with torch.no_grad():
                for features, targets, mask in val_loader:
                    logits = self.model(features.to(self.device))
                    device_targets = targets.to(self.device)
                    device_mask = mask.to(self.device)
                    loss = self._masked_loss(
                        logits,
                        device_targets,
                        device_mask,
                        self.pos_weight,
                    )
                    if loss is not None:
                        val_losses.append(float(loss.item()))
                        val_sequence_count += int(features.shape[0])
                        positive_counts, negative_counts = self._masked_class_accuracy_counts(
                            logits,
                            device_targets,
                            device_mask,
                            self.pos_weight,
                        )
                        val_positive_correct += positive_counts[0]
                        val_positive_evaluated += positive_counts[1]
                        val_negative_correct += negative_counts[0]
                        val_negative_evaluated += negative_counts[1]

            if not train_losses:
                raise ValueError("No valid temporal training samples were produced.")
            if val_split > 0 and not val_losses:
                raise ValueError("No valid temporal validation samples were produced by Val split.")

            train_loss = float(np.mean(train_losses))
            if val_losses:
                monitored_loss = float(np.mean(val_losses))
                val_correct = val_positive_correct + val_negative_correct
                val_evaluated = val_positive_evaluated + val_negative_evaluated
                val_accuracy = self._accuracy_percentage(val_correct, val_evaluated)
                positive_accuracy = self._accuracy_percentage(
                    val_positive_correct,
                    val_positive_evaluated,
                )
                negative_accuracy = self._accuracy_percentage(
                    val_negative_correct,
                    val_negative_evaluated,
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {monitored_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy} "
                    f"({val_correct}/{val_evaluated} labeled predictions), "
                    f"Positive Accuracy: {positive_accuracy} "
                    f"({val_positive_correct}/{val_positive_evaluated}), "
                    f"Negative Accuracy: {negative_accuracy} "
                    f"({val_negative_correct}/{val_negative_evaluated}), "
                    f"Val Samples: {val_sequence_count}"
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
                best_accuracy_counts = (
                    (val_positive_correct, val_positive_evaluated),
                    (val_negative_correct, val_negative_evaluated),
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if best_epoch is not None and val_split > 0:
            best_positive_counts, best_negative_counts = best_accuracy_counts
            best_correct = best_positive_counts[0] + best_negative_counts[0]
            best_evaluated = best_positive_counts[1] + best_negative_counts[1]
            best_accuracy = self._accuracy_percentage(best_correct, best_evaluated)
            best_positive_accuracy = self._accuracy_percentage(*best_positive_counts)
            best_negative_accuracy = self._accuracy_percentage(*best_negative_counts)
            print(
                f"[INFO] Best validation result: epoch={best_epoch} "
                f"loss={best_loss:.4f} "
                f"accuracy={best_accuracy} "
                f"({best_correct}/{best_evaluated} labeled predictions) "
                f"positive_accuracy={best_positive_accuracy} "
                f"({best_positive_counts[0]}/{best_positive_counts[1]}) "
                f"negative_accuracy={best_negative_accuracy} "
                f"({best_negative_counts[0]}/{best_negative_counts[1]})"
            )
        self.model.eval()

        classifier.model = self.model
        classifier.scaler = self.scaler
        classifier.label_weights = {
            label_id: float(weight)
            for label_id, weight in zip(
                classifier.label_ids,
                self.pos_weight.detach().cpu().tolist(),
            )
        }
        classifier._save_artifacts()

        try:
            from app.integrations.backend.client import backend_service

            await backend_service.save_ai_model(
                model_type="segment_classifier",
                model_data=classifier.serialize_artifacts(),
                metadata={
                    "format": MODEL_FORMAT,
                    "num_labels": len(classifier.label_ids),
                    "feature_count": input_dim,
                },
                is_active=True,
            )
        except Exception as exc:
            LOGGER.warning("segment_classifier backend upload failed: %s", exc)


segment_classifier_trainer = SegmentClassifierTrainer()


__all__ = [
    "SegmentClassifierTrainer",
    "segment_classifier_trainer",
]
