"""Complete-session training orchestration for the learned boundary cropper."""

from __future__ import annotations

import copy
import hashlib
import logging
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from app.ml.segment_cropper.data import (
    CropperSequence,
    CropperSession,
    build_sequence,
    pad_cropper_batch,
    parse_session_chunk,
)
from app.ml.segment_cropper.decoding import (
    ValidationProbabilities,
    calibrate_thresholds,
)
from app.ml.segment_cropper.model import BoundaryTCN
from app.ml.segment_cropper.service import (
    MODEL_FORMAT,
    SegmentCropperService,
    segment_cropper,
)
from app.storage import get_shared_telemetry_store


LOGGER = logging.getLogger(__name__)


class _SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[CropperSequence], scaler: StandardScaler) -> None:
        self.sequences = list(sequences)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        sequence = self.sequences[index]
        return (
            torch.tensor(self.scaler.transform(sequence.features), dtype=torch.float32),
            torch.tensor(sequence.targets, dtype=torch.float32),
        )


class SegmentCropperTrainer:
    def __init__(
        self,
        cropper_service: SegmentCropperService = segment_cropper,
        store=None,
    ) -> None:
        self.cropper_service = cropper_service
        self.store = store or get_shared_telemetry_store()
        self.device = cropper_service.device
        self.model: Optional[BoundaryTCN] = None
        self.scaler: Optional[StandardScaler] = None
        self.class_weights: Optional[torch.Tensor] = None

    @staticmethod
    def _split_order(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def load_sessions(self, source_cache_key: str) -> list[CropperSession]:
        """Read only the directly configured dataset; no source-copy lookup."""
        sessions = [
            parse_session_chunk(chunk, session_id)
            for chunk, session_id in self.store.get_cached_data_chunks(
                source_cache_key,
                include_ids=True,
            )
        ]
        if not sessions:
            raise ValueError("No complete segment_cropper sessions found in input dataset")
        ids = [session.session_id for session in sessions]
        if len(set(ids)) != len(ids):
            raise ValueError("segment_cropper input must contain one complete chunk per session")
        return sessions

    def split_sessions(
        self,
        sessions: Sequence[CropperSession],
    ) -> tuple[list[CropperSession], list[CropperSession]]:
        if len(sessions) < 2:
            raise ValueError("segment_cropper requires at least two complete sessions")
        ordered = sorted(sessions, key=lambda item: self._split_order(item.session_id))
        validation_count = max(1, int(math.floor(len(ordered) * 0.1 + 0.5)))
        validation_count = min(len(ordered) - 1, validation_count)
        validation = ordered[:validation_count]
        training = ordered[validation_count:]
        if not training or not validation:
            raise ValueError("segment_cropper requires at least one session in each split")
        return training, validation

    def fit_preprocessors(self, training: Sequence[CropperSequence]) -> None:
        scaler = StandardScaler()
        positives = np.zeros(3, dtype=np.float64)
        evaluated = np.zeros(3, dtype=np.float64)
        for sequence in training:
            scaler.partial_fit(sequence.features)
            positives += sequence.targets.sum(axis=0)
            evaluated += len(sequence.targets)
        if not training or positives.sum() == 0:
            raise ValueError("segment_cropper training split has no valid parent annotations")
        negatives = evaluated - positives
        weights = np.divide(
            negatives,
            positives,
            out=np.ones_like(negatives),
            where=positives > 0,
        )
        self.scaler = scaler
        self.class_weights = torch.tensor(
            np.minimum(weights, 100.0),
            dtype=torch.float32,
            device=self.device,
        )

    @staticmethod
    def masked_focal_binary_loss(
        logits,
        targets,
        mask,
        pos_weight,
        gamma: float = 2.0,
    ):
        raw = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        target_probability = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        weighted = raw * (1.0 - target_probability).pow(gamma) * mask
        denominator = mask.sum()
        return None if denominator.item() == 0 else weighted.sum() / denominator

    @staticmethod
    def masked_weighted_binary_loss(logits, targets, mask, pos_weight):
        raw = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        denominator = mask.sum()
        return None if denominator.item() == 0 else (raw * mask).sum() / denominator

    def _loss(self, logits, targets, mask):
        if self.class_weights is None:
            raise RuntimeError("segment_cropper class weights are not fitted")
        losses = [
            self.masked_focal_binary_loss(
                logits[0], targets[:, :, 0], mask, self.class_weights[0],
            ),
            self.masked_focal_binary_loss(
                logits[1], targets[:, :, 1], mask, self.class_weights[1],
            ),
            self.masked_weighted_binary_loss(
                logits[2], targets[:, :, 2], mask, self.class_weights[2],
            ),
        ]
        valid = [loss for loss in losses if loss is not None]
        return None if not valid else sum(valid) / len(valid)

    def _validation_probabilities(
        self,
        sessions: Sequence[CropperSession],
    ) -> list[ValidationProbabilities]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("segment_cropper model and scaler must be fitted before calibration")
        validation: list[ValidationProbabilities] = []
        self.model.eval()
        with torch.no_grad():
            for session in sessions:
                sequence = build_sequence(session, self.cropper_service.feature_names)
                scaled = self.scaler.transform(sequence.features)
                features = torch.tensor(
                    scaled,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                heads = self.model(features)
                probabilities = [torch.sigmoid(head)[0].cpu().numpy() for head in heads]
                validation.append(ValidationProbabilities(
                    start=probabilities[0],
                    end=probabilities[1],
                    inside=probabilities[2],
                    annotations=tuple(session.annotations),
                ))
        return validation

    async def train_model(
        self,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        annotation_cache_key: Optional[str] = None,
    ) -> None:
        from app.pipelines.training.config import TrainingPipelineConfig

        source_key = annotation_cache_key or TrainingPipelineConfig().annotation_cache_key
        sessions = self.load_sessions(source_key)
        training_sessions, validation_sessions = self.split_sessions(sessions)
        training_sequences = [
            build_sequence(session, self.cropper_service.feature_names)
            for session in training_sessions
        ]
        validation_sequences = [
            build_sequence(session, self.cropper_service.feature_names)
            for session in validation_sessions
        ]
        if not any(session.annotations for session in validation_sessions):
            raise ValueError("segment_cropper validation split has no valid parent annotations")
        self.fit_preprocessors(training_sequences)

        train_loader = DataLoader(
            _SequenceDataset(training_sequences, self.scaler),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_cropper_batch,
        )
        validation_loader = DataLoader(
            _SequenceDataset(validation_sequences, self.scaler),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_cropper_batch,
        )
        input_dim = len(self.cropper_service.derived_feature_names)
        self.model = BoundaryTCN(
            input_dim=input_dim,
            hidden_dim=self.cropper_service.hidden_dim,
            dilations=self.cropper_service.dilations,
            dropout=self.cropper_service.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        best_loss = float("inf")
        best_state = None
        for epoch in range(int(epochs)):
            self.model.train()
            training_losses: list[float] = []
            for features, targets, mask in train_loader:
                optimizer.zero_grad()
                loss = self._loss(
                    self.model(features.to(self.device)),
                    targets.to(self.device),
                    mask.to(self.device),
                )
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                training_losses.append(float(loss.item()))

            self.model.eval()
            validation_losses: list[float] = []
            with torch.no_grad():
                for features, targets, mask in validation_loader:
                    loss = self._loss(
                        self.model(features.to(self.device)),
                        targets.to(self.device),
                        mask.to(self.device),
                    )
                    if loss is not None:
                        validation_losses.append(float(loss.item()))
            if not training_losses or not validation_losses:
                raise ValueError("segment_cropper produced no valid padded training batches")
            train_loss = float(np.mean(training_losses))
            validation_loss = float(np.mean(validation_losses))
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {validation_loss:.4f}"
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        thresholds, metrics = calibrate_thresholds(
            self._validation_probabilities(validation_sessions),
        )

        cropper = self.cropper_service
        cropper.model = self.model
        cropper.scaler = self.scaler
        cropper.class_weights = {
            name: float(value)
            for name, value in zip(
                ("start", "end", "inside"),
                self.class_weights.detach().cpu().tolist(),
            )
        }
        cropper.thresholds = thresholds
        cropper.validation_metrics = {
            **metrics,
            "validation_loss": best_loss,
            "training_sessions": len(training_sessions),
            "validation_sessions": len(validation_sessions),
        }
        cropper.save_artifacts()

        try:
            from app.integrations.backend.client import backend_service

            await backend_service.save_ai_model(
                model_type="segment_cropper",
                model_data=cropper.serialize_artifacts(),
                metadata={
                    "format": MODEL_FORMAT,
                    "feature_count": input_dim,
                    "validation_metrics": cropper.validation_metrics,
                },
                is_active=True,
            )
        except Exception as exc:
            LOGGER.warning("segment_cropper backend upload failed: %s", exc)


segment_cropper_trainer = SegmentCropperTrainer()


__all__ = ["SegmentCropperTrainer", "segment_cropper_trainer"]
