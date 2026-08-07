"""Artifact handling and complete-session inference for the boundary cropper."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from app.ml.segment_cropper.data import prepare_feature_frame
from app.ml.segment_cropper.decoding import (
    CropCandidate,
    CropperThresholds,
    decode_probabilities,
)
from app.ml.segment_cropper.model import BoundaryTCN
from app.shared.segment_classifier_features import SEGMENT_CLASSIFIER_FEATURES


LOGGER = logging.getLogger(__name__)
MODEL_FORMAT = "segment_cropper/temporal-v1"
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DILATIONS = (1, 2, 4, 8)
DEFAULT_DROPOUT = 0.2
DEFAULT_THRESHOLDS = CropperThresholds(0.5, 0.5, 0.5)


class SegmentCropperService:
    _ARTIFACT_FILES = (
        "segment_cropper.pth",
        "segment_cropper_scaler.joblib",
        "segment_cropper_contract.json",
    )

    def __init__(self, models_directory: str = "models") -> None:
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_directory / self._ARTIFACT_FILES[0]
        self.scaler_path = self.models_directory / self._ARTIFACT_FILES[1]
        self.contract_path = self.models_directory / self._ARTIFACT_FILES[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[BoundaryTCN] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = list(SEGMENT_CLASSIFIER_FEATURES)
        self.hidden_dim = DEFAULT_HIDDEN_DIM
        self.dilations = DEFAULT_DILATIONS
        self.dropout = DEFAULT_DROPOUT
        self.class_weights: dict[str, float] = {}
        self.thresholds = DEFAULT_THRESHOLDS
        self.validation_metrics: dict[str, Any] = {}

    @property
    def derived_feature_names(self) -> list[str]:
        return [*self.feature_names, *(f"{name}_diff" for name in self.feature_names)]

    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def has_local_artifacts(self) -> bool:
        return all((self.models_directory / name).is_file() for name in self._ARTIFACT_FILES)

    def _contract(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "raw_features": list(self.feature_names),
            "features": self.derived_feature_names,
            "architecture": {
                "type": "BoundaryTCN",
                "hidden_dim": self.hidden_dim,
                "dilations": list(self.dilations),
                "dropout": self.dropout,
                "heads": ["start", "end", "inside"],
            },
            "thresholds": self.thresholds.to_dict(),
            "validation_metrics": dict(self.validation_metrics),
        }

    def _validate_contract(self, contract: Any) -> None:
        if not isinstance(contract, dict) or contract.get("format") != MODEL_FORMAT:
            raise ValueError(f"segment_cropper contract must use {MODEL_FORMAT}")
        expected_features = list(SEGMENT_CLASSIFIER_FEATURES)
        if contract.get("raw_features") != expected_features:
            raise ValueError("segment_cropper raw feature contract does not match runtime features")
        expected_derived = [
            *expected_features,
            *(f"{name}_diff" for name in expected_features),
        ]
        if contract.get("features") != expected_derived:
            raise ValueError("segment_cropper ordered derived feature contract is invalid")
        architecture = contract.get("architecture")
        expected_architecture = {
            "type": "BoundaryTCN",
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "dilations": list(DEFAULT_DILATIONS),
            "dropout": DEFAULT_DROPOUT,
            "heads": ["start", "end", "inside"],
        }
        if architecture != expected_architecture:
            raise ValueError("segment_cropper architecture contract is incompatible")
        CropperThresholds.from_dict(contract.get("thresholds"))
        if not isinstance(contract.get("validation_metrics"), dict):
            raise ValueError("segment_cropper validation metrics must be an object")

    def save_artifacts(self) -> None:
        if self.model is None or self.scaler is None:
            raise ValueError("segment_cropper cannot save before model and scaler are fitted")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "class_weights": dict(self.class_weights),
            },
            self.model_path,
        )
        joblib.dump(self.scaler, self.scaler_path)
        self.contract_path.write_text(
            json.dumps(self._contract(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_model(self) -> bool:
        if not self.has_local_artifacts():
            return False
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self._validate_contract(contract)
        scaler = joblib.load(self.scaler_path)
        if not isinstance(scaler, StandardScaler) or not hasattr(scaler, "mean_"):
            raise ValueError("segment_cropper scaler artifact is invalid")
        if int(scaler.mean_.shape[0]) != len(contract["features"]):
            raise ValueError("segment_cropper scaler feature count does not match contract")

        model = BoundaryTCN(
            input_dim=len(contract["features"]),
            hidden_dim=DEFAULT_HIDDEN_DIM,
            dilations=DEFAULT_DILATIONS,
            dropout=DEFAULT_DROPOUT,
        ).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.scaler = scaler
        self.class_weights = {
            str(name): float(value)
            for name, value in checkpoint.get("class_weights", {}).items()
        }
        self.thresholds = CropperThresholds.from_dict(contract["thresholds"])
        self.validation_metrics = dict(contract["validation_metrics"])
        return True

    def serialize_artifacts(self) -> dict[str, Any]:
        if not self.has_local_artifacts():
            raise FileNotFoundError("segment_cropper artifacts are incomplete")
        return {
            "format": MODEL_FORMAT,
            "files": {
                name: base64.b64encode((self.models_directory / name).read_bytes()).decode("ascii")
                for name in self._ARTIFACT_FILES
            },
        }

    def deserialize_artifacts(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"segment_cropper payload must use {MODEL_FORMAT}")
        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("segment_cropper payload missing files")
        missing = [name for name in self._ARTIFACT_FILES if name not in files]
        if missing:
            raise ValueError(f"segment_cropper payload missing: {', '.join(missing)}")
        try:
            decoded = {
                name: base64.b64decode(files[name], validate=True)
                for name in self._ARTIFACT_FILES
            }
            contract = json.loads(decoded["segment_cropper_contract.json"].decode("utf-8"))
        except Exception as exc:
            raise ValueError("segment_cropper payload contains invalid artifact data") from exc
        self._validate_contract(contract)
        for name, content in decoded.items():
            (self.models_directory / name).write_bytes(content)

    def predict_probabilities(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None and not self.load_model():
            raise ValueError("segment_cropper model artifact is unavailable")
        source = dataframe.reset_index(drop=True)
        if source.empty:
            empty = np.asarray([], dtype=float)
            return empty, empty.copy(), empty.copy()
        frame = prepare_feature_frame(source, self.feature_names)
        scaled = self.scaler.transform(frame.to_numpy(dtype=np.float32))
        inputs = torch.tensor(scaled, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = tuple(
                torch.sigmoid(head)[0].cpu().numpy()
                for head in logits
            )
        return probabilities

    def decode_crops(self, dataframe: pd.DataFrame) -> list[CropCandidate]:
        if dataframe.empty:
            return []
        probabilities = self.predict_probabilities(dataframe)
        return decode_probabilities(*probabilities, self.thresholds)


segment_cropper = SegmentCropperService()


__all__ = [
    "DEFAULT_DILATIONS",
    "DEFAULT_DROPOUT",
    "DEFAULT_HIDDEN_DIM",
    "MODEL_FORMAT",
    "SegmentCropperService",
    "segment_cropper",
]
