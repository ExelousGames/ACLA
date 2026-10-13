"""Per-feature scaler + running stats aggregator for transformer training.

Two utilities pulled out of the 2740-line transformer/model.py:

  - ``PerFeatureScaler``: feature-wise scaler that keeps one
    ``sklearn`` scaler per column, so feature-specific normalisation
    strategies don't bleed into each other. Owns serialisation
    (``fit``/``transform``/``inverse_transform``/``save``/``load``).
  - ``_RunningFeatureStats``: streaming aggregator that accumulates
    per-feature mean and variance across dataset chunks without
    materialising the whole dataset in memory.

Pure leaves: numpy/pandas/sklearn only — no torch, no I/O beyond
joblib-pickle for save/load. The trainer in transformer/model.py
imports these back.

Extracted from app/ml/transformer/model.py in refactor/hexagonal-v4
(Page 5 of acla-ai-service-architecture.drawio).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -----------------------------
# Per-feature scaling utilities
# -----------------------------
class PerFeatureScaler:
    """Feature-wise scaler that maintains an independent scaler per feature.

    This wrapper allows the training pipeline to apply tailor-made scaling for
    every telemetry/context feature instead of relying on a single scaler that
    treats the feature matrix homogeneously. Each feature receives its own
    ``StandardScaler`` instance by default, but a custom ``scaler_factory`` can
    be provided to build alternative scalers (e.g., ``MinMaxScaler``) on a
    per-feature basis.

    Notes:
        * The scaler enforces explicit feature ordering – callers must supply
          the feature list at fit-time and preserve that ordering for all
          subsequent transforms.
        * Zero-variance features are stabilised by forcing their scale to 1.0
          to avoid division-by-zero and keep their transformed value anchored
          at 0.
    """

    def __init__(self,
                 feature_names: Optional[List[str]] = None,
                 scaler_factory: Optional[Callable[[str], Any]] = None):
        self.feature_names: List[str] = list(feature_names) if feature_names else []
        self.scaler_factory = scaler_factory or self._default_factory
        self._scalers: Dict[str, StandardScaler] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PerFeatureScaler must be fitted before use")

    def _normalise_input(self, data: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, bool]:
        array = np.asarray(data, dtype=np.float32)
        was_one_dimensional = False
        if array.ndim == 1:
            array = array.reshape(1, -1)
            was_one_dimensional = True
        if array.ndim != 2:
            raise ValueError(f"Expected 2D data matrix, got shape {array.shape}")
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Data has {array.shape[1]} features but scaler was fitted with {len(self.feature_names)}"
            )
        return array, was_one_dimensional

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, data: Union[List[List[float]], np.ndarray], feature_names: Optional[List[str]] = None) -> 'PerFeatureScaler':
        if feature_names is not None:
            self.feature_names = list(feature_names)
        if not self.feature_names:
            raise ValueError("Feature names are required to fit PerFeatureScaler")

        array, _ = self._normalise_input(data)

        self._scalers.clear()
        for idx, feature in enumerate(self.feature_names):
            scaler = self.scaler_factory(feature)
            column = array[:, idx].reshape(-1, 1)
            scaler.fit(column)

            # Guard against zero variance to keep transforms stable.
            if hasattr(scaler, 'scale_'):
                scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
                if scale == 0.0:
                    scaler.scale_ = np.array([1.0], dtype=np.float64)
                    if hasattr(scaler, 'var_'):
                        scaler.var_ = np.array([0.0], dtype=np.float64)

            self._scalers[feature] = scaler

        self._fitted = True
        return self

    def transform(self, data: Union[List[List[float]], List[float], np.ndarray]) -> np.ndarray:
        self._require_fitted()
        array, was_one_dimensional = self._normalise_input(data)

        transformed = np.zeros_like(array, dtype=np.float32)
        for idx, feature in enumerate(self.feature_names):
            scaler = self._scalers.get(feature)
            if scaler is None:
                raise KeyError(f"No scaler fitted for feature '{feature}'")
            transformed[:, idx] = scaler.transform(array[:, idx].reshape(-1, 1)).reshape(-1)

        return transformed[0] if was_one_dimensional else transformed

    def inverse_transform(self, data: Union[List[List[float]], List[float], np.ndarray]) -> np.ndarray:
        self._require_fitted()
        array, was_one_dimensional = self._normalise_input(data)

        inversed = np.zeros_like(array, dtype=np.float32)
        for idx, feature in enumerate(self.feature_names):
            scaler = self._scalers.get(feature)
            if scaler is None:
                raise KeyError(f"No scaler fitted for feature '{feature}'")
            inversed[:, idx] = scaler.inverse_transform(array[:, idx].reshape(-1, 1)).reshape(-1)

        return inversed[0] if was_one_dimensional else inversed

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        status = 'fitted' if self._fitted else 'unfitted'
        return f"PerFeatureScaler(features={len(self.feature_names)}, status={status})"

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_factory(_: str) -> StandardScaler:
        return StandardScaler()

    @staticmethod
    def _as_1d_array(value: Any, default: float) -> np.ndarray:
        if isinstance(value, np.ndarray):
            arr = value.astype(np.float64)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float64)
        elif value is None:
            arr = np.asarray([default], dtype=np.float64)
        else:
            arr = np.asarray([value], dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def to_serializable(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of all feature scalers."""
        self._require_fitted()

        serialized_scalers: List[Dict[str, Any]] = []
        for feature in self.feature_names:
            scaler = self._scalers.get(feature)
            if scaler is None:
                raise KeyError(f"Missing scaler for feature '{feature}' during serialization")

            scaler_class = type(scaler).__name__
            if scaler_class != 'StandardScaler':
                raise ValueError(
                    f"PerFeatureScaler serialization currently supports StandardScaler instances only (feature '{feature}')"
                )

            serialized_scalers.append({
                'feature': feature,
                'class': scaler_class,
                'with_mean': getattr(scaler, 'with_mean', True),
                'with_std': getattr(scaler, 'with_std', True),
                'data': {
                    'mean': self._as_1d_array(getattr(scaler, 'mean_', None), 0.0).tolist(),
                    'scale': self._as_1d_array(getattr(scaler, 'scale_', None), 1.0).tolist(),
                    'var': self._as_1d_array(getattr(scaler, 'var_', None), 0.0).tolist(),
                    'n_samples_seen': self._as_1d_array(getattr(scaler, 'n_samples_seen_', None), 1.0).tolist()
                }
            })

        return {
            'version': 1,
            'feature_names': list(self.feature_names),
            'scalers': serialized_scalers
        }

    @classmethod
    def from_serializable(cls, payload: Dict[str, Any], scaler_factory: Optional[Callable[[str], Any]] = None) -> 'PerFeatureScaler':
        """Rehydrate a PerFeatureScaler from its serialized dictionary."""
        if not isinstance(payload, dict):
            raise ValueError("Serialized scaler payload must be a dictionary")

        feature_names = payload.get('feature_names')
        if not feature_names:
            raise ValueError("Serialized scaler payload missing 'feature_names'")

        scaler = cls(feature_names=feature_names, scaler_factory=scaler_factory)
        scaler._scalers.clear()

        scalers_payload = payload.get('scalers', [])
        if len(scalers_payload) != len(feature_names):
            raise ValueError(
                f"Serialized scaler payload contains {len(scalers_payload)} scaler entries for {len(feature_names)} features"
            )

        for entry in scalers_payload:
            feature = entry.get('feature')
            if feature not in scaler.feature_names:
                raise ValueError(f"Serialized scaler entry references unknown feature '{feature}'")

            scaler_class = entry.get('class', 'StandardScaler')
            if scaler_class != 'StandardScaler':
                raise ValueError(
                    f"PerFeatureScaler deserialization currently supports StandardScaler instances only (got '{scaler_class}')"
                )

            scaler_instance = scaler.scaler_factory(feature) if scaler_factory else StandardScaler()
            scaler_instance.with_mean = entry.get('with_mean', True)
            scaler_instance.with_std = entry.get('with_std', True)

            data = entry.get('data', {})
            scaler_instance.mean_ = cls._as_1d_array(data.get('mean'), 0.0)
            scaler_instance.scale_ = cls._as_1d_array(data.get('scale'), 1.0)
            scaler_instance.var_ = cls._as_1d_array(data.get('var'), 0.0)

            n_samples_seen = data.get('n_samples_seen', [1.0])
            if isinstance(n_samples_seen, (int, float)):
                n_samples_seen = [n_samples_seen]
            scaler_instance.n_samples_seen_ = cls._as_1d_array(n_samples_seen, 1.0)
            scaler_instance.n_features_in_ = 1

            scaler._scalers[feature] = scaler_instance

        scaler._fitted = True
        return scaler

    @classmethod
    def from_feature_statistics(cls,
                                feature_names: List[str],
                                means: np.ndarray,
                                variances: np.ndarray,
                                counts: np.ndarray,
                                scaler_factory: Optional[Callable[[str], Any]] = None) -> 'PerFeatureScaler':
        if len(feature_names) == 0:
            raise ValueError("feature_names must not be empty")

        scaler = cls(feature_names=feature_names, scaler_factory=scaler_factory)
        scaler._scalers.clear()

        means = np.asarray(means, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.float64)

        if means.shape[0] != len(feature_names) or variances.shape[0] != len(feature_names) or counts.shape[0] != len(feature_names):
            raise ValueError("Statistic arrays must match length of feature_names")

        for idx, feature in enumerate(feature_names):
            this_count = counts[idx]
            if this_count <= 0:
                this_count = 1.0

            var = max(float(variances[idx]), 0.0)
            scale_value = math.sqrt(var) if var > 0 else 1.0

            scaler_instance = scaler.scaler_factory(feature)
            scaler_instance.mean_ = np.array([float(means[idx])], dtype=np.float64)
            scaler_instance.var_ = np.array([var], dtype=np.float64)
            scaler_instance.scale_ = np.array([scale_value], dtype=np.float64)
            scaler_instance.n_samples_seen_ = np.array([this_count], dtype=np.float64)
            scaler_instance.n_features_in_ = 1

            scaler._scalers[feature] = scaler_instance

        scaler._fitted = True
        return scaler


# -----------------------------
# JSON-safe serialization utils
# -----------------------------


# -----------------------------
# Streaming feature statistics
# -----------------------------
class _RunningFeatureStats:
    """Streaming aggregator for per-feature statistics across all dataset chunks."""

    def __init__(self, num_features: int):
        self.counts = np.zeros(num_features, dtype=np.float64)
        self.sums = np.zeros(num_features, dtype=np.float64)
        self.sums_sq = np.zeros(num_features, dtype=np.float64)

    def update(self, matrix: np.ndarray) -> None:
        if matrix.size == 0:
            return
        # Ensure float64 for numerical stability
        data = matrix.astype(np.float64, copy=False)
        self.counts += data.shape[0]
        self.sums += np.sum(data, axis=0, dtype=np.float64)
        self.sums_sq += np.sum(np.square(data, dtype=np.float64), axis=0, dtype=np.float64)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = np.divide(self.sums, self.counts, out=np.zeros_like(self.sums), where=self.counts > 0)
        averaged_sq = np.divide(self.sums_sq, self.counts, out=np.zeros_like(self.sums_sq), where=self.counts > 0)
        variances = averaged_sq - np.square(means)
        variances = np.clip(variances, 0.0, None)
        counts = np.where(self.counts > 0, self.counts, 1.0)
        return counts, means, variances


__all__ = [
    "PerFeatureScaler",
    "_RunningFeatureStats",
]
