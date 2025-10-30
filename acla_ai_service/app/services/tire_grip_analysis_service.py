
"""
Driver Push-to-Limit Analysis Service for Assetto Corsa Competizione

The service answers a single question:

    "Given the driver's current control inputs and vehicle state, how close is the
    car to exceeding its learned slip-angle envelope?"

It maintains a multimodal neural encoder that learns the safe operating manifold
from telemetry. Only samples where slip angles stay inside configurable limits are
used for training; as fresh telemetry arrives the envelope expands to cover newly
observed combinations of throttle, brake, steering, load transfer, and chassis
attitude. Runtime inference compares the predicted envelope with the instantaneous
response to produce a stable **0-1 push index**:

    0.0  -> the car is operating comfortably inside the envelope
    1.0  -> the car is right at (or exceeding) the learned limit

Outputs are suitable for enriching telemetry that downstream AI models consume.
"""

import math
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from collections.abc import AsyncIterator as AsyncIteratorABC, Iterable as IterableABC
from torch import nn
# (Removed direct SciPy dependencies to keep environment minimal.)

# Import backend models
from ..models.telemetry_models import FeatureProcessor

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class TireGripFeatureCatalog:
    """Authoritative feature names emitted by TireGripAnalysisService."""

    class ContextFeature(str, Enum):
        DRIVER_PUSH_TO_LIMIT = 'driver_push_to_limit'

    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]


EPS = 1e-6
RADIAN_DETECTION_THRESHOLD = 1.0  # Slip angles below this (abs) likely expressed in radians


@dataclass
class SlipEnvelopeConfig:
    """Hyperparameters governing the slip-envelope model."""

    front_slip_limit: float = 6.0  # Units must match telemetry (deg or rad)
    rear_slip_limit: float = 7.0
    safety_margin: float = 0.25  # Keep training samples well inside the limit
    front_longitudinal_slip_limit: float = 0.12  # Dimensionless slip ratio
    rear_longitudinal_slip_limit: float = 0.15
    longitudinal_safety_margin: float = 0.02
    limit_margin_multiplier: float = 1.5  # Alpha in limit = mu + alpha * sigma
    nll_weight: float = 1.0
    penalty_weight: float = 5.0
    learning_rate: float = 3e-4
    max_epochs: int = 20
    streaming_epochs: int = 4
    batch_size: int = 256
    buffer_max_samples: int = 50000
    hidden_dims: tuple[int, ...] = (128, 128)
    dropout: float = 0.1
    device: str = "cpu"
    slip_angle_unit: str = "auto"  # 'deg', 'rad', or 'auto'

    def combined_limit(self) -> float:
        return max(self.front_slip_limit, self.rear_slip_limit)


class RunningStandardizer:
    """Maintain online mean/variance for feature standardisation."""

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self._count = 0
        self._mean = np.zeros(self.dimension, dtype=float)
        self._m2 = np.zeros(self.dimension, dtype=float)

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        batch = np.atleast_2d(batch).astype(float)
        for row in batch:
            self._count += 1
            delta = row - self._mean
            self._mean += delta / self._count
            self._m2 += delta * (row - self._mean)

    def transform(self, batch: np.ndarray) -> np.ndarray:
        if self._count < 2:
            return np.atleast_2d(batch).astype(float)
        std = self.std()
        return (np.atleast_2d(batch).astype(float) - self._mean) / std

    def mean(self) -> np.ndarray:
        return self._mean.copy()

    def var(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.dimension, dtype=float)
        return self._m2 / max(self._count - 1, 1)

    def std(self) -> np.ndarray:
        return np.sqrt(self.var() + EPS)

    @property
    def count(self) -> int:
        return int(self._count)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "count": self.count,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "RunningStandardizer":
        dimension = int(state["dimension"])
        instance = cls(dimension)
        instance._count = int(state.get("count", 0))
        instance._mean = np.array(state.get("mean", [0.0] * dimension), dtype=float)
        instance._m2 = np.array(state.get("m2", [0.0] * dimension), dtype=float)
        return instance


class SlipEnvelopeModel(nn.Module):
    """Neural network predicting slip-envelope parameters."""

    def __init__(self, input_dim: int, config: SlipEnvelopeConfig):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.mu_head = nn.Linear(prev_dim, 1)
        self.sigma_head = nn.Linear(prev_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        mu = self.mu_head(hidden).squeeze(-1)
        sigma_raw = self.sigma_head(hidden).squeeze(-1)
        sigma = torch.nn.functional.softplus(sigma_raw) + 1e-3
        return mu, sigma

    
class TireGripAnalysisService:
    """Multimodal encoder that learns a safe slip-envelope from telemetry."""

    def __init__(self, config: Optional[SlipEnvelopeConfig] = None):
        self.feature_catalog = TireGripFeatureCatalog
        self.config = config or SlipEnvelopeConfig()
        self._device = self._resolve_device(self.config.device)

        self.model: Optional[SlipEnvelopeModel] = None
        self.normalizer: Optional[RunningStandardizer] = None
        self._feature_cache: Optional[np.ndarray] = None
        self._target_cache: Optional[np.ndarray] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._trained: bool = False
        self._input_dim: Optional[int] = None
        self._feature_names: List[str] = self._default_feature_names()
        self._slip_angle_unit: Optional[str] = None
        self._config_slip_converted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def train_tire_grip_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = self._prepare_dataframe(training_data)
        return self._ingest_dataframe(df, incremental=False)

    async def train_tire_grip_model_streaming(
        self,
        chunk_iterator: Union[
            Iterator[List[Dict[str, Any]]],
            Iterable[List[Dict[str, Any]]],
            AsyncIterator[List[Dict[str, Any]]]
        ],
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        async def _iterate_chunks() -> AsyncIterator[List[Dict[str, Any]]]:
            if isinstance(chunk_iterator, AsyncIteratorABC):
                async for chunk in chunk_iterator:
                    yield chunk
            elif isinstance(chunk_iterator, IterableABC):
                for chunk in chunk_iterator:
                    yield chunk
            else:
                raise TypeError("chunk_iterator must be iterable or async iterable")

        aggregated_rows = 0
        aggregated_safe_rows = 0
        chunks_processed = 0
        last_stats: Dict[str, Any] = {}

        async for chunk in _iterate_chunks():
            if max_chunks is not None and chunks_processed >= max_chunks:
                break

            if chunk is None:
                continue

            if isinstance(chunk, pd.DataFrame):
                records = chunk.to_dict(orient="records")
            else:
                records = chunk

            if not records:
                continue

            df = self._prepare_dataframe(records)
            if df.empty:
                continue

            aggregated_rows += len(df)
            stats = self._ingest_dataframe(df, incremental=True)
            aggregated_safe_rows += stats.get("safe_rows", 0)
            chunks_processed += 1
            last_stats = stats

        if not self._trained:
            raise ValueError("Streaming training did not yield any safe telemetry samples")

        last_stats.update({
            "total_rows_seen": aggregated_rows,
            "total_safe_rows": aggregated_safe_rows,
            "chunks_processed": chunks_processed,
        })
        return last_stats

    async def extract_tire_grip_features(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        if not self._trained or self.model is None or self.normalizer is None:
            raise RuntimeError("Tire grip model must be trained before feature extraction")

        df = self._prepare_dataframe(telemetry_data)
        if df.empty:
            return []

        features, _ = self._build_feature_matrix(df)
        normalized = self.normalizer.transform(features)

        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(normalized, dtype=torch.float32)
            mu, sigma = self.model(inputs.to(self._device))
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()

        combined_utilization = self._combined_utilization(df)
        limit = 1.0
        envelope = mu + self.config.limit_margin_multiplier * sigma
        pred_ratio = np.clip(envelope / max(limit, EPS), 0.0, 2.0)
        observed_ratio = np.clip(combined_utilization / max(limit, EPS), 0.0, 2.0)
        push_index = np.clip(np.maximum(pred_ratio, observed_ratio), 0.0, 1.0)

        feature_name = self.feature_catalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value
        return [{feature_name: float(value)} for value in push_index]

    def serialize_tire_grip_model(self) -> Dict[str, Any]:
        if not self._trained or self.model is None or self.normalizer is None:
            raise ValueError("Cannot serialize tire grip model: model has not been trained yet")

        state_dict = {}
        for key, tensor in self.model.state_dict().items():
            state_dict[key] = tensor.detach().cpu().tolist()

        return {
            "version": "2.0",
            "trained": self._trained,
            "config": asdict(self.config),
            "normalizer": self.normalizer.state_dict(),
            "model_state": state_dict,
            "input_dim": self._input_dim,
            "feature_names": self._feature_names,
            "cache_size": int(self._feature_cache.shape[0]) if self._feature_cache is not None else 0,
            "serialized_timestamp": datetime.utcnow().isoformat(timespec="seconds")
        }

    def deserialize_tire_grip_model(self, model_data: Dict[str, Any]) -> "TireGripAnalysisService":
        try:
            if not isinstance(model_data, dict):
                raise ValueError("Model data must be a dictionary")

            version = str(model_data.get("version", "1.0"))
            if version != "2.0":
                print(f"[WARNING] Loading tire grip model version {version}. Expected 2.0.")

            config_payload = model_data.get("config")
            if isinstance(config_payload, dict):
                merged = asdict(self.config)
                for key, value in config_payload.items():
                    if key in merged:
                        merged[key] = value
                self.config = SlipEnvelopeConfig(**merged)
            self._device = self._resolve_device(self.config.device)

            normalizer_state = model_data.get("normalizer")
            if not isinstance(normalizer_state, dict):
                raise ValueError("Serialized model missing normalizer state")
            self.normalizer = RunningStandardizer.from_state_dict(normalizer_state)

            self._feature_names = list(model_data.get("feature_names", self._default_feature_names()))
            self._input_dim = int(model_data.get("input_dim", len(self._feature_names)))
            self._ensure_model(self._input_dim)

            model_state = model_data.get("model_state")
            if not isinstance(model_state, dict):
                raise ValueError("Serialized model missing model_state field")

            torch_state = {}
            for key, value in model_state.items():
                tensor = torch.tensor(value, dtype=torch.float32)
                torch_state[key] = tensor
            self.model.load_state_dict(torch_state)
            self.model.to(self._device)
            self.model.eval()

            self._trained = bool(model_data.get("trained", True))
            self._feature_cache = None
            self._target_cache = None
            unit = (getattr(self.config, "slip_angle_unit", "auto") or "auto").strip().lower()
            if unit in {"deg", "degree", "degrees"}:
                self._slip_angle_unit = "deg"
                self.config.slip_angle_unit = "deg"
            elif unit in {"rad", "radian", "radians"}:
                self._slip_angle_unit = "rad"
                self.config.slip_angle_unit = "rad"
            else:
                self._slip_angle_unit = None
            self._config_slip_converted = True

            print("[INFO] Tire grip model deserialization complete")
            return self

        except Exception as exc:
            raise RuntimeError(
                f"{self.__class__.__name__} failed to deserialize tire grip analysis model: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_device(self, requested: str) -> torch.device:
        requested = (requested or "cpu").lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[INFO] CUDA requested but not available. Falling back to CPU.")
            requested = "cpu"
        return torch.device(requested)

    def _default_feature_names(self) -> List[str]:
        return [
            "Physics_gas",
            "Physics_brake",
            "Physics_steer_angle",
            "Physics_speed_kmh",
            "Physics_gear",
            "Physics_rpm",
            "Physics_roll",
            "Physics_pitch",
            "Physics_local_velocity_x",
            "Physics_local_velocity_y",
            "Physics_local_velocity_z",
            "Physics_velocity_x",
            "Physics_velocity_y",
            "Physics_velocity_z",
            "Physics_suspension_travel_front_left",
            "Physics_suspension_travel_front_right",
            "Physics_suspension_travel_rear_left",
            "Physics_suspension_travel_rear_right",
            "Physics_slip_ratio_front_left",
            "Physics_slip_ratio_front_right",
            "Physics_slip_ratio_rear_left",
            "Physics_slip_ratio_rear_right",
            "Physics_tyre_core_temp_front_left",
            "Physics_tyre_core_temp_front_right",
            "Physics_tyre_core_temp_rear_left",
            "Physics_tyre_core_temp_rear_right",
            "abs_steer",
            "gas_times_speed",
            "brake_times_speed",
            "steer_times_speed",
            "pitch_times_speed",
            "roll_times_speed",
            "front_suspension_sum",
            "rear_suspension_sum",
            "front_suspension_diff",
            "rear_suspension_diff",
            "front_slip_ratio_max",
            "rear_slip_ratio_max",
            "total_slip_ratio_max",
            "gas_times_total_slip_ratio",
            "brake_times_total_slip_ratio",
            "combined_longitudinal_slip",
            "rpm_norm",
            "gear_times_gas",
            "gear_times_brake",
            "rpm_times_gas",
            "rpm_times_total_slip_ratio",
            "gear_times_total_slip_ratio",
        ]

    def _prepare_dataframe(self, telemetry_list: List[Dict[str, Any]]) -> pd.DataFrame:
        if not telemetry_list:
            return pd.DataFrame()

        df_raw = pd.DataFrame(telemetry_list)
        try:
            processor = FeatureProcessor(df_raw)
            df = processor.general_cleaning_for_analysis()
        except Exception:
            df = df_raw.fillna(0.0)

        self._ensure_slip_angle_unit(df)

        required_columns = set(self._feature_names + self._slip_columns() + self._longitudinal_slip_columns())
        required_columns.add("Physics_speed_kmh")
        required_columns.add("Physics_rpm")
        required_columns.add("Physics_gear")
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        df[list(required_columns)] = df[list(required_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return df

    def _slip_columns(self) -> List[str]:
        return [
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left",
            "Physics_slip_angle_rear_right",
        ]

    def _longitudinal_slip_columns(self) -> List[str]:
        return [
            "Physics_slip_ratio_front_left",
            "Physics_slip_ratio_front_right",
            "Physics_slip_ratio_rear_left",
            "Physics_slip_ratio_rear_right",
        ]

    def _ensure_slip_angle_unit(self, df: pd.DataFrame) -> None:
        if self._slip_angle_unit is not None:
            return

        configured = (self.config.slip_angle_unit or "auto").strip().lower()
        if configured in {"deg", "degree", "degrees"}:
            self._slip_angle_unit = "deg"
            self.config.slip_angle_unit = "deg"
            return
        if configured in {"rad", "radian", "radians"}:
            self._slip_angle_unit = "rad"
            self.config.slip_angle_unit = "rad"
            return

        slip_cols = [col for col in self._slip_columns() if col in df.columns]
        if not slip_cols:
            # Default to degrees when we cannot inspect telemetry
            self._slip_angle_unit = "deg"
            self.config.slip_angle_unit = "deg"
            return

        slip_values = df[slip_cols].to_numpy(dtype=float)
        if slip_values.size == 0:
            self._slip_angle_unit = "deg"
            self.config.slip_angle_unit = "deg"
            return

        max_slip = float(np.nanmax(np.abs(slip_values)))
        if not np.isfinite(max_slip):
            max_slip = 0.0

        if max_slip <= EPS:
            return

        if max_slip <= RADIAN_DETECTION_THRESHOLD:
            self._slip_angle_unit = "rad"
            self._convert_config_slip_limits_to_radians_if_needed()
        else:
            self._slip_angle_unit = "deg"

        self.config.slip_angle_unit = self._slip_angle_unit

    def _convert_config_slip_limits_to_radians_if_needed(self) -> None:
        if self._config_slip_converted:
            return

        max_limit = max(self.config.front_slip_limit, self.config.rear_slip_limit)
        margin = self.config.safety_margin

        if max_limit > 1.5 or margin > 0.5:
            self.config.front_slip_limit = math.radians(self.config.front_slip_limit)
            self.config.rear_slip_limit = math.radians(self.config.rear_slip_limit)
            self.config.safety_margin = math.radians(self.config.safety_margin)
            print("[INFO] Tire grip slip limits converted from degrees to radians to match telemetry")

        self._config_slip_converted = True

    def _safe_mask(self, df: pd.DataFrame) -> pd.Series:
        front_slip = df[["Physics_slip_angle_front_left", "Physics_slip_angle_front_right"]].abs().max(axis=1)
        rear_slip = df[["Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right"]].abs().max(axis=1)
        front_long_slip = df[["Physics_slip_ratio_front_left", "Physics_slip_ratio_front_right"]].abs().max(axis=1)
        rear_long_slip = df[["Physics_slip_ratio_rear_left", "Physics_slip_ratio_rear_right"]].abs().max(axis=1)
        front_limit = max(self.config.front_slip_limit - self.config.safety_margin, EPS)
        rear_limit = max(self.config.rear_slip_limit - self.config.safety_margin, EPS)
        front_long_limit = max(self.config.front_longitudinal_slip_limit - self.config.longitudinal_safety_margin, EPS)
        rear_long_limit = max(self.config.rear_longitudinal_slip_limit - self.config.longitudinal_safety_margin, EPS)
        return (
            (front_slip <= front_limit)
            & (rear_slip <= rear_limit)
            & (front_long_slip <= front_long_limit)
            & (rear_long_slip <= rear_long_limit)
        )

    def _max_lateral_slip(self, df: pd.DataFrame) -> np.ndarray:
        front_slip = df[["Physics_slip_angle_front_left", "Physics_slip_angle_front_right"]].abs().max(axis=1)
        rear_slip = df[["Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right"]].abs().max(axis=1)
        return np.maximum(front_slip.to_numpy(dtype=float), rear_slip.to_numpy(dtype=float))

    def _max_longitudinal_slip(self, df: pd.DataFrame) -> np.ndarray:
        front_long = df[["Physics_slip_ratio_front_left", "Physics_slip_ratio_front_right"]].abs().max(axis=1)
        rear_long = df[["Physics_slip_ratio_rear_left", "Physics_slip_ratio_rear_right"]].abs().max(axis=1)
        return np.maximum(front_long.to_numpy(dtype=float), rear_long.to_numpy(dtype=float))

    def _combined_utilization(self, df: pd.DataFrame) -> np.ndarray:
        lateral = self._max_lateral_slip(df)
        longitudinal = self._max_longitudinal_slip(df)
        lateral_limit = max(self.config.front_slip_limit, self.config.rear_slip_limit, EPS)
        longitudinal_limit = max(
            self.config.front_longitudinal_slip_limit, self.config.rear_longitudinal_slip_limit, EPS
        )

        lateral_ratio = np.clip(lateral / lateral_limit, 0.0, np.inf)
        longitudinal_slip_ratio = np.clip(longitudinal / longitudinal_limit, 0.0, np.inf)
        longitudinal_ratio = longitudinal_slip_ratio
        combined = np.sqrt(lateral_ratio ** 2 + longitudinal_ratio ** 2)
        return np.clip(combined, 0.0, 2.0)

    def _build_feature_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        work = df.copy()
        work["abs_steer"] = work["Physics_steer_angle"].abs()
        work["gas_times_speed"] = work["Physics_gas"] * work["Physics_speed_kmh"]
        work["brake_times_speed"] = work["Physics_brake"] * work["Physics_speed_kmh"]
        work["steer_times_speed"] = work["Physics_steer_angle"] * work["Physics_speed_kmh"]
        work["pitch_times_speed"] = work["Physics_pitch"] * work["Physics_speed_kmh"]
        work["roll_times_speed"] = work["Physics_roll"] * work["Physics_speed_kmh"]
        work["front_suspension_sum"] = (
            work["Physics_suspension_travel_front_left"] + work["Physics_suspension_travel_front_right"]
        )
        work["rear_suspension_sum"] = (
            work["Physics_suspension_travel_rear_left"] + work["Physics_suspension_travel_rear_right"]
        )
        work["front_suspension_diff"] = (
            work["Physics_suspension_travel_front_left"] - work["Physics_suspension_travel_front_right"]
        )
        work["rear_suspension_diff"] = (
            work["Physics_suspension_travel_rear_left"] - work["Physics_suspension_travel_rear_right"]
        )
        work["front_slip_ratio_max"] = (
            work[["Physics_slip_ratio_front_left", "Physics_slip_ratio_front_right"]].abs().max(axis=1)
        )
        work["rear_slip_ratio_max"] = (
            work[["Physics_slip_ratio_rear_left", "Physics_slip_ratio_rear_right"]].abs().max(axis=1)
        )
        work["total_slip_ratio_max"] = np.maximum(work["front_slip_ratio_max"], work["rear_slip_ratio_max"])
        work["combined_longitudinal_slip"] = np.sqrt(
            work["Physics_slip_ratio_front_left"] ** 2
            + work["Physics_slip_ratio_front_right"] ** 2
            + work["Physics_slip_ratio_rear_left"] ** 2
            + work["Physics_slip_ratio_rear_right"] ** 2
        )
        work["gas_times_total_slip_ratio"] = work["Physics_gas"] * work["total_slip_ratio_max"]
        work["brake_times_total_slip_ratio"] = work["Physics_brake"] * work["total_slip_ratio_max"]
        rpm = work["Physics_rpm"].clip(lower=0.0)
        max_rpm = max(float(np.nanmax(rpm.to_numpy(dtype=float))), EPS)
        work["rpm_norm"] = np.clip(rpm / max_rpm, 0.0, 1.5)
        gear = work["Physics_gear"].astype(float)
        work["gear_times_gas"] = gear * work["Physics_gas"]
        work["gear_times_brake"] = gear * work["Physics_brake"]
        work["rpm_times_gas"] = work["Physics_rpm"] * work["Physics_gas"]
        work["rpm_times_total_slip_ratio"] = work["Physics_rpm"] * work["total_slip_ratio_max"]
        work["gear_times_total_slip_ratio"] = gear * work["total_slip_ratio_max"]

        features = work[self._feature_names].to_numpy(dtype=float)
        targets = self._combined_utilization(df)
        return features, targets

    def _append_to_cache(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.size == 0:
            return
        if self._feature_cache is None:
            self._feature_cache = features
            self._target_cache = targets
        else:
            self._feature_cache = np.concatenate([self._feature_cache, features], axis=0)
            self._target_cache = np.concatenate([self._target_cache, targets], axis=0)

        max_samples = int(self.config.buffer_max_samples)
        if max_samples > 0 and self._feature_cache.shape[0] > max_samples:
            excess = self._feature_cache.shape[0] - max_samples
            self._feature_cache = self._feature_cache[excess:]
            self._target_cache = self._target_cache[excess:]

    def _ensure_model(self, input_dim: int) -> None:
        if self.model is None or self._input_dim != input_dim:
            self._input_dim = input_dim
            self.model = SlipEnvelopeModel(input_dim, self.config).to(self._device)
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.model is not None:
            self.model.to(self._device)
            if self._optimizer is None:
                self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _train_from_cache(self, epochs: int) -> Dict[str, Any]:
        if self._feature_cache is None or self._target_cache is None:
            return {"epochs": 0, "loss": None}

        dataset_size = self._feature_cache.shape[0]
        if dataset_size < 5:
            return {"epochs": 0, "loss": None}

        normalized = self.normalizer.transform(self._feature_cache)
        inputs = torch.tensor(normalized, dtype=torch.float32)
        targets = torch.tensor(self._target_cache, dtype=torch.float32)

        self.model.train()
        history_loss = None
        target_limit = torch.tensor(1.0, dtype=torch.float32, device=self._device)

        for epoch in range(max(1, epochs)):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            batches = 0

            for start in range(0, dataset_size, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset_size)
                idx = permutation[start:end]
                batch_x = inputs[idx].to(self._device)
                batch_y = targets[idx].to(self._device)

                mu, sigma = self.model(batch_x)
                residual = (batch_y - mu) / sigma
                nll = 0.5 * (residual ** 2 + 2.0 * torch.log(sigma) + math.log(2.0 * math.pi))
                penalty = torch.relu(mu + self.config.limit_margin_multiplier * sigma - target_limit)
                loss = self.config.nll_weight * nll.mean() + self.config.penalty_weight * penalty.mean()

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self._optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            history_loss = epoch_loss / max(batches, 1)

        self.model.eval()
        self._trained = True

        return {"epochs": epochs, "loss": float(history_loss) if history_loss is not None else None}

    def _ingest_dataframe(self, df: pd.DataFrame, incremental: bool) -> Dict[str, Any]:
        if df is None or df.empty:
            raise ValueError("Cannot train tire grip model on empty dataframe")

        safe_mask = self._safe_mask(df)
        safe_rows = int(safe_mask.sum())
        if safe_rows == 0:
            return {
                "success": False,
                "message": "No safe telemetry rows within slip or traction limits",
                "safe_rows": 0,
                "total_rows": int(len(df)),
            }

        safe_df = df.loc[safe_mask].reset_index(drop=True)
        features, targets = self._build_feature_matrix(safe_df)

        if self.normalizer is None or self.normalizer.dimension != features.shape[1]:
            self.normalizer = RunningStandardizer(features.shape[1])
        self.normalizer.update(features)

        self._append_to_cache(features, targets)
        self._ensure_model(features.shape[1])

        epochs = self.config.streaming_epochs if incremental else self.config.max_epochs
        train_stats = self._train_from_cache(epochs)

        return {
            "success": True,
            "safe_rows": safe_rows,
            "total_rows": int(len(df)),
            "buffer_rows": int(self._feature_cache.shape[0]) if self._feature_cache is not None else 0,
            "epochs_ran": train_stats.get("epochs"),
            "loss": train_stats.get("loss"),
        }


# Create singleton instance for import
tire_grip_analysis_service = TireGripAnalysisService()

if __name__ == "__main__":
    print("TireGripAnalysisService initialized. Ready to estimate driver push-to-limit index!")
