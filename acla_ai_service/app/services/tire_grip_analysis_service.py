
"""
Driver Push-to-Limit Analysis Service for Assetto Corsa Competizione

The service answers a single question:

    "Given the driver's current control inputs and vehicle state, how close is the
    car to exceeding its learned slip-angle envelope?"

Simplified version:
Determines DRIVER_PUSH_TO_LIMIT by checking the slip angle, slip ratio, and gas.
The limit is determined by all three factors together. The limit can go over 1.
"""

import math
import warnings
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class TireGripFeatureCatalog:
    """Authoritative feature names emitted by TireGripAnalysisService."""

    class ContextFeature(str, Enum):
        DRIVER_PUSH_TO_LIMIT = 'driver_push_to_limit'

    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]


EPS = 1e-6


@dataclass
class SlipEnvelopeConfig:
    """Hyperparameters governing the slip-envelope calculation."""

    front_slip_limit: float = 0.105  # ~6.0 degrees in radians
    rear_slip_limit: float = 0.122   # ~7.0 degrees in radians
    front_longitudinal_slip_limit: float = 0.1  # Dimensionless slip ratio
    rear_longitudinal_slip_limit: float = 0.1
    
    # Weights for the combined metric
    slip_angle_weight: float = 1.0
    slip_ratio_weight: float = 1.0
    gas_weight: float = 1.0

    def combined_limit(self) -> float:
        return max(self.front_slip_limit, self.rear_slip_limit)


class TireGripAnalysisService:
    """Simplified encoder that estimates push-to-limit from telemetry."""

    def __init__(self, config: Optional[SlipEnvelopeConfig] = None):
        self.feature_catalog = TireGripFeatureCatalog
        self.config = config or SlipEnvelopeConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def extract_tire_grip_features(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        df = self._prepare_dataframe(telemetry_data)
        if df.empty:
            return []

        # 1. Slip Angle (Lateral)
        lateral_slip = self._max_lateral_slip(df)
        lateral_limit = max(self.config.front_slip_limit, self.config.rear_slip_limit, EPS)
        norm_lateral = lateral_slip / lateral_limit

        # 2. Slip Ratio (Longitudinal)
        longitudinal_slip = self._max_longitudinal_slip(df)
        longitudinal_limit = max(
            self.config.front_longitudinal_slip_limit, self.config.rear_longitudinal_slip_limit, EPS
        )
        norm_longitudinal = longitudinal_slip / longitudinal_limit

        # 3. Gas
        gas = df["Physics_gas"].fillna(0.0).to_numpy(dtype=float)
        
        # Combine factors
        w_lat = self.config.slip_angle_weight
        w_long = self.config.slip_ratio_weight
        w_gas = self.config.gas_weight
        
        # Friction Circle for tires: sqrt((lat)^2 + (long)^2)
        # This captures that you can't brake 100% and turn 100% at the same time.
        # If we just used max(), 70% braking + 70% turning would look like 0.7 (safe),
        # but in reality sqrt(0.7^2 + 0.7^2) ~= 1.0 (at the limit).
        tire_utilization = np.sqrt((w_lat * norm_lateral)**2 + (w_long * norm_longitudinal)**2)

        # Calculate push index as the maximum of the tire utilization and gas
        # We keep gas separate because it represents engine limit/driver intent on straights
        push_index = np.maximum(tire_utilization, w_gas * gas)

        feature_name = self.feature_catalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value
        return [{feature_name: float(value)} for value in push_index]

    # ------------------------------------------------------------------
    # Compatibility / Legacy API
    # ------------------------------------------------------------------
    async def train_tire_grip_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """No-op: Simplified service does not require training."""
        return {"success": True, "message": "Training not required for simplified service"}

    async def train_tire_grip_model_streaming(
        self,
        chunk_iterator: Any,
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """No-op: Simplified service does not require training."""
        return {"success": True, "message": "Training not required for simplified service"}

    def serialize_tire_grip_model(self) -> Dict[str, Any]:
        """Serialize configuration only."""
        return {
            "version": "3.0",
            "config": asdict(self.config),
            "trained": True
        }

    def deserialize_tire_grip_model(self, model_data: Dict[str, Any]) -> "TireGripAnalysisService":
        """Load configuration."""
        if not isinstance(model_data, dict):
            return self
            
        config_payload = model_data.get("config")
        if isinstance(config_payload, dict):
            merged = asdict(self.config)
            for key, value in config_payload.items():
                if key in merged:
                    merged[key] = value
            self.config = SlipEnvelopeConfig(**merged)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, telemetry_list: List[Dict[str, Any]]) -> pd.DataFrame:
        if not telemetry_list:
            return pd.DataFrame()

        df = pd.DataFrame(telemetry_list)

        required_inputs = {"Physics_gas"}
        required_inputs.update(self._slip_columns())
        required_inputs.update(self._longitudinal_slip_columns())

        missing_columns = sorted(list(required_inputs - set(df.columns)))
        if missing_columns:
            raise ValueError(f"Input telemetry is missing required columns: {missing_columns}")

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

    def _max_lateral_slip(self, df: pd.DataFrame) -> np.ndarray:
        front_slip = df[["Physics_slip_angle_front_left", "Physics_slip_angle_front_right"]].abs().max(axis=1)
        rear_slip = df[["Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right"]].abs().max(axis=1)
        return np.maximum(front_slip.to_numpy(dtype=float), rear_slip.to_numpy(dtype=float))

    def _max_longitudinal_slip(self, df: pd.DataFrame) -> np.ndarray:
        front_long = df[["Physics_slip_ratio_front_left", "Physics_slip_ratio_front_right"]].abs().max(axis=1)
        rear_long = df[["Physics_slip_ratio_rear_left", "Physics_slip_ratio_rear_right"]].abs().max(axis=1)
        return np.maximum(front_long.to_numpy(dtype=float), rear_long.to_numpy(dtype=float))


# Create singleton instance for import
tire_grip_analysis_service = TireGripAnalysisService()

if __name__ == "__main__":
    print("TireGripAnalysisService initialized. Ready to estimate driver push-to-limit index!")
