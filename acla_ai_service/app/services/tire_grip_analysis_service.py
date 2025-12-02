
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
RADIAN_DETECTION_THRESHOLD = 1.0  # Slip angles below this (abs) likely expressed in radians


@dataclass
class SlipEnvelopeConfig:
    """Hyperparameters governing the slip-envelope calculation."""

    front_slip_limit: float = 6.0  # Units must match telemetry (deg or rad)
    rear_slip_limit: float = 7.0
    front_longitudinal_slip_limit: float = 0.1  # Dimensionless slip ratio
    rear_longitudinal_slip_limit: float = 0.1
    
    # Weights for the combined metric
    slip_angle_weight: float = 1.0
    slip_ratio_weight: float = 1.0
    gas_weight: float = 1.0

    slip_angle_unit: str = "rad"  # 'deg', 'rad', or 'auto'

    def combined_limit(self) -> float:
        return max(self.front_slip_limit, self.rear_slip_limit)


class TireGripAnalysisService:
    """Simplified encoder that estimates push-to-limit from telemetry."""

    def __init__(self, config: Optional[SlipEnvelopeConfig] = None):
        self.feature_catalog = TireGripFeatureCatalog
        self.config = config or SlipEnvelopeConfig()
        self._slip_angle_unit: Optional[str] = None
        self._config_slip_converted: bool = False

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
        
        # Calculate push index as the maximum of the normalized factors
        # If any factor is at the limit (1.0), the push index is 1.0
        push_index = np.maximum(
            w_lat * norm_lateral,
            np.maximum(w_long * norm_longitudinal, w_gas * gas)
        )

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

        self._ensure_slip_angle_unit(df)
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
        
        if max_limit > 1.5:
            self.config.front_slip_limit = math.radians(self.config.front_slip_limit)
            self.config.rear_slip_limit = math.radians(self.config.rear_slip_limit)
            print("[INFO] Tire grip slip limits converted from degrees to radians to match telemetry")

        self._config_slip_converted = True

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
