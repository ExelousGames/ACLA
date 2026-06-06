"""Tire-grip feature catalog and slip-envelope config.

Pure domain types — no behaviour beyond data definition. The actual
friction-circle math lives in app/features/tire_grip.py (target) or
in app/services/tire_grip_analysis_service.py (today, pre-Step-4).

Moved from app/services/tire_grip_analysis_service.py in
refactor/hexagonal-v1, Step 2.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class TireGripFeatureCatalog:
    """Authoritative feature names emitted by TireGripAnalysisService."""

    class ContextFeature(str, Enum):
        DRIVER_PUSH_TO_LIMIT = 'driver_push_to_limit'

    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]


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

    def combined_limit(self) -> float:
        return max(self.front_slip_limit, self.rear_slip_limit)


__all__ = [
    "TireGripFeatureCatalog",
    "SlipEnvelopeConfig",
]
