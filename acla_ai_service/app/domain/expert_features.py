"""Expert-driver feature catalog and config dataclasses.

These are pure domain types — column-name enums and configuration
hyperparameters. They live in app/domain/ because they have no
runtime behaviour beyond data definition; anyone in the codebase
imports them, they import nothing from outside app/domain/.

Moved from app/services/imitate_expert_learning_service.py in
refactor/hexagonal-v1, Step 2. The original service file now imports
these symbols back.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List


class ExpertFeatureCatalog:
    """Canonical expert feature names for downstream models.
    All expert state feature keys must be declared here and referenced via the Enum
    to avoid drifting string literals across the codebase.
    """

    class ExpertOptimalFeature(str, Enum):
        # Optimal action predictions
        EXPERT_OPTIMAL_SPEED = 'expert_optimal_speed'
        EXPERT_OPTIMAL_STEERING = 'expert_optimal_steering'
        EXPERT_OPTIMAL_THROTTLE = 'expert_optimal_throttle'
        EXPERT_OPTIMAL_BRAKE = 'expert_optimal_brake'
        EXPERT_OPTIMAL_GEAR = 'expert_optimal_gear'
        EXPERT_OPTIMAL_PLAYER_POS_X = 'expert_optimal_player_pos_x'
        EXPERT_OPTIMAL_PLAYER_POS_Y = 'expert_optimal_player_pos_y'
        EXPERT_OPTIMAL_PLAYER_POS_Z = 'expert_optimal_player_pos_z'
        EXPERT_OPTIMAL_TRACK_POSITION = 'expert_optimal_track_position'
        EXPERT_OPTIMAL_VELOCITY_X = 'expert_optimal_velocity_x'
        EXPERT_OPTIMAL_VELOCITY_Y = 'expert_optimal_velocity_y'
        EXPERT_OPTIMAL_VELOCITY_Z = 'expert_optimal_velocity_z'
        EXPERT_OPTIMAL_TIME = 'expert_optimal_time'

    class ContextFeature(str, Enum):
        # Velocity direction alignment with expert
        EXPERT_VELOCITY_ALIGNMENT = 'expert_velocity_alignment' # 1.0 if moving in the expert velocity direction, 0.0 opposite direction
        SPEED_DIFFERENCE = 'speed_difference' # Difference between current speed and expert optimal speed (km/h)
        DISTANCE_TO_EXPERT_LINE = 'distance_to_expert_line' # distance between current position and expert optimal racing line (meters)
        EXPERT_TIME_DIFFERENCE = 'expert_time_difference' # Difference between current time and expert time at this position (seconds/ms)


    class ExpertFeatures (str, Enum):
        # Optimal action predictions
        EXPERT_OPTIMAL_PLAYER_POS_X = 'expert_optimal_player_pos_x'
        EXPERT_OPTIMAL_PLAYER_POS_Y = 'expert_optimal_player_pos_y'
        EXPERT_OPTIMAL_PLAYER_POS_Z = 'expert_optimal_player_pos_z'
        EXPERT_OPTIMAL_SPEED = 'expert_optimal_speed'
        EXPERT_OPTIMAL_THROTTLE = 'expert_optimal_throttle'
        EXPERT_OPTIMAL_BRAKE = 'expert_optimal_brake'
        EXPERT_OPTIMAL_GEAR = 'expert_optimal_gear'
        EXPERT_OPTIMAL_TIME = 'expert_optimal_time'
        # Context features
        EXPERT_VELOCITY_ALIGNMENT = 'expert_velocity_alignment'
        SPEED_DIFFERENCE = 'speed_difference'
        DISTANCE_TO_EXPERT_LINE = 'distance_to_expert_line'
        EXPERT_TIME_DIFFERENCE = 'expert_time_difference'

    # Flat list for convenience (now only expert optimal + derived)
    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]


@dataclass(frozen=True)
class SegmentImprovementConfig:
    """Centralized thresholds and heuristics used during segment improvement analysis."""

    expert_velocity_alignment: float = 0.9
    expert_speed_diff_max: float = 5.0
    expert_distance_max: float = 5.0

    driver_push_high_threshold: float = 0.4
    driver_push_trend_min: float = 0.01

    smoothing_window_min: int = 2
    smoothing_window_max: int = 5
    ema_span_min: int = 2


@dataclass
class SegmentImprovementSummary:
    """Structured container for telemetry segment improvement analysis results."""

    velocity_alignment_mean: float = 0.0
    velocity_alignment_trend: float = 0.0
    velocity_consistency_rate: float = 0.0
    velocity_expert_points: int = 0

    speed_difference_mean: float = 0.0
    speed_difference_trend: float = 0.0
    speed_consistency_rate: float = 0.0
    speed_expert_points: int = 0
    distance_to_line_mean: float = 0.0
    distance_to_line_trend: float = 0.0
    distance_consistency_rate: float = 0.0
    distance_expert_points: int = 0

    time_difference_mean: float = 0.0
    time_difference_trend: float = 0.0
    time_gain_loss: float = 0.0 # Total time gained (negative) or lost (positive) in segment

    driver_push_available: bool = False
    driver_push_mean: float = 0.0
    driver_push_trend: float = 0.0
    driver_push_high_rate: float = 0.0

    overall_improvement_rate: float = 0.0
    overall_consistency_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@dataclass
class ExpertModelTrainingConfig:
    """Configuration for expert model training and binning strategy."""

    # Bin size range (in meters)
    min_bin_size: float = 20.0
    max_bin_size: float = 40.0

    # Speed sensitivity slider (exponent)
    # Controls how quickly the bin size transitions from max (low speed) to min (high speed).
    # 1.0 = Linear transition
    # > 1.0 = Stays large longer (less detail in medium speed)
    # < 1.0 = Shrinks quickly to min size (more detail in medium speed)
    speed_sensitivity: float = 2  # Controls curve of bin size reduction

    # Reference max speed for scaling (km/h) - used to normalize the curve
    reference_max_speed: float = 300.0


__all__ = [
    "ExpertFeatureCatalog",
    "SegmentImprovementConfig",
    "SegmentImprovementSummary",
    "ExpertModelTrainingConfig",
]
