"""
Tire Grip and Friction Circle Analysis Service for Assetto Corsa Competizione

This service provides comprehensive tire grip analysis and friction circle utilization
estimation using machine learning models. It extracts features related to:

- Tire grip estimation based on physics telemetry
- Friction circle utilization (how close the car is to the limit)  
- Weight transfer analysis
- Predictive load calculations
- Tire performance degradation
- Optimal grip windows

The extracted features are designed to be inserted back into telemetry data for enhanced AI analysis.
"""

import pandas as pd
import numpy as np
import warnings
import math
import asyncio  # retained if future async hooks needed
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
# (Removed direct SciPy dependencies to keep environment minimal.)

# Import backend service and models
from .backend_service import backend_service
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor, _safe_float

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


class TireGripFeatures:
    """Data class for tire grip analysis features"""
    
    def __init__(self):
        # Friction circle utilization (0-1 scale)
        self.friction_circle_utilization = 0.0
        self.friction_circle_utilization_front = 0.0
        self.friction_circle_utilization_rear = 0.0
        
        # Tire grip estimation (0-1 scale where 1 is optimal grip)
        self.estimated_tire_grip_fl = 0.0
        self.estimated_tire_grip_fr = 0.0
        self.estimated_tire_grip_rl = 0.0
        self.estimated_tire_grip_rr = 0.0
        self.overall_tire_grip = 0.0
        
        # Weight transfer analysis
        self.longitudinal_weight_transfer = 0.0  # -1 to 1 (rear to front)
        self.lateral_weight_transfer = 0.0       # -1 to 1 (right to left)
        self.dynamic_weight_distribution = 0.0   # How much weight is shifting
        
        # Predictive load on each tire
        self.predictive_load_fl = 0.0
        self.predictive_load_fr = 0.0
        self.predictive_load_rl = 0.0
        self.predictive_load_rr = 0.0
        
        # Tire performance metrics
        self.tire_saturation_level = 0.0     # How close tires are to saturation (0-1)
        self.optimal_grip_window = 0.0       # Whether in optimal temperature/pressure window (0-1)
        self.tire_degradation_factor = 0.0   # Estimated tire degradation (0-1)
        
        # Advanced grip metrics
        self.slip_angle_efficiency = 0.0     # How efficiently slip angles are being used
        self.slip_ratio_efficiency = 0.0     # How efficiently slip ratios are being used
        self.cornering_grip_utilization = 0.0 # Specific to cornering forces
        self.braking_grip_utilization = 0.0   # Specific to braking forces
        self.acceleration_grip_utilization = 0.0  # Specific to acceleration forces


class TireGripAnalysisService:
    """Heuristic Tire Grip & Friction Circle Analysis Service (Driver-Behavior Agnostic)

    PURPOSE
    -------
    Provide purely physics-derived, driver-behavior neutral enrichment of telemetry data with
    tire grip utilization, friction circle occupancy, weight transfer, slip efficiency, and
    related indicators. These enriched features are intended to feed downstream generalized
    models (e.g., transformers for imitation / reasoning) without leaking individual driver
    control habits (throttle modulation, braking aggressiveness, steering style) or car identity.

    DESIGN PRINCIPLES
    -----------------
    1. No supervised ML training inside this service (all ML code removed).
    2. Only vehicle dynamics & tire state signals are used; control inputs (brake, gas, steer, gear)
       are excluded to avoid encoding behavior profiles.
    3. Deterministic, vectorized NumPy/Pandas computations for speed & reproducibility.
    4. Safe defaults and bounded scaling (tanh, clipping) to stabilize downstream learning.

    EXTENSIBILITY
    -------------
    Hooks remain for potential future optional model-based refinement, but any reintroduction must
    maintain a strict separation between driver inputs and derived neutral features.
    """
    
    def __init__(self):
        """
        Initialize the tire grip analysis service
        """
        self.telemetry_features = TelemetryFeatures()
        # Heuristic-only mode (all ML removed). Kept flag for interface stability.
        self.heuristic_only = True
        # Whether to exclude instantaneous driver-exploitation features (reduces action leakage into context inputs)
        self.exclude_instantaneous_exploitation = True
        # Features considered high-frequency reflections of current control exploitation; better as reasoning targets or excluded.
        self.INSTANT_EXPLOIT_FEATURES = [
            'friction_circle_utilization',
            'longitudinal_weight_transfer',
            'lateral_weight_transfer',
            'dynamic_weight_distribution',
            'tire_saturation_level'
        ]
        # Allowlist of neutral, vehicle-dynamic oriented features safe for generalized heuristic enrichment.
        self.ALLOWED_GENERAL_FEATURES = [
            'Physics_speed_kmh', 'Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z',
            'Physics_slip_angle_front_left', 'Physics_slip_angle_front_right',
            'Physics_slip_angle_rear_left', 'Physics_slip_angle_rear_right',
            'Physics_slip_ratio_front_left', 'Physics_slip_ratio_front_right',
            'Physics_slip_ratio_rear_left', 'Physics_slip_ratio_rear_right',
            'Physics_wheel_slip_front_left', 'Physics_wheel_slip_front_right',
            'Physics_wheel_slip_rear_left', 'Physics_wheel_slip_rear_right',
            'Physics_suspension_travel_front_left', 'Physics_suspension_travel_front_right',
            'Physics_suspension_travel_rear_left', 'Physics_suspension_travel_rear_right',
            'Physics_tyre_core_temp_front_left', 'Physics_tyre_core_temp_front_right',
            'Physics_tyre_core_temp_rear_left', 'Physics_tyre_core_temp_rear_right',
            'Physics_brake_temp_front_left', 'Physics_brake_temp_front_right',
            'Physics_brake_temp_rear_left', 'Physics_brake_temp_rear_right',
            'Physics_wheel_pressure_front_left', 'Physics_wheel_pressure_front_right',
            'Physics_wheel_pressure_rear_left', 'Physics_wheel_pressure_rear_right'
        ]
        # Backend service integration
        self.backend_service = backend_service
        # Track runtime excluded features (if any future dynamic filtering is added)
        self._excluded_runtime_features = []  # type: List[str]
        # Cache last serialized artifact
        self._last_serialized = None  # type: Optional[Dict[str, Any]]

    # ============================= Performance Optimized (Vectorized) Helpers =============================
    def _vectorized_basic_grip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized computation of basic grip features for an entire dataframe.

        This replaces the previous per-row Python loop in extract_tire_grip_features where
        _calculate_basic_grip_features was invoked for every record. For thousands of rows this
        results in large speedups by leveraging NumPy operations.

        Args:
            df: Telemetry dataframe (already cleaned)

        Returns:
            DataFrame with columns matching the keys previously returned by _calculate_basic_grip_features
        """
        if df.empty:
            return pd.DataFrame()

        # Helper to safely get numeric series (fallback to zeros if missing)
        def col(name: str, default: float = 0.0):
            return pd.to_numeric(df.get(name, default), errors='coerce').fillna(default)

        g_x = col('Physics_g_force_x')
        g_y = col('Physics_g_force_y')

        # Friction circle utilization
        total_g = (g_x**2 + g_y**2).pow(0.5)
        friction_circle_utilization = (total_g / 2.5).clip(upper=1.0)

        # Weight transfer (tanh scaled)
        longitudinal_weight_transfer = (g_x / 1.5).clip(-10, 10).apply(np.tanh)  # clamp extreme outliers
        lateral_weight_transfer = (g_y / 1.5).clip(-10, 10).apply(np.tanh)
        dynamic_weight_distribution = (total_g / 2.0).clip(upper=1.0)

        # Tire temps
        t_fl = col('Physics_tyre_core_temp_front_left', 70)
        t_fr = col('Physics_tyre_core_temp_front_right', 70)
        t_rl = col('Physics_tyre_core_temp_rear_left', 70)
        t_rr = col('Physics_tyre_core_temp_rear_right', 70)
        avg_temp = (t_fl + t_fr + t_rl + t_rr) / 4.0
        optimal_min, optimal_max = 80.0, 110.0
        optimal_grip_window = pd.Series(np.where(
            (avg_temp >= optimal_min) & (avg_temp <= optimal_max),
            1.0,
            np.where(
                avg_temp < optimal_min,
                (avg_temp / optimal_min).clip(lower=0.0),
                (1 - ((avg_temp - optimal_max) / 50.0)).clip(lower=0.0)
            )
        ), index=df.index)

        # Slip angles & ratios (use abs mean) front for angle, all for ratios
        sa_fl = col('Physics_slip_angle_front_left')
        sa_fr = col('Physics_slip_angle_front_right')
        sa_rl = col('Physics_slip_angle_rear_left')
        sa_rr = col('Physics_slip_angle_rear_right')
        sr_fl = col('Physics_slip_ratio_front_left')
        sr_fr = col('Physics_slip_ratio_front_right')
        sr_rl = col('Physics_slip_ratio_rear_left')
        sr_rr = col('Physics_slip_ratio_rear_right')

        avg_slip_angle = (sa_fl.abs() + sa_fr.abs() + sa_rl.abs() + sa_rr.abs()) / 4.0
        avg_slip_ratio = (sr_fl.abs() + sr_fr.abs() + sr_rl.abs() + sr_rr.abs()) / 4.0
        optimal_slip_angle = 6.5
        optimal_slip_ratio = 0.125
        slip_angle_efficiency = (1 - (avg_slip_angle - optimal_slip_angle).abs() / optimal_slip_angle).clip(lower=0.0)
        slip_ratio_efficiency = (1 - (avg_slip_ratio - optimal_slip_ratio).abs() / optimal_slip_ratio).clip(lower=0.0)

        temp_factor = optimal_grip_window
        slip_factor = (slip_angle_efficiency + slip_ratio_efficiency) / 2.0
        utilization_penalty = (1 - (friction_circle_utilization * 0.3)).clip(lower=0.0)
        overall_tire_grip = (temp_factor * slip_factor * utilization_penalty).fillna(0.0)
        tire_saturation_level = (friction_circle_utilization * 1.2).clip(upper=1.0)

        basic_df = pd.DataFrame({
            'friction_circle_utilization': friction_circle_utilization,
            'longitudinal_weight_transfer': longitudinal_weight_transfer,
            'lateral_weight_transfer': lateral_weight_transfer,
            'dynamic_weight_distribution': dynamic_weight_distribution,
            'optimal_grip_window': optimal_grip_window,
            'slip_angle_efficiency': slip_angle_efficiency,
            'slip_ratio_efficiency': slip_ratio_efficiency,
            'overall_tire_grip': overall_tire_grip,
            'tire_saturation_level': tire_saturation_level
        })

        if self.exclude_instantaneous_exploitation:
            drop_cols = [c for c in self.INSTANT_EXPLOIT_FEATURES if c in basic_df.columns]
            if drop_cols:
                basic_df.drop(columns=drop_cols, inplace=True)

        # Also include prefixed copies for comparison (optional consumers can ignore)
        for col_name in list(basic_df.columns):
            basic_df[f"basic_{col_name}"] = basic_df[col_name]

        return basic_df

    def _calculate_friction_circle_utilization(self, row: pd.Series) -> float:
        """
        Calculate friction circle utilization based on G-forces
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Friction circle utilization (0-1)
        """
        try:
            # Get G-forces (assuming peak performance around 2-3G)
            g_x = _safe_float(row.get('Physics_g_force_x', 0))
            g_y = _safe_float(row.get('Physics_g_force_y', 0))
            
            # Calculate total G-force magnitude
            total_g = math.sqrt(g_x**2 + g_y**2)
            
            # Normalize to typical racing car limits (assume ~2.5G peak)
            max_g = 2.5
            utilization = min(total_g / max_g, 1.0)
            
            return utilization
            
        except Exception:
            return 0.0

    def _calculate_weight_transfer(self, row: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate weight transfer based on G-forces and suspension travel
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Tuple of (longitudinal_transfer, lateral_transfer, dynamic_distribution)
        """
        try:
            # Longitudinal weight transfer (braking/acceleration)
            g_x = _safe_float(row.get('Physics_g_force_x', 0))
            long_transfer = np.tanh(g_x / 1.5)  # Normalize and bound to [-1, 1]
            
            # Lateral weight transfer (cornering)
            g_y = _safe_float(row.get('Physics_g_force_y', 0))
            lat_transfer = np.tanh(g_y / 1.5)  # Normalize and bound to [-1, 1]
            
            # Dynamic weight distribution (how much weight is shifting)
            total_transfer = math.sqrt(g_x**2 + g_y**2)
            dynamic_dist = min(total_transfer / 2.0, 1.0)
            
            return long_transfer, lat_transfer, dynamic_dist
            
        except Exception:
            return 0.0, 0.0, 0.0

    def _estimate_tire_temperatures_optimal_window(self, row: pd.Series) -> float:
        """
        Estimate if tires are in optimal temperature window
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Optimal window factor (0-1)
        """
        try:
            # Get tire core temperatures
            temp_fl = _safe_float(row.get('Physics_tyre_core_temp_front_left', 70))
            temp_fr = _safe_float(row.get('Physics_tyre_core_temp_front_right', 70))
            temp_rl = _safe_float(row.get('Physics_tyre_core_temp_rear_left', 70))
            temp_rr = _safe_float(row.get('Physics_tyre_core_temp_rear_right', 70))
            
            avg_temp = (temp_fl + temp_fr + temp_rl + temp_rr) / 4
            
            # Optimal temperature window (typically 80-110°C for racing tires)
            optimal_min = 80
            optimal_max = 110
            
            if optimal_min <= avg_temp <= optimal_max:
                # In optimal window
                window_factor = 1.0
            elif avg_temp < optimal_min:
                # Too cold
                window_factor = max(0, avg_temp / optimal_min)
            else:
                # Too hot
                excess_heat = avg_temp - optimal_max
                window_factor = max(0, 1 - (excess_heat / 50))  # Degrade over 50°C excess
            
            return window_factor
            
        except Exception:
            return 0.7  # Default moderate value

    def _calculate_slip_efficiency(self, row: pd.Series) -> Tuple[float, float]:
        """
        Calculate slip angle and slip ratio efficiency
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Tuple of (slip_angle_efficiency, slip_ratio_efficiency)
        """
        try:
            # Get slip angles (front axle average)
            slip_angle_fl = _safe_float(row.get('Physics_slip_angle_front_left', 0))
            slip_angle_fr = _safe_float(row.get('Physics_slip_angle_front_right', 0))
            avg_slip_angle = abs((slip_angle_fl + slip_angle_fr) / 2)
            
            # Get slip ratios (all wheels average)
            slip_ratio_fl = _safe_float(row.get('Physics_slip_ratio_front_left', 0))
            slip_ratio_fr = _safe_float(row.get('Physics_slip_ratio_front_right', 0))
            slip_ratio_rl = _safe_float(row.get('Physics_slip_ratio_rear_left', 0))
            slip_ratio_rr = _safe_float(row.get('Physics_slip_ratio_rear_right', 0))
            avg_slip_ratio = abs((slip_ratio_fl + slip_ratio_fr + slip_ratio_rl + slip_ratio_rr) / 4)
            
            # Calculate efficiency (optimal slip angles around 5-8 degrees, slip ratios around 10-15%)
            optimal_slip_angle = 6.5  # degrees
            optimal_slip_ratio = 0.125  # 12.5%
            
            # Efficiency curves (inverted bell curves centered on optimal values)
            slip_angle_eff = max(0, 1 - abs(avg_slip_angle - optimal_slip_angle) / optimal_slip_angle)
            slip_ratio_eff = max(0, 1 - abs(avg_slip_ratio - optimal_slip_ratio) / optimal_slip_ratio)
            
            return slip_angle_eff, slip_ratio_eff
            
        except Exception:
            return 0.5, 0.5  # Default moderate values

    async def train_tire_grip_model(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deprecated: ML training removed. Returns informative message."""
        return {
            "skipped": True,
            "reason": "ml_removed",
            "message": "Training disabled. Service operates in heuristic-only mode.",
            "models_trained": 0
        }

    async def extract_tire_grip_features(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract tire grip features from cleaned telemetry data using saved trained models
        
        Args:
            telemetry_data: List of cleaned telemetry records to predict on
            
        Returns:
            Enhanced telemetry data with tire grip features
        """
        import time
        start_time = time.time()
        if not telemetry_data:
            return telemetry_data

        # Heuristic-only: no ML models
        models_available = False

        total_records = len(telemetry_data)
        print(f"[INFO] Extracting tire grip features for {total_records} records (models_available={models_available})")

        # DataFrame of incoming telemetry
        df = pd.DataFrame(telemetry_data)

        # Vectorized basic feature computation
        basic_df = self._vectorized_basic_grip_features(df)
        if basic_df.empty:
            print("[WARNING] Vectorized basic feature frame is empty; returning original telemetry data")
            return telemetry_data

        # Target columns potentially produced by ML models (to manage overlap)
        target_variables = [
            'friction_circle_utilization',
            'longitudinal_weight_transfer', 
            'lateral_weight_transfer',
            'dynamic_weight_distribution',
            'optimal_grip_window',
            'slip_angle_efficiency',
            'slip_ratio_efficiency'
        ]

        # Merge is just heuristic in this mode
        merged = basic_df.copy()
        for t in target_variables:
            if t not in merged.columns and f"basic_{t}" in merged.columns:
                merged[t] = merged[f"basic_{t}"]

        # Build enhanced telemetry list
        enhanced_data: List[Dict[str, Any]] = []
        for idx, original_record in enumerate(telemetry_data):
            record_features = merged.iloc[idx].to_dict()
            enhanced_record = original_record.copy()
            enhanced_record.update(record_features)
            enhanced_data.append(enhanced_record)

        elapsed = time.time() - start_time
        heuristic_cols = [c for c in merged.columns if c.startswith('basic_')]
        ml_cols = 0
        removed = self.INSTANT_EXPLOIT_FEATURES if self.exclude_instantaneous_exploitation else []
        print(f"[INFO] Tire grip feature extraction completed in {elapsed:.3f}s | records={total_records} | per_record={elapsed/total_records:.6f}s | heuristic_cols={len(heuristic_cols)} | ml_cols={ml_cols} | removed_instant={removed}")
        return enhanced_data
    # ============================= Compatibility / Summary =============================
    def get_mode_configuration(self) -> Dict[str, Any]:
        return {"heuristic_only": self.heuristic_only}

    def _calculate_basic_grip_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate basic grip features without ML models
        
        Args:
            row: Telemetry data row
            
        Returns:
            Dictionary of basic grip features
        """
        features = {}
        
        try:
            # Basic friction circle utilization
            features['friction_circle_utilization'] = self._calculate_friction_circle_utilization(row)
            
            # Weight transfer analysis
            long_transfer, lat_transfer, dynamic_dist = self._calculate_weight_transfer(row)
            features['longitudinal_weight_transfer'] = long_transfer
            features['lateral_weight_transfer'] = lat_transfer
            features['dynamic_weight_distribution'] = dynamic_dist
            
            # Tire performance metrics
            features['optimal_grip_window'] = self._estimate_tire_temperatures_optimal_window(row)
            
            slip_angle_eff, slip_ratio_eff = self._calculate_slip_efficiency(row)
            features['slip_angle_efficiency'] = slip_angle_eff
            features['slip_ratio_efficiency'] = slip_ratio_eff
            
            # Overall tire grip estimate (simple heuristic)
            temp_factor = features['optimal_grip_window']
            slip_factor = (slip_angle_eff + slip_ratio_eff) / 2
            utilization_penalty = 1 - (features['friction_circle_utilization'] * 0.3)  # Slight penalty for over-utilization
            
            features['overall_tire_grip'] = temp_factor * slip_factor * utilization_penalty
            
            # Tire saturation level (how close to limits)
            features['tire_saturation_level'] = min(features['friction_circle_utilization'] * 1.2, 1.0)
            
        except Exception as e:
            print(f"[WARNING] Error calculating basic grip features: {str(e)}")
            # Provide default values
            default_features = [
                'friction_circle_utilization', 'longitudinal_weight_transfer', 'lateral_weight_transfer',
                'dynamic_weight_distribution', 'optimal_grip_window', 'slip_angle_efficiency',
                'slip_ratio_efficiency', 'overall_tire_grip', 'tire_saturation_level'
            ]
            for feature_name in default_features:
                features[feature_name] = 0.5  # Default moderate value
        
        return features

    def has_trained_models(self) -> bool:
        return False

    def get_available_model_targets(self) -> List[str]:
        return []

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "model_type": "tire_grip_analysis",
            "mode": "heuristic_only",
            "cached_models": 0,
            "available_targets": [],
            "notes": "ML training removed; outputs are deterministic physics-derived features."
        }

    # --------------------- Serialization ---------------------
    def serialize_model(self, track_name: str, car_name: str, generated_feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Produce a JSON-safe representation of the heuristic configuration.

        Since this service is heuristic-only, 'serialization' means persisting
        configuration flags and the list of produced feature names so inference
        environments can validate presence / regenerate deterministically.
        """
        artifact = {
            'model_type': 'tire_grip_analysis',
            'track_name': track_name,
            'car_name': car_name,
            'heuristic_only': self.heuristic_only,
            'exclude_instantaneous_exploitation': self.exclude_instantaneous_exploitation,
            'instant_exploitation_features': list(self.INSTANT_EXPLOIT_FEATURES),
            'allowed_general_features': list(self.ALLOWED_GENERAL_FEATURES),
            'generated_feature_names': generated_feature_names or [],
            'excluded_runtime_features': list(self._excluded_runtime_features),
            'serialized_timestamp': datetime.now().isoformat()
        }
        self._last_serialized = artifact
        return artifact

    @classmethod
    def deserialize_model(cls, payload: Dict[str, Any]) -> 'TireGripAnalysisService':
        inst = cls()
        inst.exclude_instantaneous_exploitation = payload.get('exclude_instantaneous_exploitation', True)
        # Overwrite lists only if present
        if 'instant_exploitation_features' in payload:
            inst.INSTANT_EXPLOIT_FEATURES = payload['instant_exploitation_features']
        if 'allowed_general_features' in payload:
            inst.ALLOWED_GENERAL_FEATURES = payload['allowed_general_features']
        inst._excluded_runtime_features = payload.get('excluded_runtime_features', [])
        return inst


# Create singleton instance for import
tire_grip_analysis_service = TireGripAnalysisService()

if __name__ == "__main__":
    print("TireGripAnalysisService initialized. Ready for tire grip and friction circle analysis!")
