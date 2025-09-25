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
from enum import Enum
# (Removed direct SciPy dependencies to keep environment minimal.)

# Import backend service and models
from .backend_service import backend_service
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor, _safe_float

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class TireGripFeatureCatalog:
    """Canonical tire-grip feature names for downstream models.

    Keep this list in sync with TireGripAnalysisService outputs.
    """
    class ContextFeature(str, Enum):
        """Authoritative context feature keys for tire grip analysis."""
        LONGITUDINAL_WEIGHT_TRANSFER = 'longitudinal_weight_transfer'
        LATERAL_WEIGHT_TRANSFER = 'lateral_weight_transfer'
        DYNAMIC_WEIGHT_DISTRIBUTION = 'dynamic_weight_distribution'
        OPTIMAL_GRIP_WINDOW = 'optimal_grip_window'
        TURNING_GRIP_UTILIZATION = 'turning_grip_utilization' # Turning grip utilization: higher values indicate more grip being used


    # Derived compatibility lists – keep for consumers expecting plain strings
    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]

    
class TireGripAnalysisService:
    """Tire Grip & Friction Circle Analysis Service (Driver-Behavior Agnostic)

    PURPOSE
    -------
    Provide purely physics-derived, driver-behavior neutral enrichment of telemetry data with
    tire grip utilization, friction circle occupancy, weight transfer, slip efficiency, and
    related indicators. transformer models use the output features to reflect positive or negative effects that may arise from certain physics constraints.
    Examples include high slip angles, excessive braking, or understeer/oversteer conditions.
    These enriched features are intended to process training telemetry data and used during predicting of transformer models,
    feed downstream generalized models (e.g., transformers for imitation / reasoning) without leaking individual driver
    control habits (throttle modulation, braking aggressiveness, steering style).

    DESIGN PRINCIPLES
    -----------------
    1. Only vehicle dynamics & tire state signals are used; control inputs (brake, gas, steer, gear)
       are excluded to avoid encoding behavior profiles.
    2. Safe defaults and bounded scaling (tanh, clipping) to stabilize downstream learning.
    3. Modular design to allow easy updates as new physics insights or models become available.
    4. Clear separation of context features (for encoder input) vs. reasoning/target
    5. Output features must help transformer model to understand car physics and tire dynamics.
    """
    
    def __init__(self):
        # Store robust, data-derived scaling statistics computed during "training"
        self.stats_: Dict[str, Any] = {}
        self._trained: bool = False
        self.feature_catalog = TireGripFeatureCatalog

    async def train_tire_grip_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """"Train" tire grip analysis by computing robust, data-derived scalers.

        Note: No hard-coded physics constants are used. All thresholds and scales
        are derived from the provided dataset (percentiles, IQR, MAD). This keeps
        calculations deterministic but lets downstream ML learn any implicit values.

        Args:
            training_data: List of telemetry dicts

        Returns:
            dict with success flag and computed stats metadata
        """
        df = self._prepare_dataframe(training_data)
        self.stats_ = self._compute_robust_stats(df)
        self._trained = True
        return {"success": True, "stats_keys": list(self.stats_.keys())}

    async def extract_tire_grip_features(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Compute tire grip and friction circle features from telemetry.

        Uses data-derived statistics computed in training if available; otherwise,
        derives them from the provided data on the fly. Returns per-record feature
        dicts that can be merged as contextual inputs for other models.

        Args:
            telemetry_data: List of telemetry dicts

        Returns:
            List[Dict[str, float]]: One dict per input record with feature values
        """
        df = self._prepare_dataframe(telemetry_data)
        stats = self.stats_ if self._trained else self._compute_robust_stats(df)

        # Compute features vectorized, avoiding hard-coded constants
        features_df = self._compute_features(df, stats)
        # Ensure output ordering matches input length
        result: List[Dict[str, float]] = features_df.to_dict(orient="records")
        return result

    # ------------------------ Internal helpers ------------------------
    def _prepare_dataframe(self, telemetry_list: List[Dict[str, Any]]) -> pd.DataFrame:
        df_raw = pd.DataFrame(telemetry_list)
        try:
            processor = FeatureProcessor(df_raw)
            df = processor.general_cleaning_for_analysis()
        except Exception:
            df = df_raw.fillna(0)
        # Ensure required numeric columns exist
        needed_cols = [
            'Physics_g_force_x', 'Physics_g_force_y',
            'Physics_slip_angle_front_left', 'Physics_slip_angle_front_right',
            'Physics_slip_angle_rear_left', 'Physics_slip_angle_rear_right',
            'Physics_slip_ratio_front_left', 'Physics_slip_ratio_front_right',
            'Physics_slip_ratio_rear_left', 'Physics_slip_ratio_rear_right',
            'Physics_tyre_core_temp_front_left', 'Physics_tyre_core_temp_front_right',
            'Physics_tyre_core_temp_rear_left', 'Physics_tyre_core_temp_rear_right',
            'Physics_pitch', 'Physics_roll'
        ]
        for c in needed_cols:
            if c not in df.columns:
                df[c] = 0.0
        # Coerce numeric
        df[needed_cols] = df[needed_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return df

    def _compute_robust_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        def pct(s: pd.Series, p: float) -> float:
            return float(np.nanpercentile(s.to_numpy(dtype=float), p))

        stats: Dict[str, Any] = {}
        gx = df['Physics_g_force_x'].abs()
        gy = df['Physics_g_force_y'].abs()
        gmag = np.sqrt(df['Physics_g_force_x']**2 + df['Physics_g_force_y']**2)
        slip_angles = pd.concat([
            df['Physics_slip_angle_front_left'].abs(),
            df['Physics_slip_angle_front_right'].abs(),
            df['Physics_slip_angle_rear_left'].abs(),
            df['Physics_slip_angle_rear_right'].abs()
        ], axis=0)
        slip_ratios = pd.concat([
            df['Physics_slip_ratio_front_left'].abs(),
            df['Physics_slip_ratio_front_right'].abs(),
            df['Physics_slip_ratio_rear_left'].abs(),
            df['Physics_slip_ratio_rear_right'].abs()
        ], axis=0)
        temps = pd.concat([
            df['Physics_tyre_core_temp_front_left'],
            df['Physics_tyre_core_temp_front_right'],
            df['Physics_tyre_core_temp_rear_left'],
            df['Physics_tyre_core_temp_rear_right']
        ], axis=0)

        stats['gmag_p95'] = max(pct(gmag, 95), 1e-6)
        stats['gx_p90'] = max(pct(gx, 90), 1e-6)
        stats['gy_p90'] = max(pct(gy, 90), 1e-6)

        stats['slip_angle_med'] = pct(slip_angles, 50)
        stats['slip_angle_iqr'] = max(pct(slip_angles, 75) - pct(slip_angles, 25), 1e-6)
        stats['slip_ratio_med'] = pct(slip_ratios, 50)
        stats['slip_ratio_iqr'] = max(pct(slip_ratios, 75) - pct(slip_ratios, 25), 1e-6)

        stats['temp_med'] = pct(temps, 50)
        stats['temp_iqr'] = max(pct(temps, 75) - pct(temps, 25), 1e-6)

        # Statistics for turning grip utilization
        # Use lateral g-force and slip angles to estimate maximum grip capability
        combined_slip = pd.concat([slip_angles, slip_ratios], axis=0)
        stats['combined_slip_p85'] = max(pct(combined_slip, 85), 1e-6)
        stats['lateral_g_max'] = max(pct(gy, 95), 1e-6)

        return stats

    def _bounded(self, x: pd.Series | np.ndarray, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
        return pd.Series(np.clip(x, lo, hi))

    def _tanh_scale(self, x: pd.Series | np.ndarray) -> pd.Series:
        return pd.Series(np.tanh(x))

    def _gaussian_efficiency(self, x: pd.Series, center: float, scale: float) -> pd.Series:
        # Data-derived center/scale; produce efficiency in [0,1]
        z = (x - center) / (scale if scale > 0 else 1e-6)
        return pd.Series(np.exp(-0.5 * np.square(z)))

    def _compute_features(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        gx = df['Physics_g_force_x']
        gy = df['Physics_g_force_y']
        gmag = np.sqrt(gx**2 + gy**2)

        # Friction circle utilization: normalized by robust p95
        fr_util = (gmag / stats['gmag_p95']).clip(0.0, 1.5)
        fr_util = self._bounded(fr_util, 0.0, 1.0)

        # Weight transfer proxies using scaled g-forces (behavior-agnostic)
        long_wt = self._tanh_scale((gx.abs() / stats['gx_p90']).fillna(0.0))
        lat_wt = self._tanh_scale((gy.abs() / stats['gy_p90']).fillna(0.0))

        # Dynamic weight distribution index: front<->rear shift via pitch and longitudinal g
        # Data-derived scaling by gx_p90; bounded to [-1,1] then remap to [0,1]
        pitch = df['Physics_pitch']
        dw_raw = np.tanh((gx / stats['gx_p90']).fillna(0.0) + 0.5 * np.tanh(pitch))
        dyn_wdist = (dw_raw + 1.0) * 0.5

        # Slip efficiencies (1 near typical operating center, lower when far)
        sa_fl = df['Physics_slip_angle_front_left'].abs()
        sa_fr = df['Physics_slip_angle_front_right'].abs()
        sa_rl = df['Physics_slip_angle_rear_left'].abs()
        sa_rr = df['Physics_slip_angle_rear_right'].abs()
        sa_all = (sa_fl + sa_fr + sa_rl + sa_rr) / 4.0
        sa_eff = self._gaussian_efficiency(sa_all, stats['slip_angle_med'], stats['slip_angle_iqr'])

        sr_fl = df['Physics_slip_ratio_front_left'].abs()
        sr_fr = df['Physics_slip_ratio_front_right'].abs()
        sr_rl = df['Physics_slip_ratio_rear_left'].abs()
        sr_rr = df['Physics_slip_ratio_rear_right'].abs()
        sr_all = (sr_fl + sr_fr + sr_rl + sr_rr) / 4.0
        sr_eff = self._gaussian_efficiency(sr_all, stats['slip_ratio_med'], stats['slip_ratio_iqr'])

        # Optimal grip window: temperatures near data median score higher
        t_fl = df['Physics_tyre_core_temp_front_left']
        t_fr = df['Physics_tyre_core_temp_front_right']
        t_rl = df['Physics_tyre_core_temp_rear_left']
        t_rr = df['Physics_tyre_core_temp_rear_right']
        t_avg = (t_fl + t_fr + t_rl + t_rr) / 4.0
        grip_window = self._gaussian_efficiency(t_avg, stats['temp_med'], stats['temp_iqr'])

        # Turning grip utilization: estimate grip usage during cornering
        # Combines lateral g-force demand with slip angle efficiency
        # 0 = no utilization, 1 = max grip, >1 = slipping beyond max grip
        lateral_demand = gy.abs() / stats['lateral_g_max']  # Normalized lateral demand
        slip_factor = sa_all / stats['combined_slip_p85']   # Normalized slip usage
        
        # Turning grip utilization: higher values indicate more grip being used
        # When slip increases faster than lateral g, it indicates exceeding optimal grip
        turning_grip_util = lateral_demand + 0.5 * slip_factor
        turning_grip_util = turning_grip_util.clip(0.0, 2.0)  # Allow values >1 to indicate over-utilization

        # Overall tire grip: combine friction utilization and efficiencies
        overall = self._bounded(0.5 * fr_util + 0.25 * sa_eff + 0.25 * sr_eff, 0.0, 1.0)

        # Tire saturation: percentile rank of friction utilization (data-derived)
        # Avoid hard thresholds by using ranks within the batch
        ranks = fr_util.rank(method='average', pct=True)
        saturation = self._bounded(ranks, 0.0, 1.0)

        out = pd.DataFrame({
            self.feature_catalog.ContextFeature.LONGITUDINAL_WEIGHT_TRANSFER.value: long_wt.values,
            self.feature_catalog.ContextFeature.LATERAL_WEIGHT_TRANSFER.value: lat_wt.values,
            self.feature_catalog.ContextFeature.DYNAMIC_WEIGHT_DISTRIBUTION.value: dyn_wdist.values,
            self.feature_catalog.ContextFeature.OPTIMAL_GRIP_WINDOW.value: grip_window.values,
            self.feature_catalog.ContextFeature.TURNING_GRIP_UTILIZATION.value: turning_grip_util.values,
        })
        # Ensure float dtype and fill NaNs
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0).astype(float)
        return out


    def serialize_tire_grip_model(self) -> Dict[str, Any]:
        """Return JSON-safe representation of computed stats which can be stored and reloaded later for inference.
        
        Returns:
            Dict containing the model state including stats and training status
        """
        if not self._trained:
            raise ValueError("Cannot serialize tire grip model: model has not been trained yet")
        
        # Ensure all stats values are JSON-serializable (convert numpy types to Python types)
        serializable_stats = {}
        for key, value in self.stats_.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_stats[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_stats[key] = value.tolist()
            else:
                serializable_stats[key] = value
        
        return {
            "stats": serializable_stats,
            "trained": self._trained,
            "version": "1.0",  # For future compatibility
            "feature_count": len(self.feature_catalog.CONTEXT_FEATURES)
        }
        
    def deserialize_tire_grip_model(self, model_data: Dict[str, Any]) -> 'TireGripAnalysisService':
        """Load previously computed stats from JSON-safe dict and restore class instance state.
        
        Args:
            model_data: Dictionary containing serialized model state from serialize_tire_grip_model()
            
        Returns:
            Self (TireGripAnalysisService): The current instance with restored model state
            
        Raises:
            ValueError: If model_data is invalid or incompatible
            RuntimeError: If deserialization fails
        """
        try:
            if not isinstance(model_data, dict):
                raise ValueError("Model data must be a dictionary")
            
            if "stats" not in model_data:
                raise ValueError("Model data missing required 'stats' field")
                
            if "trained" not in model_data:
                raise ValueError("Model data missing required 'trained' field")
            
            # Validate version compatibility if present
            if "version" in model_data:
                version = model_data["version"]
                if version != "1.0":
                    print(f"[WARNING] Loading tire grip model version {version}, expected 1.0. Compatibility not guaranteed.")
            
            # Validate feature count if present
            if "feature_count" in model_data:
                expected_count = len(self.feature_catalog.CONTEXT_FEATURES)
                actual_count = model_data["feature_count"]
                if actual_count != expected_count:
                    print(f"[WARNING] Tire grip model was trained with {actual_count} features, current version expects {expected_count}")
            
            # Restore the model state
            self.stats_ = model_data["stats"].copy()
            self._trained = bool(model_data["trained"])
            
            # Validate required statistics are present
            required_stats = [
                'gmag_p95', 'gx_p90', 'gy_p90',
                'slip_angle_med', 'slip_angle_iqr',
                'slip_ratio_med', 'slip_ratio_iqr', 
                'temp_med', 'temp_iqr',
                'combined_slip_p85', 'lateral_g_max'
            ]
            
            missing_stats = [stat for stat in required_stats if stat not in self.stats_]
            if missing_stats:
                raise ValueError(f"Model data missing required statistics: {missing_stats}")

            print(f"[INFO] Successfully deserialized tire grip analysis model")
            print(f"[INFO] - Model trained: {self._trained}")
            print(f"[INFO] - Statistics loaded: {len(self.stats_)} stats")
            print(f"[INFO] - Features available: {len(self.feature_catalog.CONTEXT_FEATURES)}")

            return self
            
        except Exception as e:
            error_msg = f"{__class__.__name__} Failed to deserialize tire grip analysis model: {str(e)}"
            raise RuntimeError(error_msg) from e

# Create singleton instance for import
tire_grip_analysis_service = TireGripAnalysisService()

if __name__ == "__main__":
    print("TireGripAnalysisService initialized. Ready for tire grip and friction circle analysis!")
