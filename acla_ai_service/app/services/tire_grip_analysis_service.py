
"""
Driver Push-to-Limit Analysis Service for Assetto Corsa Competizione

This refactored service focuses on answering a single question:

    "Given the driver's current control inputs, how close is the car operating to
    its physics-derived limits?"

The service learns a lightweight, data-derived model that maps the driver's control
inputs (throttle, brake, steering) to expected chassis load and slip behaviour. It
then blends that prediction with the actual dynamic response (g-forces and slip) to
produce a **0-1 push index** per telemetry sample:

    0.0  -> the car is being driven very conservatively
    1.0  -> the car is right at (or slightly exceeding) its learned limit

Outputs are suitable for enriching telemetry that downstream AI models consume.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Iterable, Iterator, AsyncIterator, Optional, Union
from enum import Enum
from collections.abc import AsyncIterator as AsyncIteratorABC, Iterable as IterableABC
from datetime import datetime
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

    
class TireGripAnalysisService:
    """Driver push-to-limit estimator.

    The new implementation intentionally focuses on a single contextual feature that captures
    how aggressively the driver is using the car at each moment. Training is fully data-driven:

    * Robust percentiles normalise throttle, brake and steering inputs.
    * A shallow, interpretable linear mapping is fitted (least squares) to predict the dynamic
      response (g-force & slip) from those inputs.
    * Runtime inference blends the predicted demand with the actual instantaneous response to
      produce a stable 0-1 index.

    The result is behaviour-aware but physics-grounded, highlighting moments when the driver is
    asking for — and receiving — the maximum performance from the chassis.
    """

    def __init__(self):
        # Store robust, data-derived scaling statistics computed during "training"
        self.stats_: Dict[str, Any] = {}
        self._trained: bool = False
        self.feature_catalog = TireGripFeatureCatalog

    async def train_tire_grip_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute data-derived scaling and control-response mapping.

        Training fits the lightweight regression model that converts driver control
        inputs into an expected dynamic response. All parameters are pulled from the
        telemetry itself (robust percentiles and least-squares fit) so no hard-coded
        physics constants are required.

        Args:
            training_data: List of telemetry dicts

        Returns:
            dict with success flag and computed stats metadata
        """
        df = self._prepare_dataframe(training_data)
        return self._train_from_dataframe(df)

    async def train_tire_grip_model_streaming(
        self,
        chunk_iterator: Union[
            Iterator[List[Dict[str, Any]]],
            Iterable[List[Dict[str, Any]]],
            AsyncIterator[List[Dict[str, Any]]]
        ],
        max_samples: int = 200_000,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Stream-aware training routine for large telemetry corpora.

        Processes telemetry chunks iteratively, maintaining a capped sample set to
        approximate robust statistics without loading every record into memory.

        Args:
            chunk_iterator: Iterator yielding telemetry chunks as list[dict] or DataFrame rows.
            max_samples: Upper bound for rows retained for percentile estimation.
            random_state: Optional RNG seed when down-sampling the retained sample.

        Returns:
            dict with success flag and computed stats metadata, including sampling info.
        """

        async def _iter_chunks():
            if isinstance(chunk_iterator, AsyncIteratorABC):
                async for chunk in chunk_iterator:
                    yield chunk
            elif isinstance(chunk_iterator, IterableABC):
                for chunk in chunk_iterator:
                    yield chunk
            else:
                raise TypeError("chunk_iterator must be iterable or async iterable")

        aggregated_df: Optional[pd.DataFrame] = None
        total_rows = 0

        async for chunk in _iter_chunks():
            if chunk is None:
                continue

            if isinstance(chunk, pd.DataFrame):
                chunk_records = chunk.to_dict(orient="records")
            else:
                chunk_records = chunk

            if not chunk_records:
                continue

            chunk_df = self._prepare_dataframe(chunk_records)
            if chunk_df.empty:
                continue

            total_rows += len(chunk_df)

            if aggregated_df is None:
                aggregated_df = chunk_df
            else:
                aggregated_df = pd.concat([aggregated_df, chunk_df], ignore_index=True)

            if max_samples > 0 and len(aggregated_df) > max_samples:
                aggregated_df = aggregated_df.sample(
                    n=max_samples,
                    random_state=random_state,
                    replace=False
                ).reset_index(drop=True)

        if aggregated_df is None or aggregated_df.empty:
            raise ValueError("No telemetry data received while streaming tire grip training")

        training_result = self._train_from_dataframe(aggregated_df, total_rows)
        training_result["retained_rows"] = len(aggregated_df)
        training_result["total_streamed_rows"] = total_rows
        training_result["sampling_capped"] = max_samples > 0 and total_rows > len(aggregated_df)

        print(
            f"[INFO] Tire grip streaming training complete: {total_rows} rows processed, "
            f"{len(aggregated_df)} rows retained for robust statistics"
        )

        return training_result

    def _train_from_dataframe(
        self,
        df: pd.DataFrame,
        total_rows_seen: Optional[int] = None
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            raise ValueError("Cannot train tire grip model on empty dataframe")

        self.stats_ = self._compute_robust_stats(df)
        self.stats_ = self._ensure_control_model(df, self.stats_)

        if total_rows_seen is not None:
            self.stats_["streaming_total_rows"] = int(total_rows_seen)

        self._trained = True
        return {
            "success": True,
            "stats_keys": list(self.stats_.keys()),
            "training_rows": int(len(df)),
            "total_rows_seen": int(total_rows_seen or len(df))
        }

    async def extract_tire_grip_features(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Compute the driver push-to-limit index for each telemetry sample.

        The method uses training statistics when available, otherwise it derives
        them on the fly from the current telemetry window. Returns one dictionary
        per record containing the single contextual feature required downstream.

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
            df = df_raw.fillna(0.0)
        # Ensure required numeric columns exist
        needed_cols = [
            'Physics_g_force_x', 'Physics_g_force_y',
            'Physics_slip_angle_front_left', 'Physics_slip_angle_front_right',
            'Physics_slip_angle_rear_left', 'Physics_slip_angle_rear_right',
            'Physics_slip_ratio_front_left', 'Physics_slip_ratio_front_right',
            'Physics_slip_ratio_rear_left', 'Physics_slip_ratio_rear_right',
            'Physics_gas', 'Physics_brake', 'Physics_steer_angle'
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

        gas = df['Physics_gas'].clip(lower=0.0)
        brake = df['Physics_brake'].clip(lower=0.0)
        steer_abs = df['Physics_steer_angle'].abs()

        slip_combo = pd.concat([slip_angles, slip_ratios], axis=0).clip(lower=0.0)

        stats['gmag_p99'] = max(pct(gmag, 99), 1e-6)
        stats['slip_combo_p95'] = max(pct(slip_combo, 95), 1e-6)

        stats['gas_p95'] = max(pct(gas, 95), 1e-6)
        stats['brake_p95'] = max(pct(brake, 95), 1e-6)
        stats['steer_abs_p95'] = max(pct(steer_abs, 95), 1e-6)

        # Typical response baseline useful for smoothing
        slip_angle_mean_row = df[[
            'Physics_slip_angle_front_left',
            'Physics_slip_angle_front_right',
            'Physics_slip_angle_rear_left',
            'Physics_slip_angle_rear_right'
        ]].abs().mean(axis=1)
        slip_ratio_mean_row = df[[
            'Physics_slip_ratio_front_left',
            'Physics_slip_ratio_front_right',
            'Physics_slip_ratio_rear_left',
            'Physics_slip_ratio_rear_right'
        ]].abs().mean(axis=1)
        slip_combo_row = 0.5 * (slip_angle_mean_row + slip_ratio_mean_row)
        response_ratio = np.maximum(
            (gmag / stats['gmag_p99']).clip(0.0, 2.5),
            (slip_combo_row / stats['slip_combo_p95']).clip(0.0, 2.5)
        )
        stats['response_median'] = float(np.nanmedian(response_ratio))
        stats['sample_size'] = int(len(df))

        return stats

    def _bounded(self, x: pd.Series | np.ndarray, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
        return pd.Series(np.clip(np.asarray(x, dtype=float), lo, hi))

    def _default_control_coeffs(self) -> np.ndarray:
        # [bias, gas, brake, steer, gas*brake, gas*steer, brake*steer]
        return np.array([0.05, 0.45, 0.45, 0.45, 0.1, 0.1, 0.1], dtype=float)

    def _control_feature_matrix(self, df: pd.DataFrame, stats: Dict[str, Any]) -> np.ndarray:
        gas_norm = (df['Physics_gas'] / stats['gas_p95']).clip(0.0, 2.5)
        brake_norm = (df['Physics_brake'] / stats['brake_p95']).clip(0.0, 2.5)
        steer_norm = (df['Physics_steer_angle'].abs() / stats['steer_abs_p95']).clip(0.0, 2.5)

        ones = np.ones(len(df))
        cross_gb = gas_norm * brake_norm
        cross_gs = gas_norm * steer_norm
        cross_bs = brake_norm * steer_norm

        features = np.column_stack([
            ones,
            gas_norm.to_numpy(dtype=float),
            brake_norm.to_numpy(dtype=float),
            steer_norm.to_numpy(dtype=float),
            cross_gb.to_numpy(dtype=float),
            cross_gs.to_numpy(dtype=float),
            cross_bs.to_numpy(dtype=float),
        ])
        return features

    def _response_target(self, df: pd.DataFrame, stats: Dict[str, Any]) -> np.ndarray:
        gmag = np.sqrt(df['Physics_g_force_x']**2 + df['Physics_g_force_y']**2)
        slip_angles = df[[
            'Physics_slip_angle_front_left',
            'Physics_slip_angle_front_right',
            'Physics_slip_angle_rear_left',
            'Physics_slip_angle_rear_right'
        ]].abs().mean(axis=1)
        slip_ratios = df[[
            'Physics_slip_ratio_front_left',
            'Physics_slip_ratio_front_right',
            'Physics_slip_ratio_rear_left',
            'Physics_slip_ratio_rear_right'
        ]].abs().mean(axis=1)

        slip_combo = 0.5 * (slip_angles + slip_ratios)

        g_norm = (gmag / stats['gmag_p99']).clip(0.0, 2.5)
        slip_norm = (slip_combo / stats['slip_combo_p95']).clip(0.0, 2.5)
        target = np.maximum(g_norm, slip_norm)
        return target.astype(float)

    def _fit_control_model(self, df: pd.DataFrame, stats: Dict[str, Any]) -> np.ndarray:
        X = self._control_feature_matrix(df, stats)
        y = self._response_target(df, stats)

        if len(df) < X.shape[1]:
            return self._default_control_coeffs()

        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = self._default_control_coeffs()

        # Guard against pathological coefficients
        coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(coeffs).all():
            coeffs = self._default_control_coeffs()
        return coeffs

    def _ensure_control_model(self, df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
        if 'control_model_coeffs' not in stats:
            coeffs = self._fit_control_model(df, stats)
            stats['control_model_coeffs'] = coeffs.tolist()
        return stats

    def _compute_features(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        stats = self._ensure_control_model(df, stats)
        coeffs = np.array(stats['control_model_coeffs'], dtype=float)

        X = self._control_feature_matrix(df, stats)
        predicted = self._bounded(X @ coeffs, 0.0, 2.0)

        actual = self._bounded(self._response_target(df, stats), 0.0, 2.0)

        smooth_baseline = stats.get('response_median', 0.1)
        blended = 0.6 * predicted + 0.35 * actual + 0.05 * smooth_baseline
        push_index = self._bounded(blended, 0.0, 1.0)

        out = pd.DataFrame({
            self.feature_catalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value: push_index.astype(float).to_numpy()
        })

        out = out.fillna(0.0)
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
            elif isinstance(value, list) and value and isinstance(value[0], (np.integer, np.floating)):
                serializable_stats[key] = [float(v) for v in value]
            else:
                serializable_stats[key] = value
        
        return {
            "stats": serializable_stats,
            "trained": self._trained,
            "version": "1.0",  # For future compatibility
            "feature_count": len(self.feature_catalog.CONTEXT_FEATURES),
            "serialized_timestamp": datetime.utcnow().isoformat()
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
                'gmag_p99', 'slip_combo_p95',
                'gas_p95', 'brake_p95', 'steer_abs_p95',
                'control_model_coeffs'
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
            error_msg = f"{self.__class__.__name__} failed to deserialize tire grip analysis model: {str(e)}"
            raise RuntimeError(error_msg) from e

# Create singleton instance for import
tire_grip_analysis_service = TireGripAnalysisService()

if __name__ == "__main__":
    print("TireGripAnalysisService initialized. Ready to estimate driver push-to-limit index!")
