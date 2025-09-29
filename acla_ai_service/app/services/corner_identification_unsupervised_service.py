"""
Simple Corner Identification Service for Track Learning

This service learns track corner layout from multiple laps and identifies
which corner and phase the driver is currently in.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from scipy.signal import savgol_filter
from enum import Enum


class CornerFeatureCatalog:
    """Feature catalog for corner identification output."""
    
    class ContextFeature(str, Enum):
        """Corner context feature keys for output."""
        CURRENT_CORNER_NUMBER = 'current_corner_number'
        CURRENT_PHASE = 'current_phase'
        IS_IN_CORNER = 'is_in_corner'
        CORNER_PHASE_PROGRESS = 'corner_phase_progress'

    # List of all context features
    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]


class CornerIdentificationUnsupervisedService:
    """
    Simple corner identification service that learns track layout and identifies current position.
    
    1. Learn track corner layout from multiple laps
    2. Identify which corner and phase driver is currently in
    """
    
    def __init__(self):
        """Initialize the corner identification service."""
        self.corner_patterns = []  # Store learned track map
        self.track_corner_profiles = {}
        
        # Corner detection parameters
        self.min_corner_duration = 30  # Minimum data points for a valid corner
        self.smoothing_window = 7  # Window size for smoothing telemetry data
        
        # Serialization cache
        self._last_serialized: Optional[Dict[str, Any]] = None

    def _normalize_steering(self, steering_series: pd.Series) -> pd.Series:
        """Normalize steering angle to -1..1 range."""
        s = steering_series.fillna(0.0)
        return s.clip(-1.0, 1.0)

    def _to_python(self, obj: Any) -> Any:
        """Convert numpy/pandas types to Python types for JSON serialization."""
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._to_python(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._to_python(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        return str(obj)

    def serialize_corner_identification_model(self, track_name: str, car_name: str) -> Dict[str, Any]:
        """Serialize the learned corner model for saving."""
        try:
            model_key = f"{track_name}_{car_name}"
            
            payload = {
                "model_version": "1.0",
                "track_name": track_name,
                "car_name": car_name,
                "model_key": model_key,
                "export_timestamp": datetime.now().isoformat(),
                "corner_patterns": self._to_python(self.corner_patterns),
                "track_corner_profiles": self._to_python(self.track_corner_profiles),
                "total_corners": len(self.corner_patterns),
                "feature_catalog": list(CornerFeatureCatalog.CONTEXT_FEATURES),
                "config": {
                    "min_corner_duration": self.min_corner_duration,
                    "smoothing_window": self.smoothing_window
                },
                "has_learned_patterns": len(self.corner_patterns) > 0
            }
            
            self._last_serialized = payload
            return payload
            
        except Exception as e:
            print(f"[ERROR] Failed to serialize model: {str(e)}")
            return {
                "model_version": "1.0",
                "track_name": track_name,
                "car_name": car_name,
                "error": str(e),
                "corner_patterns": [],
                "total_corners": 0
            }

    def deserialize_corner_identification_model(self, payload: Dict[str, Any]) -> 'CornerIdentificationUnsupervisedService':
        """Deserialize and restore the corner model."""
        try:
            print(f"[INFO] Deserializing corner identification model...")
            
            if not isinstance(payload, dict):
                raise ValueError("Payload must be a dictionary")
            
            if "error" in payload:
                print(f"[WARNING] Payload contains error: {payload['error']}")
                return self
            
            # Restore configuration
            if "config" in payload:
                config = payload["config"]
                self.min_corner_duration = config.get("min_corner_duration", self.min_corner_duration)
                self.smoothing_window = config.get("smoothing_window", self.smoothing_window)
            
            # Restore corner patterns
            self.corner_patterns = payload.get("corner_patterns", [])
            self.track_corner_profiles = payload.get("track_corner_profiles", {})
            
            # Cache the payload
            self._last_serialized = payload
            
            track_name = payload.get("track_name", "unknown")
            car_name = payload.get("car_name", "unknown")
            total_corners = len(self.corner_patterns)
            
            print(f"[INFO] Successfully deserialized model:")
            print(f"       - Track: {track_name}")
            print(f"       - Car: {car_name}")
            print(f"       - Total Corners: {total_corners}")
            
            return self
            
        except Exception as e:
            print(f"[ERROR] Failed to deserialize model: {str(e)}")
            self.corner_patterns = []
            self.track_corner_profiles = {}
            self._last_serialized = None
            return self
        
    async def learn_track_corner_patterns(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Learn track corner layout from multiple laps.
        
        Args:
            telemetry_data: Clean telemetry data from multiple laps
            
        Returns:
            Dictionary with track layout learning results
        """
        try:
            print(f"[INFO] Learning track layout from telemetry data")
            
            if not telemetry_data:
                return {
                    "success": False,
                    "error": "No telemetry data provided",
                    "track_map": []
                }
            
            df = pd.DataFrame(telemetry_data)
            
            # Split data into individual laps
            lap_slices = self._detect_lap_slices(df)
            if len(lap_slices) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 laps to learn track layout",
                    "track_map": []
                }
            
            print(f"[INFO] Found {len(lap_slices)} laps")
            
            # Learn corner locations from all laps
            track_map = self._learn_track_layout_from_laps(df, lap_slices)
            
            if not track_map:
                return {
                    "success": False,
                    "error": "No corners detected",
                    "track_map": []
                }
            
            # Store the learned track map
            self.corner_patterns = track_map
            
            print(f"[INFO] Learned {len(track_map)} corners")
            for i, corner in enumerate(track_map, 1):
                pos_start = corner['track_position_start']
                pos_end = corner['track_position_end']
                phases = len(corner['phases'])
                print(f"  Corner {i}: {pos_start:.3f} -> {pos_end:.3f} ({phases} phases)")
            
            return {
                "success": True,
                "total_corners": len(track_map),
                "track_map": track_map
            }
            
        except Exception as e:
            raise Exception(f"[ERROR] Failed to learn track: {str(e)}")
    
    async def extract_corner_features_for_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify which corner and phase the driver is currently in.

        Args:
            telemetry_data: List of telemetry records

        Returns:
            List of corner/phase identification for each record
        """
        try:
            if not self.corner_patterns:
                print("[ERROR] No track map available. Run learn_track_corner_patterns first.")
                return [{} for _ in range(len(telemetry_data))]
            
            df = pd.DataFrame(telemetry_data)
            position_series = self._build_position_series(df)
            
            feature_only_data = []
            
            for i in range(len(df)):
                current_position = position_series.iloc[i] if i < len(position_series) else 0.0
                corner_info = self._identify_current_corner_and_phase(current_position)
                
                # Use CornerFeatureCatalog for consistent feature naming
                CF = CornerFeatureCatalog.ContextFeature
                feature_record = {
                    CF.CURRENT_CORNER_NUMBER.value: corner_info['corner_number'],
                    CF.CURRENT_PHASE.value: corner_info['phase'],
                    CF.IS_IN_CORNER.value: 1.0 if corner_info['corner_number'] > 0 else 0.0,
                    CF.CORNER_PHASE_PROGRESS.value: corner_info['phase_progress']
                }
                
                feature_only_data.append(feature_record)

            print(f"[INFO] Identified corner/phase for {len(feature_only_data)} records")
            return feature_only_data
            
        except Exception as e:
            print(f"[ERROR] Failed to extract corner features: {str(e)}")
            return [{} for _ in range(len(telemetry_data))]

    def _learn_track_layout_from_laps(self, df: pd.DataFrame, lap_slices: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Learn track layout from multiple laps."""
        try:
            # Analyze each lap to find corners
            lap_corner_data = []
            for lap_idx, (lap_start, lap_end) in enumerate(lap_slices):
                df_lap = df.iloc[lap_start:lap_end].reset_index(drop=True)
                corner_segments = self._find_corners_in_lap(df_lap, lap_start, lap_end)
                lap_corner_data.append(corner_segments)
                print(f"  Lap {lap_idx + 1}: Found {len(corner_segments)} corners")
            
            # Find consistent corners across laps
            track_corners = self._find_consistent_corners(lap_corner_data)
            
            # Define phases for each corner
            track_map = []
            for i, corner_info in enumerate(track_corners, 1):
                phases = self._define_corner_phases(corner_info, df)
                
                track_map.append({
                    'corner_number': i,
                    'track_position_start': corner_info['position_start'],
                    'track_position_end': corner_info['position_end'],
                    'phases': phases
                })
            
            return track_map
            
        except Exception as e:
            print(f"[ERROR] Failed to learn track layout: {str(e)}")
            return []

    def _find_corners_in_lap(self, df_lap: pd.DataFrame, lap_start: int, lap_end: int) -> List[Dict[str, Any]]:
        """Find corners using proper driving pattern: brake → steer in → steer out → accelerate."""
        if len(df_lap) < self.min_corner_duration:
            return []
        
        # Required columns for corner pattern detection
        required_cols = ['Physics_brake', 'Physics_steer_angle', 'Physics_gas']
        if not all(col in df_lap.columns for col in required_cols):
            print(f"[WARNING] Missing required columns for corner detection: {required_cols}")
            return []
        
        # Smooth the signals
        brake = self._smooth_signal(df_lap['Physics_brake'].fillna(0.0))
        steering = self._normalize_steering(df_lap['Physics_steer_angle'])
        steering_abs = np.abs(steering)
        gas = self._smooth_signal(df_lap['Physics_gas'].fillna(0.0))
        
        # Find corner sequences using driving pattern
        corners = []
        i = 0
        
        while i < len(df_lap) - self.min_corner_duration:
            corner_sequence = self._detect_corner_sequence(brake, steering_abs, gas, i)
            
            if corner_sequence:
                corner_start = corner_sequence['start']
                corner_end = corner_sequence['end']
                
                # Valid corner found
                lap_pos_start = corner_start / len(df_lap)
                lap_pos_end = corner_end / len(df_lap)
                
                corners.append({
                    'lap_position_start': lap_pos_start,
                    'lap_position_end': lap_pos_end,
                    'absolute_start_idx': lap_start + corner_start,
                    'absolute_end_idx': lap_start + corner_end,
                    'pattern_confidence': corner_sequence['confidence']
                })
                
                # Skip ahead to avoid overlapping corners
                i = corner_end + 5
            else:
                i += 1
        
        return corners

    def _smooth_signal(self, signal: pd.Series) -> np.ndarray:
        """Smooth a signal using appropriate filter."""
        signal_array = signal.values
        if len(signal_array) >= self.smoothing_window:
            return savgol_filter(signal_array, self.smoothing_window, 2)
        else:
            return pd.Series(signal_array).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    def _detect_corner_sequence(self, brake: np.ndarray, steering_abs: np.ndarray, gas: np.ndarray, start_idx: int) -> Optional[Dict[str, Any]]:
        """
        Detect corner sequence: progressive brake → steer in → steer out → progressive gas.
        
        Args:
            brake: Brake signal array
            steering_abs: Absolute steering signal array  
            gas: Gas/throttle signal array
            start_idx: Starting index to search from
            
        Returns:
            Dictionary with corner boundaries and confidence, or None if no valid sequence
        """
        max_search_length = min(200, len(brake) - start_idx)  # Max corner length
        if max_search_length < self.min_corner_duration:
            return None
        
        # Look for corner pattern starting from start_idx
        search_end = start_idx + max_search_length
        
        # Phase 1: Find progressive braking (brake increasing)
        brake_start = self._find_progressive_increase(brake[start_idx:search_end], min_length=5)
        if brake_start is None:
            return None
        brake_start += start_idx
        
        # Phase 2: Find steering increase after braking starts
        steer_search_start = max(brake_start - 5, start_idx)  # Allow slight overlap
        steer_start = self._find_progressive_increase(steering_abs[steer_search_start:search_end], min_length=8)
        if steer_start is None:
            return None
        steer_start += steer_search_start
        
        # Phase 3: Find steering decrease (steer out)
        steer_peak_search = steer_start + 8  # Look for peak after steer starts
        if steer_peak_search >= search_end:
            return None
        steer_end = self._find_progressive_decrease(steering_abs[steer_peak_search:search_end], min_length=8)
        if steer_end is None:
            return None
        steer_end += steer_peak_search
        
        # Phase 4: Find progressive gas increase (acceleration out of corner)
        gas_search_start = max(steer_end - 10, steer_start)  # Allow overlap with steer out
        gas_start = self._find_progressive_increase(gas[gas_search_start:search_end], min_length=5)
        if gas_start is None:
            return None
        gas_start += gas_search_start
        
        # Calculate corner boundaries
        corner_start = brake_start
        corner_end = min(gas_start + 15, search_end - 1)  # End a bit after gas starts
        
        # Validate minimum corner length
        if corner_end - corner_start < self.min_corner_duration:
            return None
        
        # Calculate confidence based on pattern strength
        confidence = self._calculate_pattern_confidence(
            brake[corner_start:corner_end],
            steering_abs[corner_start:corner_end], 
            gas[corner_start:corner_end],
            brake_start - corner_start,
            steer_start - corner_start,
            steer_end - corner_start,
            gas_start - corner_start
        )
        
        return {
            'start': corner_start,
            'end': corner_end,
            'confidence': confidence,
            'brake_start': brake_start - corner_start,
            'steer_start': steer_start - corner_start,
            'steer_end': steer_end - corner_start,
            'gas_start': gas_start - corner_start
        }

    def _find_progressive_increase(self, signal: np.ndarray, min_length: int = 5) -> Optional[int]:
        """Find start of progressive increase in signal."""
        if len(signal) < min_length:
            return None
        
        for i in range(len(signal) - min_length + 1):
            # Check if signal increases progressively over min_length points
            segment = signal[i:i + min_length]
            
            # Calculate trend (should be positive for increase)
            trend = np.polyfit(range(len(segment)), segment, 1)[0]
            
            # Check for consistent increase and meaningful magnitude
            if trend > 0.01 and segment[-1] > segment[0] + 0.05:  # Threshold for meaningful increase
                return i
        
        return None

    def _find_progressive_decrease(self, signal: np.ndarray, min_length: int = 5) -> Optional[int]:
        """Find start of progressive decrease in signal."""
        if len(signal) < min_length:
            return None
        
        for i in range(len(signal) - min_length + 1):
            # Check if signal decreases progressively over min_length points
            segment = signal[i:i + min_length]
            
            # Calculate trend (should be negative for decrease)
            trend = np.polyfit(range(len(segment)), segment, 1)[0]
            
            # Check for consistent decrease and meaningful magnitude
            if trend < -0.01 and segment[0] > segment[-1] + 0.05:  # Threshold for meaningful decrease
                return i
        
        return None

    def _calculate_pattern_confidence(self, brake: np.ndarray, steering: np.ndarray, gas: np.ndarray,
                                    brake_rel: int, steer_rel: int, steer_end_rel: int, gas_rel: int) -> float:
        """Calculate confidence score for detected corner pattern."""
        confidence = 0.0
        
        try:
            # Brake phase quality (0-0.25)
            if brake_rel < len(brake) - 5:
                brake_segment = brake[brake_rel:brake_rel + 10]
                brake_increase = brake_segment[-1] - brake_segment[0]
                confidence += min(0.25, brake_increase * 2.5)  # Max 0.25 for strong braking
            
            # Steering phase quality (0-0.50)
            if steer_rel < steer_end_rel < len(steering):
                steer_in_segment = steering[steer_rel:steer_rel + 10] if steer_rel + 10 < len(steering) else steering[steer_rel:]
                steer_out_segment = steering[steer_end_rel:steer_end_rel + 10] if steer_end_rel + 10 < len(steering) else steering[steer_end_rel:]
                
                # Reward good steer in
                if len(steer_in_segment) > 0:
                    steer_increase = steer_in_segment.max() - steer_in_segment[0]
                    confidence += min(0.25, steer_increase * 2.5)
                
                # Reward good steer out  
                if len(steer_out_segment) > 0:
                    steer_decrease = steer_out_segment[0] - steer_out_segment.min()
                    confidence += min(0.25, steer_decrease * 2.5)
            
            # Gas phase quality (0-0.25)
            if gas_rel < len(gas) - 5:
                gas_segment = gas[gas_rel:gas_rel + 10]
                gas_increase = gas_segment[-1] - gas_segment[0]
                confidence += min(0.25, gas_increase * 2.5)  # Max 0.25 for strong acceleration
            
        except Exception:
            confidence = 0.3  # Default confidence if calculation fails
        
        return min(1.0, max(0.1, confidence))  # Clamp between 0.1 and 1.0

    def _find_consistent_corners(self, lap_corner_data: List[List[Dict]]) -> List[Dict[str, Any]]:
        """Find corners that appear consistently across laps."""
        if not lap_corner_data:
            return []
            
        # Use lap with most corners as reference
        reference_lap = max(lap_corner_data, key=len)
        consistent_corners = []
        
        for ref_corner in reference_lap:
            ref_position = (ref_corner['lap_position_start'] + ref_corner['lap_position_end']) / 2
            
            # Find matching corners in other laps
            matching_corners = [ref_corner]
            
            for lap_corners in lap_corner_data:
                if lap_corners is reference_lap:
                    continue
                
                # Find closest corner
                best_match = None
                best_distance = float('inf')
                
                for corner in lap_corners:
                    corner_position = (corner['lap_position_start'] + corner['lap_position_end']) / 2
                    distance = abs(corner_position - ref_position)
                    
                    if distance < best_distance and distance <= 0.05:  # 5% tolerance
                        best_distance = distance
                        best_match = corner
                
                if best_match:
                    matching_corners.append(best_match)
            
            # Require corner in at least 70% of laps
            if len(matching_corners) >= max(2, int(len(lap_corner_data) * 0.7)):
                # Average position
                avg_start = sum(c['lap_position_start'] for c in matching_corners) / len(matching_corners)
                avg_end = sum(c['lap_position_end'] for c in matching_corners) / len(matching_corners)
                
                consistent_corners.append({
                    'position_start': avg_start,
                    'position_end': avg_end,
                    'examples': matching_corners
                })
        
        # Sort by position
        consistent_corners.sort(key=lambda x: x['position_start'])
        return consistent_corners

    def _define_corner_phases(self, corner_info: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Define phases for a corner."""
        if not corner_info['examples']:
            return []
        
        example = corner_info['examples'][0]
        start_idx = example['absolute_start_idx']
        end_idx = example['absolute_end_idx']
        
        corner_df = df.iloc[start_idx:end_idx+1]
        
        if len(corner_df) < 6:
            return [{
                'phase': 'corner',
                'position_start': corner_info['position_start'],
                'position_end': corner_info['position_end']
            }]
        
        # Find apex (minimum speed)
        speeds = corner_df['Physics_speed_kmh'].fillna(method='ffill').fillna(method='bfill')
        apex_idx = speeds.idxmin()
        apex_relative = (apex_idx - corner_df.index[0]) / len(corner_df)
        
        # Define phases
        phases = []
        
        # Turn-in phase
        turn_in_end = min(0.4, apex_relative - 0.1)
        if turn_in_end > 0.1:
            phases.append({
                'phase': 'turn_in',
                'position_start': corner_info['position_start'],
                'position_end': corner_info['position_start'] + (corner_info['position_end'] - corner_info['position_start']) * turn_in_end
            })
        
        # Apex phase
        apex_start = max(turn_in_end, apex_relative - 0.15)
        apex_end = min(1.0, apex_relative + 0.15)
        
        phases.append({
            'phase': 'apex',
            'position_start': corner_info['position_start'] + (corner_info['position_end'] - corner_info['position_start']) * apex_start,
            'position_end': corner_info['position_start'] + (corner_info['position_end'] - corner_info['position_start']) * apex_end
        })
        
        # Exit phase
        if apex_end < 0.9:
            phases.append({
                'phase': 'exit',
                'position_start': corner_info['position_start'] + (corner_info['position_end'] - corner_info['position_start']) * apex_end,
                'position_end': corner_info['position_end']
            })
        
        return phases

    def _identify_current_corner_and_phase(self, track_position: float) -> Dict[str, Any]:
        """Identify current corner and phase."""
        result = {
            'corner_number': 0,
            'phase': 'straight',
            'phase_progress': 0.0
        }
        
        for corner in self.corner_patterns:
            corner_start = corner['track_position_start']
            corner_end = corner['track_position_end']
            
            # Check if in this corner
            if corner_start <= track_position <= corner_end:
                result['corner_number'] = corner['corner_number']
                
                # Find current phase
                for phase in corner['phases']:
                    phase_start = phase['position_start']
                    phase_end = phase['position_end']
                    
                    if phase_start <= track_position <= phase_end:
                        result['phase'] = phase['phase']
                        # Calculate progress within phase
                        phase_length = phase_end - phase_start
                        if phase_length > 0:
                            result['phase_progress'] = (track_position - phase_start) / phase_length
                        break
                break
        
        return result

    def _detect_lap_slices(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Detect lap boundaries."""
        n = len(df)
        if n == 0:
            return []

        boundaries = [0]
        
        # Look for wrap-around in normalized position
        if 'Graphics_normalized_car_position' in df.columns:
            pos = df['Graphics_normalized_car_position'].fillna(0.0)
            diff = pos.diff()
            # Large negative diff indicates lap boundary
            lap_boundaries = diff[diff < -0.5].index.tolist()
            boundaries.extend(lap_boundaries)
        
        boundaries = sorted(set(b for b in boundaries if 0 <= b <= n))
        if boundaries[-1] != n:
            boundaries.append(n)

        return [(s, e) for s, e in zip(boundaries[:-1], boundaries[1:]) if e - s > 0]

    def _build_position_series(self, df: pd.DataFrame) -> pd.Series:
        """Build 0..1 position series for a lap."""
        n = len(df)
        if n == 0:
            return pd.Series([], dtype=float)
        
        if 'Graphics_normalized_car_position' in df.columns:
            pos = df['Graphics_normalized_car_position'].fillna(method='ffill').fillna(method='bfill')
            return pos.clip(0.0, 1.0).reset_index(drop=True)
        
        # Fallback to linear interpolation
        return pd.Series(np.linspace(0.0, 1.0, n)).reset_index(drop=True)

    def clear_corner_cache(self):
        """Clear cached corner identification models."""
        self.track_corner_profiles.clear()
        self.corner_patterns.clear()
        self._last_serialized = None
        print(f"[INFO] Cleared corner identification cache")
        
    def get_corner_model_summary(self) -> Dict[str, Any]:
        """Get summary of loaded corner identification models."""
        return {
            "model_type": "corner_identification",
            "cached_models": len(self.track_corner_profiles),
            "available_models": list(self.track_corner_profiles.keys()),
            "corner_patterns_count": len(self.corner_patterns),
            "has_active_patterns": len(self.corner_patterns) > 0,
            "feature_catalog": CornerFeatureCatalog.CONTEXT_FEATURES
        }

    def export_corner_patterns(self) -> Dict[str, Any]:
        """Export current corner patterns for external saving."""
        return {
            "corner_patterns": self.corner_patterns,
            "track_corner_profiles": self.track_corner_profiles,
            "export_timestamp": datetime.now().isoformat(),
            "feature_catalog": CornerFeatureCatalog.CONTEXT_FEATURES
        }


# Create service instance
corner_identification_service = CornerIdentificationUnsupervisedService()