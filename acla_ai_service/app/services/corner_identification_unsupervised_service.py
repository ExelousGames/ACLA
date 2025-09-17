"""
Corner Identification Unsupervised Service for Assetto Corsa Competizione

This service provides unsupervised corner identification and feature extraction from telemetry data.
It extracts detailed corner characteristics including:
- Entry phase duration and characteristics
- Apex phase duration and characteristics  
- Exit phase duration and characteristics
- Curvature analysis
- Speed profiles
- G-force patterns
- Brake/throttle patterns

The extracted features are designed to be inserted back into telemetry data for enhanced AI analysis.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from scipy.signal import find_peaks, savgol_filter
from scipy import interpolate
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Import backend service and models
from .backend_service import backend_service
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor, _safe_float

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class CornerFeatureCatalog:
    """Canonical corner-geometry features for downstream models.

    These are geometry/direction/curvature features suitable as encoder context.
    Keep in sync with _initialize_corner_feature_columns and assignment logic.
    """
    class ContextFeature(str, Enum):
        """Single source of truth for corner context feature keys.

        Values are the exact keys inserted into telemetry/enriched features.
        Subclassing str ensures seamless use as dict keys and JSON serialization.
        """
        CORNER_ID = 'corner_id'
        IS_IN_CORNER = 'is_in_corner'
        CORNER_PROGRESS = 'corner_progress'
        CORNER_SEQUENCE_INDEX = 'corner_sequence_index'
        CORNER_DIRECTION_NUMERIC = 'corner_direction_numeric'
        CORNER_TYPE_NUMERIC = 'corner_type_numeric'
        CORNER_TOTAL_ANGLE_DEG = 'corner_total_angle_deg'
        CORNER_ARC_LENGTH_M = 'corner_arc_length_m'
        CORNER_RADIUS_EST_M = 'corner_radius_est_m'
        CORNER_AVG_CURVATURE = 'corner_avg_curvature'
        CORNER_MAX_CURVATURE = 'corner_max_curvature'
        CORNER_CURVATURE_VARIANCE = 'corner_curvature_variance'
        CORNER_COMPLEXITY_INDEX = 'corner_complexity_index'
        DISTANCE_TO_NEXT_CORNER_M = 'distance_to_next_corner_m'
        STRAIGHT_AFTER_EXIT_LENGTH_M = 'straight_after_exit_length_m'
        CORNER_CONFIDENCE = 'corner_confidence'

    # Back-compat: keep a list view derived from the Enum (do not hand-edit).
    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]

class CornerCharacteristics:
    """Geometry-focused, driver-agnostic corner characteristics."""
    def __init__(self):
        # Temporal
        self.total_corner_duration = 0.0
        # Geometry / direction
        self.corner_type = "unknown"  # hairpin, fast_sweep, etc.
        self.corner_direction = "unknown"  # left/right
        # Curvature stats (from heading or steering proxy)
        self.avg_curvature = 0.0
        self.max_curvature = 0.0
        self.curvature_variance = 0.0
        # Derived geometric metrics
        self.total_angle_deg = 0.0
        self.arc_length_m = 0.0
        self.radius_est_m = 0.0
        self.curvature_energy = 0.0  # avg_curvature * total_angle_deg
        # Confidence (filled when segment created)
        self.confidence = 0.0


class CornerIdentificationUnsupervisedService:
    """
    Unsupervised corner identification and feature extraction service
    
    This service:
    1. Identifies corners in telemetry data using unsupervised methods
    2. Extracts detailed corner characteristics as features
    3. Provides corner classification (hairpin, chicane, sweeper, etc.)
    4. Generates new features to be inserted back into telemetry data
    """
    
    def __init__(self):
        """Initialize the corner identification service (pure geometry mode)."""
        self.telemetry_features = TelemetryFeatures()
        self.corner_models = {}
        self.scalers = {}
        self.track_corner_profiles = {}
        self.corner_patterns = []  # Store learned corner patterns for reuse
        
        # Backend service integration
        self.backend_service = backend_service
        
        # Corner identification parameters
        self.min_corner_duration = 10  # Minimum data points for a valid corner
        self.corner_detection_sensitivity = 0.7
        self.smoothing_window = 7  # Window size for smoothing telemetry data
        # Steering angle metadata: current physics feed provides 0..1 where 0.5 is neutral.
        # We normalize to a symmetric -1..1 scale for downstream logic (direction, curvature).
        self.steering_range = '0_to_1'  # allowed values: '0_to_1', 'minus1_to_1'
        # NOTE:
        #   RAW Physics_steer_angle: 0 (full left) -> 0.5 (center) -> 1 (full right)
        #   NORMALIZED (_normalize_steering): -1 (full left) -> 0 (center) -> 1 (full right)
        #   ABS NORMALIZED (np.abs(normalized)): 0 -> 1 magnitude irrespective of direction
        # Serialization cache
        self._last_serialized: Optional[Dict[str, Any]] = None
        # Advanced segmentation tuning (new)
        self.segment_label_smoothing = 3  # median filter width for label smoothing (odd, >=1)
        self.max_merge_gap = 3            # max gap (points) to merge adjacent candidate segments
        self.min_activity_ratio = 0.35    # min mean(|steer|) / global_mean(|steer|) to consider activity
        self.corner_energy_threshold = 0.15  # threshold on combined normalized steering+gforce energy
        self.require_brake_or_speed_delta = False  # optionally demand decel / brake evidence

    # --------------------- Serialization ---------------------
    def _to_python(self, obj: Any) -> Any:
        """Recursively convert numpy / pandas dtypes to plain Python for JSON serialization."""
        import numpy as _np
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._to_python(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._to_python(v) for k, v in obj.items()}
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        return str(obj)

    def serialize_model(self, track_name: str, car_name: str) -> Dict[str, Any]:
        """Return JSON-safe representation of learned corner patterns & config (geometry-only)."""
        data = {
            'model_type': 'corner_identification',
            'track_name': track_name,
            'car_name': car_name,
            'parameters': {
                'min_corner_duration': self.min_corner_duration,
                'corner_detection_sensitivity': self.corner_detection_sensitivity,
                'smoothing_window': self.smoothing_window,
                'steering_range': self.steering_range
            },
            'corner_patterns': self._to_python(self.corner_patterns),
            'corner_clusters': self._to_python(self._cluster_corner_types(self.corner_patterns) if self.corner_patterns else []),
            'total_corners': len(self.corner_patterns),
            'serialized_timestamp': datetime.now().isoformat()
        }
        self._last_serialized = data
        return data

    @classmethod
    def deserialize_model(cls, payload: Dict[str, Any]) -> 'CornerIdentificationUnsupervisedService':
        inst = cls()
        inst.corner_patterns = payload.get('corner_patterns', [])
        inst.track_corner_profiles["generic_all_cars"] = {
            'corner_patterns': inst.corner_patterns,
            'corner_clusters': payload.get('corner_clusters', []),
            'total_corners': payload.get('total_corners', len(inst.corner_patterns))
        }
        return inst

    def _normalize_steering(self, steering_series: pd.Series) -> pd.Series:
        """Normalize raw steering angle to a symmetric -1..1 range.

        If the source data is already -1..1 it is returned unchanged (with NaNs filled).
        If the source data is 0..1 we map: norm = (value - 0.5) * 2.
        """
        s = steering_series.fillna(0.0)
        if self.steering_range == '0_to_1':
            s = (s - 0.5) * 2.0
        # clip to guard against any slight overflow due to upstream noise
        return s.clip(-1.0, 1.0)
        
    async def learn_track_corner_patterns(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Learn corner patterns using unsupervised learning from cleaned telemetry data
        
        Args:
            telemetry_data: Clean telemetry data for learning
            
        Returns:
            Dictionary with learning results and extracted corner patterns
        """
        try:
            print(f"[INFO {self.__class__.__name__}] Starting unsupervised corner pattern learning with cleaned data")
            
            if not telemetry_data:
                return {
                    "success": False,
                    "error": "No telemetry data provided",
                    "corner_patterns": []
                }
            
            print(f"[INFO] Using provided clean telemetry data: {len(telemetry_data)} records")
            all_telemetry_data = telemetry_data
            
            # Convert to DataFrame - data is already clean
            df = pd.DataFrame(all_telemetry_data)

            # Identify corner segments using unsupervised methods
            corner_segments = self._identify_corner_segments_unsupervised(df)
            
            # Extract corner characteristics for each segment
            corner_patterns = []
            for corner_id, segment_data in corner_segments.items():
                characteristics = self._extract_corner_characteristics(
                    df.iloc[segment_data['start_idx']:segment_data['end_idx']+1]
                )
                
                corner_patterns.append({
                    "corner_id": corner_id,
                    "track_position_start": segment_data.get('position_start', 0.0),
                    "track_position_end": segment_data.get('position_end', 0.0),
                    "characteristics": characteristics.__dict__,
                    "data_points": segment_data['end_idx'] - segment_data['start_idx'] + 1
                })
            
            # Store the corner patterns in the class instance for later use
            self.corner_patterns = corner_patterns
            
            # Cluster similar corners to identify corner types
            corner_clusters = self._cluster_corner_types(corner_patterns)
            
            # Save learned patterns in memory
            track_model = {
                "track_name": "generic",
                "car_name": "all_cars",
                "corner_patterns": corner_patterns,
                "corner_clusters": corner_clusters,
                "total_corners": len(corner_patterns),
                "learning_timestamp": datetime.now().isoformat()
            }
            
            self.track_corner_profiles["generic_all_cars"] = track_model
            
            print(f"[INFO {self.__class__.__name__}] Successfully learned {len(corner_patterns)} corner patterns")
            print(f"[INFO {self.__class__.__name__}] Corner patterns clustered into {len(corner_clusters)} types")
            
            return {
                "success": True,
                "track_name": "generic",
                "car_name": "all_cars",
                "total_corners_identified": len(corner_patterns),
                "corner_patterns": corner_patterns,
                "corner_clusters": corner_clusters
            }
            
        except Exception as e:
            raise Exception(f"[ERROR] Failed to learn corner patterns: {str(e)}")
    
    async def extract_corner_features_for_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract corner features and insert them back into clean telemetry data using the saved model
        
        Args:
            telemetry_data: List of clean telemetry records to predict on
            geometry_only: Optional override; if provided supersedes service default.
            
        Returns:
            Enhanced telemetry data with corner features
        """
        try:
            # Check if we have a learned model
            if not self.corner_patterns:
                print("[ERROR] No corner patterns model available. Please run learn_track_corner_patterns first.")
                return telemetry_data
            
            print(f"[INFO] Using saved corner model with {len(self.corner_patterns)} learned corner patterns")
            
            # Convert telemetry data to DataFrame - data is already clean
            df = pd.DataFrame(telemetry_data)
            
            # Initialize corner feature columns
            corner_features = self._initialize_corner_feature_columns(len(df))
            
            # Use the saved corner patterns to match and assign features
            self._match_corners_with_learned_patterns(df, corner_features)

            # Post-process geometry-only relational metrics
            self._finalize_geometry_relations(df, corner_features)
            
            # Add corner features to original telemetry data
            enhanced_telemetry = []
            for i, record in enumerate(telemetry_data):
                enhanced_record = record.copy()
                
                # Add all corner features
                for feature_name, feature_values in corner_features.items():
                    if i < len(feature_values):
                        enhanced_record[feature_name] = feature_values[i]
                
                enhanced_telemetry.append(enhanced_record)
            
            print(f"[INFO] Enhanced {len(enhanced_telemetry)} telemetry records with corner features using saved model")
            return enhanced_telemetry
            
        except Exception as e:
            print(f"[ERROR] Failed to extract corner features: {str(e)}")
            return telemetry_data  # Return original data if feature extraction fails
    
    def _identify_corner_segments_unsupervised(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Identify corner segments using unsupervised clustering of telemetry patterns without inheriting driver behavior
        """
        # Required columns for corner identification
        # TODO: should identify corner height change, inner location
        required_cols = ['Physics_steer_angle', 'Physics_speed_kmh','Physics_speed_kmh', 'Physics_g_force_x', 'Physics_brake', 'Physics_gas' ]  # Physics_steer_angle raw range: 0..1 (0.5 center) before normalization
        if not all(col in df.columns for col in required_cols):
            print(f"[WARNING] Missing required columns for corner identification")
            return {}
        
        # Create feature matrix for corner detection
        features = self._create_corner_detection_features(df)
        if features is None:
            return {}
        
        # Apply clustering to identify corner regions
        corner_labels = self._cluster_corner_regions(features)
        
        # Convert cluster labels to corner segments
        corner_segments = self._extract_segments_from_clusters(df, corner_labels, features)
        
        return corner_segments
    
    def _create_corner_detection_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create feature matrix for corner detection"""
        try:
            features = []
            
            # Steering angle features (absolute and smoothed)
            # Normalize steering (incoming physics provides 0..1 so center & scale)
            steering_norm = self._normalize_steering(df['Physics_steer_angle'])  # normalized -1..1; magnitude used below (0..1 after abs)
            steering_abs = np.abs(steering_norm)
            if len(steering_abs) >= self.smoothing_window:
                steering_smooth = savgol_filter(steering_abs, self.smoothing_window, 2)
            else:
                steering_smooth = steering_abs.rolling(window=3, center=True).mean().fillna(steering_abs)
            
            features.append(steering_smooth)
            
            # Speed features
            speed = df['Physics_speed_kmh'].fillna(0)
            speed_smooth = speed.rolling(window=5).mean().fillna(speed)
            features.append(speed_smooth)
            
            # Speed change rate
            speed_change = speed_smooth.diff().fillna(0)
            features.append(speed_change)
            

            g_force_lat = np.abs(df['Physics_g_force_x'].fillna(0))
            features.append(g_force_lat)
            
            # Brake and throttle
            brake = df['Physics_brake'].fillna(0)
            features.append(brake)

            # Throttle
            throttle = df['Physics_gas'].fillna(0)
            features.append(throttle)
            
            # Combine features
            feature_matrix = np.column_stack(features)
            
            # Handle any remaining NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
            
            return feature_matrix
            
        except Exception as e:
            print(f"[ERROR] Failed to create corner detection features: {str(e)}")
            return None
    
    def _cluster_corner_regions(self, features: np.ndarray) -> np.ndarray:
        """Use clustering to identify corner regions vs straight sections"""
        try:
            # Normalize features
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use DBSCAN for adaptive clustering
            # Parameters tuned for racing telemetry patterns
            dbscan = DBSCAN(
                eps=0.5,
                min_samples=max(5, self.min_corner_duration // 2),
                metric='euclidean'
            )
            
            labels = dbscan.fit_predict(features_scaled)
            
            # If DBSCAN finds too few clusters, fallback to KMeans
            unique_labels = len(np.unique(labels))
            if unique_labels < 3:  # At least need corner vs straight distinction
                print("[INFO] DBSCAN found few clusters, trying KMeans fallback")
                
                # Try KMeans with different cluster numbers
                best_score = -1
                best_labels = labels
                
                for n_clusters in range(3, min(8, len(features) // 20)):
                    if n_clusters >= len(features):
                        break
                        
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    k_labels = kmeans.fit_predict(features_scaled)
                    
                    try:
                        score = silhouette_score(features_scaled, k_labels)
                        if score > best_score:
                            best_score = score
                            best_labels = k_labels
                    except:
                        continue
                
                labels = best_labels
            
            return labels
            
        except Exception as e:
            print(f"[ERROR] Failed to cluster corner regions: {str(e)}")
            # Fallback: use simple threshold-based detection
            return self._fallback_threshold_detection(features)
    
    def _fallback_threshold_detection(self, features: np.ndarray) -> np.ndarray:
        """Fallback method using simple thresholding"""
        try:
            # Use steering angle (first feature) for simple detection
            steering = features[:, 0]
            threshold = np.percentile(steering, 70)
            
            # Label high steering as corners (1), low steering as straights (0)
            labels = (steering > threshold).astype(int)
            
            return labels
            
        except Exception as e:
            print(f"[ERROR] Fallback detection failed: {str(e)}")
            return np.zeros(len(features), dtype=int)
    
    def _extract_segments_from_clusters(self, df: pd.DataFrame, labels: np.ndarray, features: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Extract corner segments from cluster labels with improved heuristics & merging.

        Steps:
          1. Optional median smoothing of labels to reduce flicker.
          2. Initial segmentation on label transitions (including index 0).
          3. Compute per-segment activity/energy metrics.
          4. Filter by multi-factor corner candidacy (label, activity, duration, energy, decel).
          5. Merge adjacent segments separated by small gaps if combined qualifies.
          6. Produce final dict with confidence score & metrics.
        """
        if len(labels) == 0:
            return {}

        labels_proc = labels.copy()
        # 1. Label smoothing (median filter) if window >1
        try:
            if self.segment_label_smoothing and self.segment_label_smoothing > 1:
                k = int(self.segment_label_smoothing)
                if k % 2 == 0:
                    k += 1  # enforce odd
                # simple rolling median over labels
                padded = np.pad(labels_proc, (k//2, k//2), mode='edge')
                smoothed = []
                for i in range(len(labels_proc)):
                    window = padded[i:i+k]
                    smoothed.append(int(np.median(window)))
                labels_proc = np.array(smoothed)
        except Exception:
            pass  # fall back to raw labels

        # 2. Boundaries (ensure start 0 included)
        change_idx = np.where(np.diff(labels_proc) != 0)[0] + 1
        boundaries = np.concatenate(([0], change_idx, [len(labels_proc)]))

        # Pre-compute global statistics for activity normalization
        steer_abs_global = np.abs(features[:, 0]) if features.shape[1] > 0 else np.zeros(len(labels_proc))
        global_mean_steer = steer_abs_global.mean() + 1e-6
        speed_series = features[:, 1] if features.shape[1] > 1 else np.zeros(len(labels_proc))
        gforce_lat_series = features[:, 3] if features.shape[1] > 3 else np.zeros(len(labels_proc))

        segments_raw = []
        for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
            start_idx = b_start
            end_idx = b_end - 1
            if end_idx < start_idx:
                continue
            seg_len = end_idx - start_idx + 1
            seg_label = int(labels_proc[start_idx])
            seg_slice = slice(start_idx, end_idx + 1)
            steer_seg = steer_abs_global[seg_slice]
            speed_seg = speed_series[seg_slice]
            g_lat_seg = gforce_lat_series[seg_slice]
            mean_steer = steer_seg.mean()
            steer_ratio = mean_steer / global_mean_steer
            steer_var = steer_seg.var()
            speed_drop = 0.0
            if seg_len > 2:
                speed_drop = float(max(0.0, speed_seg.max() - speed_seg.min()))
            g_lat_peak = float(g_lat_seg.max()) if seg_len > 0 else 0.0
            # corner energy: normalized steering + weighted g-force
            corner_energy = mean_steer + 0.5 * g_lat_peak
            segments_raw.append({
                'start': start_idx,
                'end': end_idx,
                'length': seg_len,
                'label': seg_label,
                'steer_ratio': steer_ratio,
                'steer_var': float(steer_var),
                'speed_drop': speed_drop,
                'g_lat_peak': g_lat_peak,
                'corner_energy': float(corner_energy)
            })

        # 3/4. Determine candidacy
        candidates = []
        for seg in segments_raw:
            if seg['length'] < self.min_corner_duration:
                # allow short segments to be merged later; keep but mark not corner yet
                seg['is_candidate'] = False
            # treat DBSCAN noise (-1) as baseline unless high energy
            baseline_label = 0
            is_noise = seg['label'] == -1
            label_support = (seg['label'] != baseline_label and not is_noise)
            activity_support = seg['steer_ratio'] >= self.min_activity_ratio or seg['steer_var'] > 0.05
            energy_support = seg['corner_energy'] >= self.corner_energy_threshold
            decel_support = True
            if self.require_brake_or_speed_delta:
                decel_support = seg['speed_drop'] > 3.0  # km/h delta heuristic
            seg['is_candidate'] = (
                seg['length'] >= self.min_corner_duration and
                (label_support or (activity_support and energy_support)) and
                decel_support
            )
            candidates.append(seg)

        # 5. Merge pass: merge adjacent candidate or near-candidate segments if small gap
        merged = []
        i = 0
        while i < len(candidates):
            current = candidates[i].copy()
            j = i + 1
            while j < len(candidates):
                gap = candidates[j]['start'] - current['end'] - 1
                if gap < 0:
                    break
                if gap <= self.max_merge_gap:
                    # prospective merged metrics
                    combined_len = candidates[j]['end'] - current['start'] + 1
                    combined_energy = (current['corner_energy'] * current['length'] + candidates[j]['corner_energy'] * candidates[j]['length']) / combined_len
                    combined_steer_ratio = (current['steer_ratio'] * current['length'] + candidates[j]['steer_ratio'] * candidates[j]['length']) / combined_len
                    # if merging improves chance to be candidate
                    if combined_len >= self.min_corner_duration and combined_energy >= self.corner_energy_threshold:
                        current['end'] = candidates[j]['end']
                        current['length'] = combined_len
                        current['corner_energy'] = combined_energy
                        current['steer_ratio'] = combined_steer_ratio
                        current['label'] = max(current['label'], candidates[j]['label'])  # keep stronger label
                        current['is_candidate'] = True
                        j += 1
                        continue
                break
            merged.append(current)
            i = j

        # Re-evaluate candidacy after merging
        final_segments = []
        for seg in merged:
            if not seg['is_candidate']:
                # second chance based on updated metrics
                seg['is_candidate'] = (
                    seg['length'] >= self.min_corner_duration and
                    (seg['steer_ratio'] >= self.min_activity_ratio and seg['corner_energy'] >= self.corner_energy_threshold)
                )
            if seg['is_candidate']:
                final_segments.append(seg)

        # 6. Build output dict with confidence
        corner_segments: Dict[int, Dict[str, Any]] = {}
        for cid, seg in enumerate(final_segments):
            start_idx, end_idx = seg['start'], seg['end']
            position_start = 0.0
            position_end = 0.0
            if 'Graphics_normalized_car_position' in df.columns:
                pos_col = df['Graphics_normalized_car_position']
                if start_idx < len(pos_col) and not pd.isna(pos_col.iloc[start_idx]):
                    position_start = float(pos_col.iloc[start_idx])
                if end_idx < len(pos_col) and not pd.isna(pos_col.iloc[end_idx]):
                    position_end = float(pos_col.iloc[end_idx])
            # confidence: blend of steer_ratio & normalized energy & length factor
            length_factor = min(1.0, seg['length'] / (self.min_corner_duration * 2))
            conf = 0.4 * min(1.0, seg['steer_ratio']) + 0.4 * min(1.0, seg['corner_energy'] / (self.corner_energy_threshold * 3)) + 0.2 * length_factor
            corner_segments[cid] = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'position_start': position_start,
                'position_end': position_end,
                'cluster_label': int(seg['label']),
                'segment_length': int(seg['length']),
                'steer_ratio': float(seg['steer_ratio']),
                'corner_energy': float(seg['corner_energy']),
                'g_lat_peak': float(seg['g_lat_peak']),
                'speed_drop': float(seg['speed_drop']),
                'confidence': float(conf)
            }

        print(f"[INFO] Extracted {len(corner_segments)} corner segments from clustering (improved)")
        return corner_segments
    
    def _is_high_activity_segment(self, segment_features: np.ndarray) -> bool:
        """Determine if a segment shows high cornering activity"""
        if len(segment_features) == 0:
            return False
        
        # Check steering activity (first feature)
        steering_activity = np.mean(segment_features[:, 0])
        steering_variance = np.var(segment_features[:, 0])
        
        # Check speed variance (second feature if available)
        speed_variance = 0.0
        if segment_features.shape[1] > 1:
            speed_variance = np.var(segment_features[:, 1])
        
        # High activity if steering is above median with some variance
        overall_steering_median = np.median(segment_features[:, 0]) if len(segment_features) > 0 else 0
        
        return (
            steering_activity > overall_steering_median * 0.5 and
            (steering_variance > 0.1 or speed_variance > 10.0)
        )
    
    def _extract_corner_characteristics(self, corner_df: pd.DataFrame) -> CornerCharacteristics:
        """Extract geometry-only corner characteristics (no driver behavior)."""
        ch = CornerCharacteristics()
        if len(corner_df) == 0:
            return ch
        try:
            ch.total_corner_duration = len(corner_df)
            # Arc length from positional data
            if all(c in corner_df.columns for c in ['car_pos_x', 'car_pos_y']) and len(corner_df) > 1:
                x = corner_df['car_pos_x'].astype(float).to_numpy()
                y = corner_df['car_pos_y'].astype(float).to_numpy()
                ch.arc_length_m = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))
            # Heading / direction
            if 'Physics_heading' in corner_df.columns and len(corner_df) > 1:
                heading = corner_df['Physics_heading'].astype(float).to_numpy()
                total_angle = heading[-1] - heading[0]
                ch.total_angle_deg = float(abs(total_angle))
                ch.corner_direction = 'right' if total_angle > 0 else 'left'
                hd_diff = np.abs(np.diff(heading))
                if len(hd_diff) > 0:
                    ch.avg_curvature = float(hd_diff.mean())
                    ch.max_curvature = float(hd_diff.max())
                    ch.curvature_variance = float(np.var(hd_diff))
            elif 'Physics_steer_angle' in corner_df.columns:
                # Fallback curvature proxy using steering normalization
                steering_norm = self._normalize_steering(corner_df['Physics_steer_angle'])
                st_abs = np.abs(steering_norm)
                ch.avg_curvature = float(st_abs.mean())
                ch.max_curvature = float(st_abs.max())
                ch.curvature_variance = float(np.var(st_abs))
                avg_sign = steering_norm.mean()
                ch.corner_direction = 'right' if avg_sign > 0 else 'left'
            if ch.avg_curvature > 1e-6:
                ch.radius_est_m = 1.0 / ch.avg_curvature
            ch.curvature_energy = ch.avg_curvature * ch.total_angle_deg
            ch.corner_type = self._classify_corner_type(ch)
        except Exception as e:
            print(f"[WARNING] Error extracting geometry characteristics: {e}")
        return ch
    
    def _analyze_corner_phases(self, corner_df: pd.DataFrame, characteristics: CornerCharacteristics):
        """Analyze entry, apex, and exit phases of the corner"""
        try:
            corner_length = len(corner_df)
            if corner_length < 6:  # Too short for phase analysis
                return
            
            # Find apex (minimum speed point)
            if 'Physics_speed_kmh' in corner_df.columns:
                speed = corner_df['Physics_speed_kmh'].fillna(0)
                apex_idx = speed.idxmin() - corner_df.index[0]  # Relative to corner start
                
                # Define phases based on apex position
                entry_end = max(0, apex_idx - 1)
                exit_start = min(corner_length - 1, apex_idx + 1)
                
                # Entry phase (start to apex)
                if entry_end > 0:
                    entry_df = corner_df.iloc[0:entry_end+1]
                    characteristics.entry_duration = len(entry_df)
                    
                    if len(entry_df) > 1:
                        entry_speed_start = entry_df['Physics_speed_kmh'].iloc[0]
                        entry_speed_end = entry_df['Physics_speed_kmh'].iloc[-1]
                        characteristics.entry_speed_delta = float(entry_speed_end - entry_speed_start)
                    
                    # Entry brake analysis
                    if 'Physics_brake' in entry_df.columns:
                        characteristics.entry_brake_intensity = float(entry_df['Physics_brake'].max())
                    
                    # Entry steering rate
                    if 'Physics_steer_angle' in entry_df.columns and len(entry_df) > 1:
                        steering_norm_entry = self._normalize_steering(entry_df['Physics_steer_angle'])  # entry steering rate computed on normalized -1..1
                        steering_diff = np.abs(np.diff(steering_norm_entry))
                        characteristics.entry_steering_rate = float(np.mean(steering_diff))
                
                # Apex phase (small region around minimum speed)
                apex_start = max(0, apex_idx - 2)
                apex_end = min(corner_length - 1, apex_idx + 2)
                apex_df = corner_df.iloc[apex_start:apex_end+1]
                characteristics.apex_duration = len(apex_df)
                
                # Exit phase (apex to end)
                if exit_start < corner_length - 1:
                    exit_df = corner_df.iloc[exit_start:]
                    characteristics.exit_duration = len(exit_df)
                    
                    if len(exit_df) > 1:
                        exit_speed_start = exit_df['Physics_speed_kmh'].iloc[0]
                        exit_speed_end = exit_df['Physics_speed_kmh'].iloc[-1]
                        characteristics.exit_speed_delta = float(exit_speed_end - exit_speed_start)
                    
                    # Exit throttle analysis
                    if 'Physics_gas' in exit_df.columns:
                        throttle = exit_df['Physics_gas'].fillna(0)
                        if len(throttle) > 1:
                            throttle_progression = np.diff(throttle).mean()
                            characteristics.exit_throttle_progression = float(throttle_progression)
                    
                    # Exit steering unwind rate
                    if 'Physics_steer_angle' in exit_df.columns and len(exit_df) > 1:
                        steering_abs_exit = np.abs(self._normalize_steering(exit_df['Physics_steer_angle']))  # exit unwind rate uses |normalized| 0..1
                        if len(steering_abs_exit) > 1:
                            unwind_rate = np.diff(steering_abs_exit).mean()
                            characteristics.exit_steering_unwind_rate = float(-unwind_rate)  # Negative because unwinding
                        
        except Exception as e:
            print(f"[WARNING] Error analyzing corner phases: {str(e)}")
    
    def _classify_corner_type(self, ch: CornerCharacteristics) -> str:
        """Geometry-only corner classification using angle & radius heuristics."""
        try:
            angle = ch.total_angle_deg
            radius = ch.radius_est_m if ch.radius_est_m > 0 else 1e9
            if angle > 120 and radius < 60:
                return 'hairpin'
            if angle < 40 and radius > 120:
                return 'fast_sweep'
            if 40 <= angle <= 100 and radius < 80:
                return 'medium_tight'
            if angle >= 100 and radius >= 80:
                return 'long_sweeper'
            return 'medium_corner'
        except Exception:
            return 'unknown'
    
    # _calculate_advanced_scores method removed (advanced metrics deprecated)
    
    def _cluster_corner_types(self, corner_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster similar corners using geometry-only features."""
        if not corner_patterns:
            return {"clusters": [], "cluster_labels": []}
        try:
            feature_names = ['total_angle_deg', 'radius_est_m', 'avg_curvature', 'total_corner_duration']
            X = []
            for p in corner_patterns:
                ch = p.get('characteristics', {})
                X.append([ch.get(n, 0.0) for n in feature_names])
            X = np.array(X)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            max_clusters = min(6, len(corner_patterns)//2)
            if max_clusters < 2:
                return {"clusters": [{"type": "single_type", "corners": list(range(len(corner_patterns)))}], "cluster_labels": [0]*len(corner_patterns)}
            best_score=-1; best_labels=None; best_n=2
            for k in range(2, max_clusters+1):
                km=KMeans(n_clusters=k, random_state=42, n_init=10)
                lbl=km.fit_predict(Xs)
                try:
                    sc=silhouette_score(Xs,lbl)
                    if sc>best_score:
                        best_score=sc; best_labels=lbl; best_n=k
                except Exception:
                    continue
            if best_labels is None:
                best_labels=np.zeros(len(corner_patterns),dtype=int)
            clusters=[]
            for cid in range(best_n):
                members=[i for i,l in enumerate(best_labels) if l==cid]
                if not members:
                    continue
                sub=[corner_patterns[i]['characteristics'] for i in members]
                avg_angle=float(np.mean([c.get('total_angle_deg',0) for c in sub]))
                avg_radius=float(np.mean([c.get('radius_est_m',0) for c in sub]))
                if avg_angle>120 and avg_radius<60:
                    ctype='hairpins'
                elif avg_radius>150:
                    ctype='fast'
                elif avg_angle>90:
                    ctype='long'
                else:
                    ctype='mixed'
                clusters.append({
                    'cluster_id': int(cid),
                    'type': ctype,
                    'corner_count': len(members),
                    'corners': members,
                    'avg_characteristics': {
                        'angle_deg': avg_angle,
                        'radius_m': avg_radius
                    }
                })
            return {'clusters': clusters, 'cluster_labels': np.array(best_labels).tolist(), 'silhouette_score': float(best_score)}
        except Exception as e:
            print(f"[ERROR] Failed to cluster corner types: {e}")
            return {"clusters": [], "cluster_labels": []}
    
    def _initialize_corner_feature_columns(self, data_length: int) -> Dict[str, List[float]]:
        """Initialize geometry-only feature columns based on the ContextFeature enum."""
        feature_names = [f.value for f in CornerFeatureCatalog.ContextFeature]
        return {f: [0.0] * data_length for f in feature_names}
    
    def _assign_corner_features_to_segment(self, corner_features: Dict[str, List[float]],
                                          start_idx: int, end_idx: int,
                                          ch: CornerCharacteristics,
                                          corner_id: int,
                                          corner_df: Optional[pd.DataFrame] = None,
                                          sequence_index: Optional[int] = None,
                                          confidence: float = 0.0):
        """Assign geometry-only metrics to a segment."""
        try:
            CF = CornerFeatureCatalog.ContextFeature
            # Allow recomputation if arc/angle missing
            total_angle_deg = ch.total_angle_deg
            arc_length = ch.arc_length_m
            if corner_df is not None and len(corner_df) > 1 and (total_angle_deg == 0 or arc_length == 0):
                if all(c in corner_df.columns for c in ['car_pos_x','car_pos_y']):
                    x = corner_df['car_pos_x'].astype(float).to_numpy(); y = corner_df['car_pos_y'].astype(float).to_numpy()
                    arc_length = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))
                if 'Physics_heading' in corner_df.columns:
                    h = corner_df['Physics_heading'].astype(float).to_numpy(); total_angle_deg = float(abs(h[-1]-h[0]))
            complexity_index = (ch.curvature_variance or 0.0) * (total_angle_deg or 0.0)
            direction_numeric = 1.0 if ch.corner_direction == 'right' else -1.0
            corner_type_numeric = float(hash(ch.corner_type) % 1000)
            length = max(1, end_idx - start_idx)
            for local_i, i in enumerate(range(start_idx, min(end_idx + 1, len(corner_features[CF.CORNER_ID.value])))):
                progress = local_i / (length - 1) if length > 1 else 0.0
                corner_features[CF.CORNER_ID.value][i] = float(corner_id)
                corner_features[CF.IS_IN_CORNER.value][i] = 1.0
                corner_features[CF.CORNER_PROGRESS.value][i] = progress
                corner_features[CF.CORNER_SEQUENCE_INDEX.value][i] = float(sequence_index if sequence_index is not None else corner_id)
                corner_features[CF.CORNER_DIRECTION_NUMERIC.value][i] = direction_numeric
                corner_features[CF.CORNER_TYPE_NUMERIC.value][i] = corner_type_numeric
                corner_features[CF.CORNER_TOTAL_ANGLE_DEG.value][i] = total_angle_deg
                corner_features[CF.CORNER_ARC_LENGTH_M.value][i] = arc_length
                corner_features[CF.CORNER_RADIUS_EST_M.value][i] = ch.radius_est_m
                corner_features[CF.CORNER_AVG_CURVATURE.value][i] = ch.avg_curvature
                corner_features[CF.CORNER_MAX_CURVATURE.value][i] = ch.max_curvature
                corner_features[CF.CORNER_CURVATURE_VARIANCE.value][i] = ch.curvature_variance
                corner_features[CF.CORNER_COMPLEXITY_INDEX.value][i] = complexity_index
                if CF.CORNER_CONFIDENCE.value in corner_features:
                    corner_features[CF.CORNER_CONFIDENCE.value][i] = confidence
        except Exception as e:
            print(f"[WARNING] Error assigning geometry corner features: {e}")

    def _finalize_geometry_relations(self, df: pd.DataFrame, corner_features: Dict[str, List[float]]):
        """Compute relational geometry metrics after all corners processed."""
        CF = CornerFeatureCatalog.ContextFeature
        if CF.CORNER_ID.value not in corner_features or CF.IS_IN_CORNER.value not in corner_features:
            return
        ids = np.array(corner_features[CF.CORNER_ID.value])
        if len(ids) == 0:
            return
        unique_ids = [int(i) for i in sorted(set(ids)) if i >= 0]
        dist_col = 'Graphics_distance_traveled'
        distances = df[dist_col].to_numpy() if dist_col in df.columns else np.arange(len(ids))
        segments = []
        for cid in unique_ids:
            idxs = np.where(ids == cid)[0]
            if len(idxs) == 0:
                continue
            segments.append((cid, idxs[0], idxs[-1]))
        for i, (cid, s, e) in enumerate(segments):
            next_start_distance = None
            if i + 1 < len(segments):
                next_start_distance = distances[segments[i + 1][1]]
            end_distance = distances[e]
            straight_after = 0.0
            if next_start_distance is not None:
                straight_after = float(max(0.0, next_start_distance - end_distance))
            for p in range(s, e + 1):
                if CF.DISTANCE_TO_NEXT_CORNER_M.value in corner_features:
                    corner_features[CF.DISTANCE_TO_NEXT_CORNER_M.value][p] = straight_after
                if CF.STRAIGHT_AFTER_EXIT_LENGTH_M.value in corner_features:
                    corner_features[CF.STRAIGHT_AFTER_EXIT_LENGTH_M.value][p] = straight_after
    
    def _determine_phase_for_point(self, point_idx: int, start_idx: int, end_idx: int, 
                                  characteristics: CornerCharacteristics) -> float:
        """Determine which phase (entry/apex/exit) a specific point belongs to"""
        try:
            corner_length = end_idx - start_idx + 1
            relative_pos = (point_idx - start_idx) / corner_length if corner_length > 0 else 0
            
            # Phase mapping: 1 = entry, 2 = apex, 3 = exit
            entry_fraction = characteristics.entry_duration / characteristics.total_corner_duration
            apex_fraction = characteristics.apex_duration / characteristics.total_corner_duration
            
            if relative_pos <= entry_fraction:
                return 1.0  # Entry
            elif relative_pos >= (1.0 - apex_fraction / 2):
                return 3.0  # Exit
            else:
                return 2.0  # Apex
                
        except Exception:
            return 2.0  # Default to apex if calculation fails
   
    def _summarize_corner_types(self, corner_patterns: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize the types of corners found"""
        type_counts = {}
        
        for pattern in corner_patterns:
            corner_type = pattern.get("characteristics", {}).get("corner_type", "unknown")
            type_counts[corner_type] = type_counts.get(corner_type, 0) + 1
        
        return type_counts
    
    def _match_corners_with_learned_patterns(self, df: pd.DataFrame, corner_features: Dict[str, List[float]]):
        """
        Match corners in new telemetry data with learned corner patterns and assign features
        
        Args:
            df: New telemetry data as DataFrame
            corner_features: Dictionary to populate with corner features
        """
        try:
            # Identify corners in the new telemetry data
            corner_segments = self._identify_corner_segments_unsupervised(df)
            
            if not corner_segments:
                print("[INFO] No corners detected in the telemetry data")
                return
            
            print(f"[INFO] Detected {len(corner_segments)} corner segments in new telemetry data")
            
            # For each detected corner, find the best matching learned pattern
            seq_counter = 0
            for corner_id, segment_data in corner_segments.items():
                start_idx = segment_data['start_idx']
                end_idx = segment_data['end_idx']
                corner_df = df.iloc[start_idx:end_idx+1]
                detected_characteristics = self._extract_corner_characteristics(corner_df)
                confidence = segment_data.get('confidence', 0.0)
                self._assign_corner_features_to_segment(
                    corner_features,
                    start_idx,
                    end_idx,
                    detected_characteristics,
                    corner_id,
                    corner_df=corner_df,
                    sequence_index=seq_counter,
                    confidence=confidence
                )
                seq_counter += 1
                
        except Exception as e:
            print(f"[ERROR] Failed to match corners with learned patterns: {str(e)}")
    
    def _find_best_matching_pattern(self, detected_characteristics: CornerCharacteristics, 
                                   segment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find the best matching learned corner pattern for a detected corner
        
        Args:
            detected_characteristics: Characteristics of the detected corner
            segment_data: Segment information (position, etc.)
            
        Returns:
            Best matching corner pattern or None if no good match found
        """
        try:
            if not self.corner_patterns:
                return None
            
            best_match = None
            best_score = float('inf')
            match_threshold = 0.5  # Similarity threshold
            
            # Calculate track position for matching
            detected_position = segment_data.get('position_start', 0.0)
            
            for pattern in self.corner_patterns:
                pattern_chars = pattern['characteristics']
                pattern_position = pattern.get('track_position_start', 0.0)
                
                # Calculate similarity score based on multiple factors
                similarity_score = self._calculate_pattern_similarity(
                    detected_characteristics, 
                    pattern_chars, 
                    detected_position, 
                    pattern_position
                )
                
                if similarity_score < best_score and similarity_score < match_threshold:
                    best_score = similarity_score
                    best_match = pattern
            
            return best_match
            
        except Exception as e:
            print(f"[ERROR] Failed to find best matching pattern: {str(e)}")
            return None
    
    def _calculate_pattern_similarity(self, detected_chars: CornerCharacteristics, 
                                    pattern_chars: Dict[str, Any],
                                    detected_position: float, 
                                    pattern_position: float) -> float:
        """
        Calculate similarity score between detected corner and learned pattern
        Lower score = better match
        
        Args:
            detected_chars: Detected corner characteristics
            pattern_chars: Learned pattern characteristics
            detected_position: Track position of detected corner
            pattern_position: Track position of learned pattern
            
        Returns:
            Similarity score (lower = more similar)
        """
        try:
            score = 0.0
            
            # Position similarity (weight 25%)
            position_diff = abs(detected_position - pattern_position)
            position_diff = min(position_diff, 1.0 - position_diff) if position_diff > 0.5 else position_diff
            score += position_diff * 0.25
            # Angle similarity (30%)
            da = detected_chars.total_angle_deg
            pa = pattern_chars.get('total_angle_deg', 0)
            if max(da, pa) > 0:
                score += (abs(da - pa) / max(da, pa)) * 0.30
            # Radius similarity (25%)
            dr = detected_chars.radius_est_m
            pr = pattern_chars.get('radius_est_m', 0)
            if max(dr, pr) > 0:
                score += (abs(dr - pr) / max(dr, pr)) * 0.25
            # Curvature similarity (20%)
            dc = detected_chars.avg_curvature
            pc = pattern_chars.get('avg_curvature', 0)
            if max(dc, pc) > 0:
                score += (abs(dc - pc) / max(dc, pc)) * 0.20
            
            return score
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate pattern similarity: {str(e)}")
            return float('inf')
    
    def _create_characteristics_from_pattern(self, pattern: Dict[str, Any]) -> CornerCharacteristics:
        """
        Create CornerCharacteristics object from a learned pattern
        
        Args:
            pattern: Learned corner pattern dictionary
            
        Returns:
            CornerCharacteristics object populated with pattern data
        """
        characteristics = CornerCharacteristics()
        pattern_chars = pattern['characteristics']
        
        try:
            # Copy all characteristics from the pattern
            for attr_name in dir(characteristics):
                if not attr_name.startswith('_') and hasattr(characteristics, attr_name):
                    if attr_name in pattern_chars:
                        setattr(characteristics, attr_name, pattern_chars[attr_name])
            
            return characteristics
            
        except Exception as e:
            print(f"[ERROR] Failed to create characteristics from pattern: {str(e)}")
            return characteristics
    
    def clear_corner_cache(self):
        """Clear cached corner identification models"""
        self.track_corner_profiles.clear()
        self.corner_patterns.clear()
        self.corner_models.clear()
        self.scalers.clear()
        print(f"[INFO] Cleared corner identification cache")
        
    def get_corner_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded corner identification models
        
        Returns:
            Model summary information
        """
        return {
            "model_type": "corner_identification",
            "cached_models": len(self.track_corner_profiles),
            "available_models": list(self.track_corner_profiles.keys()),
            "corner_patterns_count": len(self.corner_patterns),
            "has_active_patterns": len(self.corner_patterns) > 0
        }

    def predict_corner_count(self, corner_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict how many distinct corners are present in a corner DataFrame
        
        This function analyzes telemetry patterns to estimate the number of separate
        corner sections, useful for understanding track layout and corner density.
        
        Args:
            corner_df: DataFrame containing telemetry data from corner sections
            
        Returns:
            Dictionary with predicted corner count and analysis details
        """
        try:
            if len(corner_df) == 0:
                return {
                    "predicted_corners": 0,
                    "confidence": 0.0,
                    "analysis": "Empty DataFrame provided"
                }
            
            # Required columns for corner prediction
            required_cols = ['Physics_steer_angle', 'Physics_speed_kmh']
            missing_cols = [col for col in required_cols if col not in corner_df.columns]
            
            if missing_cols:
                return {
                    "predicted_corners": 0,
                    "confidence": 0.0,
                    "analysis": f"Missing required columns: {missing_cols}"
                }
            
            print(f"[INFO] Predicting corner count for DataFrame with {len(corner_df)} data points")
            
            # Method 1: Peak detection in steering angle
            corners_from_steering = self._predict_corners_from_steering_peaks(corner_df)
            
            # Method 2: Speed valley detection
            corners_from_speed = self._predict_corners_from_speed_valleys(corner_df)
            
            # Method 3: Track position analysis (if available)
            corners_from_position = self._predict_corners_from_track_position(corner_df)
            
            # Method 4: G-force pattern analysis (if available)
            corners_from_gforce = self._predict_corners_from_gforce_patterns(corner_df)
            
            # Combine predictions using weighted average
            predictions = []
            weights = []
            
            if corners_from_steering['confidence'] > 0.3:
                predictions.append(corners_from_steering['count'])
                weights.append(corners_from_steering['confidence'] * 0.4)  # High weight for steering
            
            if corners_from_speed['confidence'] > 0.3:
                predictions.append(corners_from_speed['count'])
                weights.append(corners_from_speed['confidence'] * 0.3)  # Medium weight for speed
            
            if corners_from_position['confidence'] > 0.3:
                predictions.append(corners_from_position['count'])
                weights.append(corners_from_position['confidence'] * 0.2)  # Lower weight for position
            
            if corners_from_gforce['confidence'] > 0.3:
                predictions.append(corners_from_gforce['count'])
                weights.append(corners_from_gforce['confidence'] * 0.1)  # Lowest weight for g-force
            
            # Calculate weighted prediction
            if predictions and weights:
                weighted_prediction = np.average(predictions, weights=weights)
                final_prediction = max(1, round(weighted_prediction))  # At least 1 corner
                confidence = min(1.0, sum(weights) / sum([0.4, 0.3, 0.2, 0.1]))
            else:
                # Fallback: assume at least 1 corner if we have corner data
                final_prediction = 1
                confidence = 0.2
            
            analysis_details = {
                "data_length": len(corner_df),
                "steering_analysis": corners_from_steering,
                "speed_analysis": corners_from_speed,
                "position_analysis": corners_from_position,
                "gforce_analysis": corners_from_gforce,
                "method_weights": {
                    "steering": 0.4,
                    "speed": 0.3,
                    "position": 0.2,
                    "gforce": 0.1
                }
            }
            
            print(f"[INFO] Predicted {final_prediction} corners with {confidence:.2f} confidence")
            
            return {
                "predicted_corners": final_prediction,
                "confidence": float(confidence),
                "analysis": analysis_details
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to predict corner count: {str(e)}")
            return {
                "predicted_corners": 1,
                "confidence": 0.1,
                "analysis": f"Error during prediction: {str(e)}"
            }
    
    def _predict_corners_from_steering_peaks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict corners based on steering angle peaks"""
        try:
            steering = np.abs(df['Physics_steer_angle'].fillna(0))
            
            if len(steering) < 5 or steering.max() < 0.05:  # Very little steering activity
                return {"count": 0, "confidence": 0.0, "details": "Insufficient steering activity"}
            
            # Smooth the steering data
            if len(steering) >= 7:
                steering_smooth = savgol_filter(steering, 7, 2)
            else:
                steering_smooth = steering.rolling(window=3, center=True).mean().fillna(steering)
            
            # Find peaks in steering angle
            # Use adaptive threshold based on data characteristics
            threshold = np.percentile(steering_smooth, 60)  # 60th percentile as threshold
            min_distance = max(5, len(steering) // 20)  # Minimum distance between peaks
            
            peaks, peak_properties = find_peaks(
                steering_smooth,
                height=threshold,
                distance=min_distance,
                prominence=threshold * 0.3
            )
            
            # Filter out small peaks
            significant_peaks = []
            for peak in peaks:
                if steering_smooth[peak] > np.mean(steering_smooth) * 1.5:
                    significant_peaks.append(peak)
            
            corner_count = len(significant_peaks)
            
            # Calculate confidence based on peak characteristics
            if corner_count > 0:
                peak_heights = [steering_smooth[p] for p in significant_peaks]
                peak_consistency = 1.0 - (np.std(peak_heights) / (np.mean(peak_heights) + 0.001))
                confidence = min(0.9, max(0.3, peak_consistency))
            else:
                confidence = 0.1
            
            return {
                "count": corner_count,
                "confidence": float(confidence),
                "details": {
                    "total_peaks": len(peaks),
                    "significant_peaks": len(significant_peaks),
                    "threshold": float(threshold),
                    "steering_max": float(steering.max())
                }
            }
            
        except Exception as e:
            return {"count": 0, "confidence": 0.0, "details": f"Error: {str(e)}"}
    
    def _predict_corners_from_speed_valleys(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict corners based on speed valleys (low speed points)"""
        try:
            speed = df['Physics_speed_kmh'].fillna(0)
            
            if len(speed) < 5 or speed.std() < 5:  # Not much speed variation
                return {"count": 0, "confidence": 0.0, "details": "Insufficient speed variation"}
            
            # Smooth the speed data
            speed_smooth = speed.rolling(window=5).mean().fillna(speed)
            
            # Find valleys (minimum points) in speed
            # Invert speed to find valleys as peaks
            inverted_speed = speed_smooth.max() - speed_smooth
            
            threshold = np.percentile(inverted_speed, 70)  # Look for significant dips
            min_distance = max(5, len(speed) // 15)
            
            valleys, valley_properties = find_peaks(
                inverted_speed,
                height=threshold,
                distance=min_distance,
                prominence=threshold * 0.2
            )
            
            # Filter valleys that represent significant speed reduction
            significant_valleys = []
            for valley in valleys:
                original_speed_at_valley = speed_smooth.iloc[valley]
                if original_speed_at_valley < np.percentile(speed_smooth, 40):  # Bottom 40% of speeds
                    significant_valleys.append(valley)
            
            corner_count = len(significant_valleys)
            
            # Calculate confidence
            if corner_count > 0:
                valley_depths = [inverted_speed[v] for v in significant_valleys]
                depth_consistency = 1.0 - (np.std(valley_depths) / (np.mean(valley_depths) + 0.001))
                confidence = min(0.8, max(0.2, depth_consistency))
            else:
                confidence = 0.1
            
            return {
                "count": corner_count,
                "confidence": float(confidence),
                "details": {
                    "total_valleys": len(valleys),
                    "significant_valleys": len(significant_valleys),
                    "speed_range": float(speed.max() - speed.min()),
                    "speed_std": float(speed.std())
                }
            }
            
        except Exception as e:
            return {"count": 0, "confidence": 0.0, "details": f"Error: {str(e)}"}
    
    def _predict_corners_from_track_position(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict corners based on track position changes"""
        try:
            if 'Graphics_normalized_car_position' not in df.columns:
                return {"count": 0, "confidence": 0.0, "details": "Track position data not available"}
            
            position = df['Graphics_normalized_car_position'].fillna(method='ffill').fillna(method='bfill')
            
            if len(position) < 10 or position.nunique() < 5:
                return {"count": 0, "confidence": 0.0, "details": "Insufficient position variation"}
            
            # Calculate position change rate
            position_diff = position.diff().fillna(0)
            
            # Handle position wraparound (0 to 1 or 1 to 0)
            wraparound_threshold = 0.5
            position_diff[position_diff > wraparound_threshold] -= 1.0
            position_diff[position_diff < -wraparound_threshold] += 1.0
            
            # Smooth position changes
            if len(position_diff) >= 5:
                position_smooth = position_diff.rolling(window=5).mean().fillna(position_diff)
            else:
                position_smooth = position_diff
            
            # Estimate corners based on position progression
            # Corners typically show consistent position advancement
            position_range = position.max() - position.min()
            
            if position_range < 0.05:  # Very small position range - likely single corner
                corner_count = 1
                confidence = 0.6
            elif position_range > 0.8:  # Large range - likely multiple corners or full lap
                # Estimate based on position segments
                corner_count = max(1, int(position_range * 15))  # Rough estimate
                confidence = 0.4
            else:
                # Medium range - estimate based on position variation
                position_segments = int(position_range * 20)
                corner_count = max(1, min(5, position_segments))
                confidence = 0.5
            
            return {
                "count": corner_count,
                "confidence": float(confidence),
                "details": {
                    "position_range": float(position_range),
                    "position_start": float(position.iloc[0]) if len(position) > 0 else 0.0,
                    "position_end": float(position.iloc[-1]) if len(position) > 0 else 0.0,
                    "unique_positions": position.nunique()
                }
            }
            
        except Exception as e:
            return {"count": 0, "confidence": 0.0, "details": f"Error: {str(e)}"}
    
    def _predict_corners_from_gforce_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict corners based on G-force patterns"""
        try:
            g_force_cols = ['Physics_g_force_x', 'Physics_g_force_z']
            available_g_cols = [col for col in g_force_cols if col in df.columns]
            
            if not available_g_cols:
                return {"count": 0, "confidence": 0.0, "details": "G-force data not available"}
            
            # Combine lateral and longitudinal G-forces
            g_combined = 0
            g_details = {}
            
            if 'Physics_g_force_x' in df.columns:  # Lateral G-force
                g_lat = np.abs(df['Physics_g_force_x'].fillna(0))
                g_combined += g_lat
                g_details['lateral_max'] = float(g_lat.max())
                g_details['lateral_std'] = float(g_lat.std())
            
            if 'Physics_g_force_z' in df.columns:  # Longitudinal G-force  
                g_long = np.abs(df['Physics_g_force_z'].fillna(0))
                g_combined += g_long * 0.5  # Weight longitudinal less than lateral
                g_details['longitudinal_max'] = float(g_long.max())
                g_details['longitudinal_std'] = float(g_long.std())
            
            if isinstance(g_combined, int):  # No G-force data found
                return {"count": 0, "confidence": 0.0, "details": "No usable G-force data"}
            
            # Find peaks in combined G-force
            if len(g_combined) >= 7:
                g_smooth = savgol_filter(g_combined, 7, 2)
            else:
                g_smooth = g_combined.rolling(window=3).mean().fillna(g_combined)
            
            threshold = np.percentile(g_smooth, 65)
            min_distance = max(3, len(g_smooth) // 25)
            
            peaks, _ = find_peaks(
                g_smooth,
                height=threshold,
                distance=min_distance
            )
            
            corner_count = len(peaks)
            
            # Calculate confidence based on G-force activity
            if corner_count > 0 and g_smooth.max() > 0.5:  # Reasonable G-force levels
                g_force_range = g_smooth.max() - g_smooth.min()
                confidence = min(0.7, max(0.2, g_force_range / 3.0))  # Scale with G-force range
            else:
                confidence = 0.1
            
            g_details.update({
                "peaks_found": len(peaks),
                "combined_max": float(g_smooth.max()),
                "combined_mean": float(g_smooth.mean())
            })
            
            return {
                "count": corner_count,
                "confidence": float(confidence),
                "details": g_details
            }
            
        except Exception as e:
            return {"count": 0, "confidence": 0.0, "details": f"Error: {str(e)}"}

    def export_corner_patterns(self) -> Dict[str, Any]:
        """
        Export current corner patterns for external saving
        
        Returns:
            Dictionary containing corner patterns and related data
        """
        return {
            "corner_patterns": self.corner_patterns,
            "track_corner_profiles": self.track_corner_profiles,
            "export_timestamp": datetime.now().isoformat()
        }


# Create service instance
corner_identification_service = CornerIdentificationUnsupervisedService()
