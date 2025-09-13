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


class CornerCharacteristics:
    """Data class for corner characteristics"""
    
    def __init__(self):
        # Entry phase characteristics
        self.entry_duration = 0.0
        self.entry_speed_delta = 0.0
        self.entry_brake_intensity = 0.0
        self.entry_steering_rate = 0.0
        self.entry_g_force_lat_max = 0.0
        self.entry_g_force_long_max = 0.0
        
        # Apex phase characteristics
        self.apex_duration = 0.0
        self.apex_min_speed = 0.0
        self.apex_max_steering = 0.0
        self.apex_curvature = 0.0
        self.apex_g_force_lat = 0.0
        self.apex_track_position = 0.0
        
        # Exit phase characteristics
        self.exit_duration = 0.0
        self.exit_speed_delta = 0.0
        self.exit_throttle_progression = 0.0
        self.exit_steering_unwind_rate = 0.0
        self.exit_g_force_lat_max = 0.0
        self.exit_g_force_long_max = 0.0
        
        # Overall corner characteristics
        self.total_corner_duration = 0.0
        self.corner_severity = 0.0
        self.corner_type = "unknown"  # hairpin, chicane, sweeper, etc.
        self.corner_direction = "unknown"  # left, right
        self.speed_efficiency = 0.0
        self.racing_line_adherence = 0.0
        
        # Curvature analysis
        self.avg_curvature = 0.0
        self.max_curvature = 0.0
        self.curvature_variance = 0.0
        
        # Advanced metrics
        self.trail_braking_score = 0.0
        self.throttle_discipline_score = 0.0
        self.consistency_score = 0.0


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
        """
        Initialize the corner identification service
        
        """
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
        Identify corner segments using unsupervised clustering of telemetry patterns
        """
        # Required columns for corner identification
        required_cols = ['Physics_steer_angle', 'Physics_speed_kmh']
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
            steering_abs = np.abs(df['Physics_steer_angle'].fillna(0))
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
            
            # Lateral G-force (if available)
            if 'Physics_g_force_x' in df.columns:
                g_force_lat = np.abs(df['Physics_g_force_x'].fillna(0))
                features.append(g_force_lat)
            
            # Brake and throttle (if available)
            if 'Physics_brake' in df.columns:
                brake = df['Physics_brake'].fillna(0)
                features.append(brake)
            
            if 'Physics_gas' in df.columns:
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
        """Extract corner segments from cluster labels"""
        corner_segments = {}
        corner_id = 0
        
        # Find transitions in labels to identify segments
        label_changes = np.diff(labels, prepend=labels[0])
        segment_starts = np.where(label_changes != 0)[0]
        
        # Add end of data as final segment boundary
        segment_starts = np.append(segment_starts, len(labels))
        
        for i in range(len(segment_starts) - 1):
            start_idx = segment_starts[i]
            end_idx = segment_starts[i + 1] - 1
            
            segment_label = labels[start_idx]
            segment_length = end_idx - start_idx + 1
            
            # Only consider segments that might be corners (non-zero labels or high activity)
            is_corner_candidate = (
                segment_label != 0 or  # Non-zero cluster label
                segment_length >= self.min_corner_duration or  # Long enough duration
                self._is_high_activity_segment(features[start_idx:end_idx+1])  # High activity
            )
            
            if is_corner_candidate and segment_length >= self.min_corner_duration:
                # Calculate track position if available
                position_start = 0.0
                position_end = 0.0
                
                if 'Graphics_normalized_car_position' in df.columns:
                    pos_col = df['Graphics_normalized_car_position']
                    if start_idx < len(pos_col) and not pd.isna(pos_col.iloc[start_idx]):
                        position_start = float(pos_col.iloc[start_idx])
                    if end_idx < len(pos_col) and not pd.isna(pos_col.iloc[end_idx]):
                        position_end = float(pos_col.iloc[end_idx])
                
                corner_segments[corner_id] = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'position_start': position_start,
                    'position_end': position_end,
                    'cluster_label': int(segment_label),
                    'segment_length': segment_length
                }
                
                corner_id += 1
        
        print(f"[INFO] Extracted {len(corner_segments)} corner segments from clustering")
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
        """Extract detailed characteristics from a corner segment"""
        characteristics = CornerCharacteristics()
        
        if len(corner_df) == 0:
            return characteristics
        
        try:
            # Basic duration and position
            characteristics.total_corner_duration = len(corner_df)
            
            # Speed analysis
            if 'Physics_speed_kmh' in corner_df.columns:
                speed = corner_df['Physics_speed_kmh'].fillna(0)
                characteristics.apex_min_speed = float(speed.min())
                
                # Speed efficiency (how much speed is maintained)
                if speed.max() > 0:
                    characteristics.speed_efficiency = float(speed.min() / speed.max())
            
            # Steering analysis
            if 'Physics_steer_angle' in corner_df.columns:
                steering = corner_df['Physics_steer_angle'].fillna(0)
                characteristics.apex_max_steering = float(np.abs(steering).max())
                
                # Determine corner direction
                avg_steering = steering.mean()
                characteristics.corner_direction = "right" if avg_steering > 0 else "left"
                
                # Calculate curvature metrics
                characteristics.avg_curvature = float(np.abs(steering).mean())
                characteristics.max_curvature = float(np.abs(steering).max())
                characteristics.curvature_variance = float(np.var(np.abs(steering)))
            
            # G-force analysis
            if 'Physics_g_force_x' in corner_df.columns:
                g_lat = corner_df['Physics_g_force_x'].fillna(0)
                characteristics.apex_g_force_lat = float(np.abs(g_lat).max())
            
            if 'Physics_g_force_z' in corner_df.columns:
                g_long = corner_df['Physics_g_force_z'].fillna(0)
                characteristics.entry_g_force_long_max = float(g_long.max())  # Braking
                characteristics.exit_g_force_long_max = float(g_long.min())   # Acceleration
            
            # Phase analysis
            self._analyze_corner_phases(corner_df, characteristics)
            
            # Corner type classification
            characteristics.corner_type = self._classify_corner_type(characteristics)
            
            # Advanced scoring
            self._calculate_advanced_scores(corner_df, characteristics)
            
        except Exception as e:
            print(f"[WARNING] Error extracting corner characteristics: {str(e)}")
        
        return characteristics
    
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
                        steering_diff = np.abs(np.diff(entry_df['Physics_steer_angle']))
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
                        steering_abs = np.abs(exit_df['Physics_steer_angle'])
                        if len(steering_abs) > 1:
                            unwind_rate = np.diff(steering_abs).mean()
                            characteristics.exit_steering_unwind_rate = float(-unwind_rate)  # Negative because unwinding
                        
        except Exception as e:
            print(f"[WARNING] Error analyzing corner phases: {str(e)}")
    
    def _classify_corner_type(self, characteristics: CornerCharacteristics) -> str:
        """Classify the type of corner based on characteristics"""
        try:
            duration = characteristics.total_corner_duration
            max_steering = characteristics.apex_max_steering
            speed_efficiency = characteristics.speed_efficiency
            avg_curvature = characteristics.avg_curvature
            
            # Hairpin: long duration, high steering, low speed efficiency
            if duration > 30 and max_steering > 0.5 and speed_efficiency < 0.4:
                return "hairpin"
            
            # Chicane: short duration, high steering variation
            elif duration < 20 and characteristics.curvature_variance > 0.2:
                return "chicane"
            
            # Sweeper: long duration, moderate steering, high speed efficiency  
            elif duration > 25 and max_steering < 0.4 and speed_efficiency > 0.7:
                return "sweeper"
            
            # Fast corner: short-medium duration, low-medium steering, high speed
            elif duration < 25 and max_steering < 0.3 and speed_efficiency > 0.8:
                return "fast_corner"
            
            # Tight corner: medium duration, high steering
            elif max_steering > 0.4 and speed_efficiency < 0.6:
                return "tight_corner"
            
            # Medium corner: default case
            else:
                return "medium_corner"
                
        except Exception as e:
            print(f"[WARNING] Error classifying corner type: {str(e)}")
            return "unknown"
    
    def _calculate_advanced_scores(self, corner_df: pd.DataFrame, characteristics: CornerCharacteristics):
        """Calculate advanced performance scores"""
        try:
            # Trail braking score
            if 'Physics_brake' in corner_df.columns and 'Physics_steer_angle' in corner_df.columns:
                brake = corner_df['Physics_brake'].fillna(0)
                steering = np.abs(corner_df['Physics_steer_angle'].fillna(0))
                
                # Trail braking is simultaneous braking and steering
                overlap = (brake > 0.1) & (steering > 0.1)
                characteristics.trail_braking_score = float(overlap.sum() / len(corner_df))
            
            # Throttle discipline score (smooth throttle application)
            if 'Physics_gas' in corner_df.columns:
                throttle = corner_df['Physics_gas'].fillna(0)
                if len(throttle) > 1:
                    throttle_smoothness = 1.0 - (np.var(np.diff(throttle)) / (np.mean(throttle) + 0.001))
                    characteristics.throttle_discipline_score = max(0.0, min(1.0, throttle_smoothness))
            
            # Consistency score (how consistent the inputs are)
            consistency_scores = []
            
            for col in ['Physics_steer_angle', 'Physics_speed_kmh', 'Physics_brake', 'Physics_gas']:
                if col in corner_df.columns:
                    data = corner_df[col].fillna(0)
                    if len(data) > 1 and data.std() > 0:
                        # Lower coefficient of variation = higher consistency
                        cv = data.std() / (abs(data.mean()) + 0.001)
                        consistency = 1.0 / (1.0 + cv)
                        consistency_scores.append(consistency)
            
            if consistency_scores:
                characteristics.consistency_score = float(np.mean(consistency_scores))
                
        except Exception as e:
            print(f"[WARNING] Error calculating advanced scores: {str(e)}")
    
    def _cluster_corner_types(self, corner_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster similar corners to identify common corner types on the track"""
        if not corner_patterns:
            return {"clusters": [], "cluster_labels": []}
        
        try:
            # Create feature matrix from corner characteristics
            feature_names = [
                'total_corner_duration', 'apex_max_steering', 'speed_efficiency',
                'avg_curvature', 'entry_duration', 'apex_duration', 'exit_duration'
            ]
            
            features = []
            for pattern in corner_patterns:
                char = pattern['characteristics']
                feature_row = [
                    char.get(name, 0.0) for name in feature_names
                ]
                features.append(feature_row)
            
            features_array = np.array(features)
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # Determine optimal number of clusters
            max_clusters = min(6, len(corner_patterns) // 2)
            if max_clusters < 2:
                return {"clusters": [{"type": "single_type", "corners": list(range(len(corner_patterns)))}], "cluster_labels": [0] * len(corner_patterns)}
            
            best_score = -1
            best_labels = None
            best_n_clusters = 2
            
            for n in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_scaled)
                
                try:
                    score = silhouette_score(features_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_n_clusters = n
                except:
                    continue
            
            if best_labels is None:
                best_labels = [0] * len(corner_patterns)
            
            # Create cluster descriptions
            clusters = []
            for cluster_id in range(best_n_clusters):
                cluster_corners = [i for i, label in enumerate(best_labels) if label == cluster_id]
                if not cluster_corners:
                    continue
                
                # Analyze cluster characteristics
                cluster_chars = [corner_patterns[i]['characteristics'] for i in cluster_corners]
                avg_duration = np.mean([c.get('total_corner_duration', 0) for c in cluster_chars])
                avg_steering = np.mean([c.get('apex_max_steering', 0) for c in cluster_chars])
                avg_speed_eff = np.mean([c.get('speed_efficiency', 0) for c in cluster_chars])
                
                # Classify cluster type
                if avg_duration > 30 and avg_steering > 0.5:
                    cluster_type = "hairpins"
                elif avg_duration < 20:
                    cluster_type = "chicanes_tight_corners"
                elif avg_speed_eff > 0.7:
                    cluster_type = "fast_sweepers"
                else:
                    cluster_type = "medium_corners"
                
                clusters.append({
                    "cluster_id": int(cluster_id),
                    "type": cluster_type,
                    "corner_count": len(cluster_corners),
                    "corners": cluster_corners,
                    "avg_characteristics": {
                        "duration": float(avg_duration),
                        "max_steering": float(avg_steering),
                        "speed_efficiency": float(avg_speed_eff)
                    }
                })
            
            return {
                "clusters": clusters,
                "cluster_labels": best_labels.tolist(),
                "silhouette_score": float(best_score)
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to cluster corner types: {str(e)}")
            return {"clusters": [], "cluster_labels": []}
    
    def _initialize_corner_feature_columns(self, data_length: int) -> Dict[str, List[float]]:
        """Initialize corner feature columns with default values"""
        feature_names = [
            # Corner identification
            'corner_id', 'is_in_corner', 'corner_phase',
            
            # Entry phase features
            'corner_entry_duration', 'corner_entry_speed_delta', 'corner_entry_brake_intensity',
            'corner_entry_steering_rate', 'corner_entry_g_force_lat_max', 'corner_entry_g_force_long_max',
            
            # Apex phase features  
            'corner_apex_duration', 'corner_apex_min_speed', 'corner_apex_max_steering',
            'corner_apex_curvature', 'corner_apex_g_force_lat',
            
            # Exit phase features
            'corner_exit_duration', 'corner_exit_speed_delta', 'corner_exit_throttle_progression',
            'corner_exit_steering_unwind_rate', 'corner_exit_g_force_lat_max', 'corner_exit_g_force_long_max',
            
            # Overall corner features
            'corner_total_duration', 'corner_severity', 'corner_type_numeric', 'corner_direction_numeric',
            'corner_speed_efficiency', 'corner_racing_line_adherence',
            
            # Curvature features
            'corner_avg_curvature', 'corner_max_curvature', 'corner_curvature_variance',
            
            # Advanced features
            'corner_trail_braking_score', 'corner_throttle_discipline_score', 'corner_consistency_score'
        ]
        
        corner_features = {}
        for feature_name in feature_names:
            corner_features[feature_name] = [0.0] * data_length
        
        return corner_features
    
    def _assign_corner_features_to_segment(self, corner_features: Dict[str, List[float]], 
                                          start_idx: int, end_idx: int, 
                                          characteristics: CornerCharacteristics, 
                                          corner_id: int):
        """Assign corner characteristics as features to all data points in a corner segment"""
        try:
            # Convert corner type and direction to numeric
            corner_type_map = {
                'hairpin': 1, 'chicane': 2, 'sweeper': 3, 'fast_corner': 4, 
                'tight_corner': 5, 'medium_corner': 6, 'unknown': 0
            }
            direction_map = {'left': -1, 'right': 1, 'unknown': 0}
            
            corner_type_numeric = corner_type_map.get(characteristics.corner_type, 0)
            direction_numeric = direction_map.get(characteristics.corner_direction, 0)
            
            # Assign features to all points in the corner segment
            for i in range(start_idx, min(end_idx + 1, len(corner_features['corner_id']))):
                # Basic identification
                corner_features['corner_id'][i] = float(corner_id)
                corner_features['is_in_corner'][i] = 1.0
                corner_features['corner_phase'][i] = self._determine_phase_for_point(
                    i, start_idx, end_idx, characteristics
                )
                
                # Entry phase features
                corner_features['corner_entry_duration'][i] = characteristics.entry_duration
                corner_features['corner_entry_speed_delta'][i] = characteristics.entry_speed_delta
                corner_features['corner_entry_brake_intensity'][i] = characteristics.entry_brake_intensity
                corner_features['corner_entry_steering_rate'][i] = characteristics.entry_steering_rate
                corner_features['corner_entry_g_force_lat_max'][i] = characteristics.entry_g_force_lat_max
                corner_features['corner_entry_g_force_long_max'][i] = characteristics.entry_g_force_long_max
                
                # Apex phase features
                corner_features['corner_apex_duration'][i] = characteristics.apex_duration
                corner_features['corner_apex_min_speed'][i] = characteristics.apex_min_speed
                corner_features['corner_apex_max_steering'][i] = characteristics.apex_max_steering
                corner_features['corner_apex_curvature'][i] = characteristics.apex_curvature
                corner_features['corner_apex_g_force_lat'][i] = characteristics.apex_g_force_lat
                
                # Exit phase features
                corner_features['corner_exit_duration'][i] = characteristics.exit_duration
                corner_features['corner_exit_speed_delta'][i] = characteristics.exit_speed_delta
                corner_features['corner_exit_throttle_progression'][i] = characteristics.exit_throttle_progression
                corner_features['corner_exit_steering_unwind_rate'][i] = characteristics.exit_steering_unwind_rate
                corner_features['corner_exit_g_force_lat_max'][i] = characteristics.exit_g_force_lat_max
                corner_features['corner_exit_g_force_long_max'][i] = characteristics.exit_g_force_long_max
                
                # Overall corner features
                corner_features['corner_total_duration'][i] = characteristics.total_corner_duration
                corner_features['corner_severity'][i] = characteristics.corner_severity
                corner_features['corner_type_numeric'][i] = float(corner_type_numeric)
                corner_features['corner_direction_numeric'][i] = float(direction_numeric)
                corner_features['corner_speed_efficiency'][i] = characteristics.speed_efficiency
                corner_features['corner_racing_line_adherence'][i] = characteristics.racing_line_adherence
                
                # Curvature features
                corner_features['corner_avg_curvature'][i] = characteristics.avg_curvature
                corner_features['corner_max_curvature'][i] = characteristics.max_curvature
                corner_features['corner_curvature_variance'][i] = characteristics.curvature_variance
                
                # Advanced features
                corner_features['corner_trail_braking_score'][i] = characteristics.trail_braking_score
                corner_features['corner_throttle_discipline_score'][i] = characteristics.throttle_discipline_score
                corner_features['corner_consistency_score'][i] = characteristics.consistency_score
        
        except Exception as e:
            print(f"[WARNING] Error assigning corner features: {str(e)}")
    
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
            for corner_id, segment_data in corner_segments.items():
                start_idx = segment_data['start_idx']
                end_idx = segment_data['end_idx']
                
                # Extract characteristics for this detected corner
                corner_df = df.iloc[start_idx:end_idx+1]
                detected_characteristics = self._extract_corner_characteristics(corner_df)
                
                # Find the best matching learned pattern
                best_match_pattern = self._find_best_matching_pattern(detected_characteristics, segment_data)
                
                if best_match_pattern:
                    # Use the learned pattern's characteristics
                    learned_characteristics = self._create_characteristics_from_pattern(best_match_pattern)
                    print(f"[INFO] Corner {corner_id} matched with learned pattern (type: {learned_characteristics.corner_type})")
                else:
                    # Fallback to detected characteristics if no good match found
                    learned_characteristics = detected_characteristics
                    print(f"[INFO] Corner {corner_id} using detected characteristics (no good pattern match)")
                
                # Assign corner features to all data points in this corner
                self._assign_corner_features_to_segment(
                    corner_features, 
                    start_idx, 
                    end_idx, 
                    learned_characteristics, 
                    corner_id
                )
                
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
            
            # Position similarity (weight: 30%)
            position_diff = abs(detected_position - pattern_position)
            # Handle wraparound for normalized position (0-1)
            position_diff = min(position_diff, 1.0 - position_diff) if position_diff > 0.5 else position_diff
            score += position_diff * 0.3
            
            # Duration similarity (weight: 20%)
            detected_duration = detected_chars.total_corner_duration
            pattern_duration = pattern_chars.get('total_corner_duration', 0)
            if pattern_duration > 0:
                duration_diff = abs(detected_duration - pattern_duration) / max(detected_duration, pattern_duration)
                score += duration_diff * 0.2
            
            # Steering similarity (weight: 25%)
            detected_steering = detected_chars.apex_max_steering
            pattern_steering = pattern_chars.get('apex_max_steering', 0)
            if max(detected_steering, pattern_steering) > 0:
                steering_diff = abs(detected_steering - pattern_steering) / max(detected_steering, pattern_steering)
                score += steering_diff * 0.25
            
            # Speed efficiency similarity (weight: 15%)
            detected_speed_eff = detected_chars.speed_efficiency
            pattern_speed_eff = pattern_chars.get('speed_efficiency', 0)
            if max(detected_speed_eff, pattern_speed_eff) > 0:
                speed_diff = abs(detected_speed_eff - pattern_speed_eff) / max(detected_speed_eff, pattern_speed_eff)
                score += speed_diff * 0.15
            
            # Corner type similarity (weight: 10%)
            detected_type = detected_chars.corner_type
            pattern_type = pattern_chars.get('corner_type', 'unknown')
            if detected_type != pattern_type:
                score += 0.1
            
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
