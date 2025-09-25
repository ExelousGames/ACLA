import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler


class TrackCorneringAnalyzer:
    """
    A comprehensive class for identifying and analyzing track cornering phases
    from telemetry data including steering angles, speed, brake, and throttle data.
    
    Corner Phases Defined (Based on Actual Racing Dynamics):
    - Entry: The braking zone - detected by brake application and speed reduction
    - Turn-in: Steering toward apex with speed constantly slowing down and steering movement,
               may include trail braking or natural engine braking
    - Apex: The slowest point of the corner (minimum speed)
    - Acceleration: Gradual throttle application with reducing steering after apex
    - Exit: Full throttle application as the car straightens out
    """
    
    def __init__(self, steering_threshold_percentile: float = 0.7, 
                 min_corner_duration: int = 20,
                 min_valid_percentage: float = 0.95,
                 top_laps_percentage: float = 0.05):
        """
        Initialize the TrackCorneringAnalyzer
        
        Args:
            steering_threshold_percentile: Percentile for steering angle threshold (default 0.7 for top 30%)
            min_corner_duration: Minimum duration for a corner to be considered valid (data points)
            min_valid_percentage: Minimum percentage of valid data points required for lap filtering
            top_laps_percentage: Percentage of top laps to keep for analysis (default 5%)
        """
        self.steering_threshold_percentile = steering_threshold_percentile
        self.min_corner_duration = min_corner_duration
        self.min_valid_percentage = min_valid_percentage
        self.top_laps_percentage = top_laps_percentage
    
    def identify_cornering_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify cornering phases using normalized car position and telemetry data
        
        Args:
            df: DataFrame with telemetry data including Graphics_normalized_car_position
            
        Returns:
            DataFrame with added cornering phase information
        """
        df_analysis = df.copy()
        
        # Initialize phase column
        df_analysis['cornering_phase'] = 'straight'
        df_analysis['corner_id'] = -1
        df_analysis['phase_intensity'] = 0.0
        
        # Required features check
        required_features = ['Graphics_normalized_car_position', 'Physics_steer_angle', 'Physics_speed_kmh']
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing required features: {missing_features}")
            return df_analysis
        
        # Apply Savitzky-Golay smoothing to steering data once at the beginning
        steering_raw = np.abs(df_analysis['Physics_steer_angle']).ffill().bfill()
        
        # Use savgol_filter with window length 7 and polynomial order 2 for better feature preservation
        if len(steering_raw) >= 7:
            df_analysis['steering_smooth'] = savgol_filter(steering_raw, window_length=7, polyorder=2)
        else:
            # Fallback for short data - use simple rolling mean
            df_analysis['steering_smooth'] = steering_raw.rolling(window=3, center=True).mean().fillna(steering_raw)
        
        # Step 1: Identify corners using multi-factor analysis
        corners = self._identify_corners(df_analysis)
        print(f"[INFO] Detected {len(corners)} corners using improved multi-factor analysis")
        
        # Step 2: For each corner, identify the phases
        for corner_id, corner_data in corners.items():
            start_idx, end_idx = corner_data['start_idx'], corner_data['end_idx']
            corner_df = df_analysis.iloc[start_idx:end_idx+1].copy()
            
            # Analyze this corner section
            phases = self._analyze_corner_phases(corner_df, corner_id)
            
            # Update main dataframe with phase information
            for phase_name, indices in phases.items():
                for idx in indices:
                    actual_idx = start_idx + idx
                    if actual_idx < len(df_analysis):
                        df_analysis.loc[actual_idx, 'cornering_phase'] = phase_name
                        df_analysis.loc[actual_idx, 'corner_id'] = corner_id
        
        return df_analysis

    def _identify_corners(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Identify corner sections using multi-factor analysis:
        steering angle, speed reduction, and track position changes
        """
        
        # Use pre-smoothed steering data and prepare smoothed speed data using savgol_filter
        steering_abs = df['steering_smooth']
        speed_raw = df['Physics_speed_kmh'].ffill().bfill()
        if len(speed_raw) >= 7:
            speed = pd.Series(savgol_filter(speed_raw, window_length=7, polyorder=2), index=df.index)
        else:
            speed = speed_raw.rolling(window=3, center=True).mean().fillna(speed_raw)
        
        # Multi-factor corner detection
        corner_indicators = self._calculate_corner_indicators(df, steering_abs, speed)
        
        # Find corner regions using combined indicators
        corner_regions = self._find_corner_regions(corner_indicators, len(df))
        
        # Validate and refine corners
        validated_corners = self._validate_corners(df, corner_regions, steering_abs, speed)
        
        # If no corners found, try fallback simple detection
        if not validated_corners:
            print("[WARNING] Multi-factor detection found no corners, trying simple steering-based detection")
            fallback_corners = self._fallback_steering_detection(df, steering_abs, speed)
            validated_corners = fallback_corners
        
        return validated_corners
    
    def _fallback_steering_detection(self, df: pd.DataFrame, steering_abs: pd.Series, speed: pd.Series) -> Dict[int, Dict[str, Any]]:
        """Fallback corner detection using simple steering angle thresholding"""
        
        # Use a much simpler approach - just steering angle (already smoothed)
        steering_threshold = steering_abs.quantile(self.steering_threshold_percentile * 0.8)  # Lower the threshold
        print(f"[DEBUG] Fallback steering threshold: {steering_threshold:.3f}")
        
        # Find regions where steering exceeds threshold
        in_corner = steering_abs > steering_threshold
        
        corners = {}
        corner_id = 0
        in_corner_section = False
        corner_start = 0
        
        for i, is_cornering in enumerate(in_corner):
            if is_cornering and not in_corner_section:
                # Start of new corner
                corner_start = max(0, i - 5)  # Smaller extension
                in_corner_section = True
            elif not is_cornering and in_corner_section:
                # End of corner
                corner_end = min(len(df) - 1, i + 5)  # Smaller extension
                
                # More lenient minimum duration check
                if corner_end - corner_start > max(10, self.min_corner_duration // 2):
                    # Get position data if available
                    position_start = 0.0
                    position_end = 0.0
                    
                    if 'Graphics_normalized_car_position' in df.columns:
                        pos_col = df['Graphics_normalized_car_position']
                        if not pd.isna(pos_col.iloc[corner_start]):
                            position_start = float(pos_col.iloc[corner_start])
                        if not pd.isna(pos_col.iloc[corner_end]):
                            position_end = float(pos_col.iloc[corner_end])
                    
                    corners[corner_id] = {
                        'start_idx': corner_start,
                        'end_idx': corner_end,
                        'position_start': position_start,
                        'position_end': position_end,
                        'max_steering': steering_abs.iloc[corner_start:corner_end+1].max(),
                        'min_speed': speed.iloc[corner_start:corner_end+1].min(),
                        'avg_steering': steering_abs.iloc[corner_start:corner_end+1].mean()
                    }
                    
                    print(f"[DEBUG] Fallback Corner {corner_id}: indices {corner_start}-{corner_end}")
                    corner_id += 1
                
                in_corner_section = False
        
        # Handle corner that continues to end of data
        if in_corner_section and len(df) - corner_start > max(10, self.min_corner_duration // 2):
            corner_end = len(df) - 1
            position_start = 0.0
            position_end = 0.0
            
            if 'Graphics_normalized_car_position' in df.columns:
                pos_col = df['Graphics_normalized_car_position']
                if not pd.isna(pos_col.iloc[corner_start]):
                    position_start = float(pos_col.iloc[corner_start])
                if not pd.isna(pos_col.iloc[corner_end]):
                    position_end = float(pos_col.iloc[corner_end])
            
            corners[corner_id] = {
                'start_idx': corner_start,
                'end_idx': corner_end,
                'position_start': position_start,
                'position_end': position_end,
                'max_steering': steering_abs.iloc[corner_start:corner_end+1].max(),
                'min_speed': speed.iloc[corner_start:corner_end+1].min(),
                'avg_steering': steering_abs.iloc[corner_start:corner_end+1].mean()
            }
            print(f"[DEBUG] Fallback Corner {corner_id}: indices {corner_start}-{corner_end}")
        
        print(f"[INFO] Fallback detection found {len(corners)} corners")
        return corners
    
    def _calculate_corner_indicators(self, df: pd.DataFrame, steering_abs: pd.Series, speed: pd.Series) -> pd.Series:
        """Calculate a combined corner indicator using multiple telemetry signals"""
        
        # Factor 1: Steering angle (more sensitive thresholds)
        steering_mean = steering_abs.mean()
        steering_std = steering_abs.std()
        steering_threshold = max(steering_mean + 0.5 * steering_std, steering_abs.quantile(0.5))
        steering_indicator = (steering_abs > steering_threshold).astype(float)
        
        # Add secondary steering indicator for moderate steering
        moderate_steering_threshold = max(steering_mean, steering_abs.quantile(0.3))
        moderate_steering_indicator = (steering_abs > moderate_steering_threshold).astype(float) * 0.5
        
        # Factor 2: Speed reduction (more sensitive)
        speed_diff = speed.diff().rolling(window=3).mean().fillna(0)
        # Lower threshold for speed reduction detection
        speed_reduction_indicator = (speed_diff < -1.0).astype(float)
        
        # Add speed variance indicator (corners often have varying speeds)
        speed_variance = speed.rolling(window=10).std().fillna(0)
        speed_var_threshold = speed_variance.quantile(0.6) if speed_variance.max() > 0 else 0
        speed_variance_indicator = (speed_variance > speed_var_threshold).astype(float) * 0.3
        
        # Factor 3: Brake application (if available)
        brake_indicator = pd.Series([0.0] * len(df))
        if 'Physics_brake' in df.columns:
            brake_data = df['Physics_brake'].fillna(0)
            if brake_data.max() > 0:
                brake_threshold = brake_data.quantile(0.3)  # Lower threshold
                brake_indicator = (brake_data > brake_threshold).astype(float)
        
        # Factor 4: Combined steering + speed pattern
        # Look for sustained steering with speed changes
        combined_pattern = steering_indicator * (speed_reduction_indicator + speed_variance_indicator)
        combined_pattern = combined_pattern.rolling(window=5).mean().fillna(0)
        
        # Primary score: steering-based detection
        primary_score = (
            0.6 * steering_indicator +
            0.2 * moderate_steering_indicator +
            0.15 * speed_reduction_indicator +
            0.05 * brake_indicator
        )
        
        # Secondary score: pattern-based detection
        secondary_score = (
            0.4 * combined_pattern +
            0.3 * steering_indicator +
            0.2 * speed_variance_indicator +
            0.1 * brake_indicator
        )
        
        # Take maximum of both approaches
        combined_score = pd.concat([primary_score, secondary_score], axis=1).max(axis=1)
        
        # Light smoothing to reduce noise
        return combined_score.rolling(window=3, center=True).mean().ffill().bfill()
    
    def _find_corner_regions(self, corner_indicators: pd.Series, data_length: int) -> List[Tuple[int, int]]:
        """Find regions where corner indicators suggest cornering activity"""
        
        # More lenient threshold calculation
        indicator_mean = corner_indicators.mean()
        indicator_std = corner_indicators.std()
        indicator_max = corner_indicators.max()
        
        # Try multiple threshold approaches and use the most lenient that still gives reasonable results
        thresholds_to_try = [
            max(0.2, indicator_mean + 0.3 * indicator_std),  # Very lenient
            max(0.3, indicator_mean + 0.5 * indicator_std),  # Lenient 
            corner_indicators.quantile(0.5),                 # Median
            corner_indicators.quantile(0.6),                 # 60th percentile
        ]
        
        corner_threshold = None
        regions_found = 0
        
        # Choose the most lenient threshold that finds at least some corners
        for threshold in thresholds_to_try:
            test_regions = (corner_indicators > threshold).sum()
            if test_regions >= self.min_corner_duration:  # At least minimum corner duration
                corner_threshold = threshold
                regions_found = test_regions
                break
        
        # Fallback: if no threshold works, use a very low fixed threshold
        if corner_threshold is None:
            corner_threshold = max(0.15, indicator_mean)
            
        print(f"[DEBUG] Corner detection threshold: {corner_threshold:.3f} (mean={indicator_mean:.3f}, max={indicator_max:.3f})")
        print(f"[DEBUG] Points above threshold: {(corner_indicators > corner_threshold).sum()}")
        
        # Find regions above threshold
        in_corner = corner_indicators > corner_threshold
        
        regions = []
        region_start = None
        
        for i, is_cornering in enumerate(in_corner):
            if is_cornering and region_start is None:
                # Start of new corner region
                region_start = i
            elif not is_cornering and region_start is not None:
                # End of corner region
                if i - region_start > self.min_corner_duration:
                    regions.append((region_start, i))
                region_start = None
        
        # Handle case where corner continues to end of data
        if region_start is not None and data_length - region_start > self.min_corner_duration:
            regions.append((region_start, data_length - 1))
        
        # Merge nearby regions that might be part of the same corner
        merged_regions = self._merge_nearby_regions(regions, max_gap=15)
        
        print(f"[DEBUG] Found {len(regions)} raw regions, merged to {len(merged_regions)} corners")
        
        return merged_regions
    
    def _merge_nearby_regions(self, regions: List[Tuple[int, int]], max_gap: int = 15) -> List[Tuple[int, int]]:
        """Merge corner regions that are close together (likely same corner)"""
        
        if not regions:
            return regions
        
        # Sort regions by start position
        sorted_regions = sorted(regions, key=lambda x: x[0])
        merged = [sorted_regions[0]]
        
        for current_start, current_end in sorted_regions[1:]:
            last_start, last_end = merged[-1]
            
            # If current region starts within max_gap of previous region end
            if current_start - last_end <= max_gap:
                # Merge regions
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # Keep as separate region
                merged.append((current_start, current_end))
        
        return merged
    
    def _validate_corners(self, df: pd.DataFrame, corner_regions: List[Tuple[int, int]], 
                         steering_abs: pd.Series, speed: pd.Series) -> Dict[int, Dict[str, Any]]:
        """Validate corner regions and create corner dictionary"""
        
        validated_corners = {}
        corner_id = 0
        
        for start_idx, end_idx in corner_regions:
            # Extend region slightly to include approach and exit
            extended_start = max(0, start_idx - 8)
            extended_end = min(len(df) - 1, end_idx + 8)
            
            # Validate this is actually a corner using additional checks
            if self._is_valid_corner_region(df, extended_start, extended_end, steering_abs, speed):
                # Get position data if available
                position_start = 0.0
                position_end = 0.0
                
                if 'Graphics_normalized_car_position' in df.columns:
                    pos_col = df['Graphics_normalized_car_position']
                    if not pd.isna(pos_col.iloc[extended_start]):
                        position_start = float(pos_col.iloc[extended_start])
                    if not pd.isna(pos_col.iloc[extended_end]):
                        position_end = float(pos_col.iloc[extended_end])
                
                max_steering_val = steering_abs.iloc[extended_start:extended_end+1].max()
                min_speed_val = speed.iloc[extended_start:extended_end+1].min()
                avg_steering_val = steering_abs.iloc[extended_start:extended_end+1].mean()
                
                validated_corners[corner_id] = {
                    'start_idx': extended_start,
                    'end_idx': extended_end,
                    'position_start': position_start,
                    'position_end': position_end,
                    'max_steering': max_steering_val,
                    'min_speed': min_speed_val,
                    'avg_steering': avg_steering_val
                }
                
                print(f"[DEBUG] Corner {corner_id}: indices {extended_start}-{extended_end}, "
                      f"max_steering={max_steering_val:.3f}, min_speed={min_speed_val:.1f}km/h")
                corner_id += 1
        
        return validated_corners
    
    def get_corner_detection_debug_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get debug information about corner detection process"""
        
        if len(df) < 50:
            return {"error": "Not enough data for corner detection"}
        
        # Check required features
        required_features = ['Physics_steer_angle', 'Physics_speed_kmh']
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            return {"error": f"Missing required features: {missing_features}"}
        
        # Calculate indicators using pre-smoothed steering if available, otherwise smooth it with savgol_filter
        if 'steering_smooth' in df.columns:
            steering_abs = df['steering_smooth']
        else:
            steering_raw = np.abs(df['Physics_steer_angle']).ffill().bfill()
            if len(steering_raw) >= 7:
                steering_abs = pd.Series(savgol_filter(steering_raw, window_length=7, polyorder=2), index=df.index)
            else:
                steering_abs = steering_raw.rolling(window=3, center=True).mean().fillna(steering_raw)
        
        speed_raw = df['Physics_speed_kmh'].ffill().bfill()
        if len(speed_raw) >= 7:
            speed = pd.Series(savgol_filter(speed_raw, window_length=7, polyorder=2), index=df.index)
        else:
            speed = speed_raw.rolling(window=3, center=True).mean().fillna(speed_raw)
        corner_indicators = self._calculate_corner_indicators(df, steering_abs, speed)
        
        # Try corner detection to see what happens
        try:
            regions = self._find_corner_regions(corner_indicators, len(df))
            validated = self._validate_corners(df, regions, steering_abs, speed)
        except Exception as e:
            regions = []
            validated = {}
        
        # Get statistics
        return {
            "total_data_points": len(df),
            "steering_stats": {
                "mean": float(steering_abs.mean()),
                "max": float(steering_abs.max()),
                "std": float(steering_abs.std()),
                "q30": float(steering_abs.quantile(0.3)),
                "q50": float(steering_abs.quantile(0.5)),
                "q70": float(steering_abs.quantile(0.7)),
                "q90": float(steering_abs.quantile(0.9)),
                "threshold_70pct": float(steering_abs.quantile(self.steering_threshold_percentile))
            },
            "speed_stats": {
                "mean": float(speed.mean()),
                "min": float(speed.min()),
                "max": float(speed.max()),
                "std": float(speed.std()),
                "range": float(speed.max() - speed.min())
            },
            "corner_indicator_stats": {
                "mean": float(corner_indicators.mean()),
                "max": float(corner_indicators.max()),
                "std": float(corner_indicators.std()),
                "q50": float(corner_indicators.quantile(0.5)),
                "q70": float(corner_indicators.quantile(0.7)),
                "q80": float(corner_indicators.quantile(0.8)),
                "points_above_0.1": int((corner_indicators > 0.1).sum()),
                "points_above_0.2": int((corner_indicators > 0.2).sum()),
                "points_above_0.3": int((corner_indicators > 0.3).sum()),
                "points_above_0.5": int((corner_indicators > 0.5).sum()),
                "continuous_regions_above_0.2": len([r for r in regions if (r[1] - r[0]) > 10])
            },
            "detection_results": {
                "raw_regions_found": len(regions),
                "validated_corners_found": len(validated),
                "min_corner_duration": self.min_corner_duration
            },
            "has_brake_data": 'Physics_brake' in df.columns,
            "has_position_data": 'Graphics_normalized_car_position' in df.columns
        }
    
    def _is_valid_corner_region(self, df: pd.DataFrame, start_idx: int, end_idx: int, 
                               steering_abs: pd.Series, speed: pd.Series) -> bool:
        """Additional validation to ensure detected region is actually a corner"""
        
        if end_idx <= start_idx or end_idx - start_idx < self.min_corner_duration:
            return False
        
        corner_section = df.iloc[start_idx:end_idx+1]
        
        # Much more lenient validation criteria
        # Check 1: Must have some steering activity (very lenient)
        avg_steering = steering_abs.iloc[start_idx:end_idx+1].mean()
        max_steering = steering_abs.iloc[start_idx:end_idx+1].max()
        overall_steering_mean = steering_abs.mean()
        
        # Lower thresholds for steering validation
        if max_steering < overall_steering_mean or avg_steering < overall_steering_mean * 0.5:
            print(f"[DEBUG] Rejected region {start_idx}-{end_idx}: insufficient steering (avg={avg_steering:.3f}, max={max_steering:.3f})")
            return False
        
        # Check 2: Should show some speed variation (very lenient)
        corner_speeds = speed.iloc[start_idx:end_idx+1]
        speed_range = corner_speeds.max() - corner_speeds.min()
        
        if speed_range < 2.0:  # Reduced from 5.0 to 2.0 km/h
            print(f"[DEBUG] Rejected region {start_idx}-{end_idx}: insufficient speed variation ({speed_range:.1f} km/h)")
            return False
        
        # Check 3: Steering should have some consistency (very lenient)
        moderate_steering_threshold = max(overall_steering_mean, steering_abs.quantile(0.4))
        moderate_steering_points = (steering_abs.iloc[start_idx:end_idx+1] > moderate_steering_threshold).sum()
        steering_ratio = moderate_steering_points / len(corner_section)
        
        if steering_ratio < 0.15:  # Reduced from 0.3 to 0.15 (15%)
            print(f"[DEBUG] Rejected region {start_idx}-{end_idx}: insufficient sustained steering ({steering_ratio:.1%})")
            return False
        
        print(f"[DEBUG] Validated region {start_idx}-{end_idx}: steering_ratio={steering_ratio:.1%}, speed_range={speed_range:.1f}")
        return True

    def _analyze_corner_phases(self, corner_df: pd.DataFrame, corner_id: int) -> Dict[str, List[int]]:
        """
        Analyze a single corner to identify racing phases based on actual racing dynamics.
        CRITICAL: Each phase MUST follow the previous one in STRICT sequential order:
        - Entry: The braking zone - detected by brake application and speed reduction
        - Turn-in: Steering toward apex with speed constantly slowing down and steering movement
        - Apex: The slowest point of the corner
        - Acceleration: Gradual throttle application with reducing steering after apex
        - Exit: Full throttle application
        """
        
        phases = {
            'entry': [],
            'turn_in': [],
            'apex': [],
            'acceleration': [],
            'exit': []
        }
        
        if len(corner_df) < 10:
            return phases
        
        # Reset index to work with 0-based indexing
        corner_df_indexed = corner_df.reset_index(drop=True)
        corner_length = len(corner_df_indexed)
        
        # Calculate key metrics for phase identification using pre-smoothed steering
        steering_smooth = corner_df_indexed['steering_smooth']
        speed = corner_df_indexed['Physics_speed_kmh']
        
        # Get brake and throttle data
        brake_data = corner_df_indexed.get('Physics_brake', pd.Series([0] * len(corner_df_indexed)))
        throttle_data = corner_df_indexed.get('Physics_gas', pd.Series([0] * len(corner_df_indexed)))
        
        # Smooth other data for better analysis using savgol_filter (steering already smoothed)
        if len(speed) >= 7:
            speed_smooth = pd.Series(savgol_filter(speed.ffill().bfill(), 
                                                  window_length=7, polyorder=2), index=speed.index)
        else:
            speed_smooth = speed.rolling(window=3, center=True).mean().fillna(speed)
        
        if len(brake_data) >= 7:
            brake_smooth = pd.Series(savgol_filter(brake_data.fillna(0), 
                                                  window_length=7, polyorder=2), index=brake_data.index)
        else:
            brake_smooth = brake_data.rolling(window=3, center=True).mean().fillna(brake_data)
        
        if len(throttle_data) >= 7:
            throttle_smooth = pd.Series(savgol_filter(throttle_data.fillna(0), 
                                                     window_length=7, polyorder=2), index=throttle_data.index)
        else:
            throttle_smooth = throttle_data.rolling(window=3, center=True).mean().fillna(throttle_data)
        
        # Calculate speed change (negative = deceleration)
        speed_change = speed_smooth.diff().rolling(window=3).mean().fillna(0)
        
        # Find key reference points
        min_speed_idx = speed_smooth.idxmin()  # Apex point
        
        # ENFORCE STRICT SEQUENTIAL ORDERING - each phase boundary is calculated in order
        # and MUST NOT overlap with previous phases
        
        # Phase 1: Entry - The braking zone (starts at beginning)
        entry_start = 0
        entry_end = self._find_entry_phase_end_v2(brake_smooth, speed_change, corner_length)
        entry_end = max(2, min(entry_end, corner_length // 4))  # Entry cannot exceed 25% of corner
        phases['entry'] = list(range(entry_start, entry_end))
        
        # Phase 2: Turn-in - MUST start after entry ends, MUST end before or at apex
        turn_in_start = entry_end  # STRICT: starts exactly where entry ends
        turn_in_end = self._find_turn_in_phase_end_v2(steering_smooth, speed_change, brake_smooth, 
                                                      turn_in_start, min_speed_idx, corner_length)
        # ENFORCE: turn-in cannot go past apex and must leave room for apex phase
        turn_in_end = min(turn_in_end, min_speed_idx - 1)  
        turn_in_end = max(turn_in_start + 2, turn_in_end)  # Minimum turn-in length
        phases['turn_in'] = list(range(turn_in_start, turn_in_end))
        
        # Phase 3: Apex - MUST start after turn-in ends, centered around minimum speed
        apex_start = turn_in_end  # STRICT: starts exactly where turn-in ends
        apex_window = max(3, corner_length // 12)  # Apex window size
        apex_end = min(corner_length - 4, apex_start + apex_window)  # Leave room for accel/exit
        # Ensure apex includes the minimum speed point if possible
        if min_speed_idx >= apex_start and min_speed_idx < apex_end:
            # Good, apex contains min speed point
            pass
        else:
            # Adjust apex to include min speed point while respecting boundaries
            apex_end = min(corner_length - 4, min_speed_idx + 2)
        phases['apex'] = list(range(apex_start, apex_end))
        
        # Phase 4: Acceleration - MUST start after apex ends
        accel_start = apex_end  # STRICT: starts exactly where apex ends
        accel_end = self._find_acceleration_phase_end_v2(throttle_smooth, steering_smooth, 
                                                         accel_start, corner_length)
        # ENFORCE: acceleration must leave room for exit phase
        accel_end = min(accel_end, corner_length - 2)  
        accel_end = max(accel_start + 2, accel_end)  # Minimum acceleration length
        phases['acceleration'] = list(range(accel_start, accel_end))
        
        # Phase 5: Exit - MUST start after acceleration ends, goes to corner end
        exit_start = accel_end  # STRICT: starts exactly where acceleration ends
        exit_end = corner_length  # STRICT: goes to the very end
        phases['exit'] = list(range(exit_start, exit_end))
        
        # Final validation to ensure no gaps or overlaps in the sequential phases
        phases = self._enforce_strict_sequential_phases(phases, corner_length)
        
        return phases
    
    def _validate_phases_with_position_data(self, phases: Dict[str, List[int]], corner_df: pd.DataFrame, position_data: pd.Series) -> Dict[str, List[int]]:
        """Validate and adjust phases to ensure they have position data"""
        
        # Check each phase to ensure it has valid position data
        for phase_name, indices in phases.items():
            if not indices:
                continue
                
            # Check if phase indices have position data
            valid_indices = []
            for idx in indices:
                if idx < len(position_data) and not pd.isna(position_data.iloc[idx]):
                    valid_indices.append(idx)
            
            # If no valid indices, try to expand phase to neighboring valid data
            if not valid_indices and indices:
                # Look for nearest valid position data
                start_idx = min(indices)
                end_idx = max(indices)
                
                # Expand search range slightly
                search_start = max(0, start_idx - 2)
                search_end = min(len(position_data), end_idx + 3)
                
                for idx in range(search_start, search_end):
                    if not pd.isna(position_data.iloc[idx]):
                        valid_indices.append(idx)
                
                # If still no valid data, keep original indices but log warning
                if not valid_indices:
                    print(f"Warning: Phase {phase_name} has no valid position data")
                    valid_indices = indices
            
            phases[phase_name] = valid_indices if valid_indices else indices
        
        return phases

    def _find_entry_phase_end(self, corner_df: pd.DataFrame, brake_data: pd.Series, speed_smooth: pd.Series) -> int:
        """Find where entry phase ends - when driver transitions from initial braking to trail braking"""
        
        # Entry phase: driver starts braking to slow down, preparing for trail braking
        # This ends when significant steering input begins (transition to turn_in/trail braking)
        
        steering_abs = corner_df['steering_smooth']  # Use pre-smoothed steering
        
        # Find when steering input becomes significant (start of trail braking)
        steering_threshold = steering_abs.quantile(0.3)  # Lower threshold for initial steering input
        
        # Look for sustained steering input (not just momentary) - no additional smoothing needed
        steering_window = 3
        significant_steering = steering_abs.rolling(window=steering_window).mean() > steering_threshold
        
        # Find first point where sustained steering begins
        if significant_steering.any():
            first_steering_idx = significant_steering.idxmax() - corner_df.index[0]
            # Entry phase ends just before significant steering input begins
            entry_end = max(3, first_steering_idx - 2)  # Minimum 3 points for entry
        else:
            # Fallback: use brake pressure pattern if no clear steering signal
            brake_threshold = brake_data.quantile(0.6) if brake_data.max() > 0 else 0
            
            if brake_threshold > 0:
                # Find when brake pressure stabilizes (end of initial braking phase)
                brake_diff = brake_data.diff().abs()
                stable_brake_threshold = brake_diff.quantile(0.3)
                
                for i in range(5, len(brake_data) - 3):
                    if brake_diff.iloc[i:i+3].mean() < stable_brake_threshold and brake_data.iloc[i] > brake_threshold:
                        return i
            
            # Ultimate fallback
            entry_end = len(corner_df) // 4
        
        return min(entry_end, len(corner_df) - 5)  # Ensure we leave room for other phases

    def _find_turn_in_phase_end(self, corner_df: pd.DataFrame, brake_data: pd.Series, steering_abs: pd.Series, start_idx: int) -> int:
        """Find where turn-in phase ends - when trail braking transitions to pure cornering at apex"""
        
        # Turn-in phase: trail braking (combining braking and steering)
        # This ends when braking significantly reduces and we approach the apex
        
        if start_idx >= len(corner_df) - 5:
            return len(corner_df)
        
        # Look for the point where braking pressure drops significantly
        # while steering is maintained (end of trail braking)
        brake_section = brake_data.iloc[start_idx:]
        steering_section = steering_abs.iloc[start_idx:]
        
        if len(brake_section) < 5:
            return len(corner_df)
        
        # Find where brake pressure drops while steering is still significant
        brake_threshold = brake_section.quantile(0.4)  # Moderate braking threshold
        steering_threshold = steering_section.quantile(0.7)  # High steering threshold
        
        # Look for the transition point
        for i in range(3, len(brake_section) - 2):
            current_brake = brake_section.iloc[i:i+3].mean()
            current_steering = steering_section.iloc[i:i+3].mean()
            
            # Trail braking ends when brake pressure drops significantly
            # but steering remains high (approaching apex)
            if current_brake < brake_threshold and current_steering > steering_threshold:
                return start_idx + i
        
        # Fallback: use speed minimum as indicator of apex approach
        speed_section = corner_df['Physics_speed_kmh'].iloc[start_idx:]
        min_speed_relative_idx = speed_section.idxmin() - corner_df.index[start_idx]
        
        if min_speed_relative_idx > 5:
            return start_idx + min_speed_relative_idx
        
        # Ultimate fallback
        return start_idx + max(5, len(brake_section) // 2)

    def _find_apex_index(self, corner_df: pd.DataFrame, min_speed_idx: int, max_steering_idx: int) -> int:
        """Find the apex point combining speed and steering information"""
        
        # Weight both minimum speed and maximum steering
        speed_weight = 0.6
        steering_weight = 0.4
        
        # Normalize indices to 0-1 range
        corner_length = len(corner_df)
        speed_normalized = min_speed_idx / corner_length
        steering_normalized = max_steering_idx / corner_length
        
        # Calculate weighted apex position
        apex_normalized = (speed_weight * speed_normalized + 
                          steering_weight * steering_normalized)
        
        apex_idx = int(apex_normalized * corner_length)
        return max(0, min(corner_length - 1, apex_idx))

    def _find_acceleration_phase_end(self, corner_df: pd.DataFrame, throttle_data: pd.Series, start_idx: int) -> int:
        """Find where acceleration phase ends (throttle application becomes steady)"""
        
        if start_idx >= len(corner_df) - 5:
            return len(corner_df)
        
        # Look for sustained throttle application
        throttle_section = throttle_data.iloc[start_idx:]
        
        if throttle_section.max() > 0.3:  # Significant throttle application
            # Find where throttle becomes relatively steady
            throttle_diff = throttle_section.diff().abs()
            steady_threshold = throttle_diff.quantile(0.3)  # Low variation
            
            for i in range(5, len(throttle_section) - 3):
                if throttle_diff.iloc[i:i+3].mean() < steady_threshold:
                    return start_idx + i
        
        return len(corner_df)

    def _resolve_phase_overlaps(self, phases: Dict[str, List[int]], total_length: int) -> Dict[str, List[int]]:
        """Resolve any overlapping phase assignments"""
        
        # Create a priority order for phases
        phase_priority = ['entry', 'turn_in', 'apex', 'acceleration', 'exit']
        assigned = set()
        
        for phase_name in phase_priority:
            # Remove already assigned indices
            phases[phase_name] = [i for i in phases[phase_name] if i not in assigned and 0 <= i < total_length]
            # Add to assigned set
            assigned.update(phases[phase_name])
        
        return phases

    def _find_entry_phase_end_v2(self, brake_smooth: pd.Series, speed_change: pd.Series, corner_length: int) -> int:
        """
        Find the end of the entry phase (braking zone) by detecting brake application and deceleration.
        Entry phase ends when driver transitions from pure braking to turn-in with steering.
        
        CRITICAL: Entry phase MUST NOT exceed 25% of corner length to ensure room for all subsequent phases.
        """
        # ENFORCE: Entry cannot exceed 25% of corner to leave room for turn-in, apex, acceleration, exit
        max_allowed_end = min(corner_length // 4, corner_length - 8)
        min_entry_end = 2  # Minimum entry length
        
        if max_allowed_end <= min_entry_end:
            return min_entry_end
        
        # Look for sustained braking or significant deceleration at the start
        brake_threshold = brake_smooth.quantile(0.3) if brake_smooth.max() > 0 else 0
        decel_threshold = -2.0  # km/h per data point - significant deceleration
        
        # Default entry end (conservative)
        entry_end = min(corner_length // 8, max_allowed_end)  # Default to ~12.5% of corner
        
        if brake_smooth.max() > 0.1:  # If we have brake data
            # Look for when braking starts to reduce significantly
            for i in range(min_entry_end, min(max_allowed_end, len(brake_smooth) - 2)):
                # Check if brake pressure is reducing and we're past initial heavy braking
                recent_brake = brake_smooth.iloc[max(0, i-2):i+1].mean()
                if recent_brake < brake_threshold * 0.6 and i > 5:
                    entry_end = i
                    break
        else:
            # No brake data, use speed deceleration pattern
            # Look for when deceleration starts to stabilize or reduce
            for i in range(min_entry_end, min(max_allowed_end, len(speed_change) - 2)):
                recent_decel = speed_change.iloc[max(0, i-2):i+1].mean()
                # Entry ends when deceleration becomes less severe (preparing for turn-in)
                if recent_decel > decel_threshold * 0.5 and i > 5:
                    entry_end = i
                    break
        
        # ENFORCE: Respect strict sequential boundaries
        return max(min_entry_end, min(entry_end, max_allowed_end))

    def _find_turn_in_phase_end_v2(self, steering_smooth: pd.Series, speed_change: pd.Series, 
                                   brake_smooth: pd.Series, start_idx: int, min_speed_idx: int, 
                                   corner_length: int) -> int:
        """
        Find the end of turn-in phase. Turn-in is characterized by:
        - Steering movement toward apex
        - Speed constantly slowing down (deceleration)
        - Possible continued braking (trail braking) or engine braking
        
        CRITICAL: Turn-in MUST end BEFORE apex phase begins to maintain strict sequential order.
        """
        if start_idx >= corner_length - 5:
            return min(start_idx + 3, corner_length - 5)
        
        # ENFORCE: Turn-in must end at least 2 points before min_speed_idx to leave room for apex
        max_allowed_end = min(min_speed_idx - 2, corner_length - 4)
        
        # Minimum turn-in length
        min_turn_in_end = start_idx + 3
        
        if max_allowed_end <= min_turn_in_end:
            # Not enough room - use minimum valid boundary
            return min_turn_in_end
        
        # Find where deceleration significantly reduces (approaching apex)
        steering_threshold = steering_smooth.quantile(0.5)
        turn_in_end = max_allowed_end  # Default to maximum allowed
        
        for i in range(min_turn_in_end, min(max_allowed_end, len(speed_change) - 2)):
            # Check recent speed change pattern
            recent_speed_change = speed_change.iloc[max(start_idx, i-3):i+1]
            avg_decel = recent_speed_change.mean()
            
            # Turn-in ends when we approach minimum deceleration (near apex)
            if avg_decel > -1.0 and i > start_idx + 3:  # Less than 1 km/h deceleration per point
                # Ensure we have steering activity
                if steering_smooth.iloc[i] > steering_threshold * 0.7:
                    turn_in_end = i
                    break
        
        # ENFORCE: Respect strict sequential boundaries
        return max(min_turn_in_end, min(turn_in_end, max_allowed_end))

    def _find_acceleration_phase_end_v2(self, throttle_smooth: pd.Series, steering_smooth: pd.Series,
                                        start_idx: int, corner_length: int) -> int:
        """
        Find the end of acceleration phase. Acceleration phase is characterized by:
        - Gradual throttle application after apex
        - Gradual reduction in steering angle as car straightens
        - Speed increasing
        
        CRITICAL: Acceleration MUST end BEFORE exit phase begins to maintain strict sequential order.
        """
        if start_idx >= corner_length - 3:
            return corner_length - 2  # Leave at least 1 point for exit
        
        # ENFORCE: Acceleration must end at least 2 points before corner end to leave room for exit
        max_allowed_end = corner_length - 2
        min_accel_end = start_idx + 2  # Minimum acceleration length
        
        if max_allowed_end <= min_accel_end:
            return min_accel_end
        
        # Default acceleration phase length (more conservative)
        accel_end = min(start_idx + max(3, (corner_length - start_idx) // 3), max_allowed_end)
        
        if throttle_smooth.max() > 0.3:  # If we have throttle data
            # Look for sustained throttle application (transition to full throttle/exit)
            throttle_threshold = throttle_smooth.quantile(0.7)
            
            for i in range(min_accel_end, min(max_allowed_end, len(throttle_smooth))):
                # Check for sustained high throttle (entering exit phase)
                recent_throttle = throttle_smooth.iloc[max(start_idx, i-2):i+1].mean()
                if recent_throttle > throttle_threshold and i > start_idx + 3:
                    accel_end = i
                    break
        else:
            # No throttle data, use steering reduction pattern
            # Acceleration phase ends when steering reduces significantly
            if start_idx + 3 < len(steering_smooth):
                initial_steering = steering_smooth.iloc[start_idx:start_idx+3].mean()
                
                for i in range(min_accel_end, min(max_allowed_end, len(steering_smooth))):
                    current_steering = steering_smooth.iloc[i]
                    # If steering has reduced to 50% of initial acceleration phase steering
                    if current_steering < initial_steering * 0.5 and i > start_idx + 3:
                        accel_end = i
                        break
        
        # ENFORCE: Respect strict sequential boundaries
        return max(min_accel_end, min(accel_end, max_allowed_end))

    def _enforce_strict_sequential_phases(self, phases: Dict[str, List[int]], corner_length: int) -> Dict[str, List[int]]:
        """
        ENFORCE STRICT SEQUENTIAL ORDERING: Entry → Turn-in → Apex → Acceleration → Exit
        Each phase MUST follow the previous one with NO overlaps and NO gaps.
        Every corner point must be assigned to exactly one phase in the correct order.
        """
        phase_order = ['entry', 'turn_in', 'apex', 'acceleration', 'exit']
        min_phase_size = max(1, corner_length // 15)  # Minimum size per phase
        
        # Create a clean sequential assignment
        sequential_phases = {phase: [] for phase in phase_order}
        
        # Calculate target sizes for each phase (ensure reasonable distribution)
        base_size = corner_length // 5  # Equal distribution baseline
        
        # Determine actual boundaries from current phase assignments (if valid)
        boundaries = []
        for i, phase_name in enumerate(phase_order):
            if phases[phase_name]:
                if i == 0:  # Entry
                    boundaries.append(0)  # Always start at 0
                    boundaries.append(max(phases[phase_name]) + 1)
                elif i == len(phase_order) - 1:  # Exit
                    boundaries.append(corner_length)  # Always end at corner_length
                else:
                    # For middle phases, use the computed boundaries but validate
                    phase_end = max(phases[phase_name]) + 1
                    boundaries.append(min(phase_end, corner_length - (len(phase_order) - i - 1) * min_phase_size))
            else:
                # If phase is empty, estimate boundary
                if i == 0:
                    boundaries.extend([0, min(base_size, corner_length // 4)])
                elif i == len(phase_order) - 1:
                    boundaries.append(corner_length)
                else:
                    # Estimate based on position in sequence
                    est_boundary = (i + 1) * corner_length // len(phase_order)
                    boundaries.append(min(est_boundary, corner_length - (len(phase_order) - i - 1) * min_phase_size))
        
        # Ensure we have the right number of boundaries
        if len(boundaries) != len(phase_order) + 1:
            # Fallback: create even distribution
            boundaries = []
            for i in range(len(phase_order) + 1):
                boundaries.append(i * corner_length // len(phase_order))
            boundaries[-1] = corner_length  # Ensure last boundary is exactly corner_length
        
        # Validate boundaries to ensure minimum phase sizes
        for i in range(len(phase_order)):
            phase_start = boundaries[i]
            phase_end = boundaries[i + 1]
            
            # Ensure minimum phase size
            if phase_end - phase_start < min_phase_size:
                # Adjust boundaries to meet minimum size requirements
                needed_size = min_phase_size - (phase_end - phase_start)
                
                if i < len(phase_order) - 1:  # Not the last phase
                    # Try to expand by taking from next phases
                    available_space = corner_length - phase_end - (len(phase_order) - i - 2) * min_phase_size
                    expansion = min(needed_size, available_space)
                    boundaries[i + 1] = min(corner_length, phase_end + expansion)
                elif i > 0:  # Not the first phase
                    # Try to expand by taking from previous phases
                    available_space = phase_start - i * min_phase_size
                    expansion = min(needed_size, available_space)
                    boundaries[i] = max(0, phase_start - expansion)
        
        # Final boundary validation - ensure strict ordering and no overlaps
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i-1]:
                # Fix boundary conflicts
                boundaries[i] = boundaries[i-1] + 1
        
        # Ensure last boundary is exactly corner_length
        boundaries[-1] = corner_length
        
        # Assign phases based on validated boundaries
        for i, phase_name in enumerate(phase_order):
            phase_start = boundaries[i]
            phase_end = boundaries[i + 1]
            sequential_phases[phase_name] = list(range(phase_start, phase_end))
        
        # Verify complete coverage and no gaps/overlaps
        all_indices = set()
        for phase_indices in sequential_phases.values():
            for idx in phase_indices:
                if idx in all_indices:
                    raise ValueError(f"Index {idx} assigned to multiple phases - this should never happen!")
                all_indices.add(idx)
        
        # Verify all corner indices are covered
        expected_indices = set(range(corner_length))
        if all_indices != expected_indices:
            missing = expected_indices - all_indices
            extra = all_indices - expected_indices
            raise ValueError(f"Phase assignment incomplete: missing {missing}, extra {extra}")
        
        # Log phase boundaries for debugging
        print(f"[DEBUG] Corner phases - Entry: {len(sequential_phases['entry'])} points, "
              f"Turn-in: {len(sequential_phases['turn_in'])} points, "
              f"Apex: {len(sequential_phases['apex'])} points, "
              f"Acceleration: {len(sequential_phases['acceleration'])} points, "
              f"Exit: {len(sequential_phases['exit'])} points")
        
        return sequential_phases

    def _validate_and_clean_phases(self, phases: Dict[str, List[int]], corner_length: int) -> Dict[str, List[int]]:
        """
        DEPRECATED: Use _enforce_strict_sequential_phases instead.
        This method is kept for backward compatibility but should not be used.
        """
        return self._enforce_strict_sequential_phases(phases, corner_length)


    
    def get_cornering_analysis_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for cornering analysis including detailed corner information"""
        
        if 'cornering_phase' not in df.columns:
            return {"error": "No cornering phase data found. Run identify_cornering_phases first."}
        
        corner_count = int(df['corner_id'].nunique() - (1 if -1 in df['corner_id'].values else 0))
        
        # Get detailed information for each corner
        corner_details = {}
        valid_corner_ids = [int(cid) for cid in df['corner_id'].unique() if cid >= 0]
        
        for corner_id in valid_corner_ids:
            corner_data = df[df['corner_id'] == corner_id].copy()
            
            if len(corner_data) == 0:
                continue
                
            corner_details[f"corner_{corner_id}"] = self._get_corner_phase_details(corner_data)
        
        return {
            'total_corners_detected': corner_count,
            'corner_ids': sorted(valid_corner_ids),
            'corner_details': corner_details
        }
    
    def _get_corner_phase_details(self, corner_data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed phase information for a single corner including normalized car positions"""
        
        # Helper function to ensure JSON-safe numeric values
        def safe_float(value):
            """Convert to float and handle nan/inf values"""
            try:
                result = float(value)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return result
            except (ValueError, TypeError):
                return 0.0
        
        phases = ['entry', 'turn_in', 'apex', 'acceleration', 'exit']
        corner_detail = {
            'corner_start_position': 0.0,
            'corner_end_position': 0.0,
            'total_duration_points': int(len(corner_data)),
            'phases': {}
        }
        
        # Get overall corner start and end positions
        if 'Graphics_normalized_car_position' in corner_data.columns:
            position_data = corner_data['Graphics_normalized_car_position'].dropna()
            if len(position_data) > 0:
                corner_detail['corner_start_position'] = safe_float(position_data.iloc[0])
                corner_detail['corner_end_position'] = safe_float(position_data.iloc[-1])
        
        # Get detailed information for each phase
        for phase in phases:
            phase_data = corner_data[corner_data['cornering_phase'] == phase].copy()
            
            # Initialize phase info with default values (no None values)
            phase_info = {
                'normalized_car_position': 0.0,
                'duration_points': int(len(phase_data))
            }
            
            if len(phase_data) == 0:
                # If no data for this phase, try to estimate position from corner progression
                if 'Graphics_normalized_car_position' in corner_data.columns:
                    total_corner_positions = corner_data['Graphics_normalized_car_position'].dropna()
                    if len(total_corner_positions) > 0:
                        # Estimate position based on phase order within corner
                        phase_order = {'entry': 0.1, 'turn_in': 0.3, 'apex': 0.5, 'acceleration': 0.7, 'exit': 0.9}
                        if phase in phase_order:
                            start_pos = safe_float(total_corner_positions.iloc[0])
                            end_pos = safe_float(total_corner_positions.iloc[-1])
                            estimated_pos = start_pos + (end_pos - start_pos) * phase_order[phase]
                            phase_info['normalized_car_position'] = safe_float(estimated_pos)
                
                corner_detail['phases'][phase] = phase_info
                continue
            
            # Get normalized car position for this phase - be more robust with NaN handling
            if 'Graphics_normalized_car_position' in phase_data.columns:
                position_values = phase_data['Graphics_normalized_car_position'].dropna()
                
                if len(position_values) > 0:
                    # Choose position based on phase type and available data
                    if phase == 'entry':
                        # For entry, use the starting position
                        phase_info['normalized_car_position'] = safe_float(position_values.iloc[0])
                    elif phase == 'exit':
                        # For exit, use the ending position  
                        phase_info['normalized_car_position'] = safe_float(position_values.iloc[-1])
                    else:
                        # For turn_in, apex, acceleration - use middle position
                        if len(position_values) >= 3:
                            mid_idx = len(position_values) // 2
                            phase_info['normalized_car_position'] = safe_float(position_values.iloc[mid_idx])
                        elif len(position_values) == 2:
                            # If only 2 points, average them
                            phase_info['normalized_car_position'] = safe_float(position_values.mean())
                        else:
                            # If only 1 point, use it
                            phase_info['normalized_car_position'] = safe_float(position_values.iloc[0])
                else:
                    # If no position data for this phase, try to estimate from overall corner
                    total_corner_positions = corner_data['Graphics_normalized_car_position'].dropna()
                    if len(total_corner_positions) > 0:
                        # Get phase indices within the corner
                        phase_indices = corner_data[corner_data['cornering_phase'] == phase].index
                        if len(phase_indices) > 0:
                            # Find relative position of phase within corner
                            corner_indices = corner_data.index
                            corner_start_idx = corner_indices[0]
                            corner_end_idx = corner_indices[-1]
                            phase_relative_pos = (phase_indices[len(phase_indices)//2] - corner_start_idx) / (corner_end_idx - corner_start_idx)
                            
                            # Interpolate position
                            start_pos = safe_float(total_corner_positions.iloc[0])
                            end_pos = safe_float(total_corner_positions.iloc[-1])
                            interpolated_pos = start_pos + (end_pos - start_pos) * phase_relative_pos
                            phase_info['normalized_car_position'] = safe_float(interpolated_pos)
            
            corner_detail['phases'][phase] = phase_info
        
        return corner_detail
