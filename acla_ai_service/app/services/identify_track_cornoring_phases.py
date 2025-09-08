import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


class TrackCorneringAnalyzer:
    """
    A comprehensive class for identifying and analyzing track cornering phases
    from telemetry data including steering angles, speed, and car position.
    
    Corner Phases Defined:
    - Entry: Driver starts braking to slow down, preparing for trail braking
    - Turn-in: Trail braking phase - combining braking and steering toward apex
    - Apex: The geometric center point of the corner with minimum speed
    - Acceleration: Driver begins throttle application after apex
    - Exit: Full throttle application and corner exit
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
        
        # Prepare smoothed data for analysis
        steering_abs = np.abs(df['Physics_steer_angle']).rolling(window=5, center=True).mean().fillna(method='ffill').fillna(method='bfill')
        speed = df['Physics_speed_kmh'].rolling(window=5, center=True).mean().fillna(method='ffill').fillna(method='bfill')
        
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
        
        # Use a much simpler approach - just steering angle
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
        return combined_score.rolling(window=3, center=True).mean().fillna(method='ffill').fillna(method='bfill')
    
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
        
        # Calculate indicators
        steering_abs = np.abs(df['Physics_steer_angle']).rolling(window=5, center=True).mean().fillna(method='ffill').fillna(method='bfill')
        speed = df['Physics_speed_kmh'].rolling(window=5, center=True).mean().fillna(method='ffill').fillna(method='bfill')
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
        Analyze a single corner to identify racing phases:
        - Entry: Initial braking phase preparing for trail braking
        - Turn-in: Trail braking (combining brake + steering) toward apex
        - Apex: Minimum speed point at corner's geometric center
        - Acceleration: Throttle application begins after apex
        - Exit: Full acceleration out of corner
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
        
        # Calculate key metrics for phase identification
        steering_abs = np.abs(corner_df['Physics_steer_angle'])
        speed = corner_df['Physics_speed_kmh']
        
        # Get brake and throttle data if available
        brake_data = corner_df.get('Physics_brake', pd.Series([0] * len(corner_df)))
        throttle_data = corner_df.get('Physics_gas', pd.Series([0] * len(corner_df)))
        
        # Get car position data for phase validation
        position_data = None
        if 'Graphics_normalized_car_position' in corner_df.columns:
            position_data = corner_df['Graphics_normalized_car_position'].fillna(method='ffill').fillna(method='bfill')
        
        # Smooth the data
        steering_smooth = steering_abs.rolling(window=3, center=True).mean().fillna(steering_abs)
        speed_smooth = speed.rolling(window=3, center=True).mean().fillna(speed)
        
        # Find key points
        min_speed_idx = speed_smooth.idxmin() - corner_df.index[0]  # Apex candidate
        max_steering_idx = steering_smooth.idxmax() - corner_df.index[0]  # Peak steering input
        
        # Adjust indices to be relative to corner_df
        min_speed_idx = max(0, min(len(corner_df) - 1, min_speed_idx))
        max_steering_idx = max(0, min(len(corner_df) - 1, max_steering_idx))
        
        # Phase 1: Entry (driver starts braking to slow down, preparing for trail braking)
        entry_end = self._find_entry_phase_end(corner_df, brake_data, speed_smooth)
        # Ensure entry phase has at least some data points
        entry_end = max(entry_end, 3)  # Minimum 3 points for entry
        phases['entry'] = list(range(0, min(entry_end, len(corner_df))))
        
        # Phase 2: Turn-in (trail braking - combining braking and steering toward apex)
        turn_in_start = max(entry_end, 0)
        turn_in_end = self._find_turn_in_phase_end(corner_df, brake_data, steering_abs, turn_in_start)
        
        # Ensure turn-in phase exists and doesn't exceed reasonable bounds
        if turn_in_end <= turn_in_start:
            turn_in_end = turn_in_start + max(3, (len(corner_df) - turn_in_start) // 4)
        
        phases['turn_in'] = list(range(turn_in_start, min(turn_in_end, len(corner_df))))
        
        # Phase 3: Apex (around minimum speed point - the geometric center of the corner)
        apex_idx = self._find_apex_index(corner_df, min_speed_idx, max_steering_idx)
        apex_window = max(3, len(corner_df) // 10)  # Dynamic window size
        apex_start = max(turn_in_end, apex_idx - apex_window // 2)  # Start from end of turn-in
        apex_end = min(len(corner_df), apex_idx + apex_window // 2)
        
        # Ensure apex phase has minimum size
        if apex_end <= apex_start:
            apex_end = apex_start + max(3, len(corner_df) // 15)
        
        phases['apex'] = list(range(apex_start, min(apex_end, len(corner_df))))
        
        # Phase 4: Acceleration (apex end until throttle application stabilizes)
        accel_start = apex_end
        accel_end = self._find_acceleration_phase_end(corner_df, throttle_data, accel_start)
        # Ensure acceleration phase has reasonable size
        min_accel_end = accel_start + max(3, (len(corner_df) - accel_start) // 3)
        accel_end = max(accel_end, min_accel_end)
        phases['acceleration'] = list(range(accel_start, min(accel_end, len(corner_df))))
        
        # Phase 5: Exit (acceleration end until corner end)
        exit_start = max(accel_end, apex_end)
        # Ensure exit phase exists
        exit_start = min(exit_start, len(corner_df) - 3)  # Leave at least 3 points for exit
        phases['exit'] = list(range(exit_start, len(corner_df)))
        
        # Clean up overlapping phases and ensure all points are covered
        phases = self._resolve_phase_overlaps(phases, len(corner_df))
        
        # Verify that all phases have position data if available
        if position_data is not None:
            phases = self._validate_phases_with_position_data(phases, corner_df, position_data)
        
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
        
        steering_abs = np.abs(corner_df['Physics_steer_angle'])
        
        # Find when steering input becomes significant (start of trail braking)
        steering_threshold = steering_abs.quantile(0.3)  # Lower threshold for initial steering input
        
        # Look for sustained steering input (not just momentary)
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


    def filter_top_performance_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for valid laps and select top percentage fastest laps for training
        
        Args:
            df: Processed telemetry DataFrame
            
        Returns:
            Filtered DataFrame containing only top percentage fastest valid laps
        """
        print(f"[INFO] Starting lap filtering from {len(df)} telemetry records")
        
        # We'll work with all data first, then filter by validity percentage per lap
        working_df = df.copy()
        
        # Check if we have the required columns
        has_valid_lap_column = 'Graphics_is_valid_lap' in working_df.columns
        if not has_valid_lap_column:
            print("[WARNING] Graphics_is_valid_lap column not found, cannot validate lap quality - returning all data")
            return pd.DataFrame()
        else:
            print(f"[INFO] Found Graphics_is_valid_lap column, will filter laps by validity percentage")
        
        # Group by lap and calculate lap times
        # Use both Graphics_completed_lap and Graphics_normalized_car_position together for robust lap detection
        has_completed_lap = 'Graphics_completed_lap' in working_df.columns
        has_position = 'Graphics_normalized_car_position' in working_df.columns
        
        # Only proceed if we have both fields - return empty data otherwise
        if not (has_completed_lap and has_position):
            return pd.DataFrame()
        
        completed_laps = working_df['Graphics_completed_lap'].fillna(0)
        position = working_df['Graphics_normalized_car_position'].fillna(0)
        
        # Primary method: detect when completed_lap increments (official lap completion)
        completed_lap_changes = completed_laps.diff() > 0
        
        # Use completed_lap changes as primary lap boundary indicator
        lap_boundaries = completed_lap_changes 
        
        # Create cumulative lap ID
        working_df['lap_id'] = lap_boundaries.cumsum()
        
        # Group all telemetry data by these lap ids, allowing the code to process each lap individually
        lap_groups = working_df.groupby('lap_id')
        print(f"[INFO] Detected {len(lap_groups)} individual lap segments using completed_lap changes")
        
        # Calculate lap times for each lap
        lap_times = []
        lap_data = []
        total_laps_processed = 0
        full_laps_found = 0
        
        for lap_id, lap_df in lap_groups:
            total_laps_processed += 1
            
            if len(lap_df) < 10:  # Skip very short laps (likely incomplete)
                continue
            
            # Check validity percentage if is_valid_lap column is available
            if has_valid_lap_column:
                if not self._is_lap_mostly_valid(lap_df, self.min_valid_percentage):
                    continue
            
            # Validate that this is a full lap using normalized_car_position
            if not self._is_full_lap(lap_df):
                continue
            
            full_laps_found += 1
                
            # Calculate lap time
            lap_time = 0
            if 'Graphics_current_time' in lap_df.columns:
                # Use the current lap time at the end of this lap (already in milliseconds)
                lap_time = lap_df['Graphics_current_time'].iloc[-1] / 1000.0  # Convert to seconds
            
            if lap_time > 0:  # Only include laps with valid times
                lap_times.append(lap_time)
                lap_data.append(lap_df)
        
        if not lap_times:
            return pd.DataFrame()
        
        print(f"[INFO] Calculated lap times for {len(lap_times)} qualifying laps")
        print(f"[INFO] Best lap time: {min(lap_times):.3f}s, Worst: {max(lap_times):.3f}s")
        
        # Sort laps by time (fastest first)
        sorted_indices = np.argsort(lap_times)

        # Calculate how many laps to keep based on configured percentage
        num_laps_to_keep = max(1, int(np.ceil(len(lap_times) * self.top_laps_percentage)))
        print(f"[INFO] Selecting top {num_laps_to_keep} fastest laps out of {len(lap_times)} total laps")
        
        # Select top laps
        top_lap_indices = sorted_indices[:num_laps_to_keep]
        
        # Combine data from selected laps
        filtered_data_frames = [lap_data[i] for i in top_lap_indices]
        filtered_df = pd.concat(filtered_data_frames, ignore_index=True)
        
        # Report selected lap times
        selected_lap_times = [lap_times[i] for i in top_lap_indices]
        print(f"[INFO] Selected lap times: {[f'{t:.3f}s' for t in selected_lap_times]}")
        print(f"[INFO] Filtered to {len(filtered_df)} records from top {num_laps_to_keep} fastest complete full laps")
        
        return filtered_df

    def _is_lap_mostly_valid(self, lap_df: pd.DataFrame, min_valid_percentage: float = 0.75) -> bool:
        """
        Check if a lap has a sufficient percentage of valid data points
        
        Args:
            lap_df: DataFrame containing telemetry data for one lap
            min_valid_percentage: Minimum percentage of valid points (default 75% in decimal)
            
        Returns:
            True if the lap has enough valid data points
        """
        if 'Graphics_is_valid_lap' not in lap_df.columns:
            return True  # Assume valid if we can't check
        
        valid_points = lap_df['Graphics_is_valid_lap'].fillna(False)
        total_points = len(valid_points)
        
        if total_points == 0:
            return False
        
        # Count boolean True values, handling different data types
        if valid_points.dtype == 'bool':
            valid_count = valid_points.sum()
        else:
            # Handle string or numeric representations
            valid_count = (
                (valid_points == True) | 
                (valid_points == 'True') | 
                (valid_points == 'true') | 
                (valid_points == 1) | 
                (valid_points == '1')
            ).sum()
        
        valid_percentage = valid_count / total_points
        
        if valid_percentage < min_valid_percentage:
            print(f"[DEBUG] Rejected lap: only {valid_percentage:.1%} valid points (need {min_valid_percentage:.1%})")
            return False
        
        return True
    
    def _is_full_lap(self, lap_df: pd.DataFrame) -> bool:
        """
        Validate that a lap contains a complete track progression from start to finish
        Uses both Graphics_normalized_car_position and Graphics_completed_lap when available
        
        Args:
            lap_df: DataFrame containing telemetry data for one lap
            
        Returns:
            True if the lap contains progression from ~0 to ~1 in normalized_car_position
            and shows consistent completed lap counter behavior
        """
        has_position = 'Graphics_normalized_car_position' in lap_df.columns
        has_completed_lap = 'Graphics_completed_lap' in lap_df.columns
        
        # If we have neither field, assume valid (fallback)
        if not has_position and not has_completed_lap:
            print("[WARNING] No position or completed lap data available, cannot validate full lap")
            return True
        
        # Validate using normalized car position (primary validation)
        position_valid = True
        if has_position:
            positions = lap_df['Graphics_normalized_car_position'].dropna()
            
            if len(positions) == 0:
                position_valid = False
            else:
                min_position = positions.min()
                max_position = positions.max()
                
                # Check if the lap covers most of the track
                # Allow some tolerance: lap should go from close to 0 to close to 1
                starts_near_beginning = min_position <= 0.15  # Starts at or before 15% of track
                ends_near_finish = max_position >= 0.85       # Ends at or after 85% of track
                
                # Additional check: ensure good coverage of the track
                position_range = max_position - min_position
                good_coverage = position_range >= 0.7  # Covers at least 70% of track length
                
                position_valid = starts_near_beginning and ends_near_finish and good_coverage
                
                if not position_valid:
                    print(f"[DEBUG] Position validation failed: min_pos={min_position:.3f}, max_pos={max_position:.3f}, range={position_range:.3f}")
        
        # Validate using completed lap counter (secondary validation)
        completed_lap_valid = True
        if has_completed_lap:
            completed_laps = lap_df['Graphics_completed_lap'].fillna(0)
            
            # For a valid lap, the completed lap counter should either:
            # 1. Stay constant throughout the lap (during lap progress)
            # 2. Show exactly one increment at the end (lap completion)
            unique_values = completed_laps.unique()
            
            if len(unique_values) == 1:
                # Counter stayed constant - lap in progress, this is expected
                completed_lap_valid = True
            elif len(unique_values) == 2:
                # Counter incremented once - should be at the end of the lap
                # Check that the increment happens towards the end of the data
                increment_positions = completed_laps.diff() > 0
                if increment_positions.sum() == 1:  # Exactly one increment
                    # Find where the increment occurred
                    increment_index = increment_positions.idxmax()
                    total_records = len(completed_laps)
                    increment_position_ratio = (increment_index / total_records) if total_records > 0 else 0
                    
                    # Increment should happen in the latter part of the lap (after 70% completion)
                    completed_lap_valid = increment_position_ratio >= 0.7
                    if not completed_lap_valid:
                        print(f"[DEBUG] Completed lap increment too early: {increment_position_ratio:.1%} through lap")
                else:
                    # Multiple increments - suspicious
                    completed_lap_valid = False
                    print(f"[DEBUG] Multiple completed lap increments detected: {increment_positions.sum()}")
            else:
                # Too many different values - suspicious
                completed_lap_valid = False
                print(f"[DEBUG] Too many completed lap values: {len(unique_values)} unique values")
        
        # Combine validations - both must pass if both fields are available
        if has_position and has_completed_lap:
            is_valid = position_valid and completed_lap_valid
            if not is_valid:
                print(f"[DEBUG] Rejected lap: position_valid={position_valid}, completed_lap_valid={completed_lap_valid}")
        elif has_position:
            is_valid = position_valid
        else:  # has_completed_lap only
            is_valid = completed_lap_valid
        
        return is_valid  
    

    def get_cornering_analysis_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for cornering analysis including detailed corner information"""
        
        if 'cornering_phase' not in df.columns:
            return {"error": "No cornering phase data found. Run identify_cornering_phases first."}
        
        phase_counts = df['cornering_phase'].value_counts()
        corner_count = df['corner_id'].nunique() - (1 if -1 in df['corner_id'].values else 0)
        
        # Calculate average metrics per phase
        phase_metrics = {}
        for phase in ['entry', 'turn_in', 'apex', 'acceleration', 'exit']:
            phase_data = df[df['cornering_phase'] == phase]
            if len(phase_data) > 0:
                # Ensure all metrics are JSON-serializable
                speed_values = phase_data['Physics_speed_kmh'].dropna()
                steering_values = np.abs(phase_data['Physics_steer_angle']).dropna()
                
                phase_metrics[phase] = {
                    'avg_speed': float(speed_values.mean()) if len(speed_values) > 0 else 0.0,
                    'avg_steering': float(steering_values.mean()) if len(steering_values) > 0 else 0.0,
                    'data_points': int(len(phase_data))
                }
                
                if 'Physics_brake' in df.columns:
                    brake_values = phase_data['Physics_brake'].dropna()
                    phase_metrics[phase]['avg_brake'] = float(brake_values.mean()) if len(brake_values) > 0 else 0.0
                else:
                    phase_metrics[phase]['avg_brake'] = 0.0
                    
                if 'Physics_gas' in df.columns:
                    throttle_values = phase_data['Physics_gas'].dropna()
                    phase_metrics[phase]['avg_throttle'] = float(throttle_values.mean()) if len(throttle_values) > 0 else 0.0
                else:
                    phase_metrics[phase]['avg_throttle'] = 0.0
            else:
                # If no data for this phase, fill with default values (no None values)
                phase_metrics[phase] = {
                    'avg_speed': 0.0,
                    'avg_steering': 0.0,
                    'data_points': 0,
                    'avg_brake': 0.0,
                    'avg_throttle': 0.0
                }
        
        # Get detailed information for each corner
        corner_details = {}
        valid_corner_ids = [cid for cid in df['corner_id'].unique() if cid >= 0]
        
        for corner_id in valid_corner_ids:
            corner_data = df[df['corner_id'] == corner_id].copy()
            
            if len(corner_data) == 0:
                continue
                
            corner_details[f"corner_{corner_id}"] = self._get_corner_phase_details(corner_data)
        
        return {
            'total_corners_detected': int(corner_count),
            'phase_distribution': {str(k): int(v) for k, v in phase_counts.to_dict().items()},
            'phase_metrics': phase_metrics,
            'corner_ids': [int(cid) for cid in sorted(valid_corner_ids)],
            'corner_details': corner_details
        }
    
    def _get_corner_phase_details(self, corner_data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed phase information for a single corner including normalized car positions"""
        
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
                corner_detail['corner_start_position'] = float(position_data.iloc[0])
                corner_detail['corner_end_position'] = float(position_data.iloc[-1])
        
        # Get detailed information for each phase
        for phase in phases:
            phase_data = corner_data[corner_data['cornering_phase'] == phase].copy()
            
            # Initialize phase info with default values (no None values)
            phase_info = {
                'normalized_car_position': 0.0,
                'avg_speed': 0.0,
                'avg_steering_angle': 0.0,
                'duration_points': int(len(phase_data)),
                'avg_brake': 0.0,
                'avg_throttle': 0.0
            }
            
            if len(phase_data) == 0:
                # If no data for this phase, try to estimate position from corner progression
                if 'Graphics_normalized_car_position' in corner_data.columns:
                    total_corner_positions = corner_data['Graphics_normalized_car_position'].dropna()
                    if len(total_corner_positions) > 0:
                        # Estimate position based on phase order within corner
                        phase_order = {'entry': 0.1, 'turn_in': 0.3, 'apex': 0.5, 'acceleration': 0.7, 'exit': 0.9}
                        if phase in phase_order:
                            start_pos = float(total_corner_positions.iloc[0])
                            end_pos = float(total_corner_positions.iloc[-1])
                            estimated_pos = start_pos + (end_pos - start_pos) * phase_order[phase]
                            phase_info['normalized_car_position'] = float(estimated_pos)
                
                corner_detail['phases'][phase] = phase_info
                continue
            
            # Calculate phase metrics for phases with data
            if 'Physics_speed_kmh' in phase_data.columns:
                speed_values = phase_data['Physics_speed_kmh'].dropna()
                if len(speed_values) > 0:
                    phase_info['avg_speed'] = float(speed_values.mean())
            
            if 'Physics_steer_angle' in phase_data.columns:
                steering_values = np.abs(phase_data['Physics_steer_angle']).dropna()
                if len(steering_values) > 0:
                    phase_info['avg_steering_angle'] = float(steering_values.mean())
            
            # Get normalized car position for this phase - be more robust with NaN handling
            if 'Graphics_normalized_car_position' in phase_data.columns:
                position_values = phase_data['Graphics_normalized_car_position'].dropna()
                
                if len(position_values) > 0:
                    # Choose position based on phase type and available data
                    if phase == 'entry':
                        # For entry, use the starting position
                        phase_info['normalized_car_position'] = float(position_values.iloc[0])
                    elif phase == 'exit':
                        # For exit, use the ending position  
                        phase_info['normalized_car_position'] = float(position_values.iloc[-1])
                    else:
                        # For turn_in, apex, acceleration - use middle position
                        if len(position_values) >= 3:
                            mid_idx = len(position_values) // 2
                            phase_info['normalized_car_position'] = float(position_values.iloc[mid_idx])
                        elif len(position_values) == 2:
                            # If only 2 points, average them
                            phase_info['normalized_car_position'] = float(position_values.mean())
                        else:
                            # If only 1 point, use it
                            phase_info['normalized_car_position'] = float(position_values.iloc[0])
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
                            start_pos = float(total_corner_positions.iloc[0])
                            end_pos = float(total_corner_positions.iloc[-1])
                            interpolated_pos = start_pos + (end_pos - start_pos) * phase_relative_pos
                            phase_info['normalized_car_position'] = float(interpolated_pos)
            
            # Add brake and throttle data if available
            if 'Physics_brake' in phase_data.columns:
                brake_values = phase_data['Physics_brake'].dropna()
                if len(brake_values) > 0:
                    phase_info['avg_brake'] = float(brake_values.mean())
                
            if 'Physics_gas' in phase_data.columns:
                throttle_values = phase_data['Physics_gas'].dropna()
                if len(throttle_values) > 0:
                    phase_info['avg_throttle'] = float(throttle_values.mean())
            
            corner_detail['phases'][phase] = phase_info
        
        return corner_detail
