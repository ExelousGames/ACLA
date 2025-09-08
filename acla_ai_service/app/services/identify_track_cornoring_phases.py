import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


class TrackCorneringAnalyzer:
    """
    A comprehensive class for identifying and analyzing track cornering phases
    from telemetry data including steering angles, speed, and car position.
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
        
        # Step 1: Identify corners using steering angle and position changes
        corners = self._identify_corners(df_analysis)
        
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
        """Identify corner sections using steering angle and position data"""
        
        # Calculate steering angle magnitude and smooth it
        steering_abs = np.abs(df['Physics_steer_angle']).rolling(window=5).mean()
        
        # Define steering threshold for corner detection
        steering_threshold = steering_abs.quantile(self.steering_threshold_percentile)
        
        # Find regions where steering exceeds threshold
        in_corner = steering_abs > steering_threshold
        
        corners = {}
        corner_id = 0
        in_corner_section = False
        corner_start = 0
        
        for i, is_cornering in enumerate(in_corner):
            if is_cornering and not in_corner_section:
                # Start of new corner
                corner_start = max(0, i - 10)  # Include approach phase
                in_corner_section = True
            elif not is_cornering and in_corner_section:
                # End of corner
                corner_end = min(len(df) - 1, i + 10)  # Include exit phase
                
                # Only consider significant corners (minimum duration)
                if corner_end - corner_start > self.min_corner_duration:
                    corners[corner_id] = {
                        'start_idx': corner_start,
                        'end_idx': corner_end,
                        'position_start': df.iloc[corner_start]['Graphics_normalized_car_position'],
                        'position_end': df.iloc[corner_end]['Graphics_normalized_car_position'],
                        'max_steering': steering_abs.iloc[corner_start:corner_end].max()
                    }
                    corner_id += 1
                
                in_corner_section = False
        
        return corners

    def _analyze_corner_phases(self, corner_df: pd.DataFrame, corner_id: int) -> Dict[str, List[int]]:
        """Analyze a single corner to identify Entry, Turn-in, Apex, Acceleration, Exit phases"""
        
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
        
        # Phase 1: Entry (start until significant braking ends)
        entry_end = self._find_entry_phase_end(corner_df, brake_data, speed_smooth)
        # Ensure entry phase has at least some data points
        entry_end = max(entry_end, 3)  # Minimum 3 points for entry
        phases['entry'] = list(range(0, min(entry_end, len(corner_df))))
        
        # Phase 2: Turn-in (end of entry until apex)
        turn_in_start = max(entry_end, 0)
        apex_idx = self._find_apex_index(corner_df, min_speed_idx, max_steering_idx)
        # Ensure turn-in phase exists
        if apex_idx <= turn_in_start:
            apex_idx = turn_in_start + max(3, (len(corner_df) - turn_in_start) // 4)
        phases['turn_in'] = list(range(turn_in_start, min(apex_idx, len(corner_df))))
        
        # Phase 3: Apex (around minimum speed point)
        apex_window = max(3, len(corner_df) // 10)  # Dynamic window size
        apex_start = max(0, apex_idx - apex_window // 2)
        apex_end = min(len(corner_df), apex_idx + apex_window // 2)
        # Ensure apex doesn't overlap too much with turn_in
        apex_start = max(apex_start, turn_in_start + len(phases['turn_in']) - 1)
        phases['apex'] = list(range(apex_start, apex_end))
        
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
        """Find where entry phase ends (typically when heavy braking stops)"""
        
        # Look for significant braking
        brake_threshold = brake_data.quantile(0.8) if brake_data.max() > 0 else 0
        
        if brake_threshold > 0:
            # Find last point of significant braking
            significant_braking = brake_data > brake_threshold
            if significant_braking.any():
                last_brake_idx = significant_braking[::-1].idxmax()  # Last occurrence
                return max(5, last_brake_idx - corner_df.index[0] + 3)
        
        # Fallback: use speed reduction pattern
        speed_diff = speed_smooth.diff()
        # Find where speed stops decreasing significantly
        for i in range(5, len(speed_diff) - 5):
            if speed_diff.iloc[i:i+5].mean() > -0.5:  # Speed stabilizing
                return i
        
        return len(corner_df) // 4  # Fallback to 25% of corner

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
        
        # Fallback: use remaining 70% of corner for accel + exit
        remaining_length = len(corner_df) - start_idx
        return start_idx + int(remaining_length * 0.7)

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
            return working_df
        else:
            print(f"[INFO] Found Graphics_is_valid_lap column, will filter laps by validity percentage")
        
        # Group by lap and calculate lap times
        # Use both Graphics_completed_lap and Graphics_normalized_car_position together for robust lap detection
        has_completed_lap = 'Graphics_completed_lap' in working_df.columns
        has_position = 'Graphics_normalized_car_position' in working_df.columns
        
        # Only proceed if we have both fields - return empty data otherwise
        if not (has_completed_lap and has_position):
            print("[WARNING] Lap filtering requires both Graphics_completed_lap and Graphics_normalized_car_position - returning empty DataFrame")
            return pd.DataFrame()
        
        # Use both completed_lap counter and position data for most accurate lap detection
        print("[INFO] Using both Graphics_completed_lap and Graphics_normalized_car_position for lap detection")
        
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
            print(f"[WARNING] No valid full lap times found out of {total_laps_processed} processed laps, returning empty DataFrame")
            return pd.DataFrame()
        
        print(f"[INFO] Processed {total_laps_processed} potential laps")
        if has_valid_lap_column:
            print(f"[INFO] Found {full_laps_found} complete full laps with ≥{self.min_valid_percentage:.1%} valid data points")
        else:
            print(f"[INFO] Found {full_laps_found} complete full laps (validity checking skipped)")
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
                phase_metrics[phase] = {
                    'avg_speed': phase_data['Physics_speed_kmh'].mean(),
                    'avg_steering': np.abs(phase_data['Physics_steer_angle']).mean(),
                    'data_points': len(phase_data)
                }
                
                if 'Physics_brake' in df.columns:
                    phase_metrics[phase]['avg_brake'] = phase_data['Physics_brake'].mean()
                if 'Physics_gas' in df.columns:
                    phase_metrics[phase]['avg_throttle'] = phase_data['Physics_gas'].mean()
        
        # Get detailed information for each corner
        corner_details = {}
        valid_corner_ids = [cid for cid in df['corner_id'].unique() if cid >= 0]
        
        for corner_id in valid_corner_ids:
            corner_data = df[df['corner_id'] == corner_id].copy()
            
            if len(corner_data) == 0:
                continue
                
            corner_details[f"corner_{corner_id}"] = self._get_corner_phase_details(corner_data)
        
        return {
            'total_corners_detected': corner_count,
            'phase_distribution': phase_counts.to_dict(),
            'phase_metrics': phase_metrics,
            'corner_ids': sorted(valid_corner_ids),
            'corner_details': corner_details
        }
    
    def _get_corner_phase_details(self, corner_data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed phase information for a single corner including normalized car positions"""
        
        phases = ['entry', 'turn_in', 'apex', 'acceleration', 'exit']
        corner_detail = {
            'corner_start_position': None,
            'corner_end_position': None,
            'total_duration_points': len(corner_data),
            'phases': {}
        }
        
        # Get overall corner start and end positions
        if 'Graphics_normalized_car_position' in corner_data.columns:
            position_data = corner_data['Graphics_normalized_car_position'].dropna()
            if len(position_data) > 0:
                corner_detail['corner_start_position'] = position_data.iloc[0]
                corner_detail['corner_end_position'] = position_data.iloc[-1]
        
        # Get detailed information for each phase
        for phase in phases:
            phase_data = corner_data[corner_data['cornering_phase'] == phase].copy()
            
            # Initialize phase info with None values
            phase_info = {
                'normalized_car_position': None,
                'avg_speed': None,
                'avg_steering_angle': None,
                'duration_points': len(phase_data),
                'avg_brake': None,
                'avg_throttle': None
            }
            
            if len(phase_data) == 0:
                # If no data for this phase, try to estimate position from corner progression
                if 'Graphics_normalized_car_position' in corner_data.columns:
                    total_corner_positions = corner_data['Graphics_normalized_car_position'].dropna()
                    if len(total_corner_positions) > 0:
                        # Estimate position based on phase order within corner
                        phase_order = {'entry': 0.1, 'turn_in': 0.3, 'apex': 0.5, 'acceleration': 0.7, 'exit': 0.9}
                        if phase in phase_order:
                            start_pos = total_corner_positions.iloc[0]
                            end_pos = total_corner_positions.iloc[-1]
                            estimated_pos = start_pos + (end_pos - start_pos) * phase_order[phase]
                            phase_info['normalized_car_position'] = estimated_pos
                
                corner_detail['phases'][phase] = phase_info
                continue
            
            # Calculate phase metrics for phases with data
            if 'Physics_speed_kmh' in phase_data.columns:
                speed_values = phase_data['Physics_speed_kmh'].dropna()
                if len(speed_values) > 0:
                    phase_info['avg_speed'] = speed_values.mean()
            
            if 'Physics_steer_angle' in phase_data.columns:
                steering_values = np.abs(phase_data['Physics_steer_angle']).dropna()
                if len(steering_values) > 0:
                    phase_info['avg_steering_angle'] = steering_values.mean()
            
            # Get normalized car position for this phase - be more robust with NaN handling
            if 'Graphics_normalized_car_position' in phase_data.columns:
                position_values = phase_data['Graphics_normalized_car_position'].dropna()
                
                if len(position_values) > 0:
                    # Choose position based on phase type and available data
                    if phase == 'entry':
                        # For entry, use the starting position
                        phase_info['normalized_car_position'] = position_values.iloc[0]
                    elif phase == 'exit':
                        # For exit, use the ending position  
                        phase_info['normalized_car_position'] = position_values.iloc[-1]
                    else:
                        # For turn_in, apex, acceleration - use middle position
                        if len(position_values) >= 3:
                            mid_idx = len(position_values) // 2
                            phase_info['normalized_car_position'] = position_values.iloc[mid_idx]
                        elif len(position_values) == 2:
                            # If only 2 points, average them
                            phase_info['normalized_car_position'] = position_values.mean()
                        else:
                            # If only 1 point, use it
                            phase_info['normalized_car_position'] = position_values.iloc[0]
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
                            start_pos = total_corner_positions.iloc[0]
                            end_pos = total_corner_positions.iloc[-1]
                            phase_info['normalized_car_position'] = start_pos + (end_pos - start_pos) * phase_relative_pos
            
            # Add brake and throttle data if available
            if 'Physics_brake' in phase_data.columns:
                brake_values = phase_data['Physics_brake'].dropna()
                if len(brake_values) > 0:
                    phase_info['avg_brake'] = brake_values.mean()
                
            if 'Physics_gas' in phase_data.columns:
                throttle_values = phase_data['Physics_gas'].dropna()
                if len(throttle_values) > 0:
                    phase_info['avg_throttle'] = throttle_values.mean()
            
            corner_detail['phases'][phase] = phase_info
        
        return corner_detail
