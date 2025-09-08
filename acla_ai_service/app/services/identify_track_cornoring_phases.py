import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

def identify_cornering_phases(df: pd.DataFrame) -> pd.DataFrame:
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
    corners = _identify_corners(df_analysis)
    
    # Step 2: For each corner, identify the phases
    for corner_id, corner_data in corners.items():
        start_idx, end_idx = corner_data['start_idx'], corner_data['end_idx']
        corner_df = df_analysis.iloc[start_idx:end_idx+1].copy()
        
        # Analyze this corner section
        phases = _analyze_corner_phases(corner_df, corner_id)
        
        # Update main dataframe with phase information
        for phase_name, indices in phases.items():
            for idx in indices:
                actual_idx = start_idx + idx
                if actual_idx < len(df_analysis):
                    df_analysis.loc[actual_idx, 'cornering_phase'] = phase_name
                    df_analysis.loc[actual_idx, 'corner_id'] = corner_id
    
    return df_analysis

def _identify_corners(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Identify corner sections using steering angle and position data"""
    
    # Calculate steering angle magnitude and smooth it
    steering_abs = np.abs(df['Physics_steer_angle']).rolling(window=5).mean()
    
    # Define steering threshold for corner detection
    steering_threshold = steering_abs.quantile(0.7)  # Top 30% of steering input
    
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
            if corner_end - corner_start > 20:  # At least 20 data points
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

def _analyze_corner_phases(corner_df: pd.DataFrame, corner_id: int) -> Dict[str, List[int]]:
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
    entry_end = _find_entry_phase_end(corner_df, brake_data, speed_smooth)
    phases['entry'] = list(range(0, min(entry_end, len(corner_df))))
    
    # Phase 2: Turn-in (end of entry until apex)
    turn_in_start = max(entry_end, 0)
    apex_idx = _find_apex_index(corner_df, min_speed_idx, max_steering_idx)
    phases['turn_in'] = list(range(turn_in_start, min(apex_idx, len(corner_df))))
    
    # Phase 3: Apex (around minimum speed point)
    apex_window = max(3, len(corner_df) // 10)  # Dynamic window size
    apex_start = max(0, apex_idx - apex_window // 2)
    apex_end = min(len(corner_df), apex_idx + apex_window // 2)
    phases['apex'] = list(range(apex_start, apex_end))
    
    # Phase 4: Acceleration (apex end until throttle application stabilizes)
    accel_start = apex_end
    accel_end = _find_acceleration_phase_end(corner_df, throttle_data, accel_start)
    phases['acceleration'] = list(range(accel_start, min(accel_end, len(corner_df))))
    
    # Phase 5: Exit (acceleration end until corner end)
    exit_start = max(accel_end, apex_end)
    phases['exit'] = list(range(exit_start, len(corner_df)))
    
    # Clean up overlapping phases
    phases = _resolve_phase_overlaps(phases, len(corner_df))
    
    return phases

def _find_entry_phase_end(corner_df: pd.DataFrame, brake_data: pd.Series, speed_smooth: pd.Series) -> int:
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

def _find_apex_index(corner_df: pd.DataFrame, min_speed_idx: int, max_steering_idx: int) -> int:
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

def _find_acceleration_phase_end(corner_df: pd.DataFrame, throttle_data: pd.Series, start_idx: int) -> int:
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

def _resolve_phase_overlaps(phases: Dict[str, List[int]], total_length: int) -> Dict[str, List[int]]:
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

def get_cornering_analysis_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for cornering analysis"""
    
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
    
    return {
        'total_corners_detected': corner_count,
        'phase_distribution': phase_counts.to_dict(),
        'phase_metrics': phase_metrics,
        'corner_ids': sorted([cid for cid in df['corner_id'].unique() if cid >= 0])
    }