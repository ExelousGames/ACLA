# Corner Identification Unsupervised Service

## Overview

The Corner Identification Unsupervised Service provides advanced corner identification and feature extraction capabilities for Assetto Corsa Competizione telemetry data. This service uses unsupervised machine learning techniques to:

1. **Identify corner segments** in telemetry data using clustering algorithms
2. **Extract detailed corner characteristics** including entry, apex, and exit phase analysis
3. **Generate comprehensive corner features** that can be inserted back into telemetry data
4. **Classify corner types** (hairpin, chicane, sweeper, etc.) automatically
5. **Provide curvature and performance analysis** for each identified corner

## Key Features

### Corner Identification
- **Unsupervised clustering** using DBSCAN and K-Means algorithms
- **Multi-factor analysis** combining steering angle, speed, G-forces, brake/throttle data
- **Adaptive thresholding** that adjusts to different driving styles and track characteristics
- **Fallback detection methods** ensuring robust corner identification

### Corner Characterization
The service extracts detailed characteristics for each corner:

#### Entry Phase Features
- `corner_entry_duration` - Duration of the entry phase
- `corner_entry_speed_delta` - Speed change during entry
- `corner_entry_brake_intensity` - Maximum brake application
- `corner_entry_steering_rate` - Rate of steering input
- `corner_entry_g_force_lat_max` - Maximum lateral G-force
- `corner_entry_g_force_long_max` - Maximum longitudinal G-force

#### Apex Phase Features
- `corner_apex_duration` - Duration at the apex
- `corner_apex_min_speed` - Minimum speed at apex
- `corner_apex_max_steering` - Maximum steering angle
- `corner_apex_curvature` - Curvature at the apex
- `corner_apex_g_force_lat` - Lateral G-force at apex

#### Exit Phase Features
- `corner_exit_duration` - Duration of the exit phase
- `corner_exit_speed_delta` - Speed change during exit
- `corner_exit_throttle_progression` - Throttle application rate
- `corner_exit_steering_unwind_rate` - Steering unwind rate
- `corner_exit_g_force_lat_max` - Maximum lateral G-force
- `corner_exit_g_force_long_max` - Maximum longitudinal G-force

#### Overall Corner Features
- `corner_total_duration` - Total corner duration
- `corner_severity` - Corner difficulty score
- `corner_type_numeric` - Corner type (encoded numerically)
- `corner_direction_numeric` - Corner direction (-1=left, 1=right)
- `corner_speed_efficiency` - Speed maintenance through corner
- `corner_racing_line_adherence` - Racing line adherence score

#### Curvature Analysis
- `corner_avg_curvature` - Average curvature through corner
- `corner_max_curvature` - Maximum curvature
- `corner_curvature_variance` - Curvature consistency

#### Advanced Performance Metrics
- `corner_trail_braking_score` - Trail braking technique score
- `corner_throttle_discipline_score` - Throttle smoothness score
- `corner_consistency_score` - Overall consistency score

### Corner Classification
Automatic classification into corner types:
- **Hairpin** - Long duration, high steering, low speed efficiency
- **Chicane** - Short duration, high steering variation
- **Sweeper** - Long duration, moderate steering, high speed efficiency
- **Fast Corner** - Short duration, low steering, high speed
- **Tight Corner** - High steering, low speed efficiency
- **Medium Corner** - Standard characteristics

## Usage

### Basic Usage

```python
from services.corner_identification_unsupervised_service import corner_identification_service

# Learn corner patterns for a track
results = await corner_identification_service.learn_track_corner_patterns(
    trackName="monza",
    carName="porsche_991ii_gt3_r"
)

# Enhance telemetry data with corner features
enhanced_data = await corner_identification_service.extract_corner_features_for_telemetry(
    telemetry_data=your_telemetry_data,
    trackName="monza",
    carName="porsche_991ii_gt3_r"
)
```

### Integration with Full Dataset ML Service

```python
from services.full_dataset_ml_service import Full_dataset_TelemetryMLService

ml_service = Full_dataset_TelemetryMLService()

# Learn corner characteristics
corner_results = await ml_service.learn_corner_characteristics(
    trackName="monza",
    carName="porsche_991ii_gt3_r"
)

# Enhance telemetry with corner features
enhanced_telemetry = await ml_service.enhance_telemetry_with_corner_features(
    telemetry_data=raw_telemetry,
    trackName="monza",
    carName="porsche_991ii_gt3_r"
)
```

## Data Flow

1. **Data Retrieval**: Service fetches telemetry data from backend using `backend_service.get_all_racing_sessions()`
2. **Data Preprocessing**: Telemetry data is cleaned and processed using `FeatureProcessor`
3. **Performance Filtering**: Only top-performing laps are used for corner pattern learning
4. **Corner Detection**: Unsupervised clustering identifies corner segments
5. **Feature Extraction**: Detailed characteristics are extracted for each corner
6. **Corner Classification**: Corners are automatically classified by type
7. **Feature Integration**: New corner features are added to telemetry data
8. **Model Persistence**: Learned patterns are saved for future use

## Algorithm Details

### Corner Detection Algorithm
1. **Feature Matrix Creation**: Combines steering, speed, G-forces, brake/throttle data
2. **Data Normalization**: Uses RobustScaler for outlier-resistant normalization
3. **Clustering**: DBSCAN for adaptive clustering, K-Means fallback
4. **Segment Extraction**: Converts cluster labels to corner segments
5. **Validation**: Ensures minimum duration and activity thresholds

### Phase Analysis
- **Apex Detection**: Identifies minimum speed point as apex
- **Entry Phase**: From corner start to apex
- **Exit Phase**: From apex to corner end
- **Characteristic Extraction**: Calculates metrics for each phase

### Performance Optimization
- **Top Performance Filtering**: Uses only fastest laps for pattern learning
- **Caching**: Learned patterns are cached for repeated use
- **Error Handling**: Robust fallback mechanisms prevent failures

## Configuration Parameters

- `min_corner_duration`: Minimum data points for valid corner (default: 15)
- `corner_detection_sensitivity`: Clustering sensitivity (default: 0.7)
- `smoothing_window`: Signal smoothing window (default: 7)
- `top_laps_percentage`: Percentage of top laps to use (default: 0.1)

## Output Format

The service returns enhanced telemetry data where each record contains:
- All original telemetry fields
- 30+ new corner-specific features
- Corner identification flags
- Phase classifications
- Performance metrics

## Error Handling

- **Missing Data**: Graceful handling of missing telemetry fields
- **No Corners Found**: Fallback detection methods
- **Processing Errors**: Returns original data if enhancement fails
- **Backend Failures**: Local processing continues without backend integration

## Performance Considerations

- **Memory Efficient**: Processes data in segments to manage memory usage
- **Fast Processing**: Optimized algorithms for real-time feature extraction
- **Scalable**: Can handle large telemetry datasets
- **Caching**: Avoids recomputation of learned patterns

## Future Enhancements

- **Real-time Processing**: Stream processing for live telemetry
- **Multi-car Analysis**: Comparative corner analysis across different cars
- **Weather Adaptation**: Corner characteristics in different weather conditions
- **Driver Comparison**: Corner-specific driver performance comparison
- **Setup Optimization**: Corner-specific setup recommendations
