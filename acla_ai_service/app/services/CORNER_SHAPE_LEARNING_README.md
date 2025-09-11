# Corner Shape Unsupervised Learning Service

## Overview

The Corner Shape Unsupervised Learning Service is a sophisticated AI system that automatically discovers and learns different types of corner shapes from racing telemetry data using unsupervised machine learning techniques. This service analyzes racing sessions to identify patterns in cornering behavior and creates clusters of similar corner types without requiring manually labeled data.

## Key Features

### 🔍 **Automatic Corner Discovery**
- Extracts corner characteristics from raw telemetry data
- Identifies corner entry, turn-in, apex, acceleration, and exit phases
- Analyzes speed profiles, steering patterns, and G-force signatures

### 🎯 **Unsupervised Clustering**
- Uses multiple clustering algorithms (K-Means, DBSCAN, Gaussian Mixture, Hierarchical)
- Automatically determines optimal number of corner types
- Provides performance metrics and algorithm comparison

### 📊 **Feature Extraction**
Extracts comprehensive corner shape features including:
- **Geometry Features**: Corner duration, length, steering angles
- **Speed Profile Features**: Entry/exit speeds, minimum speed, speed drop/gain
- **Steering Pattern Features**: Smoothness, peak position, aggressiveness
- **G-Force Features**: Lateral and longitudinal forces, combined magnitude
- **Input Pattern Features**: Braking and throttle application patterns
- **Trajectory Features**: Racing line consistency
- **Timing Features**: Phase duration ratios

### 🤖 **Corner Type Prediction**
- Predicts corner type for new telemetry data
- Provides confidence scores and cluster characteristics
- Offers human-readable descriptions of corner types

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Corner Shape Learning Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Data Retrieval                                          │
│     │                                                       │
│     ├── Backend Service ──→ Racing Sessions                 │
│     └── Telemetry Data Processing                           │
│                                                             │
│  2. Corner Identification                                   │
│     │                                                       │
│     ├── TrackCorneringAnalyzer ──→ Corner Detection         │
│     └── Phase Analysis (Entry/Apex/Exit)                    │
│                                                             │
│  3. Feature Extraction                                      │
│     │                                                       │
│     ├── CornerShapeFeatureExtractor                         │
│     ├── Geometry Features                                   │
│     ├── Speed/Steering Profiles                             │
│     └── G-Force/Input Patterns                              │
│                                                             │
│  4. Unsupervised Learning                                   │
│     │                                                       │
│     ├── Multiple Clustering Algorithms                      │
│     ├── Algorithm Performance Comparison                    │
│     └── Best Algorithm Selection                            │
│                                                             │
│  5. Analysis & Prediction                                   │
│     │                                                       │
│     ├── Cluster Characterization                            │
│     ├── Corner Type Descriptions                            │
│     └── Real-time Prediction                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService

# Initialize service
ml_service = Full_dataset_TelemetryMLService()

# Learn corner shapes for a track
results = await ml_service.learn_corner_shapes(
    trackName="monza",
    clustering_params={
        'n_clusters': 6,        # Try to find 6 corner types
        'eps': 0.4,             # DBSCAN epsilon parameter
        'min_samples': 3        # DBSCAN minimum samples
    }
)

# Check results
if 'error' not in results:
    print(f"Found {results['total_corners_found']} corners")
    print(f"Best algorithm: {results['clustering_results']['best_algorithm']}")
    
    # Show discovered corner types
    cluster_chars = results['cluster_analysis']['cluster_characteristics']
    for cluster_name, info in cluster_chars.items():
        print(f"{cluster_name}: {info['description']}")
```

### Corner Type Prediction

```python
# Predict corner type for current telemetry
current_telemetry = {
    'Physics_speed_kmh': 120.5,
    'Physics_steer_angle': 1500,
    'Physics_brake': 0.8,
    'Physics_gas': 0.2,
    'Physics_g_force_x': -1.2,
    'Physics_g_force_y': -0.8,
    'Graphics_normalized_car_position': 0.25
}

prediction = await ml_service.predict_corner_shape_type(
    trackName="monza",
    current_telemetry=current_telemetry
)

print(f"Corner type: {prediction['cluster_description']}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

### Analysis Summary

```python
# Get detailed analysis summary
summary = ml_service.get_corner_shape_analysis_summary("monza")

print(f"Sessions analyzed: {summary['total_sessions_analyzed']}")
print(f"Corners found: {summary['total_corners_found']}")
print(f"Features used: {len(summary['feature_names'])}")
```

## API Reference

### CornerShapeUnsupervisedService

#### `learn_corner_shapes(trackName, clustering_params=None)`
Learn corner shapes for a specific track.

**Parameters:**
- `trackName` (str): Name of the track to analyze
- `clustering_params` (dict, optional): Clustering algorithm parameters

**Returns:**
Dictionary with learning results including:
- `total_corners_found`: Number of corners detected
- `clustering_results`: Algorithm performance and selection
- `cluster_analysis`: Characteristics of each corner type
- `feature_names`: List of features used for clustering

#### `predict_corner_shape(trackName, current_telemetry)`
Predict corner type for current telemetry data.

**Parameters:**
- `trackName` (str): Track name
- `current_telemetry` (dict): Current corner telemetry data

**Returns:**
Dictionary with:
- `predicted_cluster`: Cluster ID
- `confidence`: Prediction confidence (0-1)
- `cluster_description`: Human-readable corner type
- `cluster_size`: Number of corners in this cluster

#### `get_corner_shape_summary(trackName)`
Get summary of learned corner shapes.

**Parameters:**
- `trackName` (str): Track name

**Returns:**
Dictionary with complete analysis summary and statistics.

### CornerShapeFeatureExtractor

#### `extract_corner_shape_features(corner_df, corner_info)`
Extract comprehensive features characterizing corner shape.

**Parameters:**
- `corner_df` (DataFrame): Corner telemetry data
- `corner_info` (dict): Corner metadata (positions, phases, etc.)

**Returns:**
Dictionary of extracted features including geometry, speed profile, steering patterns, G-forces, and timing characteristics.

## Clustering Algorithms

The service uses multiple clustering algorithms and automatically selects the best one:

### K-Means Clustering
- **Best for**: Well-separated, spherical clusters
- **Parameters**: `n_clusters` (number of corner types to find)
- **Advantages**: Fast, works well with balanced cluster sizes

### DBSCAN
- **Best for**: Irregular cluster shapes, handling noise
- **Parameters**: `eps` (neighborhood radius), `min_samples` (core point threshold)
- **Advantages**: Discovers clusters of arbitrary shape, identifies outliers

### Gaussian Mixture Model
- **Best for**: Overlapping clusters, probabilistic assignments
- **Parameters**: `n_components` (number of mixture components)
- **Advantages**: Provides probability scores, handles overlapping distributions

### Hierarchical Clustering
- **Best for**: Hierarchical cluster relationships
- **Parameters**: `n_clusters` (number of final clusters)
- **Advantages**: Creates cluster hierarchy, deterministic results

### Algorithm Selection

The service automatically selects the best algorithm based on:
1. **Silhouette Score**: Measures cluster separation quality
2. **Cluster Count**: Preference for reasonable number of clusters (3-8)
3. **Algorithm-specific metrics**: BIC/AIC for GMM, noise ratio for DBSCAN

## Feature Categories

### Geometry Features
- `corner_duration`: Number of telemetry points in corner
- `corner_length`: Track distance covered by corner
- `corner_density`: Data density (points per track distance)
- `max_steering_angle`: Maximum absolute steering angle
- `avg_steering_angle`: Average absolute steering angle
- `steering_direction`: Left (-1) or right (+1) corner

### Speed Profile Features
- `entry_speed`: Speed at corner entry
- `exit_speed`: Speed at corner exit
- `min_speed`: Minimum speed in corner
- `speed_drop`: Entry speed minus minimum speed
- `speed_gain`: Exit speed minus minimum speed
- `min_speed_position`: Relative position of minimum speed (0=entry, 1=exit)

### Steering Pattern Features
- `steering_smoothness`: Inverse of steering input roughness
- `max_steering_position`: Relative position of maximum steering
- `steering_aggressiveness`: Maximum rate of steering change

### G-Force Features
- `max_lateral_g`: Maximum lateral G-force
- `max_braking_g`: Maximum braking G-force
- `max_accel_g`: Maximum acceleration G-force
- `max_combined_g`: Maximum combined G-force magnitude

### Input Pattern Features
- `max_brake_input`: Maximum brake pedal input
- `brake_usage_ratio`: Fraction of corner spent braking
- `max_throttle_input`: Maximum throttle input
- `throttle_application_point`: Relative position where throttle is applied
- `throttle_usage_ratio`: Fraction of corner spent on throttle

## Corner Types Discovered

The system typically discovers these corner types:

### **Tight Hairpins**
- High maximum steering angles (>300°)
- Large speed drops (>60 km/h)
- Heavy braking usage
- Late throttle application

### **High-Speed Sweepers**
- Low maximum steering angles (<100°)
- Small speed drops (<20 km/h)
- Minimal braking
- Early throttle application

### **Technical Corners**
- Moderate steering angles (100-250°)
- Complex speed profiles
- Trail-braking patterns
- Precise throttle control

### **Chicanes**
- Rapid steering direction changes
- Multiple speed variations
- Complex G-force patterns
- Mixed input patterns

## Integration

### Backend Integration
The service automatically saves learned models to the backend and retrieves racing session data for analysis.

### Cache Management
- Learned models are cached for fast access
- Cache can be cleared per track or globally
- Automatic cache validation and refresh

### Error Handling
Comprehensive error handling for:
- Missing or insufficient data
- Backend communication failures
- Clustering algorithm failures
- Feature extraction issues

## Performance Considerations

### Data Requirements
- Minimum 3 corners needed for clustering
- Better results with 20+ corners
- Multiple racing sessions recommended for robust learning

### Computational Complexity
- Feature extraction: O(n) where n is telemetry points
- Clustering: O(n²) for most algorithms
- Prediction: O(1) for trained models

### Memory Usage
- Scales with number of corners and features
- Models are compressed for storage
- Efficient caching to minimize memory footprint

## Examples

See `app/examples/corner_shape_learning_demo.py` for complete usage examples and demonstrations of all service capabilities.

## Future Enhancements

### Planned Features
- **Dynamic clustering**: Automatic optimal cluster count detection
- **Temporal analysis**: Corner evolution over time
- **Multi-track learning**: Cross-track pattern recognition
- **Real-time adaptation**: Online learning from new data
- **Visual analytics**: Interactive cluster visualization

### Research Opportunities
- **Deep learning**: Neural network-based corner embeddings
- **Causal analysis**: Understanding corner performance factors
- **Transfer learning**: Apply learned patterns to new tracks
- **Ensemble methods**: Combine multiple clustering approaches
