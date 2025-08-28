# ACLA AI Service - AC Competizione Telemetry Integration

This document explains how to integrate all the AC Competizione telemetry features you specified into the ACLA AI Service for comprehensive racing performance analysis.

## Overview

The ACLA AI Service now supports comprehensive telemetry analysis with **220+ features** from Assetto Corsa Competizione, including:

- **Physics telemetry** (120+ features): Speed, G-forces, tyre data, suspension, brakes, engine
- **Graphics telemetry** (85+ features): Lap times, gaps, fuel, track status, display info  
- **Static telemetry** (22+ features): Car setup, track info, session parameters

## Your Input Features

The system now supports all the features you specified:

```
Physics_pad_life_front_left, Static_sector_count, Physics_wheel_angular_s_front_left,
Physics_brake_pressure_rear_left, Graphics_ideal_line_on, Graphics_is_valid_lap,
Graphics_packed_id, Graphics_delta_lap_time_str, Graphics_mfd_tyre_pressure_rear_left,
... (and 211 more features)
```

## New Features Added

### 1. Comprehensive Telemetry Models (`telemetry_models.py`)

- **TelemetryFeatures**: Complete catalog of all AC Competizione features
- **FeatureProcessor**: Data validation and preprocessing
- **TelemetryDataModel**: Pydantic model for data validation

### 2. Enhanced AI Analysis (`advanced_analyzer.py`)

- **Telemetry Summary**: Feature validation and data quality assessment
- **Advanced Performance Analysis**: Speed, cornering, braking, tyre management
- **Setup Analysis**: TC/ABS effectiveness, brake bias optimization
- **Enhanced Scoring**: Multi-dimensional performance scoring with telemetry confidence

### 3. New API Endpoints (`main.py`)

#### `POST /telemetry/upload`
Upload comprehensive telemetry data with automatic feature validation.

```python
{
    "session_id": "your_session_id",
    "telemetry_data": {
        "Physics_speed_kmh": [120.5, 125.2, ...],
        "Physics_brake": [0.8, 0.9, ...],
        # ... all your telemetry features
    },
    "metadata": {
        "track": "spa_francorchamps",
        "car": "mercedes_amg_gt3"
    }
}
```

#### `POST /telemetry/analyze`
Perform comprehensive telemetry analysis.

```python
{
    "session_id": "your_session_id", 
    "analysis_type": "comprehensive", # or "performance", "setup", "telemetry_summary"
    "features": ["Physics_speed_kmh", "Physics_brake"] # optional specific features
}
```

#### `GET /telemetry/features`
Get information about all available telemetry features.

#### `POST /telemetry/validate`
Validate telemetry data structure and feature coverage.

## Analysis Capabilities

### 1. Performance Score (Enhanced)
- **Speed Consistency** (30% weight): Smooth throttle and racing line
- **Cornering Performance** (25% weight): G-force analysis and consistency  
- **Braking Efficiency** (20% weight): Threshold braking and consistency
- **Tyre Management** (15% weight): Temperature optimization
- **Setup Optimization** (10% weight): TC/ABS effectiveness

### 2. Advanced Performance Analysis
- **Speed Patterns**: Acceleration, top speed zones, variance analysis
- **Cornering Analysis**: Lateral G-forces, high-G percentage, consistency
- **Braking Performance**: Pressure patterns, temperature analysis
- **Tyre Performance**: Temperature balance across all four wheels
- **Setup Analysis**: Brake bias, TC/ABS intervention rates

### 3. Data Quality Assessment
- **Feature Coverage**: Percentage of expected features present
- **Data Completeness**: Missing values analysis per feature
- **Anomaly Detection**: Outlier identification in key metrics
- **Consistency Scoring**: Overall data reliability assessment

## Usage Examples

### Basic Integration

```python
from telemetry_demo import ACLATelemetryClient

# Initialize client
client = ACLATelemetryClient("http://localhost:8000")

# Upload your telemetry data
result = client.upload_telemetry_data(
    session_id="my_session",
    telemetry_data=your_telemetry_dict
)

# Perform comprehensive analysis
analysis = client.analyze_telemetry("my_session", "comprehensive")
print(f"Performance Score: {analysis['result']['performance_score']['overall_score']}")
```

### Feature Validation

```python
# Check which features are available
features_info = client.get_available_features()
print(f"Total features: {features_info['total_features']}")

# Validate your data structure
validation = client.validate_telemetry(your_telemetry_dict)
print(f"Coverage: {validation['validation_result']['coverage_percentage']}%")
```

## Real-World Integration Steps

### 1. Data Collection
Collect telemetry from AC Competizione using your preferred method:
- Shared memory interface
- UDP broadcasting
- Log file parsing

### 2. Data Formatting
Format data as dictionary with your feature names as keys:

```python
telemetry_data = {
    "Physics_speed_kmh": [speed_values_list],
    "Physics_brake": [brake_values_list],
    "Graphics_last_time": [lap_time_values_list],
    # ... all other features
}
```

### 3. Upload and Analysis
```python
# Upload data
upload_result = client.upload_telemetry_data(session_id, telemetry_data)

# Get comprehensive analysis
analysis = client.analyze_telemetry(session_id, "comprehensive")

# Extract insights
performance_score = analysis['result']['performance_score']['overall_score']
recommendations = analysis['result']['performance_score']['recommendations']
```

## Analysis Outputs

### Performance Score Response
```json
{
    "overall_score": 78.5,
    "grade": "B",
    "components": {
        "speed_consistency": 85.2,
        "cornering_performance": 72.1,
        "braking_efficiency": 80.3,
        "tyre_management": 65.8,
        "setup_optimization": 45.0
    },
    "recommendations": [
        "Focus on maintaining consistent speed - work on racing line precision",
        "Cornering needs improvement - practice trail braking",
        "Poor tyre temperature management - adjust driving style"
    ],
    "telemetry_coverage": 95.4,
    "analysis_confidence": "high"
}
```

### Advanced Analysis Response
```json
{
    "speed_analysis": {
        "max_speed": 287.5,
        "avg_speed": 145.2,
        "max_acceleration": 15.3,
        "speed_variance": 1250.8
    },
    "cornering_analysis": {
        "max_lateral_g": 2.8,
        "avg_lateral_g": 1.2,
        "high_g_percentage": 15.5,
        "cornering_consistency": 0.3
    },
    "tyre_analysis": {
        "tyre_temperatures": {
            "front_left": {"max": 105.2, "avg": 92.1},
            "front_right": {"max": 108.5, "avg": 94.3}
        },
        "optimal_temp_percentage": {
            "front_left": 78.5,
            "front_right": 82.1
        }
    }
}
```

## Running the Demo

1. Start the ACLA AI service:
```bash
cd acla_ai_service
python main.py
```

2. Run the telemetry demo:
```bash
python telemetry_demo.py
```

This will demonstrate:
- Feature validation with your input features
- Sample data upload
- Comprehensive analysis
- Performance scoring
- Recommendations generation

## Configuration

### Environment Variables
- `BACKEND_URL`: Backend service URL (default: http://backend:7001)
- `OPENAI_API_KEY`: OpenAI API key for advanced AI features

### Customization Options
- Modify `TelemetryFeatures` class to add custom feature categories
- Adjust performance scoring weights in `racing_performance_score()`
- Customize analysis parameters in `AdvancedRacingAnalyzer`

## Performance Recommendations

The enhanced AI provides specific recommendations based on your telemetry:

- **Speed Consistency**: Racing line optimization, throttle control
- **Cornering**: Trail braking, apex positioning, gradual throttle application  
- **Braking**: Threshold braking, brake point consistency
- **Tyre Management**: Temperature window optimization, driving style adjustments
- **Setup**: Brake bias, TC/ABS levels, differential settings

## Next Steps

1. **Replace Sample Data**: Use real AC Competizione telemetry
2. **Real-time Integration**: Set up live telemetry streaming
3. **Custom Analysis**: Add track-specific or car-specific analysis
4. **Data Storage**: Implement persistent storage for session history
5. **Advanced AI**: Integrate machine learning models for predictive analysis

## Support

For questions or issues:
- Check the console output for detailed error messages
- Validate your telemetry data structure using `/telemetry/validate`
- Ensure all required dependencies are installed
- Verify the AI service is running on the correct port

The system is now ready to provide comprehensive analysis of your AC Competizione telemetry data with all 220+ features integrated!
