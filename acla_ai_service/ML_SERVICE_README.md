 🏎️ ACLA Telemetry ML Service

A comprehensive machine learning service for training AI models on Assetto Corsa Competizione telemetry data. This service leverages your `TelemetryFeatures` and `FeatureProcessor` classes to provide intelligent racing analysis.

## 🚀 Features

### 🎯 Supported Prediction Tasks
- **Lap Time Prediction** - Predict lap times based on telemetry features
- **Performance Classification** - Classify driver performance (Fast/Medium/Slow)
- **Speed Prediction** - Predict speed at different track points
- **Tire Strategy Analysis** - Predict tire temperatures and wear
- **Brake Performance** - Analyze braking efficiency and temperatures
- **Fuel Consumption** - Predict fuel usage patterns
- **Setup Optimization** - Recommend car setup changes
- **Weather Adaptation** - Analyze performance in different conditions

### 🤖 Machine Learning Models
- **Random Forest** (Regression & Classification)
- **Gradient Boosting** (Regression & Classification)
- **Linear/Logistic Regression**
- **Support Vector Machines**
- **Ridge/Lasso Regression**

### 📊 Features
- Automatic feature selection based on telemetry data
- Hyperparameter tuning with cross-validation
- Feature importance analysis
- Model performance metrics
- REST API endpoints
- Batch prediction capabilities

## 🛠️ Installation

The ML service is already integrated into your ACLA AI service. All required dependencies are listed in `requirements.txt`:

```bash
# Already included in your requirements.txt:
scikit-learn==1.3.2
joblib==1.3.2
pandas==2.1.4
numpy==1.24.3
```

## 📖 Quick Start

### 1. Using the Python API

```python
from app.services.scikit_ml_service import TelemetryMLService
import pandas as pd

# Initialize the service
ml_service = TelemetryMLService("models")

# Load your telemetry data
df = pd.read_csv("your_telemetry_data.csv")

# Train a lap time prediction model
results = ml_service.train_regression_model(
    df=df,
    target_column='Graphics_last_time',
    model_name='random_forest',
    model_type='lap_time_prediction'
)

print(f"Model ID: {results['model_id']}")
print(f"Test R²: {results['test_metrics']['r2']:.4f}")

# Make predictions
predictions = ml_service.predict(results['model_id'], new_data)
```

### 2. Using the REST API

#### Train a Model
```bash
curl -X POST "http://localhost:8000/ml/train/regression" \
  -F "file=@telemetry_data.csv" \
  -F "target_column=Graphics_last_time" \
  -F "model_name=random_forest" \
  -F "model_type=lap_time_prediction"
```

#### Make Predictions
```bash
curl -X POST "http://localhost:8000/ml/predict/csv" \
  -F "model_id=your_model_id" \
  -F "file=@new_telemetry_data.csv"
```

#### List Models
```bash
curl "http://localhost:8000/ml/models"
```

## 🎛️ API Endpoints

### Training Endpoints
- `POST /ml/train/regression` - Train regression models
- `POST /ml/train/classification` - Train classification models
- `POST /ml/train/specialized` - Train multiple specialized models

### Prediction Endpoints
- `POST /ml/predict` - Make predictions with JSON data
- `POST /ml/predict/csv` - Make predictions with CSV file

### Model Management
- `GET /ml/models` - List all trained models
- `GET /ml/models/{model_id}` - Get model details
- `GET /ml/models/{model_id}/importance` - Get feature importance
- `DELETE /ml/models/{model_id}` - Delete a model

### Analysis Endpoints
- `GET /ml/features/available` - List available telemetry features
- `POST /ml/features/analyze` - Analyze telemetry data features

## 📊 Example Use Cases

### 1. Lap Time Prediction

```python
# Train a model to predict lap times
results = ml_service.train_regression_model(
    df=telemetry_df,
    target_column='Graphics_last_time',
    model_type='lap_time_prediction',
    feature_selection='performance'
)

# Features automatically selected:
# - Physics_speed_kmh, Physics_brake, Physics_gas
# - Physics_g_force_x, Physics_g_force_y
# - Physics_tyre_core_temp_*, Physics_brake_temp_*
# - Graphics_delta_lap_time, Graphics_current_sector_index
```

### 2. Performance Classification

```python
# Classify driver performance
df['performance_category'] = ml_service.create_performance_categories(
    df, 'Graphics_last_time'
)

results = ml_service.train_classification_model(
    df=df,
    target_column='performance_category',
    model_type='performance_classification'
)

# Predicts: 'Fast', 'Medium', or 'Slow' performance
```

### 3. Tire Strategy Analysis

```python
# Predict tire temperatures for strategy
results = ml_service.train_regression_model(
    df=df,
    target_column='Physics_tyre_core_temp_front_left',
    model_type='tire_strategy',
    feature_selection='auto'
)

# Uses tire-specific features:
# - Tire pressures, slip angles, temperatures
# - Track conditions, driving style
# - Weather data
```

### 4. Specialized Models Training

```python
# Train multiple models at once
results = ml_service.train_specialized_models(df)

# Automatically trains:
# - Lap time prediction
# - Performance classification  
# - Speed prediction
# - Brake performance
# - Tire temperature prediction
```

## 🔧 Feature Selection

The service automatically selects relevant features based on the prediction task:

### Performance Critical Features
- Speed, gear, RPM, brake, gas, steering
- G-forces, slip angles, tire temperatures
- Lap timing data

### Tire Strategy Features
- Tire temperatures and pressures
- Slip angles and ratios
- Track and air temperature
- Driving style indicators

### Brake Performance Features
- Brake pressures and temperatures
- Brake pad/disc life
- Speed and G-force data
- Brake bias settings

### Setup Features
- Brake bias, TC, ABS levels
- Engine mapping
- Suspension settings
- Tire compound choices

## 📈 Model Performance Metrics

### Regression Models
- **R² Score** - Coefficient of determination
- **Mean Absolute Error (MAE)** - Average prediction error
- **Mean Squared Error (MSE)** - Squared prediction error
- **Cross-validation scores** - Model reliability

### Classification Models
- **Accuracy** - Overall classification accuracy
- **Precision/Recall** - Class-specific performance
- **F1 Score** - Balanced performance metric
- **Confusion Matrix** - Detailed classification results

## 🎯 Advanced Usage

### Custom Feature Selection

```python
# Use specific feature sets
from app.models.telemetry_models import TelemetryFeatures

features = TelemetryFeatures()

# Get features for specific tasks
brake_features = features.get_brake_performance_features()
tire_features = features.get_tire_strategy_features()
setup_features = features.get_setup_features()

# Filter available features in your data
available_features = features.filter_available_features(
    brake_features, 
    list(df.columns)
)
```

### Hyperparameter Tuning

```python
# Enable hyperparameter tuning (default: True)
results = ml_service.train_regression_model(
    df=df,
    target_column='Graphics_last_time',
    hyperparameter_tuning=True,  # Performs GridSearchCV
    cv_folds=5  # 5-fold cross-validation
)
```

### Feature Importance Analysis

```python
# Get feature importance after training
importance = ml_service.analyze_feature_importance(
    model_id='your_model_id',
    top_n=15
)

# Results show most influential features:
# {'Physics_speed_kmh': 0.245,
#  'Physics_brake': 0.189,
#  'Physics_g_force_y': 0.156, ...}
```

## 📁 Data Format

Your telemetry CSV should contain AC Competizione telemetry data with columns like:

```
Physics_speed_kmh,Physics_gear,Physics_rpm,Physics_brake,Physics_gas,...
150.5,4,6500,0.2,0.8,...
148.2,4,6400,0.0,0.9,...
```

The service automatically:
- Handles missing values
- Converts data types
- Processes complex telemetry fields
- Validates feature availability

## 🚦 Getting Started

1. **Prepare your telemetry data** in CSV format
2. **Start the ACLA AI service** (includes ML endpoints)
3. **Upload data and train models** via API or Python
4. **Make predictions** on new telemetry data
5. **Analyze results** and feature importance

### Example Script

```python
from app.services.scikit_ml_service import train_models_from_csv

# Train all specialized models from CSV
results = train_models_from_csv(
    csv_path="session_recording/your_telemetry.csv",
    output_dir="models"
)

print(f"Trained models: {list(results.keys())}")
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing target column**
   - Ensure the target column exists in your CSV
   - Check column names with `/ml/features/analyze`

2. **No valid features found**
   - Your data might not contain standard AC telemetry features
   - Use `feature_selection='all_numeric'` as fallback

3. **Poor model performance**
   - Check data quality and missing values
   - Try different model types (random_forest, gradient_boosting)
   - Increase dataset size

4. **Memory issues with large datasets**
   - Process data in smaller chunks
   - Use simpler models (linear_regression)
   - Reduce feature count

## 🤝 Integration

The ML service integrates seamlessly with your existing ACLA system:

- **Uses your TelemetryFeatures class** for feature definitions
- **Leverages FeatureProcessor** for data preprocessing  
- **Follows your project structure** and coding patterns
- **Supports your telemetry data format**
- **Extends your FastAPI application**

## 📚 Examples

Check out `examples/ml_service_examples.py` for comprehensive examples of:
- Basic training
- Classification models
- Multiple model training
- Predictions and analysis
- Feature importance analysis

## 🎉 Ready to Race!

Your ACLA Telemetry ML Service is ready to help you gain insights from your racing data. Start by training your first model and see what patterns the AI can discover in your telemetry! 🏁
