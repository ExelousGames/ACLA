# Multi-Algorithm Machine Learning System for Racing Telemetry

This system provides a sophisticated multi-algorithm approach to racing telemetry analysis, allowing different machine learning algorithms to be used for different types of predictions. Each algorithm is optimized for specific prediction tasks.

## Overview

Instead of using a single algorithm for all predictions, this system intelligently selects the best algorithm for each task type:

- **Lap Time Prediction**: Gradient Boosting (complex non-linear relationships)
- **Fuel Consumption**: Linear Regression (linear relationships)
- **Setup Recommendation**: Random Forest Classifier (multi-class decisions)
- **Brake Performance**: Support Vector Regression (robust to outliers)
- **Tire Strategy**: Decision Tree Classifier (rule-based decisions)
- **Weather Adaptation**: Random Forest (handles missing data well)
- **Damage Prediction**: Gradient Boosting Classifier (complex pattern detection)

## Supported Algorithms

### Regression Algorithms
- **Linear Regression**: Fast, interpretable, good for linear relationships
- **Ridge Regression**: Handles multicollinearity, regularized
- **Lasso Regression**: Feature selection, sparse solutions
- **Elastic Net**: Combines Ridge and Lasso benefits
- **Random Forest**: Handles non-linear relationships, feature importance
- **Gradient Boosting**: High accuracy, handles complex patterns
- **Extra Trees**: Fast, reduces overfitting
- **Support Vector Regression (SVR)**: Robust to outliers
- **Neural Networks (MLP)**: Complex non-linear patterns
- **K-Nearest Neighbors**: Local patterns, non-parametric
- **SGD Regressor**: Incremental learning support

### Classification Algorithms
- **Random Forest Classifier**: Robust, feature importance
- **Gradient Boosting Classifier**: High accuracy for complex patterns
- **Logistic Regression**: Fast, interpretable probabilities
- **Support Vector Classifier**: Good for high-dimensional data
- **Neural Network Classifier**: Complex decision boundaries
- **K-Nearest Neighbors**: Local decision boundaries
- **Naive Bayes**: Fast, works well with small datasets
- **Decision Tree**: Interpretable rules

## Key Features

### 1. Automatic Algorithm Selection
```python
# The system automatically selects the best algorithm for each task
result = await telemetry_service.train_ai_model(
    telemetry_data=data,
    target_variable="lap_time",
    model_type="lap_time_prediction"  # Uses Gradient Boosting automatically
)
```

### 2. Algorithm Override
```python
# You can override with a specific algorithm
result = await telemetry_service.train_ai_model(
    telemetry_data=data,
    target_variable="lap_time", 
    model_type="lap_time_prediction",
    preferred_algorithm="random_forest"  # Override default
)
```

### 3. Task-Specific Feature Selection
Each prediction task uses features most relevant to that specific goal:
- **Lap Time**: Speed, acceleration, steering, throttle, brake
- **Fuel Consumption**: Throttle, speed, engine load
- **Brake Performance**: Brake pressure, temperature, speed
- **Setup**: Suspension, aerodynamics, tire pressure

### 4. Incremental Learning Support
Some algorithms support learning from new data without full retraining:
```python
# Update existing model with new data
result = await telemetry_service.train_ai_model(
    telemetry_data=new_data,
    target_variable="lap_time",
    model_type="lap_time_prediction",
    existing_model_data=previous_model  # Incremental update
)
```

### 5. Algorithm Comparison
```python
# Compare multiple algorithms on the same dataset
comparison = await compare_algorithms({
    "telemetry_data": data,
    "target_variable": "lap_time",
    "model_type": "lap_time_prediction",
    "algorithms": ["gradient_boosting", "random_forest", "neural_network"]
})
```

## API Endpoints

### Core Training Endpoints
- `POST /racing-session/train-model` - Train model with automatic algorithm selection
- `POST /racing-session/batch-train-model` - Batch training across sessions
- `POST /racing-session/evaluate-model` - Evaluate model performance

### Algorithm Information Endpoints
- `GET /racing-session/algorithms/available` - List all algorithms and tasks
- `GET /racing-session/algorithms/{model_type}` - Get algorithm info for specific task
- `POST /racing-session/compare-algorithms` - Compare algorithm performance

### Analysis Endpoints
- `POST /racing-session/validate-telemetry` - Validate data quality
- `POST /racing-session/predict` - Make predictions with trained models

## Usage Examples

### 1. Basic Training with Automatic Algorithm Selection
```python
training_request = {
    "session_id": "session_123",
    "telemetry_data": telemetry_records,
    "target_variable": "lap_time",
    "model_type": "lap_time_prediction"
}

response = requests.post("/racing-session/train-model", json=training_request)
```

### 2. Training with Specific Algorithm
```python
training_request = {
    "session_id": "session_123", 
    "telemetry_data": telemetry_records,
    "target_variable": "fuel_consumption",
    "model_type": "fuel_consumption",
    "preferred_algorithm": "ridge"  # Override default linear_regression
}
```

### 3. Algorithm Comparison
```python
comparison_request = {
    "telemetry_data": telemetry_records,
    "target_variable": "lap_time",
    "model_type": "lap_time_prediction",
    "algorithms": ["gradient_boosting", "random_forest", "neural_network"]
}

response = requests.post("/racing-session/compare-algorithms", json=comparison_request)
```

### 4. Get Available Algorithms
```python
response = requests.get("/racing-session/algorithms/available")
algorithms = response.json()

# Show all supported tasks
for task in algorithms["supported_tasks"]:
    print(f"Task: {task}")
    print(f"Description: {algorithms['task_descriptions'][task]}")
    print(f"Primary Algorithm: {algorithms['algorithm_options'][task]['primary']}")
```

## Response Format

### Training Response
```json
{
    "success": true,
    "model_data": "base64_encoded_model...",
    "model_type": "lap_time_prediction",
    "algorithm_used": "gradient_boosting",
    "algorithm_type": "regression",
    "target_variable": "lap_time",
    "training_metrics": {
        "test_r2": 0.892,
        "test_mse": 2.34,
        "test_mae": 1.12,
        "training_samples": 800,
        "test_samples": 200
    },
    "feature_names": ["Physics_speed_kmh", "Physics_gas", "Physics_brake", ...],
    "feature_count": 15,
    "feature_importance": {
        "Physics_speed_kmh": 0.285,
        "Physics_gas": 0.192,
        "Physics_brake": 0.134,
        ...
    },
    "supports_incremental": false,
    "recommendations": [
        "Model performance is excellent with R² > 0.85",
        "Feature importance shows speed is most critical factor"
    ],
    "alternative_algorithms": ["random_forest", "neural_network", "svr"]
}
```

### Algorithm Comparison Response
```json
{
    "message": "Algorithm comparison completed",
    "model_type": "lap_time_prediction",
    "target_variable": "lap_time",
    "best_algorithm": "gradient_boosting",
    "best_score": 0.892,
    "comparison_results": {
        "gradient_boosting": {
            "success": true,
            "metrics": {"test_r2": 0.892, "test_mse": 2.34},
            "algorithm_type": "regression"
        },
        "random_forest": {
            "success": true, 
            "metrics": {"test_r2": 0.856, "test_mse": 2.87},
            "algorithm_type": "regression"
        }
    }
}
```

## Performance Considerations

### Algorithm Speed Comparison
- **Fastest**: Linear Regression, Naive Bayes, Decision Tree
- **Medium**: Random Forest, K-Nearest Neighbors, Logistic Regression
- **Slower**: Gradient Boosting, SVR, Neural Networks

### Memory Usage
- **Low Memory**: Linear models, Decision Trees
- **Medium Memory**: Random Forest, K-NN
- **High Memory**: Neural Networks, SVR with RBF kernel

### Data Size Recommendations
- **Small Data (<100 samples)**: Linear Regression, Naive Bayes, Decision Tree
- **Medium Data (100-1000)**: Random Forest, Gradient Boosting
- **Large Data (>1000)**: SGD, Neural Networks, Linear models

## Configuration

### Algorithm Parameters
Each algorithm can be customized through the `AlgorithmConfiguration` class:

```python
# Custom algorithm parameters
algorithm_config.algorithms["random_forest"]["params"] = {
    "n_estimators": 200,  # More trees
    "max_depth": 10,      # Limit depth
    "random_state": 42
}
```

### Task-Specific Features
Define which features are most important for each task:

```python
# Custom feature recommendations
feature_recommendations = {
    "custom_prediction_task": ["speed", "throttle", "custom_feature"]
}
```

## Best Practices

1. **Start with Default Algorithms**: The system chooses optimal algorithms for each task
2. **Use Algorithm Comparison**: Compare performance before production deployment
3. **Monitor Performance**: Track metrics across different algorithms
4. **Incremental Learning**: Use for real-time model updates where supported
5. **Feature Engineering**: Task-specific features improve all algorithms
6. **Data Quality**: Validate telemetry data before training

## Demo Script

Run the included demo to see the system in action:

```bash
cd scripts
python multi_algorithm_demo.py
```

This demonstrates:
- Automatic algorithm selection
- Multiple prediction tasks
- Algorithm comparison
- Incremental learning
- Performance metrics
- Feature importance analysis

## Future Enhancements

- **AutoML Integration**: Automatic hyperparameter optimization
- **Ensemble Methods**: Combine multiple algorithms
- **Deep Learning**: Advanced neural network architectures
- **Online Learning**: Real-time model updates
- **Multi-objective Optimization**: Balance accuracy vs speed
- **Explainable AI**: Enhanced model interpretability
