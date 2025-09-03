"""
FastAPI endpoints for the Telemetry ML Service

This module provides REST API endpoints for training and using AI models
with AC Competizione telemetry data.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime

from ..services.scikit_ml_service import TelemetryMLService
from ..models.telemetry_models import TelemetryFeatures

# Initialize router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Initialize ML service (will be created once)
ml_service = TelemetryMLService("models/ml_models")

# Pydantic models for request/response
class TrainingRequest(BaseModel):
    target_column: str = Field(..., description="Name of the target variable column")
    model_name: str = Field("random_forest", description="Type of ML model to train")
    model_type: str = Field("performance_classification", description="Type of prediction task")
    feature_selection: str = Field("auto", description="Feature selection method")
    hyperparameter_tuning: bool = Field(True, description="Whether to perform hyperparameter tuning")
    test_size: float = Field(0.2, description="Proportion of data for testing", ge=0.1, le=0.5)
    cv_folds: int = Field(5, description="Number of cross-validation folds", ge=3, le=10)

class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="ID of the trained model to use")
    data: List[Dict[str, Any]] = Field(..., description="Telemetry data for prediction")

class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    created_at: str
    feature_count: int
    performance_metrics: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_id: str
    predictions: List[float]
    prediction_count: int

class FeatureImportanceResponse(BaseModel):
    model_id: str
    feature_importance: Dict[str, float]
    top_features: List[str]

@router.post("/train/regression", response_model=ModelResponse)
async def train_regression_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with telemetry data")
):
    """
    Train a regression model for continuous target prediction
    
    Supports tasks like:
    - Lap time prediction
    - Speed prediction
    - Temperature prediction
    - Fuel consumption prediction
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"[INFO] Received CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Validate target column exists
        if request.target_column not in df.columns:
            available_columns = list(df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_column}' not found. Available columns: {available_columns[:20]}"
            )
        
        # Train model
        results = ml_service.train_regression_model(
            df=df,
            target_column=request.target_column,
            model_name=request.model_name,
            model_type=request.model_type,
            feature_selection=request.feature_selection,
            hyperparameter_tuning=request.hyperparameter_tuning,
            test_size=request.test_size,
            cv_folds=request.cv_folds
        )
        
        return ModelResponse(
            model_id=results['model_id'],
            model_name=results['model_name'],
            model_type=results['model_type'],
            created_at=datetime.now().isoformat(),
            feature_count=results['feature_count'],
            performance_metrics=results['test_metrics']
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/train/classification", response_model=ModelResponse)
async def train_classification_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with telemetry data")
):
    """
    Train a classification model for categorical target prediction
    
    Supports tasks like:
    - Performance classification (Fast/Medium/Slow)
    - Driver behavior classification
    - Setup recommendation
    - Weather condition classification
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"[INFO] Received CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # If target column is lap time, create performance categories
        if request.target_column == 'Graphics_last_time' and 'performance_category' not in df.columns:
            df['performance_category'] = ml_service.create_performance_categories(df, request.target_column)
            request.target_column = 'performance_category'
            print(f"[INFO] Created performance categories from lap times")
        
        # Validate target column exists
        if request.target_column not in df.columns:
            available_columns = list(df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_column}' not found. Available columns: {available_columns[:20]}"
            )
        
        # Train model
        results = ml_service.train_classification_model(
            df=df,
            target_column=request.target_column,
            model_name=request.model_name,
            model_type=request.model_type,
            feature_selection=request.feature_selection,
            hyperparameter_tuning=request.hyperparameter_tuning,
            test_size=request.test_size,
            cv_folds=request.cv_folds
        )
        
        return ModelResponse(
            model_id=results['model_id'],
            model_name=results['model_name'],
            model_type=results['model_type'],
            created_at=datetime.now().isoformat(),
            feature_count=results['feature_count'],
            performance_metrics=results['test_metrics']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/train/specialized")
async def train_specialized_models(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with telemetry data")
):
    """
    Train multiple specialized models for common racing scenarios
    
    This endpoint automatically trains several models:
    - Lap time prediction
    - Performance classification
    - Speed prediction
    - Brake performance analysis
    - Tire temperature prediction
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"[INFO] Training specialized models on {len(df)} rows of data")
        
        # Train specialized models
        results = ml_service.train_specialized_models(df)
        
        # Format response
        trained_models = []
        for model_type, result in results.items():
            if isinstance(result, dict) and 'model_id' in result:
                trained_models.append({
                    'model_type': model_type,
                    'model_id': result['model_id'],
                    'model_name': result['model_name'],
                    'feature_count': result['feature_count'],
                    'performance_metrics': result.get('test_metrics', {})
                })
        
        return {
            'message': f'Successfully trained {len(trained_models)} specialized models',
            'models': trained_models,
            'training_data_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Specialized training failed: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
async def make_predictions(request: PredictionRequest):
    """
    Make predictions using a trained model
    """
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided for prediction")
        
        # Make predictions
        predictions = ml_service.predict(request.model_id, df)
        
        return PredictionResponse(
            model_id=request.model_id,
            predictions=predictions.tolist(),
            prediction_count=len(predictions)
        )
        
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/csv", response_model=PredictionResponse)
async def make_predictions_from_csv(
    model_id: str,
    file: UploadFile = File(..., description="CSV file with telemetry data for prediction")
):
    """
    Make predictions from CSV file
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Make predictions
        predictions = ml_service.predict(model_id, df)
        
        return PredictionResponse(
            model_id=model_id,
            predictions=predictions.tolist(),
            prediction_count=len(predictions)
        )
        
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/features/available")
async def get_available_features():
    """
    Get information about available telemetry features
    """
    try:
        features = TelemetryFeatures()
        
        return {
            'feature_categories': {
                'physics': {
                    'count': len(features.PHYSICS_FEATURES),
                    'features': features.PHYSICS_FEATURES[:10]  # Show first 10 as example
                },
                'graphics': {
                    'count': len(features.GRAPHICS_FEATURES),
                    'features': features.GRAPHICS_FEATURES[:10]
                },
                'static': {
                    'count': len(features.STATIC_FEATURES),
                    'features': features.STATIC_FEATURES[:10]
                }
            },
            'specialized_feature_sets': {
                'performance_critical': features.get_performance_critical_features(),
                'tire_strategy': features.get_tire_strategy_features(),
                'brake_performance': features.get_brake_performance_features(),
                'setup_features': features.get_setup_features(),
                'fuel_consumption': features.get_fuel_consumption_features(),
                'weather_adaptation': features.get_weather_adaptation_features()
            },
            'total_features': len(features.get_all_features())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get features: {str(e)}")

@router.post("/features/analyze")
async def analyze_telemetry_features(
    file: UploadFile = File(..., description="CSV file with telemetry data")
):
    """
    Analyze which telemetry features are present in uploaded data
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Analyze features
        from ..models.telemetry_models import FeatureProcessor
        processor = FeatureProcessor(df)
        
        validation = processor.validate_features()
        metrics = processor.extract_performance_metrics()
        
        return {
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'feature_analysis': validation,
            'performance_metrics': metrics,
            'available_columns': list(df.columns)[:50]  # Limit for API response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a trained model
    """
    try:
        model_path = ml_service.models_directory / f"{model_id}.pkl"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model_path.unlink()  # Delete the file
        
        return {
            'message': f'Model {model_id} deleted successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the ML service
    """
    return {
        'status': 'healthy',
        'service': 'TelemetryMLService',
        'timestamp': datetime.now().isoformat(),
        'available_endpoints': [
            '/ml/train/regression',
            '/ml/train/classification', 
            '/ml/train/specialized',
            '/ml/predict',
            '/ml/models',
            '/ml/features/available'
        ]
    }
