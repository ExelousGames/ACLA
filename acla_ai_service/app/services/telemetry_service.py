"""
Telemetry data processing and analysis service
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import base64
from datetime import datetime, timezone
# Using River for online learning instead of scikit-learn
from river import preprocessing, metrics
import io
from app.models.telemetry_models import TelemetryFeatures, FeatureProcessor
from app.models.ml_algorithms import AlgorithmConfiguration
from app.services.river_ml_service import RiverMLService
from app.analyzers import AdvancedRacingAnalyzer


# Utility functions to replace sklearn functionality
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Simple train/test split replacement"""
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]
    
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def mean_squared_error(y_true, y_pred):
    """Calculate MSE"""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Calculate MAE"""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Calculate R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def accuracy_score(y_true, y_pred):
    """Calculate accuracy"""
    return np.mean(y_true == y_pred)


class SimpleStandardScaler:
    """Simple StandardScaler replacement"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1  # Avoid division by zero
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class TelemetryService:
    """Service for telemetry data processing and analysis with River-based online learning capabilities"""
    
    def __init__(self):
        self.telemetry_features = TelemetryFeatures()
        self.algorithm_config = AlgorithmConfiguration()
        self.river_ml_service = RiverMLService()  # New River-based ML service
        
        # Model types supported for different prediction tasks
        self.model_types = {
            "lap_time_prediction": "regression",
            "sector_time_prediction": "regression", 
            "performance_classification": "classification", 
            "setup_recommendation": "classification",
            "tire_strategy": "classification",
            "fuel_consumption": "regression",
            "brake_performance": "regression",
            "overtaking_opportunity": "classification",
            "racing_line_optimization": "regression",
            "weather_adaptation": "regression",
            "consistency_analysis": "regression",
            "damage_prediction": "classification"
        }
    
    async def train_ai_model(self, 
                           telemetry_data: List[Dict[str, Any]], 
                           target_variable: str,
                           model_type: str = "lap_time_prediction",
                           preferred_algorithm: Optional[str] = None,
                           existing_model_data_from_db: List[Dict[str, Any]] = None,
                           user_id: Optional[str] = None,) -> Dict[str, Any]:
        """
        Train AI model on telemetry data with support for online learning using River
        
        Args:
            telemetry_data: List of telemetry data dictionaries
            target_variable: The variable to predict (e.g., 'lap_time', 'sector_time')
            model_type: Type of model to train
            preferred_algorithm: Override the default algorithm for this task
            existing_model_data_from_db:  data about existing model for incremental training
            user_id: User identifier for tracking
            use_river: Whether to use River for online learning (default: True)
         
        Returns:
            Dict containing trained model data and metrics for backend storage
        """
        try:

            # Use the new River-based online learning service
            return await self.river_ml_service.train_online_model(
                telemetry_data=telemetry_data,
                target_variable=target_variable,
                ai_model_type=model_type,
                preferred_algorithm=preferred_algorithm,
                existing_model_data_for_db=existing_model_data_from_db,
                user_id=user_id
            )
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}",
                "model_type": model_type,
                "algorithm_used": preferred_algorithm or "unknown"
            }
    
    async def predict_with_model(self, 
                               telemetry_data: Dict[str, Any],
                               model_data: str,
                               model_type: str,
                               use_river: bool = True) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Args:
            telemetry_data: Current telemetry data for prediction
            model_data: Base64 encoded model data
            model_type: Type of model
            use_river: Whether to use River model (default: True)
        
        Returns:
            Prediction results
        """
        try:
            if use_river:
                # Use River-based prediction
                return await self.river_ml_service.predict_online(
                    telemetry_data=telemetry_data,
                    model_data=model_data,
                    model_type=model_type
                )
            else:
                # Legacy scikit-learn prediction (deprecated)
                return await self._predict_legacy_model(
                    telemetry_data=telemetry_data,
                    model_data=model_data,
                    model_type=model_type
                )
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "model_type": model_type
            }
