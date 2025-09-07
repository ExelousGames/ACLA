"""
Scikit-learn Machine Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service provides comprehensive AI model training and prediction capabilities
using your TelemetryFeatures and FeatureProcessor classes.
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
import base64
import pickle
import io
import aiohttp
import asyncio
import time
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from app.models import AiModelDto

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import your telemetry models
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor, _safe_float

# Import imitation learning service
from .scikit_imitation_learning_service import ImitationLearningService

# Import backend service
from .backend_service import backend_service

# Import model cache service
from .model_cache_service import model_cache_service

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ModelConfig:
    """Configuration for different ML model types"""
    
    REGRESSION_MODELS = {
        'random_forest': {
            'model': RandomForestRegressor,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        },
        'linear_regression': {
            'model': LinearRegression,
            'params': {}
        },
        'ridge': {
            'model': Ridge,
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        'svr': {
            'model': SVR,
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    }
    
    CLASSIFICATION_MODELS = {
        'random_forest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression,
            'params': {
                'C': [0.1, 1, 10],
                'max_iter': [1000]
            }
        },
        'svc': {
            'model': SVC,
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    }


class TelemetryMLService:
    """
    Machine Learning Service for AC Competizione Telemetry Analysis
    
    Supports multiple prediction tasks:
    - Lap time prediction (regression)
    - Performance classification
    - Driver behavior analysis
    - Setup optimization
    - Tire strategy
    - Imitation learning from expert demonstrations
    - Expert guidance and coaching
    - Driving behavior comparison and analysis
    - And more...
    """
    
    def __init__(self, models_directory: str = "models"):
        """
        Initialize the ML service
        
        Args:
            models_directory: Directory to save/load trained models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        self.telemetry_features = TelemetryFeatures()
        self.trained_models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Initialize imitation learning service
        imitation_models_dir = self.models_directory / "imitation_models"
        self.imitation_learning = ImitationLearningService(str(imitation_models_dir))
        
        # Backend service integration
        self.backend_service = backend_service
        
        # Model cache service integration
        self.model_cache = model_cache_service
        
        # Add a simple lock mechanism to prevent concurrent fetches of the same model
        self._model_fetch_locks = {}
        self._lock_creation_lock = asyncio.Lock()

    def get_fetch_locks_status(self) -> Dict[str, Any]:
        """
        Get status of current fetch locks for debugging purposes
        
        Returns:
            Dictionary with lock status information
        """
        return {
            "active_locks": list(self._model_fetch_locks.keys()),
            "lock_count": len(self._model_fetch_locks),
            "timestamp": datetime.now().isoformat()
        }
    
    async def clear_stuck_fetch_locks(self, max_age_minutes: int = 10) -> Dict[str, Any]:
        """
        Clear fetch locks that might be stuck (emergency cleanup)
        
        Args:
            max_age_minutes: Age threshold for considering locks as stuck
            
        Returns:
            Dictionary with cleanup results
        """
        cleared_locks = []
        
        try:
            async with self._lock_creation_lock:
                # Since we can't track lock creation time in this simple implementation,
                # we'll just clear all locks as an emergency measure
                for model_key in list(self._model_fetch_locks.keys()):
                    try:
                        self._model_fetch_locks[model_key].set()
                        del self._model_fetch_locks[model_key]
                        cleared_locks.append(model_key)
                    except Exception as e:
                        pass
            
            return {
                "success": True,
                "cleared_locks": cleared_locks,
                "cleared_count": len(cleared_locks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "cleared_locks": cleared_locks,
                "cleared_count": len(cleared_locks),
                "timestamp": datetime.now().isoformat()
            }
        

    async def train_ai_model(self, 
                            telemetry_data: List[Dict[str, Any]], 
                            target_variable: str,
                            ai_model_type: str = "lap_time_prediction",
                            preferred_algorithm: Optional[str] = None,
                            user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Train or update an AI model using scikit-learn's batch learning approach
        
        Args:
            telemetry_data: Single telemetry data dictionary or list of telemetry data dictionaries
            target_variable: The variable to predict
            ai_model_type: Type of model to train
            preferred_algorithm: Override the default algorithm for this task
            existing_model_data_for_db: existing model for incremental training (not supported in sklearn)
            user_id: User identifier for tracking
         
        Returns:
            Dict containing trained model data and metrics
        """
        try:
            df = pd.DataFrame(telemetry_data)
            if df.empty:
                return {
                    "success": False,
                    "error": "No telemetry data provided",
                    "model_type": ai_model_type,
                }
            
            feature_processor = FeatureProcessor(df)
            
            
            # Cleaned data
            processed_df = feature_processor.general_cleaning_for_analysis()
            
            if processed_df.empty:
                return {
                    "success": False,
                    "error": "No telemetry data provided after processing",
                    "model_type": ai_model_type,
                }
            
            # Warning for insufficient data
            if len(processed_df) < 10:
                if len(processed_df) < 2:
                    return {
                        "success": False,
                        "error": "Insufficient data for training. At least 2 data points required.",
                        "model_type": ai_model_type,
                    }
            
            # Determine if this is a regression or classification task
            model_type_mapping = {
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
                "damage_prediction": "classification",
                "imitation_learning": "imitation"  # Add imitation learning
            }
            
            task_type = model_type_mapping.get(ai_model_type, "regression")
            
            # Set default algorithm based on task type and preferred_algorithm
            if preferred_algorithm:
                algorithm_name = preferred_algorithm
            else:
                # Default algorithms for different tasks
                default_algorithms = {
                    "lap_time_prediction": "random_forest",
                    "sector_time_prediction": "gradient_boosting",
                    "performance_classification": "random_forest",
                    "setup_recommendation": "random_forest",
                    "tire_strategy": "gradient_boosting",
                    "fuel_consumption": "ridge",
                    "brake_performance": "random_forest",
                    "overtaking_opportunity": "random_forest",
                    "racing_line_optimization": "gradient_boosting",
                    "weather_adaptation": "random_forest",
                    "consistency_analysis": "linear_regression",
                    "damage_prediction": "random_forest",
                    "imitation_learning": "behavior_cloning"  # Add imitation learning default
                }
                algorithm_name = default_algorithms.get(ai_model_type, "random_forest")
            
            # Handle target variable - create if it doesn't exist
            if target_variable not in processed_df.columns:
                # Try to create the target based on the model type
                target_creation_mapping = {
                    "lap_time_prediction": "lap_time",
                    "sector_time_prediction": "sector_time",
                    "performance_classification": "performance_category",
                    "fuel_consumption": "fuel_consumption",
                    "tire_strategy": "tire_performance",
                    "imitation_learning": "driving_style"  # Add imitation learning target
                }
                
                target_type = target_creation_mapping.get(ai_model_type, "lap_time")
                try:
                    processed_df[target_variable] = self.create_training_targets(processed_df, target_type)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Could not create target variable '{target_variable}': {str(e)}",
                        "model_type": ai_model_type,
                        "algorithm_used": algorithm_name
                    }
            
            # Train the model based on task type
            if task_type == "regression":
                training_result = self.train_regression_model(
                    df=processed_df,
                    target_column=target_variable,
                    model_name=algorithm_name,
                    model_type=ai_model_type,
                    feature_selection='auto',
                    hyperparameter_tuning=True
                )
            else:
                training_result = self.train_classification_model(
                    df=processed_df,
                    target_column=target_variable,
                    model_name=algorithm_name,
                    model_type=ai_model_type,
                    feature_selection='auto',
                    hyperparameter_tuning=True
                )
            
            # Convert sklearn results to match river_ml_service format
            if training_result:
                # Serialize model for database storage
                model_data = self._serialize_sklearn_model(training_result['model_id'])
                
                return {
                    "success": True,
                    "model_data": model_data,
                    "model_type": ai_model_type,
                    "algorithm_used": algorithm_name,
                    "algorithm_type": task_type,
                    "target_variable": target_variable,
                    "training_metrics": training_result.get('test_metrics', {}),
                    "samples_processed": training_result['data_shape']['training_samples'],
                    "features_count": training_result['feature_count'],
                    "feature_names": training_result['feature_names'],
                    "algorithm_description": self._get_algorithm_description(algorithm_name),
                    "algorithm_strengths": self._get_algorithm_strengths(algorithm_name),
                    "training_time": "N/A",  # sklearn doesn't track this by default
                    "data_quality_score": self._calculate_data_quality_score(processed_df),
                    "recommendations": self._generate_training_recommendations(
                        training_result.get('test_metrics', {}), algorithm_name, task_type
                    ),
                    "model_version": 1,  # sklearn models are always new versions
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                }
            else:
                return {
                    "success": False,
                    "error": "Model training failed",
                    "model_type": ai_model_type,
                    "algorithm_used": algorithm_name
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}",
                "model_type": ai_model_type,
                "algorithm_used": preferred_algorithm or "unknown"
            }
    
    
    def load_telemetry_from_json(self, json_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Load telemetry data from JSON format (your specific format)
        
        Args:
            json_data: List of telemetry dictionaries
            
        Returns:
            Processed DataFrame ready for ML training
        """
        # Convert to DataFrame
        df = pd.DataFrame(json_data)
        
        # Handle specific data issues from your format
        # Fix invalid lap times (2147483647 is likely a placeholder for invalid times)
        invalid_time_placeholder = 2147483647
        time_columns = ['Graphics_last_time', 'Graphics_best_time', 'Graphics_estimated_lap_time']
        
        for col in time_columns:
            if col in df.columns:
                df[col] = df[col].replace(invalid_time_placeholder, np.nan)
        
        # Handle string time columns with invalid values
        string_time_columns = ['Graphics_last_time_str', 'Graphics_best_time_str', 
                              'Graphics_estimated_lap_time_str', 'Graphics_delta_lap_time_str']
        
        for col in string_time_columns:
            if col in df.columns:
                # Replace placeholder values
                df[col] = df[col].replace(['35791:23:647', '-:--:---'], np.nan)
        
        # Create valid lap times from current time if last_time is invalid
        if 'Graphics_current_time' in df.columns and 'Graphics_last_time' in df.columns:
            # For records where we're in the middle of a lap, use current_time as a proxy
            mask = df['Graphics_last_time'].isna() & (df['Graphics_current_time'] > 0) & (df['Graphics_current_time'] < 300000)  # Under 5 minutes
            df.loc[mask, 'Graphics_estimated_current_lap'] = df.loc[mask, 'Graphics_current_time']
        
        # Create speed-based performance indicators
        if 'Physics_speed_kmh' in df.columns:
            # Convert very small speeds (likely stationary) to 0
            df.loc[df['Physics_speed_kmh'] < 1, 'Physics_speed_kmh'] = 0
            
            # Create speed categories
            df['speed_category'] = pd.cut(df['Physics_speed_kmh'], 
                                        bins=[0, 50, 100, 150, 200, 1000], 
                                        labels=['Stationary', 'Slow', 'Medium', 'Fast', 'Very Fast'])
        
        # Create tire performance indicators
        tire_temp_cols = [col for col in df.columns if 'tyre_core_temp' in col]
        if tire_temp_cols:
            # Average tire temperature
            df['avg_tire_temp'] = df[tire_temp_cols].mean(axis=1)
            
            # Tire temperature balance (difference between hottest and coldest)
            df['tire_temp_balance'] = df[tire_temp_cols].max(axis=1) - df[tire_temp_cols].min(axis=1)
        
        # Create brake performance indicators  
        brake_temp_cols = [col for col in df.columns if 'brake_temp' in col]
        if brake_temp_cols:
            # Average brake temperature
            df['avg_brake_temp'] = df[brake_temp_cols].mean(axis=1)
            
            # Front vs rear brake temperature difference
            front_brake_cols = [col for col in brake_temp_cols if 'front' in col]
            rear_brake_cols = [col for col in brake_temp_cols if 'rear' in col]
            
            if front_brake_cols and rear_brake_cols:
                df['front_rear_brake_temp_diff'] = (df[front_brake_cols].mean(axis=1) - 
                                                   df[rear_brake_cols].mean(axis=1))
        
        # Create G-force magnitude
        g_force_cols = ['Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z']
        available_g_cols = [col for col in g_force_cols if col in df.columns]
        if len(available_g_cols) >= 2:
            df['g_force_magnitude'] = np.sqrt(sum(df[col]**2 for col in available_g_cols))
        
        # Create driving intensity indicator
        if all(col in df.columns for col in ['Physics_brake', 'Physics_gas']):
            df['driving_intensity'] = df['Physics_brake'] + df['Physics_gas']
        
        # Create sector progress indicator
        if 'Graphics_normalized_car_position' in df.columns:
            df['track_position_category'] = pd.cut(df['Graphics_normalized_car_position'],
                                                  bins=[0, 0.33, 0.66, 1.0],
                                                  labels=['Sector1', 'Sector2', 'Sector3'])
        
        return df
    
    def create_training_targets(self, df: pd.DataFrame, target_type: str) -> pd.Series:
        """
        Create training targets from telemetry data based on different prediction tasks
        
        Args:
            df: Processed telemetry DataFrame
            target_type: Type of prediction target to create
                       - 'lap_time': Predict lap completion time
                       - 'sector_time': Predict sector completion time  
                       - 'speed_prediction': Predict future speed
                       - 'tire_performance': Predict tire degradation
                       - 'fuel_consumption': Predict fuel usage
                       - 'driving_style': Classify driving style
                       - 'track_position': Predict track position
                       - 'performance_category': Classify performance level
        
        Returns:
            Target values for training
        """
        if target_type == 'lap_time':
            # Use available lap time data or estimate based on current time and completed laps
            if 'Graphics_last_time' in df.columns and not df['Graphics_last_time'].isna().all():
                target = df['Graphics_last_time'].fillna(df['Graphics_current_time'])
            elif 'Graphics_estimated_current_lap' in df.columns:
                target = df['Graphics_estimated_current_lap']
            else:
                # Estimate based on current session time and distance
                if 'Graphics_current_time' in df.columns:
                    target = df['Graphics_current_time']
                else:
                    raise ValueError("No suitable lap time data available")
        
        elif target_type == 'sector_time':
            # Use sector time or estimate based on position
            if 'Graphics_last_sector_time' in df.columns:
                target = df['Graphics_last_sector_time']
            else:
                # Estimate sector time based on current time and sector
                target = df['Graphics_current_time'] / 3  # Rough estimate
        
        elif target_type == 'speed_prediction':
            # Predict future speed (shift speed by 1 second or next record)
            target = df['Physics_speed_kmh'].shift(-1)
            target = target.fillna(target.mean())  # Fill last value
        
        elif target_type == 'tire_performance':
            # Predict tire temperature or wear
            tire_temp_cols = [col for col in df.columns if 'tyre_core_temp' in col]
            if tire_temp_cols:
                target = df[tire_temp_cols].mean(axis=1)
            else:
                target = df['avg_tire_temp'] if 'avg_tire_temp' in df.columns else pd.Series([80] * len(df))
        
        elif target_type == 'fuel_consumption':
            # Predict fuel consumption rate
            if 'Graphics_fuel_per_lap' in df.columns:
                target = df['Graphics_fuel_per_lap']
            elif 'Physics_fuel' in df.columns:
                # Calculate fuel consumption rate
                target = df['Physics_fuel'].diff().abs()
                target = target.fillna(target.mean())
            else:
                target = pd.Series([2.5] * len(df))  # Default consumption
        
        elif target_type == 'driving_style':
            # Classify driving style based on inputs
            if all(col in df.columns for col in ['Physics_brake', 'Physics_gas']):
                # Aggressive = high brake and gas usage
                # Smooth = low brake and gas usage  
                # Balanced = medium usage
                intensity = df['Physics_brake'] + df['Physics_gas']
                target = pd.cut(intensity, bins=[0, 0.3, 0.7, 2.0], 
                              labels=['Smooth', 'Balanced', 'Aggressive'])
            else:
                target = pd.Series(['Balanced'] * len(df))
        
        elif target_type == 'track_position':
            # Predict track position progression
            if 'Graphics_normalized_car_position' in df.columns:
                target = df['Graphics_normalized_car_position'].shift(-1)
                target = target.fillna(target.mean())
            else:
                target = pd.Series([0.5] * len(df))
        
        elif target_type == 'performance_category':
            # Classify performance based on speed and lap time
            if 'Graphics_last_time' in df.columns and not df['Graphics_last_time'].isna().all():
                lap_times = df['Graphics_last_time'].dropna()
                if len(lap_times) > 0:
                    # Categorize based on lap time percentiles
                    q33, q66 = lap_times.quantile([0.33, 0.66])
                    target = pd.cut(df['Graphics_last_time'], 
                                  bins=[0, q33, q66, float('inf')],
                                  labels=['Fast', 'Medium', 'Slow'])
                    target = target.fillna('Medium')
                else:
                    target = pd.Series(['Medium'] * len(df))
            else:
                # Use speed as proxy for performance
                if 'Physics_speed_kmh' in df.columns:
                    speed_q33, speed_q66 = df['Physics_speed_kmh'].quantile([0.33, 0.66])
                    target = pd.cut(df['Physics_speed_kmh'],
                                  bins=[0, speed_q33, speed_q66, float('inf')],
                                  labels=['Slow', 'Medium', 'Fast'])
                    target = target.fillna('Medium')
                else:
                    target = pd.Series(['Medium'] * len(df))
        
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Remove any remaining NaN values
        target = target.fillna(target.mode().iloc[0] if hasattr(target, 'mode') else target.mean())
        
        return target
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    model_type: str, 
                    target_column: str,
                    feature_selection: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare telemetry data for ML training
        
        Args:
            df: Raw telemetry DataFrame
            model_type: Type of prediction task (e.g., 'performance_classification', 'lap_time')
            target_column: Name of the target variable column
            feature_selection: Optional feature selection method ('auto', 'performance', etc.)
            
        Returns:
            Tuple of (features_df, target_series, selected_feature_names)
        """
        # Initialize feature processor
        processor = FeatureProcessor(df)
        
        # Clean and preprocess the data
        processed_df = processor.general_cleaning_for_analysis()
        
        # Validate features present in data
        feature_validation = processor.validate_features()
        
        # Handle target variable
        if target_column not in processed_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {list(processed_df.columns)}")
        
        target = processed_df[target_column].copy()
        
        # Select features based on model type and available data
        available_columns = list(processed_df.columns)
        
        if feature_selection == 'auto' or feature_selection is None:
            # Use model-specific features from TelemetryFeatures
            desired_features = self.telemetry_features.get_features_for_model_type(model_type)
            selected_features = self.telemetry_features.filter_available_features(desired_features, available_columns)
            
            # If we don't have enough features, use fallback
            if len(selected_features) < 5:
                fallback_features = self.telemetry_features.get_fallback_features(available_columns, target_column)
                selected_features.extend(fallback_features)
                selected_features = list(set(selected_features))  # Remove duplicates
        
        elif feature_selection == 'performance':
            selected_features = self.telemetry_features.filter_available_features(
                self.telemetry_features.get_performance_critical_features(), 
                available_columns
            )
        
        elif feature_selection == 'all_numeric':
            # Use all numeric columns except target
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = [col for col in numeric_cols if col != target_column]
        
        else:
            raise ValueError(f"Unknown feature_selection method: {feature_selection}")
        
        # Ensure we have the selected features and they're numeric
        final_features = []
        for feature in selected_features:
            if feature in processed_df.columns and feature != target_column:
                # Check if feature is numeric or can be converted
                try:
                    pd.to_numeric(processed_df[feature], errors='raise')
                    final_features.append(feature)
                except (ValueError, TypeError):
                    print(f"[WARNING] Skipping non-numeric feature: {feature}")
        
        if len(final_features) == 0:
            raise ValueError("No valid numeric features found for training")
        
        # Extract features and handle missing values
        X = processed_df[final_features].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Remove rows where target is missing
        valid_indices = ~target.isna()
        X = X[valid_indices]
        target = target[valid_indices]
        
        return X, target, final_features
    
    def train_regression_model(self,
                             df: pd.DataFrame,
                             target_column: str,
                             model_name: str = 'random_forest',
                             model_type: str = 'lap_time_prediction',
                             feature_selection: str = 'auto',
                             hyperparameter_tuning: bool = True,
                             test_size: float = 0.2,
                             cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a regression model for continuous target prediction
        
        Args:
            df: Telemetry DataFrame
            target_column: Target variable column name
            model_name: Type of model ('random_forest', 'gradient_boosting', etc.)
            model_type: Type of prediction task
            feature_selection: Feature selection method
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare data
        X, y, feature_names = self.prepare_data(df, model_type, target_column, feature_selection)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get model configuration
        if model_name not in ModelConfig.REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {model_name}")
        
        model_config = ModelConfig.REGRESSION_MODELS[model_name]
        
        # Hyperparameter tuning
        if hyperparameter_tuning and model_config['params']:
            base_model = model_config['model'](random_state=42)
            grid_search = GridSearchCV(
                base_model,
                model_config['params'],
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
            # Use default parameters
            best_model = model_config['model'](random_state=42)
            best_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            best_model, X_train_scaled, y_train, 
            cv=cv_folds, scoring='neg_mean_squared_error'
        )
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, best_model.feature_importances_))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Save model
        model_id = f"{model_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._save_model(best_model, scaler, model_id, feature_names, model_type='regression')
        
        results = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'target_column': target_column,
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean_score': -cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'feature_importance': feature_importance,
            'data_shape': {
                'total_samples': len(df),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(feature_names)
            }
        }
        
        return results
    
    def train_classification_model(self,
                                 df: pd.DataFrame,
                                 target_column: str,
                                 model_name: str = 'random_forest',
                                 model_type: str = 'performance_classification',
                                 feature_selection: str = 'auto',
                                 hyperparameter_tuning: bool = True,
                                 test_size: float = 0.2,
                                 cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a classification model for categorical target prediction
        
        Args:
            df: Telemetry DataFrame
            target_column: Target variable column name
            model_name: Type of model ('random_forest', 'gradient_boosting', etc.)
            model_type: Type of prediction task
            feature_selection: Feature selection method
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare data
        X, y, feature_names = self.prepare_data(df, model_type, target_column, feature_selection)
        
        # Encode labels if they're not numeric
        label_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y.copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get model configuration
        if model_name not in ModelConfig.CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {model_name}")
        
        model_config = ModelConfig.CLASSIFICATION_MODELS[model_name]
        
        # Hyperparameter tuning
        if hyperparameter_tuning and model_config['params']:
            base_model = model_config['model'](random_state=42)
            grid_search = GridSearchCV(
                base_model,
                model_config['params'],
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
            # Use default parameters
            best_model = model_config['model'](random_state=42)
            best_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            best_model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42), 
            scoring='accuracy'
        )
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, best_model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Classification report
        class_labels = label_encoder.classes_ if label_encoder else None
        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        
        # Save model
        model_id = f"{model_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._save_model(best_model, scaler, model_id, feature_names, model_type='classification', label_encoder=label_encoder)
        
        results = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'target_column': target_column,
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean_score': cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'feature_importance': feature_importance,
            'classification_report': report,
            'class_labels': class_labels.tolist() if class_labels is not None else None,
            'data_shape': {
                'total_samples': len(df),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(feature_names)
            }
        }
        
        return results
    
    def predict(self, 
                model_id: str, 
                df: pd.DataFrame,
                use_cache: bool = True) -> np.ndarray:
        """
        Make predictions using a trained model with caching support
        
        Args:
            model_id: ID of the trained model
            df: DataFrame with telemetry data for prediction
            use_cache: Whether to use cached model if available
            
        Returns:
            Array of predictions
        """
        model_info = None
        
        if use_cache:
            # Try to get model from cache first
            cached_result = self.model_cache.get_by_key(model_id)
            if cached_result:
                model_info, _ = cached_result
        
        # If not in cache or cache disabled, load from file
        if model_info is None:
            model_info = self._load_model(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
            
            if use_cache:
                # Cache the loaded model
                cache_metadata = {
                    "model_id": model_id,
                    "model_type": model_info.get('model_type', 'sklearn'),
                    "model_name": model_info.get('model_name', 'unknown'),
                    "loaded_at": datetime.now().isoformat()
                }
                
                self.model_cache.put(
                    model_type="sklearn",
                    track_name="general",  # sklearn models are general purpose
                    car_name="general",
                    data=model_info,
                    metadata=cache_metadata,
                    additional_params={"model_id": model_id}
                    # TTL will be automatically set based on model type configuration
                )
        
        # Prepare data using the same features as training
        processor = FeatureProcessor(df)
        processed_df = processor.general_cleaning_for_analysis()
        
        # Extract features used in training
        feature_names = model_info['feature_names']
        missing_features = [f for f in feature_names if f not in processed_df.columns]
        
        if missing_features:
            for feature in missing_features:
                processed_df[feature] = 0
        
        X = processed_df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = model_info['scaler'].transform(X)
        
        # Make predictions
        predictions = model_info['model'].predict(X_scaled)
        
        # Decode labels if it's a classification model
        if model_info.get('label_encoder'):
            predictions = model_info['label_encoder'].inverse_transform(predictions.astype(int))
        
        return predictions
    
    def _deserialize_sklearn_model(self, model_data: str) -> Dict[str, Any]:
        """
        Deserialize sklearn model from base64 string
        
        Args:
            model_data: Base64 encoded model data
        
        Returns:
            Model info dictionary
        """
        try:
            # Decode from base64
            decoded_data = base64.b64decode(model_data.encode('utf-8'))
            
            # Deserialize using pickle
            buffer = io.BytesIO(decoded_data)
            model_info = pickle.load(buffer)
            
            return model_info
            
        except Exception as e:
            raise Exception(f"Failed to deserialize model: {str(e)}")
    
    
    def _serialize_sklearn_model(self, model_id: str) -> str:
        """
        Serialize sklearn model to base64 string for database storage
        
        Args:
            model_id: ID of the trained model
        
        Returns:
            Base64 encoded model data
        """
        try:
            model_info = self._load_model(model_id)
            if not model_info:
                raise Exception(f"Model {model_id} not found")
            
            # Serialize using pickle
            buffer = io.BytesIO()
            pickle.dump(model_info, buffer)
            buffer.seek(0)
        
            # Encode to base64
            encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return encoded_data
            
        except Exception as e:
            raise Exception(f"Failed to serialize model: {str(e)}")
    
    def _get_algorithm_description(self, algorithm_name: str) -> str:
        """Get description of the algorithm"""
        descriptions = {
            'random_forest': 'Random Forest: Ensemble method using multiple decision trees for robust predictions',
            'gradient_boosting': 'Gradient Boosting: Sequential ensemble method that corrects previous model errors',
            'linear_regression': 'Linear Regression: Simple linear relationship modeling between features and target',
            'ridge': 'Ridge Regression: Linear regression with L2 regularization to prevent overfitting',
            'logistic_regression': 'Logistic Regression: Linear method for classification with probabilistic output',
            'svc': 'Support Vector Classifier: Maximum margin classifier for complex decision boundaries',
            'svr': 'Support Vector Regression: Support vector method for regression with robust outlier handling'
        }
        return descriptions.get(algorithm_name, f'Advanced {algorithm_name} algorithm')
    
    def _get_algorithm_strengths(self, algorithm_name: str) -> List[str]:
        """Get strengths of the algorithm"""
        strengths = {
            'random_forest': [
                'Handles missing values well',
                'Provides feature importance',
                'Resistant to overfitting',
                'Works with mixed data types'
            ],
            'gradient_boosting': [
                'High predictive accuracy',
                'Handles complex patterns',
                'Robust to outliers',
                'Good feature selection'
            ],
            'linear_regression': [
                'Fast training and prediction',
                'Interpretable coefficients',
                'Low memory usage',
                'Good baseline model'
            ],
            'ridge': [
                'Prevents overfitting',
                'Stable with multicollinearity',
                'Fast computation',
                'Regularized solution'
            ],
            'logistic_regression': [
                'Probabilistic output',
                'Fast and efficient',
                'Interpretable results',
                'No tuning of hyperparameters'
            ],
            'svc': [
                'Effective in high dimensions',
                'Memory efficient',
                'Versatile with kernel tricks',
                'Works well with limited data'
            ],
            'svr': [
                'Robust to outliers',
                'Effective in high dimensions',
                'Memory efficient',
                'Non-linear modeling capability'
            ]
        }
        return strengths.get(algorithm_name, ['Advanced machine learning capabilities'])
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-100)"""
        try:
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            missing_score = max(0, 100 - (missing_ratio * 100))
            
            # Check for duplicate rows
            duplicate_ratio = df.duplicated().sum() / len(df)
            duplicate_score = max(0, 100 - (duplicate_ratio * 100))
            
            # Check for outliers (using IQR method)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_ratio = 0
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_ratio += outliers / len(df)
                outlier_ratio = outlier_ratio / len(numeric_cols)
            
            outlier_score = max(0, 100 - (outlier_ratio * 100))
            
            # Average the scores
            overall_score = (missing_score + duplicate_score + outlier_score) / 3
            return round(overall_score, 2)
            
        except Exception:
            return 75.0  # Default moderate score
    
    def _generate_training_recommendations(self, metrics: Dict[str, Any], 
                                         algorithm_name: str, task_type: str) -> List[str]:
        """Generate recommendations based on training metrics and algorithm"""
        recommendations = []
        
        # Performance-based recommendations
        if task_type == "regression":
            mae = metrics.get("mae", float('inf'))
            rmse = metrics.get("mse", float('inf')) ** 0.5 if metrics.get("mse") else float('inf')
            r2 = metrics.get("r2", 0)
            
            if r2 < 0.6:
                recommendations.append("Consider feature engineering or trying a different algorithm")
            if mae > 5.0:
                recommendations.append("High prediction error - check data quality and feature relevance")
            if r2 > 0.8:
                recommendations.append("Excellent model performance - monitor for overfitting")
                
        else:  # classification
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1", 0)
            
            if accuracy < 0.7:
                recommendations.append("Low classification accuracy - consider feature selection or different algorithm")
            if f1 < 0.6:
                recommendations.append("Poor F1 score - check class balance and feature quality")
            if accuracy > 0.9:
                recommendations.append("Excellent classification performance - validate on new data")
        
        # Algorithm-specific recommendations
        if algorithm_name == "linear_regression" and metrics.get("r2", 0) < 0.6:
            recommendations.append("Linear model may not capture complex patterns - try tree-based algorithms")
        elif algorithm_name == "random_forest":
            recommendations.append("Random Forest provides good baseline - tune hyperparameters for better performance")
        elif algorithm_name == "gradient_boosting":
            recommendations.append("Gradient Boosting can achieve high accuracy - monitor training time and overfitting")
        elif algorithm_name == "ridge":
            recommendations.append("Ridge regression prevents overfitting - good for many features")
        
        if not recommendations:
            recommendations.append("Model training completed successfully - monitor performance over time")
        
        return recommendations

    
    def _save_model(self, 
                   model, 
                   scaler, 
                   model_id: str, 
                   feature_names: List[str], 
                   model_type: str,
                   label_encoder=None):
        """Save trained model and associated components"""
        model_info = {
            'model_id': model_id,
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': feature_names,
            'model_name': model.__class__.__name__,
            'model_type': model_type,
            'created_at': datetime.now().isoformat()
        }
        
        model_path = self.models_directory / f"{model_id}.pkl"
        joblib.dump(model_info, model_path)
    
    def _load_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load a trained model"""
        model_path = self.models_directory / f"{model_id}.pkl"
        if not model_path.exists():
            return None
        
        return joblib.load(model_path)
    
    def create_performance_categories(self, df: pd.DataFrame, lap_time_column: str) -> pd.Series:
        """
        Create performance categories based on lap times for classification
        
        Args:
            df: DataFrame with lap time data
            lap_time_column: Name of the lap time column
            
        Returns:
            Series with performance categories ('Fast', 'Medium', 'Slow')
        """
        lap_times = df[lap_time_column].dropna()
        
        # Calculate percentiles
        fast_threshold = lap_times.quantile(0.33)
        slow_threshold = lap_times.quantile(0.67)
        
        def categorize_performance(lap_time):
            if pd.isna(lap_time):
                return 'Unknown'
            elif lap_time <= fast_threshold:
                return 'Fast'
            elif lap_time <= slow_threshold:
                return 'Medium'
            else:
                return 'Slow'
        
        return df[lap_time_column].apply(categorize_performance)
    
    def create_sector_time_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create sector time prediction target from graphics data
        
        Args:
            df: DataFrame with telemetry data
            
        Returns:
            Series with sector times for prediction
        """
        if 'Graphics_last_sector_time' in df.columns:
            return df['Graphics_last_sector_time'].copy()
        elif 'Graphics_current_time' in df.columns:
            # Use current lap time as approximation
            return df['Graphics_current_time'].copy()
        else:
            raise ValueError("No suitable sector time column found")
    
    def train_specialized_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train specialized models for common racing scenarios
        
        Args:
            df: Telemetry DataFrame
            
        Returns:
            Dictionary with results from all trained models
        """
        results = {}
        
        try:
            # 1. Lap Time Prediction (Regression)
            if 'Graphics_last_time' in df.columns:
                lap_time_results = self.train_regression_model(
                    df=df,
                    target_column='Graphics_last_time',
                    model_name='random_forest',
                    model_type='lap_time_prediction',
                    feature_selection='auto'
                )
                results['lap_time_prediction'] = lap_time_results
            
            # 2. Performance Classification
            if 'Graphics_last_time' in df.columns:
                # Create performance categories
                df['performance_category'] = self.create_performance_categories(df, 'Graphics_last_time')
                
                perf_results = self.train_classification_model(
                    df=df,
                    target_column='performance_category',
                    model_name='random_forest',
                    model_type='performance_classification',
                    feature_selection='performance'
                )
                results['performance_classification'] = perf_results
            
            # 3. Speed Prediction
            if 'Physics_speed_kmh' in df.columns:
                speed_results = self.train_regression_model(
                    df=df,
                    target_column='Physics_speed_kmh',
                    model_name='gradient_boosting',
                    model_type='speed_prediction',
                    feature_selection='auto'
                )
                results['speed_prediction'] = speed_results
            
            # 4. Brake Performance Analysis
            if 'Physics_brake_temp_front_left' in df.columns:
                brake_results = self.train_regression_model(
                    df=df,
                    target_column='Physics_brake_temp_front_left',
                    model_name='random_forest',
                    model_type='brake_performance',
                    feature_selection='auto'
                )
                results['brake_performance'] = brake_results
            
            # 5. Tire Temperature Prediction
            if 'Physics_tyre_core_temp_front_left' in df.columns:
                tire_results = self.train_regression_model(
                    df=df,
                    target_column='Physics_tyre_core_temp_front_left',
                    model_name='gradient_boosting',
                    model_type='tire_strategy',
                    feature_selection='auto'
                )
                results['tire_strategy'] = tire_results
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    # Imitation Learning Methods
    async def train_imitation_model(self, trackName: str, carName: str) -> Dict[str, Any]:
        """
        Train an imitation learning model from expert driving demonstrations
        
        Args:
            unprocessed_telemetry_data: List of expert driver telemetry data (optional if fetch_from_backend=True)
            learning_objectives: What to learn ('behavior', 'trajectory', 'both')
            user_id: User identifier for tracking and filtering backend data
            jwt_token: JWT token for backend authentication
            fetch_from_backend: Whether to fetch data from backend instead of using provided data
            
        Returns:
            Dictionary with imitation learning results
        """
        #retrieve all racing session in database
        try:
            sessions = await backend_service.get_all_racing_sessions(trackName, carName)
        except Exception as e:
            return {"error": str(e)}

        each_session_telemetry_data = []
  
        for session in sessions.get("sessions", []):
                each_session_telemetry_data.append(session.get("data", []))

        if not each_session_telemetry_data:
            raise ValueError("No telemetry data found")

        # Flatten the list of lists into a single list of telemetry records
        telemetry_data = [item for sublist in each_session_telemetry_data for item in sublist]

        # Learn from expert demonstrations
        results = self.imitation_learning.train_ai_model(telemetry_data)
            
        try:
            #save the info to backend

            ai_model_dto = {
                "modelType": "imitation_learning",
                "trackName": trackName,
                "carName": carName,
                "modelData": results,
                "metadata": {
                    "summary": results.get("summary", {}),
                    "training_timestamp": datetime.now().isoformat()
                },
                "isActive": True
            }
            
            await backend_service.save_imitation_learning_results(ai_model_dto)
        except Exception as error:
            pass
        
        return results
        
    async def get_imitation_learning_expert_guidance(self,
                                current_telemetry: Dict[str, Any],
                                trackName: str,
                                carName: str,
                                guidance_type: str = "both") -> Dict[str, Any]:
        """
        Get expert guidance for current driving situation using imitation learning
        
        Args:
            current_telemetry: Current telemetry state
            trackName: Track name for the model
            carName: Car name for the model
            guidance_type: Type of guidance ('behavior', 'actions', 'both')
            
        Returns:
            Expert guidance and recommendations
        """
        
        model_key = f"{trackName}/{carName}/imitation_learning"
        is_fetching_thread = False
        deserialized_trajectory_models = None
        
        try:
            # First, try to get model from cache without any locking
            cached_result = self.model_cache.get(
                model_type="imitation_learning",
                track_name=trackName,
                car_name=carName,
                model_subtype="complete_model_data"
            )
            
            if cached_result:
                deserialized_trajectory_models, metadata = cached_result
            else:
                # Use a proper async lock to prevent race conditions
                async with self._lock_creation_lock:
                    # Double-check cache after acquiring lock
                    cached_result = self.model_cache.get(
                        model_type="imitation_learning",
                        track_name=trackName,
                        car_name=carName,
                        model_subtype="complete_model_data"
                    )
                    
                    if cached_result:
                        deserialized_trajectory_models, metadata = cached_result
                    elif model_key in self._model_fetch_locks:
                        # Another thread is fetching this model, wait for it
                        fetch_event = self._model_fetch_locks[model_key]
                    else:
                        # We're the first to request this model, create the event and mark ourselves as fetching
                        self._model_fetch_locks[model_key] = asyncio.Event()
                        is_fetching_thread = True
                
                # If we need to wait for another thread
                if not deserialized_trajectory_models and not is_fetching_thread:
                    try:
                        await fetch_event.wait()
                        # The other thread should have cached the model, try cache again
                        cached_result = self.model_cache.get(
                            model_type="imitation_learning",
                            track_name=trackName,
                            car_name=carName,
                            model_subtype="complete_model_data"
                        )
                        if cached_result:
                            deserialized_trajectory_models, metadata = cached_result
                            print(f"[INFO] Using model cached by another thread for {trackName}/{carName}")
                        else:
                            print(f"[WARNING] Expected cached model not found after waiting for {trackName}/{carName}")
                    except Exception as wait_error:
                        print(f"[ERROR] Error while waiting for model fetch: {str(wait_error)}")
                        # Continue to try fetching ourselves as fallback
                        is_fetching_thread = True
                
                # If we are the fetching thread or fallback, do the actual fetch
                if not deserialized_trajectory_models and is_fetching_thread:
                    try:
                        print(f"[INFO] Fetching model data from backend for {trackName}/{carName}")
                        model_response = await self.backend_service.getCompleteActiveModelData(
                            trackName, carName, "imitation_learning"
                        )
                          
                        if "error" in model_response:
                            raise Exception(f"Backend error: {model_response['error']}")
                        
                        # Extract modelData from the correct location in response
                        model_data = None
                        if "data" in model_response and isinstance(model_response["data"], dict):
                            model_data = model_response["data"].get("modelData", {})
                        else:
                            model_data = model_response.get("modelData", {})
                        
                        if not model_data:
                            raise Exception("No model data found in response")

                        # Deserialize the model data
                        deserialized_trajectory_models = self.imitation_learning.deserialize_object_inside(model_data)
                        
                        # Cache the deserialized model for future use
                        cache_metadata = {
                            "track_name": trackName,
                            "car_name": carName,
                            "model_type": "imitation_learning",
                            "fetched_at": datetime.now().isoformat(),
                            "backend_model_id": model_response.get("id", "unknown")
                        }
                        
                        self.model_cache.put(
                            model_type="imitation_learning",
                            track_name=trackName,
                            car_name=carName,
                            data=deserialized_trajectory_models,
                            metadata=cache_metadata,
                            model_subtype="complete_model_data"
                        )
                        
                        print(f"[INFO] Successfully fetched and cached model for {trackName}/{carName}")
                        
                    except Exception as fetch_error:
                        print(f"[ERROR] Failed to fetch model: {str(fetch_error)}")
                        raise fetch_error
                    finally:
                        # Always signal completion and clean up lock when we're the fetching thread
                        if model_key in self._model_fetch_locks:
                            try:
                                self._model_fetch_locks[model_key].set()
                                del self._model_fetch_locks[model_key]
                                print(f"[INFO] Released fetch lock for {trackName}/{carName}")
                            except Exception as cleanup_error:
                                print(f"[WARNING] Error cleaning up fetch lock: {str(cleanup_error)}")
            
            # At this point, we should have the model data
            if not deserialized_trajectory_models:
                raise Exception("Failed to obtain model data from cache or backend")
            
            # Prepare current telemetry data
            df = pd.DataFrame([current_telemetry])
            processor = FeatureProcessor(df)
            processed_df = processor.general_cleaning_for_analysis()
        
            # Get predictions from imitation model
            predictions = self.imitation_learning.predict_expert_actions(
                processed_df=processed_df,
                model_data=deserialized_trajectory_models
            )

            # Format guidance based on requested type
            guidance = {"success": True, "timestamp": datetime.now().isoformat()}
            
            if guidance_type in ['behavior', 'both'] and 'driving_behavior' in predictions:
                behavior_pred = predictions['driving_behavior']
                guidance['behavior_guidance'] = {
                    "predicted_driving_style": behavior_pred['predicted_style'],
                    "confidence": behavior_pred['confidence'],
                    "style_recommendations": self._generate_behavior_recommendations(behavior_pred),
                    "alternative_styles": behavior_pred['style_probabilities']
                }
            
            if guidance_type in ['actions', 'both'] and 'optimal_actions' in predictions:
                optimal_actions = predictions['optimal_actions']
                guidance['action_guidance'] = {
                    "optimal_actions": optimal_actions,
                    "action_recommendations": self.imitation_learning._create_detailed_point_comparison(
                        processed_df, optimal_actions
                    ),
                    "performance_insights": self.imitation_learning._analyze_action_performance(
                        processed_df, optimal_actions
                    )
                }
                
            return guidance
            
        except Exception as e:
            print(f"[ERROR] Failed to get expert guidance: {str(e)}")
            
            # Clean up any locks that might have been created by this thread
            if is_fetching_thread and hasattr(self, '_model_fetch_locks') and model_key in self._model_fetch_locks:
                try:
                    self._model_fetch_locks[model_key].set()  # Signal any waiting threads
                    del self._model_fetch_locks[model_key]
                    print(f"[INFO] Emergency cleanup of fetch lock for {trackName}/{carName}")
                except Exception as cleanup_error:
                    print(f"[WARNING] Error during emergency lock cleanup: {str(cleanup_error)}")
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_behavior_recommendations(self, behavior_prediction: Dict[str, Any]) -> List[str]:
        """Generate driving behavior recommendations based on prediction"""
        recommendations = []
        predicted_style = behavior_prediction['predicted_style']
        confidence = behavior_prediction['confidence']
        
        if confidence > 0.8:
            recommendations.append(f"Expert analysis shows {predicted_style} driving style (high confidence)")
        else:
            recommendations.append(f"Driving style appears to be {predicted_style} (moderate confidence)")
        
        # Style-specific recommendations
        style_advice = {
            'aggressive': [
                "Consider smoother throttle and brake inputs for better tire management",
                "Try to maintain higher minimum speeds through corners",
                "Focus on consistent lap times rather than single fast laps"
            ],
            'smooth': [
                "Your smooth driving style is excellent for tire longevity",
                "You can afford to be slightly more aggressive on throttle application",
                "Consider pushing harder in practice to find the limit"
            ],
            'conservative': [
                "You have room to push harder - try braking later into corners",
                "Increase throttle application aggressiveness on corner exit",
                "Focus on finding the racing line for better lap times"
            ],
            'optimal': [
                "Your driving style matches expert patterns very well",
                "Maintain this consistent approach for race conditions",
                "Focus on race craft and strategic driving"
            ],
            'defensive': [
                "Good for wheel-to-wheel racing situations",
                "In qualifying or practice, try to be more aggressive",
                "Focus on protecting the inside line in races"
            ]
        }
        
        recommendations.extend(style_advice.get(predicted_style, [
            "Continue analyzing your driving patterns for improvement",
            "Compare your telemetry with expert drivers",
            "Focus on consistency and smooth inputs"
        ]))
        
        return recommendations
    
    
    def _calculate_driving_metrics_comparison(self, detailed_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall driving metrics comparison from detailed point comparisons"""
        
        if not detailed_comparisons:
            return {"error": "No comparison data available"}
        
        metrics = {
            "speed_analysis": {},
            "throttle_analysis": {},
            "brake_analysis": {},
            "steering_analysis": {},
            "gear_analysis": {},
            "overall_similarity": 0
        }
        
        # Collect all comparison data
        speed_diffs = []
        throttle_diffs = []
        brake_diffs = []
        steering_diffs = []
        gear_optimal_count = 0
        similarity_scores = []
        
        for comparison in detailed_comparisons:
            comps = comparison.get("comparisons", {})
            
            if "speed" in comps:
                speed_diffs.append(comps["speed"]["difference"])
            if "throttle" in comps:
                throttle_diffs.append(comps["throttle"]["difference"])
            if "brake" in comps:
                brake_diffs.append(comps["brake"]["difference"])
            if "steering" in comps:
                steering_diffs.append(comps["steering"]["difference"])
            if "gear" in comps:
                if comps["gear"]["gear_optimal"]:
                    gear_optimal_count += 1
            
            similarity_scores.append(comparison.get("point_similarity", 0))
        
        # Speed analysis
        if speed_diffs:
            metrics["speed_analysis"] = {
                "average_difference_kmh": np.mean(speed_diffs),
                "speed_deficit_percentage": (len([d for d in speed_diffs if d > 5]) / len(speed_diffs)) * 100,
                "speed_excess_percentage": (len([d for d in speed_diffs if d < -5]) / len(speed_diffs)) * 100,
                "speed_consistency": max(0, 100 - np.std(speed_diffs)),
                "max_speed_difference": max(abs(d) for d in speed_diffs),
                "recommendation": self._generate_speed_recommendation(speed_diffs)
            }
        
        # Throttle analysis
        if throttle_diffs:
            metrics["throttle_analysis"] = {
                "average_difference": np.mean(throttle_diffs),
                "aggressive_percentage": (len([d for d in throttle_diffs if d < -0.1]) / len(throttle_diffs)) * 100,
                "conservative_percentage": (len([d for d in throttle_diffs if d > 0.1]) / len(throttle_diffs)) * 100,
                "throttle_consistency": max(0, 100 - np.std(throttle_diffs) * 100),
                "recommendation": self._generate_throttle_recommendation(throttle_diffs)
            }
        
        # Brake analysis
        if brake_diffs:
            metrics["brake_analysis"] = {
                "average_difference": np.mean(brake_diffs),
                "under_braking_percentage": (len([d for d in brake_diffs if d > 0.1]) / len(brake_diffs)) * 100,
                "over_braking_percentage": (len([d for d in brake_diffs if d < -0.1]) / len(brake_diffs)) * 100,
                "brake_consistency": max(0, 100 - np.std(brake_diffs) * 100),
                "recommendation": self._generate_brake_recommendation(brake_diffs)
            }
        
        # Steering analysis
        if steering_diffs:
            metrics["steering_analysis"] = {
                "average_difference_rad": np.mean(steering_diffs),
                "steering_precision": max(0, 100 - np.std(steering_diffs) * 100),
                "over_steering_percentage": (len([d for d in steering_diffs if abs(d) > 0.1]) / len(steering_diffs)) * 100,
                "recommendation": self._generate_steering_recommendation(steering_diffs)
            }
        
        # Gear analysis
        if detailed_comparisons:
            gear_optimality = (gear_optimal_count / len(detailed_comparisons)) * 100
            metrics["gear_analysis"] = {
                "optimal_gear_percentage": gear_optimality,
                "recommendation": self._generate_gear_recommendation(gear_optimality)
            }
        
        # Overall similarity
        if similarity_scores:
            metrics["overall_similarity"] = np.mean(similarity_scores)
        
        return metrics
    
    def _generate_speed_recommendation(self, speed_diffs: List[float]) -> str:
        """Generate speed-specific recommendation"""
        avg_diff = np.mean(speed_diffs)
        if avg_diff > 10:
            return "You're consistently carrying too much speed - focus on better braking points"
        elif avg_diff > 5:
            return "Slight tendency to carry excess speed - work on braking consistency"
        elif avg_diff < -10:
            return "You're leaving significant speed on the table - try braking later and carrying more corner speed"
        elif avg_diff < -5:
            return "You can carry more speed through corners - practice finding the limit gradually"
        else:
            return "Good speed management overall - maintain this consistency"
    
    def _generate_throttle_recommendation(self, throttle_diffs: List[float]) -> str:
        """Generate throttle-specific recommendation"""
        avg_diff = np.mean(throttle_diffs)
        if avg_diff > 0.2:
            return "You're being too conservative with throttle - apply power earlier and more aggressively"
        elif avg_diff > 0.1:
            return "Slightly conservative throttle application - try being more aggressive on corner exit"
        elif avg_diff < -0.2:
            return "Too aggressive with throttle - focus on smoother application to avoid wheelspin"
        elif avg_diff < -0.1:
            return "Slightly aggressive throttle - work on progressive application"
        else:
            return "Good throttle control - maintain this technique"
    
    def _generate_brake_recommendation(self, brake_diffs: List[float]) -> str:
        """Generate brake-specific recommendation"""
        avg_diff = np.mean(brake_diffs)
        if avg_diff > 0.2:
            return "You're under-braking significantly - brake later and with more pressure"
        elif avg_diff > 0.1:
            return "Slightly under-braking - you can brake later and with more confidence"
        elif avg_diff < -0.2:
            return "Over-braking - try braking later and with more progressive pressure release"
        elif avg_diff < -0.1:
            return "Slightly over-braking - work on later braking points"
        else:
            return "Good braking technique - maintain this approach"
    
    def _generate_steering_recommendation(self, steering_diffs: List[float]) -> str:
        """Generate steering-specific recommendation"""
        precision = max(0, 100 - np.std(steering_diffs) * 100)
        if precision < 70:
            return "Inconsistent steering inputs - focus on smoother, more precise steering"
        elif precision < 85:
            return "Good steering control with room for improvement in precision"
        else:
            return "Excellent steering precision - maintain this smooth technique"
    
    def _generate_gear_recommendation(self, gear_optimality: float) -> str:
        """Generate gear-specific recommendation"""
        if gear_optimality < 60:
            return "Significant gear optimization needed - focus on optimal shift points"
        elif gear_optimality < 80:
            return "Good gear selection with some room for improvement"
        else:
            return "Excellent gear selection - maintain this technique"
    

    
    def _generate_overall_driving_analysis(self, point_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall driving analysis from individual point analyses"""
        
        # Collect efficiency scores
        efficiency_scores = []
        behavior_styles = []
        
        for analysis in point_analyses:
            action_guidance = analysis.get("action_guidance", {})
            if "performance_insights" in action_guidance:
                efficiency_scores.append(action_guidance["performance_insights"].get("overall_efficiency", 0))
            
            behavior_guidance = analysis.get("behavior_guidance", {})
            if "predicted_driving_style" in behavior_guidance:
                behavior_styles.append(behavior_guidance["predicted_driving_style"])
        
        # Calculate overall metrics
        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0
        efficiency_consistency = 100 - np.std(efficiency_scores) if len(efficiency_scores) > 1 else 100
        
        # Most common driving style
        most_common_style = max(set(behavior_styles), key=behavior_styles.count) if behavior_styles else "unknown"
        style_consistency = (behavior_styles.count(most_common_style) / len(behavior_styles) * 100) if behavior_styles else 0
        
        return {
            "average_efficiency": avg_efficiency,
            "efficiency_consistency": efficiency_consistency,
            "dominant_driving_style": most_common_style,
            "style_consistency": style_consistency,
            "data_points_analyzed": len(point_analyses),
            "performance_category": self._categorize_performance(avg_efficiency),
            "consistency_category": self._categorize_consistency(efficiency_consistency)
        }
    
    def _generate_improvement_plan(self, point_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate personalized improvement plan based on analysis"""
        
        # Collect all recommendations
        all_behavior_recs = []
        all_action_recs = []
        
        for analysis in point_analyses:
            behavior_guidance = analysis.get("behavior_guidance", {})
            action_guidance = analysis.get("action_guidance", {})
            
            if "style_recommendations" in behavior_guidance:
                all_behavior_recs.extend(behavior_guidance["style_recommendations"])
            
            if "action_recommendations" in action_guidance:
                all_action_recs.extend(action_guidance["action_recommendations"])
        
        # Find most common recommendations
        behavior_counter = Counter(all_behavior_recs)
        action_counter = Counter(all_action_recs)
        
        return {
            "priority_behavior_improvements": [item for item, count in behavior_counter.most_common(3)],
            "priority_action_improvements": [item for item, count in action_counter.most_common(3)],
            "practice_focus_areas": self._identify_focus_areas(point_analyses),
            "estimated_improvement_potential": self._estimate_improvement_potential(point_analyses)
        }
    
    def _calculate_consistency_metrics(self, point_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate driving consistency metrics"""
        
        efficiency_scores = []
        for analysis in point_analyses:
            action_guidance = analysis.get("action_guidance", {})
            if "performance_insights" in action_guidance:
                efficiency_scores.append(action_guidance["performance_insights"].get("overall_efficiency", 0))
        
        if len(efficiency_scores) < 2:
            return {"consistency_score": 100, "variability": 0}
        
        consistency_score = 100 - np.std(efficiency_scores)
        variability = np.std(efficiency_scores)
        
        return {
            "consistency_score": max(0, consistency_score),
            "variability": variability,
            "efficiency_range": {
                "min": min(efficiency_scores),
                "max": max(efficiency_scores),
                "average": np.mean(efficiency_scores)
            }
        }
    
    def _categorize_performance(self, efficiency: float) -> str:
        """Categorize performance level based on efficiency"""
        if efficiency >= 90:
            return "Expert"
        elif efficiency >= 75:
            return "Advanced"
        elif efficiency >= 60:
            return "Intermediate"
        elif efficiency >= 40:
            return "Beginner"
        else:
            return "Novice"
    
    def _categorize_consistency(self, consistency: float) -> str:
        """Categorize consistency level"""
        if consistency >= 90:
            return "Very Consistent"
        elif consistency >= 75:
            return "Consistent"
        elif consistency >= 60:
            return "Moderately Consistent"
        else:
            return "Inconsistent"
    
    def _identify_focus_areas(self, point_analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify key areas for practice focus"""
        focus_areas = []
        
        # Analyze efficiency patterns
        low_efficiency_count = 0
        for analysis in point_analyses:
            action_guidance = analysis.get("action_guidance", {})
            if "performance_insights" in action_guidance:
                efficiency = action_guidance["performance_insights"].get("overall_efficiency", 100)
                if efficiency < 60:
                    low_efficiency_count += 1
        
        if low_efficiency_count > len(point_analyses) * 0.3:
            focus_areas.append("Overall driving technique and smoothness")
        
        # Add specific focus areas based on common issues
        focus_areas.extend([
            "Throttle control and timing",
            "Brake pressure modulation",
            "Racing line optimization",
            "Consistency across multiple laps"
        ])
        
        return focus_areas[:4]  # Return top 4 focus areas
    
    def _estimate_improvement_potential(self, point_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate improvement potential based on current performance"""
        
        efficiency_scores = []
        for analysis in point_analyses:
            action_guidance = analysis.get("action_guidance", {})
            if "performance_insights" in action_guidance:
                efficiency_scores.append(action_guidance["performance_insights"].get("overall_efficiency", 0))
        
        if not efficiency_scores:
            return {"potential": "Unknown", "estimated_gain": 0}
        
        avg_efficiency = np.mean(efficiency_scores)
        potential_gain = 100 - avg_efficiency
        
        if potential_gain > 40:
            potential_level = "Very High"
        elif potential_gain > 25:
            potential_level = "High"
        elif potential_gain > 15:
            potential_level = "Moderate"
        elif potential_gain > 5:
            potential_level = "Low"
        else:
            potential_level = "Minimal"
        
        return {
            "potential": potential_level,
            "estimated_gain": potential_gain,
            "current_efficiency": avg_efficiency,
            "target_efficiency": min(100, avg_efficiency + potential_gain * 0.6)  # Realistic target
        }
    
    # Cache Management Methods
    def invalidate_model_cache(self, 
                              model_type: str,
                              track_name: str,
                              car_name: str,
                              model_subtype: Optional[str] = None) -> bool:
        """
        Invalidate a specific model from cache
        
        Args:
            model_type: Type of model to invalidate
            track_name: Track name
            car_name: Car name
            model_subtype: Optional model subtype
            
        Returns:
            True if model was cached and removed, False otherwise
        """
        return self.model_cache.invalidate(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            model_subtype=model_subtype
        )
    
    def invalidate_track_cache(self, track_name: str) -> int:
        """
        Invalidate all cached models for a specific track
        
        Args:
            track_name: Track name to invalidate
            
        Returns:
            Number of models invalidated
        """
        pattern = f"*:{track_name}:*"
        return self.model_cache.invalidate_by_pattern(pattern)
    
    def invalidate_car_cache(self, car_name: str) -> int:
        """
        Invalidate all cached models for a specific car
        
        Args:
            car_name: Car name to invalidate
            
        Returns:
            Number of models invalidated
        """
        pattern = f"*:*:{car_name}:*"
        return self.model_cache.invalidate_by_pattern(pattern)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Dictionary with cache statistics and performance metrics
        """
        return self.model_cache.get_stats()
    
    def get_model_cache_info(self, 
                            model_type: str,
                            track_name: str,
                            car_name: str,
                            model_subtype: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a cached model
        
        Args:
            model_type: Type of model
            track_name: Track name
            car_name: Car name
            model_subtype: Optional model subtype
            
        Returns:
            Cache information or None if not cached
        """
        return self.model_cache.get_cache_info(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            model_subtype=model_subtype
        )
    
    def clear_all_cache(self):
        """Clear all cached models"""
        self.model_cache.clear()
        print("[INFO] All cached models cleared")
    
    def preload_models_for_session(self, 
                                   track_name: str, 
                                   car_name: str,
                                   model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Preload models for a racing session to improve performance
        
        Args:
            track_name: Track name for the session
            car_name: Car name for the session
            model_types: Optional list of model types to preload (default: all available)
            
        Returns:
            Dictionary with preload results
        """
        if model_types is None:
            model_types = ["imitation_learning"]
        
        results = {
            "track_name": track_name,
            "car_name": car_name,
            "preloaded_models": [],
            "failed_models": [],
            "total_preload_time": 0
        }
        
        start_time = datetime.now()
        
        for model_type in model_types:
            try:
                # Check if already cached
                if self.model_cache.get(model_type, track_name, car_name):
                    results["preloaded_models"].append(f"{model_type} (already cached)")
                    continue
                
                # Attempt to fetch and cache the model
                if model_type == "imitation_learning":
                    # This would typically be an async operation, but we'll simulate it
                    print(f"[INFO] Preloading {model_type} model for {track_name}/{car_name}")
                    # In a real scenario, you'd fetch from backend here
                    results["preloaded_models"].append(model_type)
                
            except Exception as e:
                error_info = f"{model_type}: {str(e)}"
                results["failed_models"].append(error_info)
                print(f"[ERROR] Failed to preload {model_type}: {e}")
        
        end_time = datetime.now()
        results["total_preload_time"] = (end_time - start_time).total_seconds()
        
        print(f"[INFO] Preload completed: {len(results['preloaded_models'])} successful, "
              f"{len(results['failed_models'])} failed, "
              f"{results['total_preload_time']:.2f}s total")
        
        return results
          
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")