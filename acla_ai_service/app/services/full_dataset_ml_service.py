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
from .Corner_imitation_learning_service import CornerImitationLearningService, CornerSpecificLearner
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

# Import cornering analysis
from .identify_track_cornoring_phases import TrackCorneringAnalyzer

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


class Full_dataset_TelemetryMLService:
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
            
    def _deserialize_model_data(self, model_data: str) -> Dict[str, Any]:
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
    
    
    def _serialize_model_data(self, model_id: str) -> str:
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
            
            await backend_service.save_ai_model(ai_model_dto)
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
        
        try:
            # Get model from cache or fetch from backend using the separated function
            deserialized_trajectory_models, metadata = await self._get_cached_model_or_fetch(
                model_type="imitation_learning",
                track_name=trackName,
                car_name=carName,
                model_subtype="complete_model_data",
                deserializer_func=self.imitation_learning.deserialize_object_inside
            )
            
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
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_track_cornering(self, trackName: str) -> Dict[str, Any]:
        try:
            sessions = await backend_service.get_all_racing_sessions(trackName)
        except Exception as e:
            return {"error": str(e)}

        each_session_telemetry_data = []
  
        for session in sessions.get("sessions", []):
                each_session_telemetry_data.append(session.get("data", []))

        if not each_session_telemetry_data:
            raise ValueError("No telemetry data found")

        # Flatten the list of lists into a single list of telemetry records
        telemetry_data = [item for sublist in each_session_telemetry_data for item in sublist]
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(telemetry_data)
        
        # Process telemetry data
        feature_processor = FeatureProcessor(df)
        processed_df = feature_processor.general_cleaning_for_analysis()
        
        # Create track cornering analyzer and filter top performance laps
        track_analyzer = TrackCorneringAnalyzer()
        filtered_df = track_analyzer.filter_top_performance_laps(processed_df)
        
        # Identify cornering phases
        cornering_df = track_analyzer.identify_cornering_phases(filtered_df)
        
        # Get analysis summary
        analysis_summary = track_analyzer.get_cornering_analysis_summary(cornering_df)
        try:
            #save the info to backend
            ai_model_dto = {
                "modelType": "track_corner_analysis",
                "trackName": trackName,
                "modelData": analysis_summary,
                "metadata": {
                },
                "isActive": True
            }
            
            await backend_service.save_ai_model(ai_model_dto)
        except Exception as error:
            pass
        return analysis_summary
    
    
    async def train_track_cornering_model(self, trackName: str) -> Dict[str, Any]:
        """
        Train a model to predict optimal cornering speeds based on telemetry data
        
        Args:
            trackName: Track name for the model
            carName: Car name for the model
            
        Returns:
            Dictionary with cornering model training results
        """
        try:
            # Load previously saved cornering analysis from backend
            model_response = await backend_service.getCompleteActiveModelData(trackName, None, "track_corner_analysis")
            
            if not model_response or not model_response.get('success', False) or 'data' not in model_response:
                raise ValueError("No cornering analysis data found for this track")
            
            analysis_data = model_response.get('data', {}).get('modelData', {})
            if not analysis_data:
                raise ValueError("Cornering analysis data is empty")
            
            #retrieve all racing session in database
            try:
                sessions = await backend_service.get_all_racing_sessions(trackName)
            except Exception as e:
                return {"error": str(e)}

            each_session_telemetry_data = []
  
            for session in sessions.get("sessions", []):
                each_session_telemetry_data.append(session.get("data", []))

            if not each_session_telemetry_data:
                raise ValueError("No telemetry data found")

            # Flatten the list of lists into a single list of telemetry records
            telemetry_data = [item for sublist in each_session_telemetry_data for item in sublist]
            
            service = CornerImitationLearningService()
            serialized_model_results,results = service.train_corner_specific_model(telemetry_data, analysis_data)

            # Save results to backend
            try:
                ai_model_dto = {
                    "modelType": "track_corner_training",
                    "trackName": trackName,
                    "modelData": serialized_model_results,
                    "metadata": {
                        "training_timestamp": datetime.now().isoformat()
                    },
                    "isActive": True
                }
                await backend_service.save_ai_model(ai_model_dto)
            except Exception as error:
                print(f"[ERROR] Failed to save AI model: {error}")
                pass
            
            return results
        except Exception as e:
            return {"error": str(e)}
    

    async def predict_optimal_cornering(self, trackName: str,corner_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal cornering using trained cornering model
        
        Args:
            trackName: Track name for the model
            normalized_car_position: Current car position on track (0.0 to 1.0)
        
        Returns:
            Dictionary with optimal cornering actions and guidance
        """
        try:
            # Get model from cache or fetch from backend
            model_data, metadata = await self._get_cached_model_or_fetch(
                model_type="track_corner_training",
                track_name=trackName,
                model_subtype="complete_model_data"
            )

            if not model_data:
                raise ValueError({"error": "No valid cornering model found for this track"})
            
            # Create CornerImitationLearningService instance
            service = CornerImitationLearningService()

            # Deserialize the model data using the service's deserialize method
            deserialized_data = service.receive_serialized_model_data(model_data)
            
            # Reconstruct the trained models from the saved data
            if 'corner_models' in deserialized_data:
                # Get optimal actions for the current position
                predictions = service.get_all_corner_predictions(corner_analysis_result)
                simple_corner_guidance = service.get_simple_corner_guidance(corner_analysis_result)
                return {"predictions": predictions, "simple_guidance": simple_corner_guidance}
            else:
                raise  RuntimeError({"error": "Invalid model data structure - missing corner_models"}) 
                
        except Exception as e:
            raise ValueError({
                "error": f"predict_optimal_cornering(), Failed to predict optimal cornering: {str(e)}",
                "track_name": trackName,
            })

    async def predict_optimal_cornering_from_telemetry(self, trackName: str, current_telemetry: Dict[str, Any]):
        """
        Predict optimal cornering using current telemetry data
        
        Args:
            trackName: Track name for the model
            current_telemetry: Current telemetry data containing normalized_car_position
        
        Returns:
            Dictionary with optimal cornering actions and guidance
        """
        # Extract normalized car position from telemetry
        normalized_position = current_telemetry.get('Graphics_normalized_car_position')
        
        if normalized_position is None:
            return {"error": "No normalized car position found in telemetry data"}
        
        return await self.predict_optimal_cornering(trackName, normalized_position)

    def clear_all_cache(self):
        """Clear all cached models"""
        self.model_cache.clear()
        print("[INFO] All cached models cleared")
    
    async def _get_cached_model_or_fetch(self,
                                        model_type: str,
                                        track_name: Optional[str] = None,
                                        car_name: Optional[str] = None,
                                        model_subtype: str = "complete_model_data",
                                        deserializer_func=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Get model from cache or fetch from backend with thread-safe locking
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            track_name: Track name for the model (optional)
            car_name: Car name for the model (optional)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize model data from backend
            
        Returns:
            Tuple of (deserialized_model_data, metadata)
        """
        # Unique key for the model to manage locks
        model_key = f"{track_name or 'any'}/{car_name or 'any'}/{model_type}"
        
        # Use flags to track if this thread is responsible for fetching
        is_fetching_thread = False
        deserialized_model_data = None
        metadata = {}
        
        try:
            # First, try to get model from cache without any locking
            cached_result = self.model_cache.get(
                model_type=model_type,
                track_name=track_name,
                car_name=car_name,
                model_subtype=model_subtype
            )
            
            if cached_result:
                deserialized_model_data, metadata = cached_result
                return deserialized_model_data, metadata
            
            # If no cached result, handle fetching with proper locking
            async with self._lock_creation_lock:
                # Double-check cache after acquiring lock
                cached_result = self.model_cache.get(
                    model_type=model_type,
                    track_name=track_name,
                    car_name=car_name,
                    model_subtype=model_subtype
                )
                
                if cached_result:
                    deserialized_model_data, metadata = cached_result
                    return deserialized_model_data, metadata
                elif model_key in self._model_fetch_locks:
                    # Another thread is fetching this model, wait for it
                    fetch_event = self._model_fetch_locks[model_key]
                else:
                    # We're the first to request this model, create the event and mark ourselves as fetching
                    self._model_fetch_locks[model_key] = asyncio.Event()
                    is_fetching_thread = True
            
            # If we need to wait for another thread
            if not deserialized_model_data and not is_fetching_thread:
                try:
                    await fetch_event.wait()
                    # The other thread should have cached the model, try cache again
                    cached_result = self.model_cache.get(
                        model_type=model_type,
                        track_name=track_name,
                        car_name=car_name,
                        model_subtype=model_subtype
                    )
                    if cached_result:
                        deserialized_model_data, metadata = cached_result
                        print(f"[INFO] Using model cached by another thread for {track_name or 'any'}/{car_name or 'any'}")
                    else:
                        print(f"[WARNING] Expected cached model not found after waiting for {track_name or 'any'}/{car_name or 'any'}")
                        # Continue to try fetching ourselves as fallback
                        is_fetching_thread = True
                except Exception as wait_error:
                    print(f"[ERROR] Error while waiting for model fetch: {str(wait_error)}")
                    # Continue to try fetching ourselves as fallback
                    is_fetching_thread = True

            # If we are the fetching thread or no data in cache, do the actual fetch
            if not deserialized_model_data and is_fetching_thread:
                try:
                    deserialized_model_data, metadata = await self._fetch_and_cache_model(
                        model_type=model_type,
                        track_name=track_name,
                        car_name=car_name,
                        model_subtype=model_subtype,
                        deserializer_func=deserializer_func
                    )
                    print(f"[INFO] Successfully fetched and cached model for {track_name or 'any'}/{car_name or 'any'}")
                    
                except Exception as fetch_error:
                    print(f"[ERROR] Failed to fetch model: {str(fetch_error)}")
                    raise fetch_error
                finally:
                    # Always signal completion and clean up lock when we're the fetching thread
                    self._cleanup_fetch_lock(model_key, track_name or 'any', car_name or 'any')
            
            # At this point, we should have the model data
            if not deserialized_model_data:
                raise Exception("Failed to obtain model data from cache or backend")
            
            return deserialized_model_data, metadata
            
        except Exception as e:
            # Clean up any locks that might have been created by this thread
            if is_fetching_thread:
                self._emergency_cleanup_fetch_lock(model_key, track_name or 'any', car_name or 'any')
            
            raise e
    
    async def _fetch_and_cache_model(self,
                                    model_type: str,
                                    track_name: Optional[str] = None,
                                    car_name: Optional[str] = None,
                                    model_subtype: str = "complete_model_data",
                                    deserializer_func=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Fetch model from backend and cache it
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            track_name: Track name for the model (optional)
            car_name: Car name for the model (optional)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize model data from backend
            
        Returns:
            Tuple of (deserialized_model_data, metadata)
        """
        model_response = await self.backend_service.getCompleteActiveModelData(
            track_name, car_name, model_type
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

        # Deserialize the model data using provided function or default
        if deserializer_func:
            deserialized_model_data = deserializer_func(model_data)
        elif model_type == "imitation_learning":
            deserialized_model_data = self.imitation_learning.deserialize_object_inside(model_data)
        else:
            # For other model types, you might need to add more deserializers
            deserialized_model_data = model_data
        
        # Cache the deserialized model for future use
        cache_metadata = {
            "track_name": track_name,
            "car_name": car_name,
            "model_type": model_type,
            "fetched_at": datetime.now().isoformat(),
            "backend_model_id": model_response.get("id", "unknown")
        }
        
        self.model_cache.put(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            data=deserialized_model_data,
            metadata=cache_metadata,
            model_subtype=model_subtype
        )
        
        return deserialized_model_data, cache_metadata
    
    def _cleanup_fetch_lock(self, model_key: str, track_name: str, car_name: str):
        """
        Clean up fetch lock for a specific model key
        
        Args:
            model_key: The model key used for locking
            track_name: Track name for logging
            car_name: Car name for logging
        """
        if model_key in self._model_fetch_locks:
            try:
                self._model_fetch_locks[model_key].set()
                del self._model_fetch_locks[model_key]
                print(f"[INFO] Released fetch lock for {track_name}/{car_name}")
            except Exception as cleanup_error:
                print(f"[WARNING] Error cleaning up fetch lock: {str(cleanup_error)}")
    
    def _emergency_cleanup_fetch_lock(self, model_key: str, track_name: str, car_name: str):
        """
        Emergency cleanup of fetch lock with additional safety checks
        
        Args:
            model_key: The model key used for locking
            track_name: Track name for logging
            car_name: Car name for logging
        """
        if hasattr(self, '_model_fetch_locks') and model_key in self._model_fetch_locks:
            try:
                self._model_fetch_locks[model_key].set()  # Signal any waiting threads
                del self._model_fetch_locks[model_key]
                print(f"[INFO] Emergency cleanup of fetch lock for {track_name}/{car_name}")
            except Exception as cleanup_error:
                print(f"[WARNING] Error during emergency lock cleanup: {str(cleanup_error)}")
    
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