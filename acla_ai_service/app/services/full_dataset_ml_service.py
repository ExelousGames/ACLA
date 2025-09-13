"""
Scikit-learn Machine Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service provides comprehensive AI model training and prediction capabilities
using your TelemetryFeatures and FeatureProcessor classes.
"""

import os
import pandas as pd
from .imitate_expert_learning_service import ImitateExpertLearningService
import numpy as np
import joblib
import warnings
import base64
import pickle
import io
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

# Import backend service
from .backend_service import backend_service

# Import model cache service
from .model_cache_service import model_cache_service

# PyTorch imports (for transformer model)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available - transformer functionality will be disabled")

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
        
        # Initialize corner identification unsupervised service
        from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
        self.corner_identification = CornerIdentificationUnsupervisedService()
        
        # Initialize tire grip analysis service
        from .tire_grip_analysis_service import TireGripAnalysisService
        self.tire_grip_analysis = TireGripAnalysisService()

        
        # Backend service integration
        self.backend_service = backend_service
        
        # Model cache service integration
        self.model_cache = model_cache_service
        
        # Add a simple lock mechanism to prevent concurrent fetches of the same model
        self._model_fetch_locks = {}
        self._lock_creation_lock = asyncio.Lock()

    def clear_all_cache(self):
        """Clear all cached models including corner identification, and tire grip analysis"""
        self.model_cache.clear()
        self.corner_identification.clear_corner_cache()
        self.tire_grip_analysis.clear_models_cache()
        print("[INFO] All cached models cleared (including corner identification, and tire grip analysis)")
    
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
            imitation_learning = ImitateExpertLearningService()
            deserialized_model_data = imitation_learning.deserialize_object_inside(model_data)
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
            model_types = ["imitation_learning", "corner_shape_unsupervised"]
        
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
                elif model_type == "corner_shape_unsupervised":
                    # Corner shape analysis not yet implemented
                    print(f"[INFO] Corner shape analysis not yet implemented")
                    results["failed_models"].append(f"{model_type}: Not yet implemented")
                
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

        imitation_learning = ImitateExpertLearningService()
        results = imitation_learning.train_ai_model(telemetry_data)
            
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

    # Corner Shape Unsupervised Learning Methods
    async def learn_corner_shapes(self, trackName: str, 
                                 clustering_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Learn corner shapes for a specific track using unsupervised machine learning
        
        This method uses multiple clustering algorithms to automatically discover
        different types of corners on a track based on their shape characteristics
        extracted from racing telemetry data.
        
        Args:
            trackName: Name of the track to analyze
            clustering_params: Optional parameters for clustering algorithms
                             e.g., {'n_clusters': 6, 'eps': 0.5, 'min_samples': 3}
                             
        Returns:
            Dictionary with corner shape learning results including:
            - Discovered corner types and their characteristics
            - Clustering algorithm performance metrics
            - Feature importance and analysis
        """
        try:
            # Corner shape learning is not yet implemented
            results = {
                "error": f"Corner shape learning not yet implemented for track {trackName}",
                "track_name": trackName,
                "implemented": False
            }
            
            return results
            
        except Exception as e:
            return {
                "error": f"Failed to learn corner shapes for track {trackName}: {str(e)}",
                "track_name": trackName
            }
    
    # Corner Identification Unsupervised Learning Methods
    async def learn_corner_characteristics(self, trackName: str, carName: Optional[str] = None) -> Dict[str, Any]:
        """
        Learn detailed corner characteristics for a track using unsupervised methods
        
        Args:
            trackName: Track name for corner identification
            carName: Optional car name filter
            
        Returns:
            Dictionary with corner identification and feature extraction results
        """
        try:
            results = await self.corner_identification.learn_track_corner_patterns(trackName, carName)
            
            # Save results to backend if successful
            if results.get("success"):
                model_data = {
                    "modelType": "corner_identification",
                    "trackName": trackName,
                    "carName": carName or "all_cars",
                    "modelData": results,
                    "timestamp": datetime.now().isoformat()
                }
                
                try:
                    await self.backend_service.save_ai_model(
                        AiModelDto(**model_data)
                    )
                    print(f"[INFO] Corner identification model saved to backend for {trackName}")
                except Exception as save_error:
                    print(f"[WARNING] Failed to save corner identification model to backend: {str(save_error)}")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Failed to learn corner characteristics for {trackName}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "corner_patterns": []
            }
    
    async def extract_corner_features_for_telemetry(self, 
                                                    telemetry_data: List[Dict[str, Any]], 
                                                    trackName: str,
                                                    carName: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Enhance telemetry data with extracted corner features
        
        Args:
            telemetry_data: List of telemetry records
            trackName: Track name for corner pattern matching
            carName: Optional car name
            
        Returns:
            Enhanced telemetry data with corner identification features
        """
        try:
            enhanced_telemetry = await self.corner_identification.extract_corner_features_for_telemetry(
                telemetry_data, trackName, carName
            )
            
            print(f"[INFO] Enhanced {len(enhanced_telemetry)} telemetry records with corner features")
            return enhanced_telemetry
            
        except Exception as e:
            print(f"[ERROR] Failed to enhance telemetry with corner features: {str(e)}")
            return telemetry_data  # Return original data on failure
    

    # Tire Grip Analysis Methods
    async def train_tire_grip_model(self, trackName: str, carName: Optional[str] = None) -> Dict[str, Any]:
        """
        Train tire grip analysis models for a specific track/car combination
        
        Args:
            trackName: Name of the track
            carName: Name of the car (optional)
            
        Returns:
            Training results and model performance metrics
        """
        return await self.tire_grip_analysis.train_tire_grip_model(trackName, carName)
    
    async def extract_tire_grip_features(self, 
                                       telemetry_data: List[Dict[str, Any]], 
                                       trackName: str,
                                       carName: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract tire grip features from telemetry data using trained models
        
        Args:
            telemetry_data: List of telemetry records
            trackName: Track name for model selection
            carName: Car name for model selection
            
        Returns:
            Enhanced telemetry data with tire grip features
        """
        return await self.tire_grip_analysis.extract_tire_grip_features(telemetry_data, trackName, carName)
    

    async def transformerLearning(self, trackName: str):
        try:
            sessions = await backend_service.get_all_racing_sessions(trackName)
        except Exception as e:
            return {"error": str(e)}

        each_session_telemetry_data = []
  
        for session in sessions.get("sessions", []):
                each_session_telemetry_data.append(session.get("data", []))
        if not each_session_telemetry_data:
            raise ValueError("No telemetry data found")
        
        # Flatten the list of lists into a single list of telemetry records (dictionaries)
        # This ensures each telemetry record maintains its field names as dictionary keys
        flattened_telemetry_data = []
        for session_data in each_session_telemetry_data:
            if isinstance(session_data, list):
                flattened_telemetry_data.extend(session_data)
            else:
                flattened_telemetry_data.append(session_data)
        
        # make sure every ai model is using the same data for training
        # Convert to DataFrame from list of dictionaries (this preserves column names)
        telemetry_df = pd.DataFrame(flattened_telemetry_data)
        feature_processor = FeatureProcessor(telemetry_df)
        
        # Cleaned data
        processed_df = feature_processor.general_cleaning_for_analysis()
        
        # Filter to only relevant features for analysis
        telemetry_features = TelemetryFeatures()
        relevant_features = telemetry_features.get_features_for_imitate_expert()
        processed_df = feature_processor.filter_features_by_list(processed_df, relevant_features)
        processed_df,laps = feature_processor._filter_top_performance_laps(processed_df,1)


        if processed_df.empty:
            raise ValueError("No valid telemetry data available after filtering for training.")
        
        # Filter top 1% laps as expert demonstrations, but ensure at least 1 lap
        top_laps_count = max(1, int(len(laps) * 0.01))
        top_laps = laps[:top_laps_count]
        # get rest of laps
        rest_laps = laps[top_laps_count:]

        # Flatten the DataFrames to list of dictionaries for imitation learning
        flatten_laps = []
        for lap_df in top_laps:
            # Convert DataFrame to list of dictionaries
            lap_records = lap_df.to_dict('records')
            flatten_laps.extend(lap_records)
        
        
        # Learn from expert demonstrations
        imitation_learning = ImitateExpertLearningService()
        imitation_learning.train_ai_model(flatten_laps)
        
        # Convert rest_laps DataFrames to the format expected by enriched_contextual_data
        rest_laps_dict_format = []
        for lap_df in rest_laps:
            # Convert DataFrame to list of dictionaries
            lap_records = lap_df.to_dict('records')
            # Wrap in a dictionary with 'data' key to match expected format
            rest_laps_dict_format.append({
                'data': lap_records,
                'lap_index': len(rest_laps_dict_format)
            })
        
        #enrich data
        enriched_contextual_data = await self.enriched_contextual_data(rest_laps_dict_format)

        #process enriched data, Generate training pairs with performance sections
        comparison_results = imitation_learning.compare_telemetry_with_expert(enriched_contextual_data.get("data"), 5, 5)
        training_and_expert_action = comparison_results.get('transformer_training_pairs', [])
        
        # train transformer model
        transformer_results = await self._train_expert_action_transformer(
            training_and_expert_action=training_and_expert_action,
            trackName=trackName
        )
        
        return {
            "success": True,
            "transformer_training": transformer_results,
            "expert_imitation_trained": True,
            "contextual_data_enriched": len(enriched_contextual_data),
            "training_pairs_generated": len(training_and_expert_action),
            "comparison_results": {
                "total_data_points": comparison_results.get('total_data_points', 0),
                "overall_score": comparison_results.get('overall_score', 0.0),
                "performance_sections_count": len(comparison_results.get('performance_sections', []))
            },
            "track_name": trackName
        }

    async def _train_expert_action_transformer(self, 
                                             training_and_expert_action: List[Dict[str, Any]],
                                             trackName: str) -> Dict[str, Any]:
        """
        Train the transformer model to predict expert actions from current telemetry
        
        Args:
            training_and_expert_action: List of training pairs with telemetry sections and expert actions
            trackName: Track name for model identification
            
        Returns:
            Training results and model performance metrics
        """
        try:
            # Check if PyTorch is available
            if not TORCH_AVAILABLE:
                return {
                    "success": False,
                    "error": "PyTorch is not available - please install torch to use transformer functionality"
                }
            
            # Import the transformer model (inside try block to handle import errors gracefully)
            from ..models.transformer_model import (
                ExpertActionTransformer, 
                ExpertActionTrainer,
                TelemetryActionDataset,
                create_expert_action_transformer
            )
            
            print(f"[INFO] Starting transformer training for {trackName} with {len(training_and_expert_action)} training pairs")
            
            if not training_and_expert_action:
                return {
                    "success": False,
                    "error": "No training pairs available for transformer training"
                }
            
            # Extract telemetry and expert action data from training pairs
            telemetry_data = []
            expert_actions = []
            
            for pair in training_and_expert_action:
                # Extract telemetry section (rising/peak/falling performance sections)
                telemetry_section = pair.get("telemetry_section", [])
                # Extract corresponding expert actions
                expert_action_section = pair.get("expert_actions", [])
                
                if telemetry_section and expert_action_section:
                    telemetry_data.extend(telemetry_section)
                    
                    # Process expert actions to extract action values
                    for expert_action in expert_action_section:
                        if isinstance(expert_action, dict):
                            # Extract optimal actions if available
                            optimal_actions = expert_action.get('optimal_actions', {})
                            
                            # Create action dictionary with steering, throttle, brake
                            action_dict = {
                                'Physics_steer_angle': optimal_actions.get('optimal_steering', 0.0),
                                'Physics_gas': optimal_actions.get('optimal_throttle', 0.0),
                                'Physics_brake': optimal_actions.get('optimal_brake', 0.0)
                            }
                            expert_actions.append(action_dict)
                        else:
                            # If expert_action is already in the right format
                            expert_actions.append(expert_action)
            
            if not telemetry_data or not expert_actions:
                return {
                    "success": False,
                    "error": "No valid telemetry-action pairs found in training data"
                }
            
            print(f"[INFO] Extracted {len(telemetry_data)} telemetry records and {len(expert_actions)} expert actions")
            
            # Determine input feature count from telemetry data
            if isinstance(telemetry_data[0], dict):
                input_features = len(telemetry_data[0])
            else:
                # Fallback to a reasonable number if structure is different
                input_features = 32
            
            # Create transformer model and trainer
            model, trainer = create_expert_action_transformer(
                telemetry_features=input_features,
                action_features=3,  # steering, throttle, brake
                d_model=256, # embedding dimension
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=1024,
                dropout=0.1,
                max_sequence_length=100
            )
            
            print(f"[INFO] Created transformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Create dataset and split into train/validation
            dataset = TelemetryActionDataset(
                telemetry_data=telemetry_data,
                expert_actions=expert_actions,
                sequence_length=50,
                prediction_horizon=20
            )
            
            print(f"[INFO] Created dataset with {len(dataset)} sequences")
            
            if len(dataset) == 0:
                return {
                    "success": False,
                    "error": "Dataset creation resulted in 0 sequences - check data compatibility"
                }
            
            # Split into train/validation sets
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders
            batch_size = min(16, len(train_dataset))  # Adjust batch size based on data availability
            
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                drop_last=True if len(train_dataset) > batch_size else False
            )
            
            val_dataloader = None
            if len(val_dataset) > 0:
                val_batch_size = min(8, len(val_dataset))
                val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=val_batch_size, 
                    shuffle=False,
                    drop_last=False
                )
            
            print(f"[INFO] Created data loaders: train_size={len(train_dataset)}, val_size={len(val_dataset)}")
            
            # Train the model
            model_save_path = str(self.models_directory / f"transformer_{trackName.replace(' ', '_')}.pth")
            
            training_results = trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=50,  # Adjust based on data size
                patience=10,
                save_path=model_save_path
            )
            
            print(f"[INFO] Transformer training completed for {trackName}")
            
            # Save model metadata to backend
            try:
                transformer_model_data = {
                    "model_path": model_save_path,
                    "input_features": input_features,
                    "training_results": training_results,
                    "model_parameters": {
                        "d_model": 256,
                        "nhead": 8,
                        "num_encoder_layers": 6,
                        "num_decoder_layers": 6,
                        "sequence_length": 50,
                        "prediction_horizon": 20
                    },
                    "scaler_data": dataset.scaler.__dict__ if hasattr(dataset, 'scaler') else None
                }
                
                ai_model_dto = {
                    "modelType": "expert_action_transformer",
                    "trackName": trackName,
                    "carName": None,  # Track-specific model
                    "modelData": transformer_model_data,
                    "metadata": {
                        "training_pairs": len(training_and_expert_action),
                        "dataset_sequences": len(dataset),
                        "training_results": training_results,
                        "training_timestamp": datetime.now().isoformat()
                    },
                    "isActive": True
                }
                
                await backend_service.save_ai_model(ai_model_dto)
                print(f"[INFO] Saved transformer model data to backend for track: {trackName}")
                
            except Exception as backend_error:
                print(f"[WARNING] Failed to save transformer model to backend: {str(backend_error)}")
            
            return {
                "success": True,
                "model_path": model_save_path,
                "training_results": training_results,
                "dataset_info": {
                    "total_sequences": len(dataset),
                    "train_sequences": len(train_dataset),
                    "val_sequences": len(val_dataset),
                    "input_features": input_features
                },
                "model_info": {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                }
            }
            
        except ImportError as import_error:
            return {
                "success": False,
                "error": f"Failed to import transformer dependencies: {str(import_error)}. Please install PyTorch.",
                "track_name": trackName
            }
        except Exception as e:
            print(f"[ERROR] Failed to train transformer model for {trackName}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "track_name": trackName
            }

    async def predict_expert_actions(self, 
                                   current_telemetry: List[Dict[str, Any]],
                                   trackName: str,
                                   sequence_length: int = 20,
                                   temperature: float = 1.0) -> Dict[str, Any]:
        """
        Predict expert action sequence from current telemetry using trained transformer
        
        Args:
            current_telemetry: Current telemetry data
            trackName: Track name for model selection
            sequence_length: Number of future actions to predict
            temperature: Sampling temperature (1.0 = normal, lower = more conservative)
            
        Returns:
            Predicted expert actions and confidence scores
        """
        try:
            # Check if PyTorch is available
            if not TORCH_AVAILABLE:
                return {
                    "success": False,
                    "error": "PyTorch is not available - cannot perform transformer predictions"
                }
            
            # Load the trained model from backend
            try:
                model_response = await self.backend_service.getCompleteActiveModelData(
                    trackName, None, "expert_action_transformer"
                )
                
                if "error" in model_response:
                    return {
                        "success": False,
                        "error": f"No trained transformer model found for track {trackName}"
                    }
                
                model_data = model_response.get("data", {}).get("modelData", {})
                if not model_data:
                    return {
                        "success": False,
                        "error": f"No model data found for track {trackName}"
                    }
                
            except Exception as backend_error:
                return {
                    "success": False,
                    "error": f"Failed to load model from backend: {str(backend_error)}"
                }
            
            # Import transformer model
            from ..models.transformer_model import (
                ExpertActionTransformer, 
                create_expert_action_transformer
            )
            
            # Extract model parameters
            model_params = model_data.get("model_parameters", {})
            input_features = model_data.get("input_features", 32)
            model_path = model_data.get("model_path", "")
            
            # Create model with same architecture
            model, trainer = create_expert_action_transformer(
                telemetry_features=input_features,
                action_features=3,
                d_model=model_params.get("d_model", 256),
                nhead=model_params.get("nhead", 8),
                num_encoder_layers=model_params.get("num_encoder_layers", 6),
                num_decoder_layers=model_params.get("num_decoder_layers", 6),
                max_sequence_length=model_params.get("sequence_length", 50)
            )
            
            # Load model weights if available
            if model_path and os.path.exists(model_path):
                trainer.load_model(model_path)
                print(f"[INFO] Loaded transformer model from {model_path}")
            else:
                print("[WARNING] Model weights not found - using untrained model")
            
            # Prepare input telemetry data
            if not current_telemetry:
                return {
                    "success": False,
                    "error": "No telemetry data provided for prediction"
                }
            
            # Convert telemetry to tensor format
            telemetry_df = pd.DataFrame(current_telemetry)
            
            # Extract features (similar to training)
            feature_columns = [
                'Physics_speed_kmh', 'Physics_gear', 'Physics_rpm', 'Physics_brake',
                'Physics_gas', 'Physics_steer_angle', 'Physics_slip_angle_front_left',
                'Physics_slip_angle_front_right', 'Physics_g_force_x', 'Physics_g_force_y',
                'Physics_g_force_z', 'Physics_tyre_core_temp_front_left',
                'Physics_tyre_core_temp_front_right', 'Physics_brake_temp_front_left',
                'Physics_brake_temp_front_right', 'Graphics_delta_lap_time'
            ]
            
            available_features = [col for col in feature_columns if col in telemetry_df.columns]
            if not available_features:
                available_features = telemetry_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not available_features:
                return {
                    "success": False,
                    "error": "No numeric features found in telemetry data"
                }
            
            features = telemetry_df[available_features].fillna(0).values
            
            # Apply scaling if scaler data is available
            scaler_data = model_data.get("scaler_data")
            if scaler_data:
                try:
                    scaler = StandardScaler()
                    # Restore scaler state
                    scaler.mean_ = np.array(scaler_data.get("mean_", [0] * features.shape[1]))
                    scaler.scale_ = np.array(scaler_data.get("scale_", [1] * features.shape[1]))
                    scaler.var_ = np.array(scaler_data.get("var_", [1] * features.shape[1]))
                    scaler.n_samples_seen_ = scaler_data.get("n_samples_seen_", 1)
                    
                    features = scaler.transform(features)
                except Exception as scaling_error:
                    print(f"[WARNING] Failed to apply scaling: {scaling_error}")
            
            # Convert to PyTorch tensor
            telemetry_tensor = torch.FloatTensor(features).unsqueeze(1)  # [seq_len, batch_size=1, features]
            
            # Predict expert actions
            predicted_actions, performance_scores = model.predict_expert_sequence(
                telemetry_tensor, sequence_length, temperature
            )
            
            # Convert predictions back to numpy/list format
            actions_numpy = predicted_actions.squeeze(1).numpy()  # [seq_len, action_features]
            performance_numpy = performance_scores.squeeze().numpy()  # [seq_len]
            
            # Format results
            predicted_sequence = []
            for i in range(len(actions_numpy)):
                action_dict = {
                    "timestep": i + 1,
                    "steering_angle": float(actions_numpy[i, 0]),
                    "throttle": float(actions_numpy[i, 1]) if len(actions_numpy[i]) > 1 else 0.0,
                    "brake": float(actions_numpy[i, 2]) if len(actions_numpy[i]) > 2 else 0.0,
                    "performance_score": float(performance_numpy[i]) if len(performance_numpy) > i else 0.0
                }
                predicted_sequence.append(action_dict)
            
            return {
                "success": True,
                "track_name": trackName,
                "predicted_actions": predicted_sequence,
                "prediction_info": {
                    "sequence_length": sequence_length,
                    "temperature": temperature,
                    "input_features": len(available_features),
                    "input_timesteps": len(current_telemetry)
                },
                "model_info": {
                    "model_parameters": model_params,
                    "input_features": input_features
                }
            }
            
        except ImportError as import_error:
            return {
                "success": False,
                "error": f"Failed to import transformer dependencies: {str(import_error)}"
            }
        except Exception as e:
            print(f"[ERROR] Failed to predict expert actions for {trackName}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "track_name": trackName
            }

    async def enriched_contextual_data(self, laps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich laps data with contextual features using trained models
        
        Args:
            laps: List of lap telemetry data dictionaries
            
        Returns:
            Enriched telemetry data with corner identification and tire grip features
        """
        print(f"[INFO] Starting contextual data enrichment for {len(laps)} laps")
        
        if not laps:
            print("[WARNING] No laps data provided for enrichment")
            return []
        
        try:
            # Split laps for training and feature extraction (70% training, 30% extraction)
            split_index = int(len(laps) * 0.7)
            training_laps = laps[:split_index]
            extraction_laps = laps[split_index:]
            
            print(f"[INFO] Split data: {len(training_laps)} laps for training both corner identification and tire grip analysis, {len(extraction_laps)} laps for extraction for both too")
            
            # Flatten training data for model training
            training_telemetry = []
            for lap_data in training_laps:
                if isinstance(lap_data, dict) and 'data' in lap_data:
                    training_telemetry.extend(lap_data['data'])
                elif isinstance(lap_data, list):
                    training_telemetry.extend(lap_data)
                else:
                    training_telemetry.append(lap_data)
            
            # Train corner identification model using training data
            print("[INFO] Training corner identification model...")
            try:
                corner_results = await self.corner_identification.learn_track_corner_patterns(training_telemetry)
                
                if corner_results.get("success"):
                    print(f"[INFO] Corner identification training successful: {corner_results.get('total_corners_identified', 0)} corners identified")
                else:
                    print(f"[WARNING] Corner identification training failed: {corner_results.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[ERROR] Corner identification training failed: {str(e)}")
                corner_results = {"success": False, "error": str(e)}
            
            # Train tire grip analysis model using training data
            print("[INFO] Training tire grip analysis model...")
            try:
                tire_grip_results = await self.tire_grip_analysis.train_tire_grip_model(training_telemetry)
                if tire_grip_results.get("success"):
                    print(f"[INFO] Tire grip model training successful: {tire_grip_results.get('models_trained', 0)} models trained")
                else:
                    print(f"[WARNING] Tire grip model training failed: {tire_grip_results.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[ERROR] Tire grip analysis training failed: {str(e)}")
                tire_grip_results = {"success": False, "error": str(e)}
            
            # Now extract features for extraction telemetry data only
            enriched_telemetry_data = []
            total_laps = len(extraction_laps)
            
            for lap_idx, lap_data in enumerate(extraction_laps):
                print(f"[INFO] Processing lap {lap_idx + 1}/{total_laps} for feature extraction...")
                
                try:
                    # Extract telemetry records from lap
                    if isinstance(lap_data, dict) and 'data' in lap_data:
                        telemetry_records = lap_data['data']
                        lap_metadata = {k: v for k, v in lap_data.items() if k != 'data'}
                    elif isinstance(lap_data, list):
                        telemetry_records = lap_data
                        lap_metadata = {}
                    else:
                        telemetry_records = [lap_data]
                        lap_metadata = {}
                    
                    if not telemetry_records:
                        continue
                    
                    # Extract corner features for this lap's telemetry
                    corner_enhanced_telemetry = telemetry_records
                    if corner_results.get("success"):
                        try:
                            corner_enhanced_telemetry = await self.corner_identification.extract_corner_features_for_telemetry(
                                telemetry_records      
                            )
                            print(f"[INFO] Added corner features to {len(corner_enhanced_telemetry)} records")
                            
                            cornerPrediction = self.corner_identification.predict_corner_count(pd.DataFrame(telemetry_records))
                            print(f"[INFO] Predicted corner count for lap {lap_idx + 1}: predicted {cornerPrediction.predicted_corners} corners with confidence of {cornerPrediction.confidence}")
                        except Exception as e:
                            print(f"[WARNING] Failed to extract corner features for lap {lap_idx}: {str(e)}")
                            corner_enhanced_telemetry = telemetry_records
                    
                    # Extract tire grip features for this lap's telemetry
                    fully_enhanced_telemetry = corner_enhanced_telemetry
                    if tire_grip_results.get("success"):
                        try:
                            fully_enhanced_telemetry = await self.tire_grip_analysis.extract_tire_grip_features(
                                corner_enhanced_telemetry
                            )
                            print(f"[INFO] Added tire grip features to {len(fully_enhanced_telemetry)} records")
                        except Exception as e:
                            print(f"[WARNING] Failed to extract tire grip features for lap {lap_idx}: {str(e)}")
                            fully_enhanced_telemetry = corner_enhanced_telemetry
                    
                    # Reconstruct lap data with enhanced telemetry
                    enriched_lap = {
                        **lap_metadata,
                        'data': fully_enhanced_telemetry,
                        'enrichment_info': {
                            'corner_features_added': corner_results.get("success", False),
                            'tire_grip_features_added': tire_grip_results.get("success", False),
                            'original_records_count': len(telemetry_records),
                            'enriched_records_count': len(fully_enhanced_telemetry),
                            'enrichment_timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    enriched_telemetry_data.append(enriched_lap)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process lap {lap_idx}: {str(e)}")
                    # Add original lap data on failure
                    enriched_telemetry_data.append(lap_data)
            
            print(f"[INFO] Contextual data enrichment completed: {len(enriched_telemetry_data)} laps processed")
            
            # Calculate enrichment statistics
            total_original_records = sum(
                len(lap.get('data', [])) if isinstance(lap.get('data'), list) else 1 
                for lap in extraction_laps
            )
            total_enriched_records = sum(
                len(lap.get('data', [])) if isinstance(lap.get('data'), list) else 1 
                for lap in enriched_telemetry_data
            )
            
            enrichment_summary = {
                'total_laps_processed': len(enriched_telemetry_data),
                'total_original_records': total_original_records,
                'total_enriched_records': total_enriched_records,
                'corner_identification_success': corner_results.get("success", False),
                'tire_grip_analysis_success': tire_grip_results.get("success", False),
                'training_laps_used': len(training_laps),
                'extraction_laps_processed': len(extraction_laps)
            }
            
            print(f"[INFO] Enrichment summary: {enrichment_summary}")
            
            return enriched_telemetry_data
            
        except Exception as e:
            print(f"[ERROR] Failed to enrich contextual data: {str(e)}")
            return laps  # Return original data on failure

          
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")