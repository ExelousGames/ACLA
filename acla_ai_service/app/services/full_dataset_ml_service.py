"""
Scikit-learn Machine Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service provides comprehensive AI model training and prediction capabilities
using your TelemetryFeatures and FeatureProcessor classes.
"""

import os
import pandas as pd
from .imitate_expert_learning_service import ExpertImitateLearningService
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
    from ..models.transformer_model import ExpertActionTransformer, ExpertActionTrainer, TelemetryActionDataset
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
            imitation_learning = ExpertImitateLearningService()
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
    
    def _print_section_divider(self, title: str, width: int = 80):
        """
        Print a large, visually prominent section divider for console output
        
        Args:
            title: Section title to display
            width: Width of the divider (default 80 characters)
        """
        # Create the divider lines
        border_line = "=" * width
        title_line = f"║ {title.center(width - 4)} ║"
        empty_line = f"║{' ' * (width - 2)}║"
        
        print()
        print(border_line)
        print(empty_line)
        print(title_line)
        print(empty_line)
        print(border_line)
        print()
        
    
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

        imitation_learning = ExpertImitateLearningService()
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
    
    
    def _deserialize_transformer_model(self, model_data: Dict[str, Any]) -> 'ExpertActionTransformer':
        """
        Deserialize transformer model data specifically for transformer models
        
        Args:
            model_data: Serialized transformer model data from backend
            
        Returns:
            Deserialized ExpertActionTransformer instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available - transformer functionality is disabled")
        
        try:
            # Check if this is the expected transformer model data structure
            if model_data.get('model_type') != 'ExpertActionTransformer':
                raise ValueError(f"Expected ExpertActionTransformer, got {model_data.get('model_type', 'unknown')}")
            
            # Use the class method to deserialize
            transformer_model = ExpertActionTransformer.deserialize_model(model_data)
            transformer_model.eval()  # Set to evaluation mode
            
            print(f"[INFO] Successfully deserialized ExpertActionTransformer model")
            return transformer_model
            
        except Exception as e:
            print(f"[ERROR] Failed to deserialize transformer model: {str(e)}")
            raise e
    
    
    async def predict_expert_actions(self, 
                                   telemetry_dict: Dict[str, Any],
                                   trackName: str, 
                                   carName: Optional[str] = None,
                                   sequence_length: int = 20,
                                   temperature: float = 1.0) -> Dict[str, Any]:
        """
        Fetch transformer model and predict expert actions from telemetry data
        
        Args:
            telemetry_dict: Single telemetry data record as dictionary
            trackName: Track name for model selection
            carName: Optional car name for model selection
            sequence_length: Length of action sequence to predict (default: 20)
            temperature: Temperature for prediction sampling (1.0 = normal, lower = more conservative)
            
        Returns:
            Dictionary containing:
            - predicted_actions: List of predicted expert actions
            - performance_scores: List of performance scores for each predicted action
            - metadata: Additional prediction metadata
        """
        if not TORCH_AVAILABLE:
            return {
                "error": "PyTorch is not available - transformer functionality is disabled",
                "predicted_actions": [],
                "performance_scores": [],
                "metadata": {}
            }
        
        try:
            # Fetch and deserialize the transformer model
            print(f"[INFO] Fetching transformer model for {trackName}/{carName or 'any'}")
            transformer_model, model_metadata = await self._get_cached_model_or_fetch(
                model_type="transformer_learning",
                track_name=trackName,
                car_name=carName,
                model_subtype="transformer_model_data",
                deserializer_func=self._deserialize_transformer_model
            )
            
            # Prepare telemetry data for the model
            telemetry_features = TelemetryFeatures()
            feature_names = telemetry_features.get_features_for_imitate_expert()
            
            # Extract relevant features from the telemetry dictionary
            feature_values = []
            for feature_name in feature_names:
                value = telemetry_dict.get(feature_name, 0.0)
                try:
                    feature_values.append(float(value))
                except (ValueError, TypeError):
                    print(f"[WARNING] Invalid value for feature {feature_name}: {value}, using 0.0")
                    feature_values.append(0.0)
            
            # Convert to tensor format expected by the transformer
            # Shape: [seq_len=1, batch_size=1, input_features]
            src_telemetry = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            
            print(f"[INFO] Input telemetry tensor shape: {src_telemetry.shape}")
            print(f"[INFO] Predicting {sequence_length} expert actions...")
            
            # Make prediction using the transformer model
            with torch.no_grad():
                predicted_sequence, performance_sequence = transformer_model.predict_expert_sequence(
                    src_telemetry=src_telemetry,
                    sequence_length=sequence_length,
                    temperature=temperature
                )
            
            # Convert predictions back to lists for JSON serialization
            predicted_actions = predicted_sequence.squeeze(1).tolist()  # Remove batch dimension
            performance_scores = performance_sequence.squeeze(1).squeeze(2).tolist()  # Remove batch and feature dims
            
            # Calculate metadata
            avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
            prediction_confidence = max(0.0, min(1.0, (avg_performance + 1.0) / 2.0))  # Normalize to 0-1 range
            
            print(f"[INFO] Successfully predicted {len(predicted_actions)} expert actions")
            print(f"[INFO] Average predicted performance: {avg_performance:.4f}")
            print(f"[INFO] Prediction confidence: {prediction_confidence:.2%}")
            
            return {
                "success": True,
                "predicted_actions": predicted_actions,
                "performance_scores": performance_scores,
                "metadata": {
                    "track_name": trackName,
                    "car_name": carName,
                    "sequence_length": sequence_length,
                    "temperature": temperature,
                    "input_features_count": len(feature_values),
                    "avg_predicted_performance": avg_performance,
                    "prediction_confidence": prediction_confidence,
                    "model_metadata": model_metadata,
                    "prediction_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to predict expert actions: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "predicted_actions": [],
                "performance_scores": [],
                "metadata": {
                    "track_name": trackName,
                    "car_name": carName,
                    "error_timestamp": datetime.now().isoformat()
                }
            }
    

    async def StartImitateExpertPipeline(self, trackName: str):
        
        """
        returns: success, transformer_training, expert_imitation_trained    , contextual_data_enriched, training_pairs_generated, comparison_results, track_name
        """
        self._print_section_divider("FETCHING TELEMETRY DATA FROM BACKEND")
        #retrieve all racing session in database
        try:
            sessions_summary = await backend_service.get_all_racing_sessions(trackName)
        except Exception as e:
            return {"error": str(e)}

        #list of telemetry data for each session
        each_session_telemetry_data = []
  
        for session_summary in sessions_summary.get("sessions", []):
                each_session_telemetry_data.append(session_summary.get("data", []))
        if not each_session_telemetry_data:
            raise ValueError("No telemetry data found")
        
        # Flatten the list of lists into a single list of telemetry records (dictionaries)
        # This ensures each telemetry record maintains its field names as dictionary keys
        flattened_telemetry_data = []
        for session_data in each_session_telemetry_data:
                flattened_telemetry_data.extend(session_data)
        
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
        processed_df,lap_df_list = feature_processor._filter_top_performance_laps(processed_df,1)

        if processed_df.empty:
            raise ValueError("No valid telemetry data available after filtering for training.")
        
        # Filter top 1% laps as expert demonstrations, but ensure at least 1 lap
        top_laps_df_count = max(1, int(len(lap_df_list) * 0.01))
        top_laps_df = lap_df_list[:top_laps_df_count]
        # get rest of laps
        bottom_laps_df = lap_df_list[top_laps_df_count:]

        # Flatten the DataFrames to list of laps for imitation learning
        top_laps_telemetry_list = []
        for lap_df in top_laps_df:
            # Convert DataFrame to list of dictionaries
            lap_records = lap_df.to_dict('records')
            top_laps_telemetry_list.extend(lap_records)
        
        
        # Learn from expert demonstrations
        self._print_section_divider("TRAINING IMITATION LEARNING MODEL")
        imitation_learning = ExpertImitateLearningService()
        imitation_learning.train_ai_model(top_laps_telemetry_list)
        
        # Convert rest_laps DataFrames to list of lap records (dictionaries)
        bottom_laps_telemetry_list = []
        for lap_df in bottom_laps_df:
            # Convert DataFrame to list of dictionaries
            lap_records = lap_df.to_dict('records')
            # Add lap records directly to the list
            bottom_laps_telemetry_list.extend(lap_records)
        
        #enrich data
        self._print_section_divider("ENRICHING CONTEXTUAL DATA")
        enriched_contextual_data = await self.enriched_contextual_data(bottom_laps_telemetry_list)

        #process enriched data, Generate training pairs with performance sections
        self._print_section_divider("GENERATING TRAINING PAIRS")
        comparison_results = imitation_learning.compare_telemetry_with_expert(enriched_contextual_data, 5, 5)
        training_and_expert_action = comparison_results.get('transformer_training_pairs', [])
        
        # train transformer model
        self._print_section_divider("TRAINING TRANSFORMER MODEL")
        transformer_results = await self._train_expert_action_transformer(
            training_and_expert_action=training_and_expert_action,
            trackName=trackName
        )
        
        self._print_section_divider("TRANSFORMER LEARNING COMPLETED")
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
            
            # Create dataset first to determine actual feature count
            temp_dataset = TelemetryActionDataset(
                telemetry_data=telemetry_data,
                expert_actions=expert_actions,
                sequence_length=50,
                prediction_horizon=20
            )
            
            if len(temp_dataset) == 0:
                return {
                    "success": False,
                    "error": "Dataset creation resulted in 0 sequences - check data compatibility"
                }
            
            # Get actual input features from processed data
            sample_input, _ = temp_dataset[0]
            input_features = sample_input.size(-1)  # Last dimension is feature dimension
            
            print(f"[INFO] Detected {input_features} input features from processed telemetry data")
            
            # Create transformer model and trainer with correct feature count
            model = ExpertActionTransformer(
                input_features=input_features,
                action_features=3,  # steering, throttle, brake
                d_model=256, # embedding dimension
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=1024,
                dropout=0.1,
                max_sequence_length=100
            )
            
            trainer = ExpertActionTrainer(model)
            
            print(f"[INFO] Created transformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Use the already created dataset
            # Use the already created dataset
            dataset = temp_dataset
            
            print(f"[INFO] Dataset contains {len(dataset)} sequences")
            
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
            
            transformer_model_data = trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=50,  # Adjust based on data size
                patience=10,
            )
            
            print(f"[INFO] Transformer training completed for {trackName}")
            
            # Save model metadata to backend
            try:
                
                ai_model_dto = {
                    "modelType": "expert_action_transformer",
                    "trackName": trackName,
                    "carName": 'AllCars',  # Track-specific model
                    "modelData": model.serialize_model(),
                    "metadata": {
                        "training_pairs": len(training_and_expert_action),
                        "dataset_sequences": len(dataset),
                        "training_timestamp": datetime.now().isoformat()
                    },
                    "isActive": True
                }
                
                await backend_service.save_ai_model(ai_model_dto)
                print(f"[INFO] Saved transformer model data to backend for track: {trackName}")
                
            except Exception as backend_error:
                raise RuntimeError(f"[WARNING] Failed to save transformer model to backend: {str(backend_error)}")
            
            return {
                "success": True,
                "training_results": transformer_model_data,
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
            raise ImportError(f"Failed to import transformer dependencies: {str(import_error)}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to train transformer model for {trackName}: {str(e)}")


    async def enriched_contextual_data(self, telemetry_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich telemetry data with contextual features using trained models
        
        Args:
            telemetry_list: List of telemetry record dictionaries (flat list)
            
        Returns:
            Enriched telemetry data with corner identification and tire grip features
        """
        print(f"[INFO] Starting contextual data enrichment for {len(telemetry_list)} telemetry records")
        
        if not telemetry_list:
            print("[WARNING] No telemetry data provided for enrichment")
            return []
        
        try:
            # Split telemetry records for training and feature extraction (70% training, 30% extraction)
            split_index = int(len(telemetry_list) * 0.7)
            training_telemetry_list = telemetry_list[:split_index]
            extraction_telemetry_list = telemetry_list[split_index:]
            
            print(f"[INFO] Split data: {len(training_telemetry_list)} records for training, {len(extraction_telemetry_list)} records for extraction")
            
            # Train corner identification model using training data
            self._print_section_divider("Training corner identification model...")
            try:
                corner_model = await self.corner_identification.learn_track_corner_patterns(training_telemetry_list)
                
                if corner_model.get("success"):
                    print(f"[INFO] Corner identification training successful: {corner_model.get('total_corners_identified', 0)} corners identified")
                else:
                    print(f"[WARNING] Corner identification training failed: {corner_model.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[ERROR] Corner identification training failed: {str(e)}")
                corner_model = {"success": False, "error": str(e)}
            
            # Train tire grip analysis model using training data
            self._print_section_divider("Training tire grip analysis model...")
            try:
                tire_grip_model = await self.tire_grip_analysis.train_tire_grip_model(training_telemetry_list)
                if tire_grip_model.get("success"):
                    print(f"[INFO] Tire grip model training successful: {tire_grip_model.get('models_trained', 0)} models trained")
                else:
                    print(f"[WARNING] Tire grip model training failed: {tire_grip_model.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[ERROR] Tire grip analysis training failed: {str(e)}")
                tire_grip_model = {"success": False, "error": str(e)}
            
            # Now extract features from extraction telemetry data
            enriched_telemetry_data = extraction_telemetry_list.copy()  # Start with original extraction data
            
            # Extract corner features for extraction telemetry
            if corner_model.get("success"):
                try:
                    self._print_section_divider("Enriching corner features for telemetry data...")
                    enriched_telemetry_data = await self.corner_identification.extract_corner_features_for_telemetry(
                        enriched_telemetry_data      
                    )
                    print(f"[INFO] Added corner features to {len(enriched_telemetry_data)} records")
                except Exception as e:
                    raise ValueError(f"[WARNING] Failed to extract corner features: {str(e)}")
            
            # Extract tire grip features for extraction telemetry
            if tire_grip_model.get("success"):
                try:
                    self._print_section_divider("Enriching tire grip features for telemetry data...")
                    enriched_telemetry_data = await self.tire_grip_analysis.extract_tire_grip_features(
                        enriched_telemetry_data
                    )
                    print(f"[INFO] Added tire grip features to {len(enriched_telemetry_data)} records")
                except Exception as e:
                    raise ValueError(f"[WARNING] Failed to extract tire grip features: {str(e)}")
            
            print(f"[INFO] Contextual data enrichment completed: {len(enriched_telemetry_data)} records processed")
            
            # Print enrichment summary
            enrichment_summary = {
                'total_records_processed': len(enriched_telemetry_data),
                'total_original_records': len(extraction_telemetry_list),
                'corner_identification_success': corner_model.get("success", False),
                'tire_grip_analysis_success': tire_grip_model.get("success", False),
                'training_records_used': len(training_telemetry_list),
                'extraction_records_processed': len(extraction_telemetry_list)
            }
            
            print(f"[INFO] Enrichment summary: {enrichment_summary}")
            
            return enriched_telemetry_data
            
        except Exception as e:
            print(f"[ERROR] Failed to enrich contextual data: {str(e)}")
            return telemetry_list  # Return original data on failure

    
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")