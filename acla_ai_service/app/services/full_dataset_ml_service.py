"""
Scikit-learn Machine Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service provides comprehensive AI model training and prediction capabilities
using your TelemetryFeatures and FeatureProcessor classes.
"""

import os
import pandas as pd

from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
from .tire_grip_analysis_service import TireGripAnalysisService
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
        # Cache frequently used feature lists to avoid recreating TelemetryFeatures each prediction
        try:
            self._imitate_expert_feature_names = self.telemetry_features.get_features_for_imitate_expert()
        except Exception:
            self._imitate_expert_feature_names = []
        
    # NOTE: Corner identification & tire grip analysis services are now created on-demand
    # during training or prediction to avoid holding stale state between different
    # transformer trainings or runtime sessions. Any previous persistent instances
    # have been removed (self.corner_identification / self.tire_grip_analysis).

        
        # Backend service integration
        self.backend_service = backend_service
        
        # Model cache service integration
        self.model_cache = model_cache_service
        
        # Add a simple lock mechanism to prevent concurrent fetches of the same model
        self._model_fetch_locks = {}
        self._lock_creation_lock = asyncio.Lock()

    def clear_all_cache(self):
        """Clear cached transformer / imitation models (corner & tire services now on-demand)."""
        self.model_cache.clear()
        print("[INFO] All cached model_cache entries cleared (corner & tire services are on-demand so no persistent cache to clear)")
    
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
            # On-demand instantiate corner identification service
            from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
            corner_service = CornerIdentificationUnsupervisedService()
            results = await corner_service.learn_track_corner_patterns(trackName, carName)
            
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
            # Fetch serialized corner model (first try car-specific then fallback to all_cars)
            model_payload = None
            metadata_used = None
            fetch_attempts = []
            for attempt_car in [carName, 'all_cars']:
                try:
                    model_payload, metadata_used = await self._get_cached_model_or_fetch(
                        model_type="corner_identification",
                        track_name=trackName,
                        car_name=attempt_car,
                        model_subtype="complete_model_data"
                    )
                    fetch_attempts.append(f"success:{attempt_car}")
                    if model_payload:
                        break
                except Exception as fetch_err:
                    fetch_attempts.append(f"fail:{attempt_car}:{fetch_err}")
                    continue

            if not model_payload:
                print(f"[WARNING] No corner identification model available for {trackName} ({fetch_attempts}); returning original telemetry")
                return telemetry_data

            # Deserialize model using service class
            from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
            try:
                corner_service = CornerIdentificationUnsupervisedService.deserialize_model(model_payload if isinstance(model_payload, dict) else model_payload.get('modelData', {}))
            except Exception as deser_err:
                print(f"[WARNING] Failed to deserialize corner model payload: {deser_err}; returning original telemetry")
                return telemetry_data

            # Extract features (geometry_only flag inside serialized payload)
            enhanced_telemetry = await corner_service.extract_corner_features_for_telemetry(
                telemetry_data,
                geometry_only=model_payload.get('geometry_only', True) if isinstance(model_payload, dict) else True
            )
            print(f"[INFO] Enhanced {len(enhanced_telemetry)} telemetry records with corner features (model fetch attempts: {fetch_attempts})")
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
        from .tire_grip_analysis_service import TireGripAnalysisService
        tire_service = TireGripAnalysisService()
        return await tire_service.train_tire_grip_model(trackName, carName)
    
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
        try:
            # Attempt to fetch serialized tire grip heuristic config (car-specific then fallback)
            model_payload = None
            metadata_used = None
            fetch_attempts = []
            for attempt_car in [carName, 'all_cars']:
                try:
                    model_payload, metadata_used = await self._get_cached_model_or_fetch(
                        model_type="tire_grip_analysis",
                        track_name=trackName,
                        car_name=attempt_car,
                        model_subtype="complete_model_data"
                    )
                    fetch_attempts.append(f"success:{attempt_car}")
                    if model_payload:
                        break
                except Exception as fetch_err:
                    fetch_attempts.append(f"fail:{attempt_car}:{fetch_err}")
                    continue

            from .tire_grip_analysis_service import TireGripAnalysisService
            if model_payload:
                try:
                    tire_service = TireGripAnalysisService.deserialize_model(model_payload if isinstance(model_payload, dict) else model_payload.get('modelData', {}))
                except Exception as deser_err:
                    print(f"[WARNING] Failed to deserialize tire grip model payload: {deser_err}; using fresh instance")
                    tire_service = TireGripAnalysisService()
            else:
                print(f"[INFO] No serialized tire grip artifact found ({fetch_attempts}); using fresh heuristic instance")
                tire_service = TireGripAnalysisService()

            enhanced = await tire_service.extract_tire_grip_features(telemetry_data, trackName, carName)
            print(f"[INFO] Enhanced {len(enhanced)} telemetry records with tire grip features (model fetch attempts: {fetch_attempts})")
            return enhanced
        except Exception as e:
            print(f"[ERROR] Failed to extract tire grip features: {str(e)}")
            return telemetry_data
    
    
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
                                   carName: Optional[str] = 'AllCars',
                                   sequence_length: int = 20,
                                   temperature: float = 1.0,
                                   deterministic: bool = False,
                                   clamp_actions: bool = True) -> Dict[str, Any]:
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
            from time import perf_counter
            t_start_total = perf_counter()
            timings = {}
            def _mark(name, start):
                timings[name] = round((perf_counter() - start)*1000.0, 3)
            t0 = perf_counter()
            # Fetch and deserialize the transformer model
            print(f"[INFO] Fetching transformer model for {trackName}/{carName or 'any'}")
            transformer_model, model_metadata = await self._get_cached_model_or_fetch(
                model_type="expert_action_transformer",
                track_name=trackName,
                car_name=carName,
                model_subtype="transformer_model_data",
                deserializer_func=self._deserialize_transformer_model
            )
            _mark('model_fetch_ms', t0)
            # Specify the type for transformer_model (for type hints and IDEs)
            transformer_model: ExpertActionTransformer
            
            # Prepare telemetry data for the model using the same feature filtering as training
            # Use cached feature names (fallback to dynamic if empty)
            feature_names = self._imitate_expert_feature_names or TelemetryFeatures().get_features_for_imitate_expert()
            
            print(f"[INFO] Using {len(feature_names)} features for prediction (same as training)")
            print(f"[INFO] Expected input features: {len(feature_names)}")
            
            # Extract relevant features from the telemetry dictionary in the same order as training
            t_feat = perf_counter()
            missing_features = [fn for fn in feature_names if fn not in telemetry_dict]
            def _coerce_float(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return 0.0
            feature_values = [_coerce_float(telemetry_dict.get(fn, 0.0)) for fn in feature_names]
            
            if missing_features:
                print(f"[WARNING] {len(missing_features)} features missing from input, using defaults: {missing_features[:5]}...")
            _mark('feature_vector_build_ms', t_feat)
            
            # Convert to tensor format expected by the transformer
            # Shape: [seq_len=1, batch_size=1, input_features]
            src_telemetry = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

            # ================= Inference-time enrichment for corner + tire grip =================
            t_enrich = perf_counter()
            enriched_context_source = dict(telemetry_dict)
            corner_enrichment_applied = False
            tire_enrichment_applied = False
            enrichment_errors: List[str] = []
            pre_context_keys = set(enriched_context_source.keys())
            if transformer_model.context_features and getattr(transformer_model, 'context_feature_names', []):
                try:
                    # Run corner and tire enrichment concurrently
                    import asyncio as _asyncio
                    corner_task = self.extract_corner_features_for_telemetry([enriched_context_source], trackName, carName)
                    tire_task = self.extract_tire_grip_features([enriched_context_source], trackName, carName)
                    corner_result, tire_result = await _asyncio.gather(corner_task, tire_task, return_exceptions=True)
                    if isinstance(corner_result, list) and corner_result:
                        enriched_context_source.update(corner_result[0])
                        corner_enrichment_applied = True
                    elif isinstance(corner_result, Exception):
                        enrichment_errors.append(f"corner:{corner_result}")
                    if isinstance(tire_result, list) and tire_result:
                        enriched_context_source.update(tire_result[0])
                        tire_enrichment_applied = True
                    elif isinstance(tire_result, Exception):
                        enrichment_errors.append(f"tire:{tire_result}")
                except Exception as enrich_err:
                    enrichment_errors.append(f"parallel_enrich:{enrich_err}")
                if enrichment_errors:
                    print(f"[WARNING] Inference enrichment partial failures: {enrichment_errors}")
            post_context_keys = set(enriched_context_source.keys())
            added_context_feature_count = len(post_context_keys - pre_context_keys)
            _mark('context_enrichment_ms', t_enrich)
            # ===============================================================================

            # Attempt to reconstruct context features if model expects them
            src_context = None
            ctx_build_start = perf_counter()
            try:
                model_cfg = getattr(transformer_model, 'context_feature_names', [])
                context_feature_names = model_cfg if isinstance(model_cfg, list) else []
                if transformer_model.context_features and transformer_model.context_features > 0 and context_feature_names:
                    # Retrieve scaler stats from cached metadata if available
                    cached = self.model_cache.get(
                        model_type="expert_action_transformer",
                        track_name=trackName,
                        car_name=carName,
                        model_subtype="transformer_model_data"
                    )
                    telemetry_context_scaler = None
                    if cached:
                        _, cached_meta = cached
                        scalers_meta = cached_meta.get('metadata', {}).get('scalers') if cached_meta.get('metadata') else None
                        if scalers_meta and scalers_meta.get('context_scaler'):
                            telemetry_context_scaler = scalers_meta['context_scaler']
                    context_values = []
                    missing_ctx = []
                    for cname in context_feature_names:
                        # Prefer enriched source (which may now include corner/grip features)
                        val = enriched_context_source.get(cname, 0.0)
                        if cname not in enriched_context_source:
                            missing_ctx.append(cname)
                        try:
                            context_values.append(float(val))
                        except (ValueError, TypeError):
                            context_values.append(0.0)
                    ctx_tensor = torch.from_numpy(np.array(context_values, dtype=np.float32)) if 'np' in globals() else torch.tensor(context_values, dtype=torch.float32)
                    # Apply scaling if stats available
                    if telemetry_context_scaler and telemetry_context_scaler.get('mean') and telemetry_context_scaler.get('scale'):
                        try:
                            import numpy as np
                            mean = np.array(telemetry_context_scaler['mean'])
                            scale = np.array(telemetry_context_scaler['scale'])
                            if mean.shape[0] == ctx_tensor.numel():
                                ctx_tensor = (ctx_tensor - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(scale, dtype=torch.float32)
                        except Exception as scale_err:
                            print(f"[WARNING] Failed to scale context features: {scale_err}")
                    src_context = ctx_tensor.unsqueeze(0).unsqueeze(1)  # [1,1,ctx_features]
                    if missing_ctx:
                        print(f"[INFO] Missing {len(missing_ctx)} context features (filled with 0): {missing_ctx[:5]}{'...' if len(missing_ctx)>5 else ''}")
                elif transformer_model.context_features == 0:
                    # No context path: skip building/scaling entirely
                    src_context = None
            except Exception as ctx_err:
                print(f"[WARNING] Context reconstruction failed, proceeding without context: {ctx_err}")
            _mark('context_tensor_build_ms', ctx_build_start)
            
            print(f"[INFO] Input telemetry tensor shape: {src_telemetry.shape}")
            print(f"[INFO] Model expects input_features: {transformer_model.input_features}")
            
            # Verify feature count consistency
            if len(feature_values) != transformer_model.input_features:
                print(f"[ERROR] Feature count mismatch! Expected: {transformer_model.input_features}, Got: {len(feature_values)}")
                return {
                    "error": f"Feature count mismatch: model expects {transformer_model.input_features} features, got {len(feature_values)}",
                    "predicted_actions": [],
                    "performance_scores": [],
                    "metadata": {}
                }
            
            print(f"[INFO] Predicting {sequence_length} expert actions...")
            t_infer = perf_counter()
            
            # Make prediction using the transformer model and also generate human-readable steps
            with torch.no_grad():
                predicted_sequence, predicted_reasoning, performance_sequence = transformer_model.predict_expert_sequence(
                    src_telemetry=src_telemetry,
                    sequence_length=sequence_length,
                    temperature=(0.01 if deterministic else temperature),
                    src_context=src_context
                )
                # Generate expert plan steps (use same context)
                instructions = transformer_model.generate_expert_action_instructions(
                    src_telemetry=src_telemetry,
                    sequence_length=sequence_length,
                    temperature=(0.01 if deterministic else temperature),
                    src_context=src_context
                )
            _mark('inference_ms', t_infer)
            
            # Convert predictions back to lists for JSON serialization
            predicted_actions = predicted_sequence.squeeze(1).tolist()  # Remove batch dimension
            predictions_are_deltas_flag = bool(instructions.get('metadata', {}).get('predictions_are_deltas', False))
            if clamp_actions and not predictions_are_deltas_flag:
                # Simple safety clamp: steering [-1,1], throttle/brake [0,1] if shapes align (assumes 3 dims)
                clamped = []
                for step in predicted_actions:
                    if isinstance(step, list) and len(step) >= 3:
                        steer = max(-1.0, min(1.0, step[0]))
                        throttle = max(0.0, min(1.0, step[1]))
                        brake = max(0.0, min(1.0, step[2]))
                        clamped.append([steer, throttle, brake] + step[3:])
                    else:
                        clamped.append(step)
                predicted_actions = clamped
            predicted_reasoning_features = predicted_reasoning.squeeze(1).tolist()  # Remove batch dimension: [seq_len, batch_size, 43] -> [seq_len, 43]
            
            # Fix: For performance_sequence [seq_len, batch_size, 1], squeeze both batch and feature dimensions
            performance_scores = performance_sequence.squeeze(1).squeeze(-1).tolist()  # [seq_len, batch_size, 1] -> [seq_len, 1] -> [seq_len]
            
            # Create interpretable reasoning labels
            reasoning_labels = self._create_reasoning_labels(predicted_reasoning_features)
            
            # Calculate metadata
            avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
            prediction_confidence = max(0.0, min(1.0, (avg_performance + 1.0) / 2.0))  # Normalize to 0-1 range
            
            print(f"[INFO] Successfully predicted {len(predicted_actions)} expert actions")
            print(f"[INFO] Average predicted performance: {avg_performance:.4f}")
            print(f"[INFO] Prediction confidence: {prediction_confidence:.2%}")
            _mark('post_processing_ms', t_infer)
            _mark('total_ms', t_start_total)
            
            return {
                "success": True,
                "predicted_actions": predicted_actions,
                "predicted_reasoning": predicted_reasoning_features,
                "reasoning_explanations": reasoning_labels,
                "performance_scores": performance_scores,
                "expert_action_steps": instructions.get('steps', []),
                "expert_action_text": instructions.get('text_instructions', []),
                "recommended_actions": instructions.get('recommended_actions', []),
                "metadata": {
                    "track_name": trackName,
                    "car_name": carName,
                    "sequence_length": sequence_length,
                    "temperature": temperature,
                    "input_features_count": len(feature_values),
                    "expected_input_features": transformer_model.input_features,
                    "feature_filtering": "get_features_for_imitate_expert()",
                    "context_features_expected": transformer_model.context_features,
                    "context_features_provided": int(src_context.size(-1)) if src_context is not None else 0,
                    "context_applied": bool(src_context is not None and src_context.size(-1) == transformer_model.context_features),
                    "context_missing_count": (
                        (transformer_model.context_features - int(src_context.size(-1))) if src_context is not None else transformer_model.context_features
                    ) if transformer_model.context_features else 0,
                    "corner_enrichment_applied": corner_enrichment_applied,
                    "tire_grip_enrichment_applied": tire_enrichment_applied,
                    "enrichment_error_count": len(enrichment_errors),
                    "missing_features_count": len(missing_features),
                    "reasoning_features_count": len(predicted_reasoning_features[0]) if predicted_reasoning_features else 0,
                    "avg_predicted_performance": avg_performance,
                    "prediction_confidence": prediction_confidence,
                    "model_metadata": model_metadata,
                    "prediction_timestamp": datetime.now().isoformat(),
                    "timings_ms": timings,
                    "added_context_feature_count": added_context_feature_count,
                    "deterministic": deterministic,
                    "clamp_actions": clamp_actions,
                    "instructions_meta": instructions.get('metadata', {})
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to predict expert actions: \n\t{str(e)}"
            raise Exception(error_msg)
    
    def _create_reasoning_labels(self, predicted_reasoning_features: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Create interpretable labels for predicted reasoning features
        
        Args:
            predicted_reasoning_features: List of reasoning feature vectors for each prediction step
            
        Returns:
            List of dictionaries containing interpretable reasoning explanations
        """
        reasoning_explanations = []
        
        for step_idx, features in enumerate(predicted_reasoning_features):
            explanation = {
                "step": step_idx + 1,
                "reasoning": {}
            }
            
            if len(features) >= 12:  # Basic reasoning features
                # Speed-based reasoning (features 0-2)
                speed_norm = features[0]
                is_slow_corner = features[1] > 0.5
                is_high_speed = features[2] > 0.5
                
                explanation["reasoning"]["speed_context"] = {
                    "normalized_speed": round(speed_norm, 3),
                    "is_slow_corner": is_slow_corner,
                    "is_high_speed_section": is_high_speed,
                    "interpretation": (
                        "Approaching slow corner - prepare for heavy braking" if is_slow_corner
                        else "High speed section - maintain momentum" if is_high_speed
                        else "Medium speed section - balanced approach"
                    )
                }
                
                # G-force based reasoning (features 3-5)
                g_lat_norm = features[3]
                g_long_norm = features[4]
                high_cornering = features[5] > 0.5
                
                explanation["reasoning"]["cornering_dynamics"] = {
                    "lateral_g_load": round(g_lat_norm, 3),
                    "longitudinal_g_load": round(g_long_norm, 3),
                    "high_cornering_detected": high_cornering,
                    "interpretation": (
                        "High lateral forces - approaching or in corner" if high_cornering
                        else "Moderate cornering forces" if g_lat_norm > 0.3
                        else "Straight line or gentle curve"
                    )
                }
                
                # Steering reasoning (features 6-7)
                steer_norm = features[6]
                sharp_turn = features[7] > 0.5
                
                explanation["reasoning"]["steering_input"] = {
                    "steering_demand": round(steer_norm, 3),
                    "sharp_turn_detected": sharp_turn,
                    "interpretation": (
                        "Sharp steering input required" if sharp_turn
                        else "Moderate steering correction" if steer_norm > 0.3
                        else "Minimal steering input needed"
                    )
                }
                
                # Brake/throttle reasoning (features 8-11)
                brake_level = features[8]
                throttle_level = features[9]
                heavy_braking = features[10] > 0.5
                full_throttle = features[11] > 0.5
                
                explanation["reasoning"]["pedal_strategy"] = {
                    "brake_demand": round(brake_level, 3),
                    "throttle_demand": round(throttle_level, 3),
                    "heavy_braking_zone": heavy_braking,
                    "full_acceleration_zone": full_throttle,
                    "interpretation": (
                        "Heavy braking required - corner entry" if heavy_braking
                        else "Full acceleration - corner exit or straight" if full_throttle
                        else "Trail braking or throttle modulation" if brake_level > 0.2 and throttle_level > 0.2
                        else "Coasting or maintenance throttle"
                    )
                }
            
            # Additional enriched features interpretation (if available)
            if len(features) > 12:
                explanation["reasoning"]["additional_context"] = {
                    "enriched_features_detected": True,
                    "corner_identification": "Available" if len(features) > 20 else "Limited",
                    "tire_grip_analysis": "Available" if len(features) > 25 else "Limited",
                    "interpretation": "Enhanced contextual analysis from corner and grip models"
                }
            
            reasoning_explanations.append(explanation)
        
        return reasoning_explanations  


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
        print(f"[INFO] Filtering to {len(relevant_features)} features for training using get_features_for_imitate_expert()")
        processed_df = feature_processor.filter_features_by_list(processed_df, relevant_features)
        print(f"[INFO] DataFrame after filtering has {processed_df.shape[1]} columns")
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
        enrichment_result = await self.enriched_contextual_data(bottom_laps_telemetry_list)
        
        # Handle the new structure
        if enrichment_result and enrichment_result.get("enriched_features"):
            enriched_contextual_data = enrichment_result["original_telemetry"]  # Use original data for comparison
            enrichment_count = len(enrichment_result["enriched_features"])
            feature_metadata = enrichment_result["feature_metadata"]
            print(f"[INFO] Enrichment successful: {enrichment_count} records with {feature_metadata.get('feature_count', 0)} enriched features")
        else:
            enriched_contextual_data = bottom_laps_telemetry_list  # Fallback to original data
            enrichment_count = 0
            print("[WARNING] Enrichment failed or returned no features, using original data")

        #process enriched data, Generate training pairs with performance sections
        self._print_section_divider("GENERATING TRAINING PAIRS")
        comparison_results = imitation_learning.compare_telemetry_with_expert(enriched_contextual_data, 5, 5)
        training_and_expert_action = comparison_results.get('transformer_training_pairs', [])
        
        # train transformer model
        self._print_section_divider("TRAINING TRANSFORMER MODEL")
        transformer_results = await self._train_expert_action_transformer(
            training_and_expert_action=training_and_expert_action,
            trackName=trackName,
            enrichment_result=enrichment_result  # Pass the already computed enrichment
        )
        
        self._print_section_divider("TRANSFORMER LEARNING COMPLETED")
        return {
            "success": True,
            "transformer_training": transformer_results,
            "expert_imitation_trained": True,
            "contextual_data_enriched": enrichment_count,
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
                                             trackName: str,
                                             enrichment_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the transformer model to predict expert actions from current telemetry
        
        Args:
            training_and_expert_action: List of training pairs with telemetry sections and expert actions
            trackName: Track name for model identification
            enrichment_result: Pre-computed enrichment result to avoid re-training models
            
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

            if not enrichment_result or not enrichment_result.get("enriched_features"):
                raise ValueError("[Error] No enriched contextual data generated, proceeding without enriched features")
            else:
                # Extract the separated data
                original_telemetry = enrichment_result["original_telemetry"]
                enriched_contextual_data = enrichment_result["enriched_features"]
                feature_metadata = enrichment_result["feature_metadata"]

                print(f"[INFO] Generated {len(enriched_contextual_data)} enriched contextual records (forced retrain)")
                print(f"[INFO] Feature sources (forced): {feature_metadata.get('sources', [])}")
                print(f"[INFO] Total enriched features (forced): {feature_metadata.get('feature_count', 0)}")

                # Ensure matching lengths. They’re parallel sequences that must line up 1:1 per timestep
                min_len = min(len(telemetry_data), len(enriched_contextual_data))
                telemetry_data = telemetry_data[:min_len]
                expert_actions = expert_actions[:min_len]
                enriched_contextual_data = enriched_contextual_data[:min_len]
            
            # Create dataset with enriched contextual features.
            # Enriched data is split internally into context (encoder exogenous inputs) and reasoning (aux targets).
            temp_dataset = TelemetryActionDataset(
                telemetry_data=telemetry_data,
                expert_actions=expert_actions,
                enriched_contextual_data=enriched_contextual_data,
                sequence_length=50,
                prediction_horizon=20,
                predict_action_deltas=True,
            )
            
            if len(temp_dataset) == 0:
                return {
                    "success": False,
                    "error": "Dataset creation resulted in 0 sequences - check data compatibility"
                }
            
            # Get actual input features from processed data
            sample_input, sample_actions, sample_reasoning, sample_context = temp_dataset[0]
            input_features = sample_input.size(-1)
            reasoning_features = sample_reasoning.size(-1)
            context_feature_count = sample_context.size(-1) if sample_context is not None and sample_context.numel() > 0 else 0

            context_feature_names = temp_dataset.context_feature_names if hasattr(temp_dataset, 'context_feature_names') else []
            reasoning_feature_names = temp_dataset.reasoning_feature_names if hasattr(temp_dataset, 'reasoning_feature_names') else []

            print(f"[INFO] Context feature count: {context_feature_count}")
            print(f"[INFO] First context features: {context_feature_names[:5]}" + (f" ... and {len(context_feature_names)-5} more" if len(context_feature_names) > 5 else ""))
            print(f"[INFO] Reasoning feature count: {reasoning_features}")
            
            print(f"[INFO] Detected {input_features} input features from processed telemetry data")
            print(f"[INFO] Detected {reasoning_features} reasoning features from enriched contextual data")
            
            # Create transformer model and trainer with correct feature counts
            model = ExpertActionTransformer(
                input_features=input_features,
                action_features=3,
                reasoning_features=reasoning_features,
                context_features=context_feature_count,
                context_feature_names=context_feature_names,
                context_fusion="gate",  # default fusion strategy; could be parameterized
                d_model=256,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=1024,
                dropout=0.1,
                max_sequence_length=256
            )
            setattr(model, 'targets_are_deltas', True)
            
            trainer = ExpertActionTrainer(model)
            
            print(f"[INFO] Created transformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
            
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
                
                # Export scalers for telemetry and context to allow reconstruction at inference
                scaler_payload = temp_dataset.export_scalers() if hasattr(temp_dataset, 'export_scalers') else {}
                ai_model_dto = {
                    "modelType": "expert_action_transformer",
                    "trackName": trackName,
                    "carName": 'AllCars',  # Track-specific model
                    "modelData": model.serialize_model(),
                    "metadata": {
                        "training_pairs": len(training_and_expert_action),
                        "dataset_sequences": len(dataset),
                        "training_timestamp": datetime.now().isoformat()
                        ,
                        "context_feature_names": context_feature_names,
                        "reasoning_feature_names": reasoning_feature_names,
                        "context_fusion": "gate",
                        "scalers": scaler_payload
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
                    "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
                    "targets_are_deltas": True
                }
            }
            
        except ImportError as import_error:
            raise ImportError(f"Failed to import transformer dependencies: {str(import_error)}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to train transformer model for {trackName}: {str(e)}")


    async def enriched_contextual_data(self, telemetry_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract enriched contextual features from telemetry data using trained models
        
        This method keeps original telemetry separate from enriched features for clean training.
        
        Args:
            telemetry_list: List of telemetry record dictionaries (flat list)
            
        Returns:
            Dictionary containing:
            - original_telemetry: Original telemetry data (unchanged)
            - enriched_features: List of enriched feature dictionaries
            - feature_metadata: Information about feature sources and types
        """
        print(f"[INFO] Starting contextual data enrichment for {len(telemetry_list)} telemetry records")
        
        if not telemetry_list:
            print("[WARNING] No telemetry data provided for enrichment")
            return {
                "original_telemetry": [],
                "enriched_features": [],
                "feature_metadata": {"sources": [], "feature_count": 0}
            }
        
        try:
            # Use all telemetry data for both training enrichment models and feature extraction
            # This ensures consistency and maximizes available data
            training_telemetry_list = telemetry_list.copy()
            
            print(f"[INFO] Using all {len(training_telemetry_list)} records for enrichment model training and feature extraction")
            
            # Train corner identification model using training data
            self._print_section_divider("Training corner identification model...")
            try:
                from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
                corner_service = CornerIdentificationUnsupervisedService()
                corner_model = await corner_service.learn_track_corner_patterns(training_telemetry_list)
                
                if corner_model.get("success"):
                    print(f"[INFO] Corner identification training successful: {corner_model.get('total_corners_identified', 0)} corners identified")
                    # Serialize and persist corner model artifact
                    try:
                        corner_serialized = corner_service.serialize_model(track_name="generic", car_name="all_cars")
                        corner_model_dto = {
                            "modelType": "corner_identification",
                            "trackName": corner_serialized.get("track_name"),
                            "carName": corner_serialized.get("car_name"),
                            "modelData": corner_serialized,
                            "metadata": {
                                "total_corners": corner_serialized.get("total_corners"),
                                "clusters": len(corner_serialized.get("corner_clusters", [])),
                                "serialization_timestamp": corner_serialized.get("serialized_timestamp")
                            },
                            "isActive": True
                        }
                        await self.backend_service.save_ai_model(corner_model_dto)
                        print("[INFO] Saved serialized corner identification model to backend")
                    except Exception as ser_err:
                        print(f"[WARNING] Failed to serialize/save corner model: {ser_err}")
                else:
                    print(f"[WARNING] Corner identification training failed: {corner_model.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[ERROR] Corner identification training failed: {str(e)}")
                corner_model = {"success": False, "error": str(e)}
            
            # Train tire grip analysis model using training data
            self._print_section_divider("Training tire grip analysis model...")
            try:
                from .tire_grip_analysis_service import TireGripAnalysisService
                tire_service = TireGripAnalysisService()
                tire_grip_model = await tire_service.train_tire_grip_model(training_telemetry_list)
                if tire_grip_model.get("success"):
                    print(f"[INFO] Tire grip model training successful: {tire_grip_model.get('models_trained', 0)} models trained")
                    # Serialize and persist tire grip heuristic configuration
                    try:
                        # We don't yet have generated feature names at this stage; will infer later
                        tire_serialized = tire_service.serialize_model(track_name="generic", car_name="all_cars")
                        tire_model_dto = {
                            "modelType": "tire_grip_analysis",
                            "trackName": tire_serialized.get("track_name"),
                            "carName": tire_serialized.get("car_name"),
                            "modelData": tire_serialized,
                            "metadata": {
                                "heuristic_only": tire_serialized.get("heuristic_only"),
                                "exclude_instantaneous_exploitation": tire_serialized.get("exclude_instantaneous_exploitation"),
                                "serialization_timestamp": tire_serialized.get("serialized_timestamp")
                            },
                            "isActive": True
                        }
                        await self.backend_service.save_ai_model(tire_model_dto)
                        print("[INFO] Saved serialized tire grip analysis model to backend")
                    except Exception as ser_err:
                        print(f"[WARNING] Failed to serialize/save tire grip model: {ser_err}")
                else:
                    print(f"[WARNING] Tire grip model training failed: {tire_grip_model.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[ERROR] Tire grip analysis training failed: {str(e)}")
                tire_grip_model = {"success": False, "error": str(e)}
            
            # Now extract enriched features separately (don't mix back into original data)
            enriched_features_data = []
            feature_sources = []
            
            # Initialize with empty enriched features for each telemetry record
            for i in range(len(training_telemetry_list)):
                enriched_features_data.append({})
            
            # Extract corner features as separate enriched data
            if corner_model.get("success"):
                try:
                    self._print_section_divider("Extracting corner features...")
                    corner_enriched_data = await self._extract_corner_features_only(training_telemetry_list, corner_service)
                    
                    # Add corner features to enriched features data
                    for i, corner_features in enumerate(corner_enriched_data):
                        if i < len(enriched_features_data):
                            enriched_features_data[i].update(corner_features)
                    
                    feature_sources.append("corner_identification")
                    print(f"[INFO] Extracted corner features for {len(corner_enriched_data)} records")
                except Exception as e:
                    print(f"[WARNING] Failed to extract corner features: {str(e)}")
            
            # Extract tire grip features as separate enriched data
            if tire_grip_model.get("success"):
                try:
                    self._print_section_divider("Extracting tire grip features...")
                    grip_enriched_data = await self._extract_tire_grip_features_only(training_telemetry_list, tire_service)
                    
                    # Add tire grip features to enriched features data
                    for i, grip_features in enumerate(grip_enriched_data):
                        if i < len(enriched_features_data):
                            enriched_features_data[i].update(grip_features)
                    
                    feature_sources.append("tire_grip_analysis")
                    print(f"[INFO] Extracted tire grip features for {len(grip_enriched_data)} records")
                except Exception as e:
                    print(f"[WARNING] Failed to extract tire grip features: {str(e)}")
            
            print(f"[INFO] Contextual data enrichment completed: {len(enriched_features_data)} enriched feature records created")
            
            # Create feature metadata
            sample_enriched = enriched_features_data[0] if enriched_features_data else {}
            feature_metadata = {
                'sources': feature_sources,
                'feature_count': len(sample_enriched),
                'feature_names': list(sample_enriched.keys()),
                'corner_features': [k for k in sample_enriched.keys() if 'corner' in k.lower()],
                'grip_features': [k for k in sample_enriched.keys() if 'grip' in k.lower() or 'friction' in k.lower()],
                'total_original_records': len(training_telemetry_list),
                'corner_identification_success': corner_model.get("success", False),
                'tire_grip_analysis_success': tire_grip_model.get("success", False)
            }
            
            print(f"[INFO] Feature metadata: {feature_metadata['feature_count']} enriched features from {len(feature_sources)} sources")
            
            return {
                "original_telemetry": training_telemetry_list,  # Keep original data unchanged
                "enriched_features": enriched_features_data,    # Separate enriched features
                "feature_metadata": feature_metadata           # Metadata about features
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to enrich contextual data: {str(e)}")
            return {
                "original_telemetry": telemetry_list,  # Return original data on failure
                "enriched_features": [],
                "feature_metadata": {"sources": [], "feature_count": 0, "error": str(e)}
            }

    async def _extract_corner_features_only(self, telemetry_data: List[Dict[str, Any]], corner_service: CornerIdentificationUnsupervisedService) -> List[Dict[str, float]]:
        """
        Extract ONLY corner-related features without mixing them back into telemetry
        
        Args:
            telemetry_data: List of telemetry records
            
        Returns:
            List of dictionaries containing only corner-related features
        """
        try:
            # Use the corner identification service to extract features
            enhanced_data = await corner_service.extract_corner_features_for_telemetry(telemetry_data)
            
            # Extract only the corner-related features (not original telemetry)
            corner_features_only = []
            
            for enhanced_record in enhanced_data:
                corner_features = {}
                
                # Extract only features that are corner-related
                for key, value in enhanced_record.items():
                    if any(corner_keyword in key.lower() for corner_keyword in [
                        'corner', 'entry', 'apex', 'exit', 'curvature', 'steering', 
                        'phase', 'severity', 'type', 'direction', 'racing_line'
                    ]):
                        try:
                            corner_features[key] = float(value) if value is not None else 0.0
                        except (ValueError, TypeError):
                            corner_features[key] = 0.0
                
                corner_features_only.append(corner_features)
            
            return corner_features_only
            
        except Exception as e:
            print(f"[ERROR] Failed to extract corner features only: {str(e)}")
            # Return empty feature dictionaries
            return [{}] * len(telemetry_data)

    async def _extract_tire_grip_features_only(self, telemetry_data: List[Dict[str, Any]], tire_service: TireGripAnalysisService) -> List[Dict[str, float]]:
        """
        Extract ONLY tire grip-related features without mixing them back into telemetry
        
        Args:
            telemetry_data: List of telemetry records
            
        Returns:
            List of dictionaries containing only tire grip-related features
        """
        try:
            # Use the tire grip analysis service to extract features
            enhanced_data = await tire_service.extract_tire_grip_features(telemetry_data)
            
            # Extract only the tire grip-related features (not original telemetry)
            grip_features_only = []
            
            for enhanced_record in enhanced_data:
                grip_features = {}
                
                # Extract only features that are tire grip-related
                for key, value in enhanced_record.items():
                    if any(grip_keyword in key.lower() for grip_keyword in [
                        'friction', 'grip', 'utilization', 'weight_transfer', 'load', 
                        'tire', 'saturation', 'degradation', 'slip_efficiency', 'optimal'
                    ]):
                        try:
                            grip_features[key] = float(value) if value is not None else 0.0
                        except (ValueError, TypeError):
                            grip_features[key] = 0.0
                
                grip_features_only.append(grip_features)
            
            return grip_features_only
            
        except Exception as e:
            print(f"[ERROR] Failed to extract tire grip features only: {str(e)}")
            # Return empty feature dictionaries
            return [{}] * len(telemetry_data)

    
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")