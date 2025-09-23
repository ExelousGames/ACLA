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
import os

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
    import torch.utils.data
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
            # Some deserializers only mutate internal state and return None.
            # Treat a None return as success and keep the original serialized data
            # so callers and cache have a non-None payload.
            _ret = deserializer_func(model_data)
            deserialized_model_data = _ret if _ret is not None else model_data
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
                print(f"[INFO] Released fetch lock for {model_key}")
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
                    print(f"[INFO] Successfully fetched and cached model for {model_key}")
                    
                except Exception as fetch_error:
                    print(f"[ERROR] Failed to fetch model: {str(fetch_error)}")
                    raise fetch_error
                finally:
                    # Always signal completion and clean up lock when we're the fetching thread
                    self._cleanup_fetch_lock(model_key, track_name or 'any', car_name or 'any')
            
            # At this point, we should have the model data
            # Use explicit None check to allow empty dicts/lists as valid payloads
            if deserialized_model_data is None:
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
        
        print("\n" + border_line + "\n" + empty_line + "\n" + title_line + "\n" + empty_line + "\n" + border_line + "\n", flush=True)
        
    
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
                if self.model_cache.get(model_type=model_type, track_name=track_name, car_name=car_name):
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
    
    
    
    async def predict_expert_actions(self, 
                                   telemetry_dict: Dict[str, Any],
                                   trackName: str, 
                                   carName: Optional[str] = 'AllCars',
                                   sequence_length: int = 10) -> Dict[str, Any]:
        """
        Fetch transformer model and predict expert actions from telemetry data
        
        Args:
            telemetry_dict: Single telemetry data record as dictionary
            trackName: Track name for model selection
            carName: Optional car name for model selection
            sequence_length: Length of action sequence to predict (default: 10)
            
        Returns:
            Dictionary with human-readable predictions directly from transformer model
        """
        if not TORCH_AVAILABLE:
            return {
                "status": "error",
                "error_message": "PyTorch is not available - transformer functionality is disabled",
                "error_type": "ImportError"
            }
        
        try:
            print(f"[INFO] Fetching transformer model for {trackName}/{carName or 'any'}")
            
            # Initialize transformer model
            transformer_model = ExpertActionTransformer()
            
            # Fetch and load the trained model
            transformer_model_data, model_metadata = await self._get_cached_model_or_fetch(
                model_type="transformer_expert_action",
                track_name=trackName,
                car_name=carName,
                model_subtype="transformer_model_data",
                deserializer_func=transformer_model.deserialize_transformer_model
            )
            
            # Extract context data by running corner and tire grip analysis
            context_data = []
            
            # Get corner identification features
            try:
                corner_service = CornerIdentificationUnsupervisedService()
                corner_service_data, corner_metadata = await self._get_cached_model_or_fetch(
                    model_type="corner_identification",
                    track_name=trackName,
                    car_name='all_cars',
                    model_subtype="corner_model_data",
                    deserializer_func=corner_service.deserialize_corner_identification_model
                )
                
                # Extract corner features from telemetry
                # Service is async and returns List[Dict[str, Any]]
                # Use the single telemetry record as a list for extraction
                corner_features_list = await corner_service.extract_corner_features_for_telemetry([telemetry_dict])
                context_data.append(corner_features_list)
                
            except Exception as e:
                print(f"[Error] Corner identification failed: {e}")
                context_data['corner_info'] = {}
            
            # Get tire grip features
            try:
                tire_grip_service = TireGripAnalysisService()
                tire_grip_data, tire_metadata = await self._get_cached_model_or_fetch(
                    model_type="tire_grip_analysis",
                    track_name='generic',
                    car_name='all_cars',
                    model_subtype="tire_grip_model_data",
                    deserializer_func=tire_grip_service.deserialize_tire_grip_model
                )
                
                # Extract tire grip features from telemetry (async; returns List[Dict])
                tire_features_list = await tire_grip_service.extract_tire_grip_features([telemetry_dict])
                context_data.append(tire_features_list)
                
            except Exception as e:
                print(f"[Error] Tire grip analysis failed: {e}")
            
            try:
                expert_service = ExpertImitateLearningService()
                imitation_model_data, imitation_metadata = await self._get_cached_model_or_fetch(
                    model_type="imitation_learning",
                    track_name=trackName,
                    car_name=carName,
                    model_subtype="imitation_model_data",
                    deserializer_func=expert_service.deserialize_imitation_model
                )
                context_data.append(expert_service.extract_expert_state_for_telemetry([telemetry_dict]))
            except Exception as e:
                print(f"[Error] Tireexpert_service failed: {e}")   
                
            # Use transformer model's predict_human_readable method
            print(f"[INFO] Generating predictions with sequence length: {sequence_length}")
            predictions = transformer_model.predict_human_readable(
                current_telemetry=telemetry_dict,
                context_data=context_data,
                sequence_length=sequence_length
            )
            
            return predictions
            
        except Exception as e:
            error_msg = f"Failed to predict expert actions: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                "status": "error",
                "error_message": error_msg,
                "error_type": type(e).__name__
            }

    async def StartImitateExpertPipeline(self, trackName: str):
        
        """
        returns: success, transformer_training, expert_imitation_trained    , contextual_data_enriched, comparison_results, track_name
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

        # Filter top 1% laps as expert demonstrations, but ensure at least 3 laps
        top_laps_df_count = max(3, int(len(lap_df_list) * 0.01))
        top_laps_df = lap_df_list[:top_laps_df_count]
        # get rest of laps
        
        bottom_laps_df = lap_df_list[top_laps_df_count:]

        # Console plot summary for top laps using FeatureProcessor.plot_features_console
        try:
            self._print_section_divider("TOP LAP TELEMETRY SUMMARY PLOTS")
            if top_laps_df:
                combined_top_df = pd.concat(top_laps_df, ignore_index=True)
                # Use a small, informative default feature set and filter to available columns
                features_to_plot = [
                    "Physics_speed_kmh",
                    "Physics_gas",
                    "Physics_brake",
                    "Physics_steer_angle",
                    "Physics_gear",
                    "Physics_rpm",
                    "Physics_velocity_x",
                ]
                available_features = [f for f in features_to_plot if f in combined_top_df.columns]
                if available_features:
                    FeatureProcessor(combined_top_df).plot_features_console(
                        features=available_features,
                        width=72,
                        window=None,
                        use_unicode=True,
                        title="Top Laps (Expert) Overview"
                    )
                else:
                    print("[INFO] No selected features available to plot for top laps.")
            else:
                print("[INFO] No top laps available to plot.")
        except Exception as plot_err:
            print(f"[WARNING] Failed to render top laps console plots: {plot_err}")

        # Flatten the DataFrames to list of laps for imitation learning
        top_laps_telemetry_list = []
        for lap_df in top_laps_df:
            # Convert DataFrame to list of dictionaries
            lap_records = lap_df.to_dict('records')
            top_laps_telemetry_list.extend(lap_records)
        
        # Convert rest_laps DataFrames to list of lap records (dictionaries)
        bottom_laps_telemetry_list = []
        for lap_df in bottom_laps_df:
            # Convert DataFrame to list of dictionaries
            lap_records = lap_df.to_dict('records')
            # Add lap records directly to the list
            bottom_laps_telemetry_list.extend(lap_records)
        
        #enrich data
        self._print_section_divider("ENRICHING CONTEXTUAL DATA")
        enrichment_result = await self.enriched_contextual_data(top_laps_telemetry_list, bottom_laps_telemetry_list,trackName)
        enrichment_count = len(enrichment_result["enriched_features"])
        feature_metadata = enrichment_result["feature_metadata"]
        print(f"[INFO] Bottom laps telemetry has {len(bottom_laps_telemetry_list[0].keys())} keys which are: {bottom_laps_telemetry_list[0].keys()}")
        print(f"[INFO] Enrichment successful: {enrichment_count} records with {feature_metadata.get('feature_count', 0)} enriched features")
        print(f"[INFO] Enrichment details: {feature_metadata.get('feature_names', [])}")
        
        # train transformer model
        self._print_section_divider("TRAINING TRANSFORMER MODEL")
        transformer_results = await self._train_expert_action_transformer(
            original_telemetry=enrichment_result["original_telemetry"],
            enriched_contextual_data=enrichment_result["enriched_features"],
            trackName=trackName,
            enrichment_result=enrichment_result  # Pass the already computed enrichment
        )
        
        self._print_section_divider("TRANSFORMER LEARNING COMPLETED")
        return {
            "success": True,
            "transformer_training": transformer_results,
            "expert_imitation_trained": True,
            "contextual_data_enriched": enrichment_count,
            "track_name": trackName
        }

    async def _train_expert_action_transformer(self, 
                                             original_telemetry: List[Dict[str, Any]],
                                             enriched_contextual_data: List[Dict[str, Any]],
                                             trackName: str,
                                             enrichment_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the transformer model to learn non-expert driver progression toward expert performance
        
        Args:
            original_telemetry: List of non-expert telemetry records (contains telemetry and non-expert actions)
            enriched_contextual_data: List of enriched contextual features (expert targets, gaps, corner/tire features)
            trackName: Track name for model identification
            enrichment_result: Pre-computed enrichment result to avoid re-training models
            
        Returns:
            Training results and model performance metrics
        """
        try:
            # Check if PyTorch is available
            if not TORCH_AVAILABLE:
                return {
                    "error": "PyTorch is not available - transformer training is disabled",
                    "success": False   
                }
            
            print(f"[INFO] Starting transformer training with {len(original_telemetry)} telemetry records")
            
            # Verify data consistency
            if len(original_telemetry) != len(enriched_contextual_data):
                raise ValueError(f"Data length mismatch: {len(original_telemetry)} telemetry records vs {len(enriched_contextual_data)} enriched records")
            
            # Data is already prepared:
            # - original_telemetry: Non-expert telemetry data
            # - enriched_contextual_data: Expert targets, delta-to-expert gaps, corner/tire features (context)
            telemetry_data = original_telemetry
            contextual_data = enriched_contextual_data
            
            # Create dataset
            dataset = TelemetryActionDataset(
                telemetry_data=telemetry_data,
                enriched_contextual_data=contextual_data if contextual_data else None,
                sequence_length=20
            )
            
            # Configure DataLoader for better GPU throughput when available
            use_cuda = torch.cuda.is_available()
            num_cpu = os.cpu_count() or 2
            num_workers = min(4, max(0, num_cpu - 1)) if use_cuda else 0
            loader_kwargs = {
                'batch_size': 16,
                'shuffle': False,  # preserve temporal order for time series
                'num_workers': num_workers,
                'pin_memory': use_cuda,
                'persistent_workers': num_workers > 0,
            }
            if num_workers > 0:
                loader_kwargs['prefetch_factor'] = 2

            # Create data loader for training
            train_loader = DataLoader(dataset, **loader_kwargs)

            if use_cuda:
                # Enable TF32 on Ampere+ for faster matmuls while keeping accuracy
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
            
            print(f"[INFO] Using all {len(dataset)} samples for training")
            print(f"[INFO] No validation split - using full dataset for training")
            
            # Get feature dimensions from dataset
            telemetry_features, non_expert_action_features = dataset.get_feature_names()
            context_features_count = len(contextual_data[0]) if contextual_data else 0
            
            print(f"[INFO] Dataset info: {len(telemetry_features)} telemetry features, "
                  f"{len(non_expert_action_features)} non-expert action features (from telemetry), {context_features_count} context features")
            print(f"[INFO] Model will output fixed 5 action features: throttle, brake, steering, gear, speed")
            
            # Create model
            model = ExpertActionTransformer(
                telemetry_features_count=len(telemetry_features),
                context_features_count=context_features_count,
                d_model=256,
                nhead=8,
                num_layers=4,  # Smaller model for faster training
                sequence_length=20
            )
            
            # Create trainer
            device = 'cuda' if use_cuda else 'cpu'
            trainer = ExpertActionTrainer(model, device=device, learning_rate=1e-4)
            
            # Train model (without validation)
            training_history = trainer.train(
                train_dataloader=train_loader,
                val_dataloader=None,
                epochs=30,
                patience=10
            )
            
            # Evaluate model on training data
            test_metrics = trainer.evaluate(train_loader)
            
            # Serialize model
            serialized_model = model.serialize_model()
            
            # Save to backend
            transformer_model_dto = {
                "modelType": "transformer_expert_action",
                "trackName": trackName,
                "carName": 'AllCars',
                "modelData": serialized_model,
                "metadata": {
                    "training_history": training_history,
                    "test_metrics": test_metrics,
                    "feature_names": {
                        "telemetry": telemetry_features,
                    },
                    "training_timestamp": datetime.now().isoformat()
                },
                "isActive": True
            }
            
            await backend_service.save_ai_model(transformer_model_dto)
            
            return {
                "success": True,
                "training_history": training_history,
                "test_metrics": test_metrics,
                "model_info": trainer.get_model_info(),
                "serialized_model": serialized_model
            }
            
        except Exception as e:
            print(f"[ERROR] Transformer training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enriched_contextual_data(self, top_telemetry_list: List[Dict[str, Any]], bottom_telemetry_list: List[Dict[str, Any]], track_name: str) -> Dict[str, Any]:
        """
        Extract enriched contextual features from telemetry data using trained models. it adds expert state, corner identification, and tire grip features,
        and helps transformer model to better understand track geometry, physics constraints, extra expert insights to differentiate actions that
        converge towards feature expert state.
        
        This method keeps original telemetry separate from enriched features for clean training.
        
        Args:
            telemetry_list: List of telemetry record dictionaries (flat list)
            
        Returns:
            Dictionary containing:
            - original_telemetry: Original telemetry data (unchanged)
            - enriched_features: List of enriched feature dictionaries
            - feature_metadata: Information about feature sources and types
        """


        if not bottom_telemetry_list:
            print("[WARNING] No telemetry data provided for enrichment")
            return {
                "original_telemetry": [],
                "enriched_features": [],
                "feature_metadata": {"sources": [], "feature_count": 0}
            }
        
        try:
            # Use all telemetry data for both training enrichment models and feature extraction
            # This ensures consistency and maximizes available data
            bottom_training_telemetry_list = bottom_telemetry_list.copy()
            top_training_telemetry_list = top_telemetry_list.copy()

            self._print_section_divider("TRAINING IMITATION LEARNING MODEL")
            imitation_learning = ExpertImitateLearningService()
            # Train imitation model only on top (expert) telemetry laps
            imitation_result = imitation_learning.train_ai_model(top_training_telemetry_list)

            ai_model_dto = {
                "modelType": "imitation_learning",
                "trackName": track_name,
                "carName": 'AllCars',
                "modelData": imitation_result,
                "metadata": {
                    "summary": imitation_result.get("summary", {}),
                    "training_timestamp": datetime.now().isoformat()
                },
                "isActive": True
            }
            
            await backend_service.save_ai_model(ai_model_dto)
            
            # Extract expert state features for each bottom (non-expert) telemetry record
            
            self._print_section_divider("EXTRACT FROM IMITATION LEARNING MODEL")
            try:
                expert_state_features = imitation_learning.extract_expert_state_for_telemetry(bottom_training_telemetry_list)
                print(f"[INFO] Extracted expert state features for {len(expert_state_features)} non-expert records")
            except Exception as e:
                print(f"[WARNING] Failed to extract expert state features: {e}")
                expert_state_features = []
            
            print(f"[INFO] Using all {len(bottom_training_telemetry_list)} records for enrichment model training and feature extraction")
            
            # Train corner identification model using training data
            self._print_section_divider("Training corner identification model...")
            try:
                from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
                corner_service = CornerIdentificationUnsupervisedService()
                corner_model = await corner_service.learn_track_corner_patterns(top_training_telemetry_list)
                
                if corner_model.get("success"):
                    print(f"[INFO] Corner identification training successful: {corner_model.get('total_corners_identified', 0)} corners identified")
                    # Serialize and persist corner model artifact
                    try:
                        corner_serialized = corner_service.serialize_corner_identification_model(track_name=track_name, car_name="all_cars")
                        corner_model_dto = {
                            "modelType": "corner_identification",
                            "trackName": corner_serialized.get("track_name"),
                            "carName": 'all_cars',
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
                        raise Exception(f"[WARNING] Failed to serialize/save corner model: {ser_err}")
                else:
                    raise Exception(f"[WARNING] Corner identification training failed: {corner_model.get('error', 'Unknown error')}")
            except Exception as e:
                raise Exception(f"[ERROR] Corner identification training failed: {str(e)}")
            
            # Train tire grip analysis model using training data
            self._print_section_divider("Training tire grip analysis model...")
            try:
                # The tire grip service is now heuristic-only: it computes features deterministically from physics telemetry
                from .tire_grip_analysis_service import TireGripAnalysisService
                tire_service = TireGripAnalysisService()
                tire_grip_model = await tire_service.train_tire_grip_model(bottom_training_telemetry_list)
                
                tire_service_serialized = tire_service.serialize_tire_grip_model()
                tire_grip_model_dto = {
                    "modelType": "tire_grip_analysis",
                    "trackName": "generic",
                    "carName": "all_cars",
                    "modelData": tire_service_serialized,
                    "metadata": {
                        "model_info": tire_service_serialized.get("model_info", {}),
                        "serialization_timestamp": tire_service_serialized.get("serialized_timestamp")
                    },
                    "isActive": True
                }
                await self.backend_service.save_ai_model(tire_grip_model_dto)
                print("[INFO] Saved serialized tire grip analysis model to backend")
            except Exception as e:
                raise Exception(f"[ERROR] Tire grip analysis training failed: {str(e)}")
            
            # Now extract enriched features separately (don't mix back into original data)
            enriched_features_data = []
            feature_sources = []
            
            # Initialize with expert state features (if available) else empty for each record
            if expert_state_features and len(expert_state_features) == len(bottom_training_telemetry_list):
                enriched_features_data = [dict(esf) for esf in expert_state_features]
                feature_sources.append("expert_state")
            else:
                for _ in range(len(bottom_training_telemetry_list)):
                    enriched_features_data.append({})
            
            # Extract corner features as separate enriched data
            if corner_model.get("success"):
                try:
                    self._print_section_divider("Extracting corner features...")
                    corner_enriched_data = await corner_service.extract_corner_features_for_telemetry(bottom_training_telemetry_list)
                    
                    # Add corner features to enriched features data
                    for i, corner_features in enumerate(corner_enriched_data):
                        if i < len(enriched_features_data):
                            enriched_features_data[i].update(corner_features)
                    
                    feature_sources.append("corner_identification")
                    print(f"[INFO] Extracted corner features for {len(corner_enriched_data)} records")
                except Exception as e:
                    raise Exception(f"[WARNING] Failed to extract corner features: {str(e)}")
            
            # Extract tire grip features as separate enriched data

            try:
                self._print_section_divider("Extracting tire grip features...")
                grip_enriched_data = await tire_service.extract_tire_grip_features(bottom_training_telemetry_list)
                # Validate expected keys exist in at least one record
                try:
                    from .tire_grip_analysis_service import TireGripFeatureCatalog
                    expected_keys = set(TireGripFeatureCatalog.CONTEXT_FEATURES)
                    if grip_enriched_data:
                        sample_keys = set(grip_enriched_data[0].keys())
                        missing = expected_keys - sample_keys
                        if missing:
                            print(f"[WARNING] Tire grip features missing keys: {sorted(list(missing))}")
                except Exception as v_err:
                    raise Exception(f"[WARNING] Tire grip feature validation skipped: {v_err}")
                    
                # Add tire grip features to enriched features data
                for i, grip_features in enumerate(grip_enriched_data):
                    if i < len(enriched_features_data):
                        enriched_features_data[i].update(grip_features)
                    
                feature_sources.append("tire_grip_analysis")
                print(f"[INFO] Extracted tire grip features for {len(grip_enriched_data)} records")
            except Exception as e:
                raise Exception(f"[WARNING] Failed to extract tire grip features: {str(e)}")
            
            print(f"[INFO] Contextual data enrichment completed: {len(enriched_features_data)} enriched feature records created")
            
            # Create feature metadata
            sample_enriched = enriched_features_data[0] if enriched_features_data else {}
            grip_sample_enriched = grip_enriched_data[0] if grip_enriched_data else {}
            corner_sample_enriched = corner_enriched_data[0] if corner_enriched_data else {}
            feature_metadata = {
                'sources': feature_sources,
                'feature_count': len(sample_enriched),
                'feature_names': list(sample_enriched.keys()),
                'corner_features': list(corner_sample_enriched.keys()),
                'grip_features': list(grip_sample_enriched.keys()),
                'total_original_records': len(bottom_training_telemetry_list),
                'corner_identification_success': corner_model.get("success", False),
                'tire_grip_analysis_success': tire_grip_model.get("success", False)
            }
            
            print(f"[INFO] Feature metadata: {feature_metadata['feature_count']} enriched features from {len(feature_sources)} sources")
            
            return {
                "original_telemetry": bottom_training_telemetry_list,  # Keep original data unchanged
                "enriched_features": enriched_features_data,    # Separate enriched features
                "feature_metadata": feature_metadata           # Metadata about features
            }
            
        except Exception as e:
            raise Exception(f"{self.__dir__} Failed to enrich contextual data: {str(e)}")

    
    def _compare_telemetry_performance(self, 
                                      expert_telemetry: List[Dict[str, Any]], 
                                      non_expert_telemetry: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance between expert and non-expert telemetry
        
        Args:
            expert_telemetry: Expert driver telemetry records
            non_expert_telemetry: Non-expert driver telemetry records
            
        Returns:
            Dictionary with comparison results and performance metrics
        """
        try:
            # Convert to DataFrames for analysis
            expert_df = pd.DataFrame(expert_telemetry) if expert_telemetry else pd.DataFrame()
            non_expert_df = pd.DataFrame(non_expert_telemetry) if non_expert_telemetry else pd.DataFrame()
            
            if expert_df.empty or non_expert_df.empty:
                return {
                    'total_data_points': 0,
                    'overall_score': 0.0,
                    'performance_sections': [],
                    'error': 'Insufficient data for comparison'
                }
            
            comparison_metrics = {}
            performance_sections = []
            
            # Speed comparison
            if 'Physics_speed_kmh' in expert_df.columns and 'Physics_speed_kmh' in non_expert_df.columns:
                expert_speed = expert_df['Physics_speed_kmh'].mean()
                non_expert_speed = non_expert_df['Physics_speed_kmh'].mean()
                speed_ratio = non_expert_speed / expert_speed if expert_speed > 0 else 0
                
                comparison_metrics['speed'] = {
                    'expert_avg': expert_speed,
                    'non_expert_avg': non_expert_speed,
                    'efficiency_ratio': speed_ratio
                }
                
                performance_sections.append({
                    'metric': 'speed',
                    'expert_value': expert_speed,
                    'driver_value': non_expert_speed,
                    'score': min(speed_ratio, 1.0) * 100  # Cap at 100%
                })
            
            # Lap time comparison (if available)
            if 'Graphics_current_time' in expert_df.columns and 'Graphics_current_time' in non_expert_df.columns:
                expert_lap_time = expert_df['Graphics_current_time'].max() - expert_df['Graphics_current_time'].min()
                non_expert_lap_time = non_expert_df['Graphics_current_time'].max() - non_expert_df['Graphics_current_time'].min()
                
                if expert_lap_time > 0 and non_expert_lap_time > 0:
                    time_ratio = expert_lap_time / non_expert_lap_time  # Ratio < 1 means non-expert is faster (unlikely)
                    
                    comparison_metrics['lap_time'] = {
                        'expert_time': expert_lap_time,
                        'non_expert_time': non_expert_lap_time,
                        'efficiency_ratio': time_ratio
                    }
                    
                    performance_sections.append({
                        'metric': 'lap_time',
                        'expert_value': expert_lap_time,
                        'driver_value': non_expert_lap_time,
                        'score': min(time_ratio * 100, 100)  # Expert should be faster
                    })
            
            # Smoothness comparison (steering, throttle, brake)
            smoothness_scores = []
            for control in ['Physics_steer_angle', 'Physics_gas', 'Physics_brake']:
                if control in expert_df.columns and control in non_expert_df.columns:
                    expert_std = expert_df[control].std()
                    non_expert_std = non_expert_df[control].std()
                    
                    # Lower standard deviation indicates smoother control
                    smoothness_ratio = expert_std / non_expert_std if non_expert_std > 0 else 1.0
                    smoothness_score = min(smoothness_ratio * 100, 100)
                    smoothness_scores.append(smoothness_score)
                    
                    performance_sections.append({
                        'metric': f'{control}_smoothness',
                        'expert_value': expert_std,
                        'driver_value': non_expert_std,
                        'score': smoothness_score
                    })
            
            # Calculate overall performance score
            all_scores = [section['score'] for section in performance_sections]
            overall_score = np.mean(all_scores) if all_scores else 0.0
            
            return {
                'total_data_points': len(non_expert_telemetry),
                'overall_score': overall_score,
                'performance_sections': performance_sections,
                'comparison_metrics': comparison_metrics,
                'expert_data_points': len(expert_telemetry),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[ERROR] Performance comparison failed: {str(e)}")
            return {
                'total_data_points': len(non_expert_telemetry) if non_expert_telemetry else 0,
                'overall_score': 0.0,
                'performance_sections': [],
                'error': str(e)
            }
    
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")