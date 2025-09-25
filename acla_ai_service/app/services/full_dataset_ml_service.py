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
        
        # Configure default caching strategies for common model types
        self._caching_strategies = {
            "imitation_learning": {"cache_raw_data": True, "configured_at": datetime.now().isoformat()},
            "transformer_expert_action": {"cache_raw_data": True, "configured_at": datetime.now().isoformat()},
            "corner_identification": {"cache_raw_data": True, "configured_at": datetime.now().isoformat()},
            "tire_grip_analysis": {"cache_raw_data": True, "configured_at": datetime.now().isoformat()},
        }
        
        # Log cache configuration on startup
        self._log_cache_configuration()
    
    def _log_cache_configuration(self):
        """Log cache configuration details"""
        cache_stats = self.model_cache.get_stats()
        print("\n" + "="*60)
        print("CACHE CONFIGURATION")
        print("="*60)
        print(f"Max Cache Size: {self.model_cache.max_cache_size} models")
        print(f"Max Memory: {self.model_cache.max_memory_mb}MB ({self.model_cache.max_memory_mb/1024:.1f}GB)")
        print(f"Environment: {self.model_cache.environment}")
        print(f"Default TTL: {self.model_cache.default_ttl_seconds}s ({self.model_cache.default_ttl_seconds/3600:.1f}h)")
        print(f"Large Model Priority: {self.model_cache.config.get('performance', {}).get('large_model_priority', False)}")
        print("Model-specific TTLs:")
        for model_type, strategy in self._caching_strategies.items():
            ttl = self.model_cache.get_model_ttl(model_type)
            print(f"  - {model_type}: {ttl}s ({ttl/3600:.1f}h), raw_data={strategy['cache_raw_data']}")
        print("="*60 + "\n")

    def clear_all_cache(self):
        """Clear cached transformer / imitation models (corner & tire services now on-demand)."""
        self.model_cache.clear()
        print("[INFO] All cached model_cache entries cleared (corner & tire services are on-demand so no persistent cache to clear)")
    
    async def _fetch_and_cache_model(self,
                                    model_type: str,
                                    track_name: Optional[str] = None,
                                    car_name: Optional[str] = None,
                                    model_subtype: str = "complete_model_data",
                                    deserializer_func=None,
                                    cache_raw_data: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Fetch model from backend and cache it with improved caching strategy.
        Now supports caching raw data for better cache efficiency and on-demand deserialization.
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            track_name: Track name for the model (optional)
            car_name: Car name for the model (optional)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize model data from backend and return model instance
            cache_raw_data: If True, cache raw data and deserialize on-demand. If False, cache deserialized model.
            
        Returns:
            Tuple of (model_instance, metadata)
        """
        print(f"[DEBUG] Fetching model from backend: {model_type}/{track_name}/{car_name}")
        
        model_response = await self.backend_service.getCompleteActiveModelData(
            track_name, car_name, model_type
        )
          
        if "error" in model_response:
            raise Exception(f"Backend error: {model_response['error']}")
        
        # Debug: Log the structure we received
        print(f"[DEBUG] Model response keys: {list(model_response.keys()) if isinstance(model_response, dict) else 'Not a dict'}")
        if "data" in model_response:
            print(f"[DEBUG] Response data type: {type(model_response['data'])}")
            if isinstance(model_response["data"], dict):
                print(f"[DEBUG] Response data keys: {list(model_response['data'].keys())}")
        
        # Extract modelData from the correct location in response
        model_data = None
        if "data" in model_response and isinstance(model_response["data"], dict):
            model_data = model_response["data"].get("modelData", {})
            print(f"[DEBUG] Found modelData in response['data'], type: {type(model_data)}")
        else:
            model_data = model_response.get("modelData", {})
            print(f"[DEBUG] Found modelData in response root, type: {type(model_data)}")
        
        if not model_data:
            print(f"[ERROR] No modelData found. Response structure: {model_response}")
            raise Exception("No modelData found in response")

        # Prepare cache metadata
        cache_metadata = {
            "track_name": track_name,
            "car_name": car_name,
            "model_type": model_type,
            "fetched_at": datetime.now().isoformat(),
            "backend_model_id": model_response.get("id", "unknown"),
            "model_subtype": model_subtype
        }
        
        # Decide caching strategy
        if cache_raw_data and deserializer_func:
            # Strategy 1: Cache raw data, deserialize on-demand (recommended)
            print(f"[DEBUG] Caching raw data for on-demand deserialization: {model_type}")
            
            # Add flag to metadata to indicate raw data storage
            cache_metadata["is_raw_model_data"] = True
            cache_metadata["has_deserializer"] = True
            
            # Cache the raw model data
            self.model_cache.put(
                model_type=model_type,
                track_name=track_name,
                car_name=car_name,
                data=model_data,  # Store raw data
                metadata=cache_metadata,
                model_subtype=model_subtype
            )
            
            # Deserialize for immediate return
            model_instance = deserializer_func(model_data)
            if model_instance is None:
                raise Exception("Deserializer function returned None - must return model instance")
                
        else:
            # Strategy 2: Cache deserialized model (legacy behavior)
            print(f"[DEBUG] Caching deserialized model: {model_type}")
            
            if deserializer_func:
                # Deserialize the model data using provided function
                model_instance = deserializer_func(model_data)
                if model_instance is None:
                    raise Exception("Deserializer function returned None - must return model instance")
                cache_metadata["is_raw_model_data"] = False
                cache_metadata["has_deserializer"] = True
            else:
                # For other model types without deserializer, just use the raw data
                model_instance = model_data
                cache_metadata["is_raw_model_data"] = False
                cache_metadata["has_deserializer"] = False
            
            # Cache the processed model instance
            self.model_cache.put(
                model_type=model_type,
                track_name=track_name,
                car_name=car_name,
                data=model_instance,
                metadata=cache_metadata,
                model_subtype=model_subtype
            )
        
        print(f"[DEBUG] Successfully cached model: {model_type} (raw_data: {cache_raw_data and deserializer_func is not None})")
        
        return model_instance, cache_metadata
    
    def _cleanup_fetch_lock(self, cache_key: str, track_name: str, car_name: str):
        """
        Clean up fetch lock for a specific cache key
        
        Args:
            cache_key: The cache key used for locking
            track_name: Track name for logging
            car_name: Car name for logging
        """
        if cache_key in self._model_fetch_locks:
            try:
                self._model_fetch_locks[cache_key].set()
                del self._model_fetch_locks[cache_key]
                print(f"[DEBUG] Released fetch lock for {cache_key}")
            except Exception as cleanup_error:
                print(f"[WARNING] Error cleaning up fetch lock: {str(cleanup_error)}")
    
    def _emergency_cleanup_fetch_lock(self, cache_key: str, track_name: str, car_name: str):
        """
        Emergency cleanup of fetch lock with additional safety checks
        
        Args:
            cache_key: The cache key used for locking
            track_name: Track name for logging
            car_name: Car name for logging
        """
        if hasattr(self, '_model_fetch_locks') and cache_key in self._model_fetch_locks:
            try:
                self._model_fetch_locks[cache_key].set()  # Signal any waiting threads
                del self._model_fetch_locks[cache_key]
                print(f"[INFO] Emergency cleanup of fetch lock for {cache_key}")
            except Exception as cleanup_error:
                print(f"[WARNING] Error during emergency lock cleanup: {str(cleanup_error)}")
    

    async def _get_cached_model_or_fetch(self,
                                        model_type: str,
                                        track_name: Optional[str] = None,
                                        car_name: Optional[str] = None,
                                        model_subtype: str = "complete_model_data",
                                        deserializer_func=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Get model from cache or fetch from backend with thread-safe locking.
        Now supports both raw data caching and on-demand deserialization.
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            track_name: Track name for the model (optional)
            car_name: Car name for the model (optional)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize raw model data and return model instance
            
        Returns:
            Tuple of (model_instance, metadata)
        """
        # Generate consistent cache key using the same method as ModelCacheService
        cache_key = self.model_cache._generate_cache_key(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            model_subtype=model_subtype
        )
        
        # Use cache_key for locking to ensure consistency
        is_fetching_thread = False
        model_instance = None
        metadata = {}
        
        try:
            # First, check cache info without accessing the data to avoid unnecessary hits
            cache_info = self.model_cache.get_cache_info(
                model_type=model_type,
                track_name=track_name,
                car_name=car_name,
                model_subtype=model_subtype
            )
            
            if cache_info and not cache_info.get('is_expired', True):
                print(f"[DEBUG] Cache hit for {cache_key} - age: {cache_info.get('ttl_remaining_seconds', 'N/A')}s remaining")
                
                # Get the actual cached data
                cached_result = self.model_cache.get(
                    model_type=model_type,
                    track_name=track_name,
                    car_name=car_name,
                    model_subtype=model_subtype
                )
                
                if cached_result:
                    raw_data, metadata = cached_result
                    
                    # Handle deserialization based on storage strategy
                    if deserializer_func and isinstance(raw_data, dict) and 'is_raw_model_data' in metadata:
                        # Raw data stored - deserialize on demand
                        try:
                            print(f"[DEBUG] Deserializing raw cached data for {cache_key}")
                            model_instance = deserializer_func(raw_data)
                            if model_instance is None:
                                raise Exception("Deserializer returned None for cached raw data")
                        except Exception as deser_error:
                            print(f"[ERROR] Failed to deserialize cached data for {cache_key}: {str(deser_error)}")
                            # Clear corrupted cache entry and continue to refetch
                            self.model_cache.invalidate(
                                model_type=model_type,
                                track_name=track_name,
                                car_name=car_name,
                                model_subtype=model_subtype
                            )
                            cached_result = None
                    else:
                        # Pre-deserialized data or no deserializer needed
                        model_instance = raw_data
                    
                    if model_instance is not None:
                        return model_instance, metadata
            else:
                print(f"[DEBUG] Cache miss or expired for {cache_key}")
            
            # If no valid cached result, handle fetching with proper locking
            async with self._lock_creation_lock:
                # Double-check cache after acquiring lock
                cached_result = self.model_cache.get(
                    model_type=model_type,
                    track_name=track_name,
                    car_name=car_name,
                    model_subtype=model_subtype
                )
                
                if cached_result:
                    raw_data, metadata = cached_result
                    
                    # Handle deserialization
                    if deserializer_func and isinstance(raw_data, dict) and 'is_raw_model_data' in metadata:
                        try:
                            model_instance = deserializer_func(raw_data)
                            if model_instance is None:
                                raise Exception("Deserializer returned None")
                        except Exception as deser_error:
                            print(f"[ERROR] Failed to deserialize double-checked cache: {str(deser_error)}")
                            # Clear and continue to fetch
                            self.model_cache.invalidate(
                                model_type=model_type,
                                track_name=track_name,
                                car_name=car_name,
                                model_subtype=model_subtype
                            )
                            model_instance = None
                    else:
                        model_instance = raw_data
                    
                    if model_instance is not None:
                        return model_instance, metadata
                
                # Check if another thread is already fetching
                if cache_key in self._model_fetch_locks:
                    fetch_event = self._model_fetch_locks[cache_key]
                else:
                    # We're the first to request this model, create the event and mark ourselves as fetching
                    self._model_fetch_locks[cache_key] = asyncio.Event()
                    is_fetching_thread = True
            
            # If we need to wait for another thread
            if not model_instance and not is_fetching_thread:
                try:
                    print(f"[DEBUG] Waiting for another thread to fetch {cache_key}")
                    await fetch_event.wait()
                    # The other thread should have cached the model, try cache again
                    cached_result = self.model_cache.get(
                        model_type=model_type,
                        track_name=track_name,
                        car_name=car_name,
                        model_subtype=model_subtype
                    )
                    if cached_result:
                        raw_data, metadata = cached_result
                        
                        if deserializer_func and isinstance(raw_data, dict) and 'is_raw_model_data' in metadata:
                            try:
                                model_instance = deserializer_func(raw_data)
                            except Exception as deser_error:
                                print(f"[ERROR] Failed to deserialize after wait: {str(deser_error)}")
                                model_instance = None
                        else:
                            model_instance = raw_data
                        
                        if model_instance:
                            print(f"[INFO] Using model cached by another thread for {cache_key}")
                            return model_instance, metadata
                    
                    print(f"[WARNING] Expected cached model not found after waiting for {cache_key}")
                    # Continue to try fetching ourselves as fallback
                    is_fetching_thread = True
                except Exception as wait_error:
                    print(f"[ERROR] Error while waiting for model fetch: {str(wait_error)}")
                    # Continue to try fetching ourselves as fallback
                    is_fetching_thread = True

            # If we are the fetching thread or no data in cache, do the actual fetch
            if not model_instance and is_fetching_thread:
                try:
                    # Use configured caching strategy
                    cache_raw_data = self.get_caching_strategy(model_type)
                    
                    model_instance, metadata = await self._fetch_and_cache_model(
                        model_type=model_type,
                        track_name=track_name,
                        car_name=car_name,
                        model_subtype=model_subtype,
                        deserializer_func=deserializer_func,
                        cache_raw_data=cache_raw_data
                    )
                    print(f"[INFO] Successfully fetched and cached model for {cache_key}")
                    
                except Exception as fetch_error:
                    print(f"[ERROR] Failed to fetch model {cache_key}: {str(fetch_error)}")
                    raise fetch_error
                finally:
                    # Always signal completion and clean up lock when we're the fetching thread
                    self._cleanup_fetch_lock(cache_key, track_name or 'any', car_name or 'any')
            
            # At this point, we should have the model instance
            if model_instance is None:
                raise Exception("Failed to obtain model instance from cache or backend")
            
            return model_instance, metadata
            
        except Exception as e:
            # Clean up any locks that might have been created by this thread
            if is_fetching_thread:
                self._emergency_cleanup_fetch_lock(cache_key, track_name or 'any', car_name or 'any')
            
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
    
    def get_cache_debug_info(self) -> Dict[str, Any]:
        """
        Get comprehensive cache debugging information
        
        Returns:
            Dictionary with cache and lock information
        """
        cache_stats = self.model_cache.get_stats()
        lock_status = self.get_fetch_locks_status()
        
        return {
            "cache_stats": cache_stats,
            "fetch_locks": lock_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def print_cache_debug_info(self):
        """
        Print cache debugging information to console
        """
        debug_info = self.get_cache_debug_info()
        
        print("\n" + "="*60)
        print("CACHE DEBUG INFORMATION")
        print("="*60)
        
        cache_stats = debug_info["cache_stats"]
        print(f"Cache Size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        print(f"Memory Usage: {cache_stats['memory_usage_mb']:.2f}/{cache_stats['max_memory_mb']} MB")
        print(f"Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
        print(f"Evictions: {cache_stats['evictions']}, Cleanups: {cache_stats['cleanups']}")
        
        print(f"\nActive Fetch Locks: {debug_info['fetch_locks']['lock_count']}")
        for lock_key in debug_info['fetch_locks']['active_locks']:
            print(f"  - {lock_key}")
        
        print(f"\nCached Models:")
        for entry in cache_stats.get('entries', []):
            print(f"  - {entry['key']} ({entry['size_mb']:.2f} MB, accessed {entry['access_count']} times)")
            if entry.get('ttl_remaining_seconds'):
                print(f"    TTL remaining: {entry['ttl_remaining_seconds']:.0f}s")
        
        print("="*60 + "\n")
    
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
                for cache_key in list(self._model_fetch_locks.keys()):
                    try:
                        self._model_fetch_locks[cache_key].set()
                        del self._model_fetch_locks[cache_key]
                        cleared_locks.append(cache_key)
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
    
    async def validate_and_preload_model(self, 
                                        model_type: str,
                                        track_name: Optional[str] = None,
                                        car_name: Optional[str] = None,
                                        model_subtype: str = "complete_model_data",
                                        deserializer_func=None,
                                        force_refresh: bool = False) -> Dict[str, Any]:
        """
        Validate a cached model or preload it from backend.
        Useful for warming up cache or verifying model integrity.
        
        Args:
            model_type: Type of model
            track_name: Track name (optional)
            car_name: Car name (optional)
            model_subtype: Subtype identifier
            deserializer_func: Deserializer function
            force_refresh: If True, bypass cache and refetch from backend
            
        Returns:
            Dictionary with validation/preload results
        """
        cache_key = self.model_cache._generate_cache_key(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            model_subtype=model_subtype
        )
        
        start_time = datetime.now()
        
        try:
            if force_refresh:
                # Clear existing cache entry
                self.model_cache.invalidate(
                    model_type=model_type,
                    track_name=track_name,
                    car_name=car_name,
                    model_subtype=model_subtype
                )
                print(f"[DEBUG] Force refresh - cleared cache for {cache_key}")
            
            # Get or fetch the model
            model_instance, metadata = await self._get_cached_model_or_fetch(
                model_type=model_type,
                track_name=track_name,
                car_name=car_name,
                model_subtype=model_subtype,
                deserializer_func=deserializer_func
            )
            
            end_time = datetime.now()
            load_time = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "cache_key": cache_key,
                "load_time_seconds": load_time,
                "model_loaded": model_instance is not None,
                "metadata": metadata,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            end_time = datetime.now()
            load_time = (end_time - start_time).total_seconds()
            
            return {
                "success": False,
                "cache_key": cache_key,
                "load_time_seconds": load_time,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": end_time.isoformat()
            }
   
    def configure_caching_strategy(self, model_type: str, cache_raw_data: bool = True):
        """
        Configure caching strategy for a specific model type
        
        Args:
            model_type: Type of model to configure
            cache_raw_data: If True, cache raw data and deserialize on-demand
        """
        if not hasattr(self, '_caching_strategies'):
            self._caching_strategies = {}
        
        self._caching_strategies[model_type] = {
            "cache_raw_data": cache_raw_data,
            "configured_at": datetime.now().isoformat()
        }
        
        print(f"[INFO] Configured caching strategy for {model_type}: cache_raw_data={cache_raw_data}")
    
    def get_caching_strategy(self, model_type: str) -> bool:
        """
        Get the caching strategy for a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            True if should cache raw data, False if should cache deserialized model
        """
        if hasattr(self, '_caching_strategies') and model_type in self._caching_strategies:
            return self._caching_strategies[model_type]["cache_raw_data"]
        
        # Default strategy: cache raw data for better efficiency
        return True
    
    async def test_caching_performance(self, 
                                      model_type: str,
                                      track_name: Optional[str] = None,
                                      car_name: Optional[str] = None,
                                      deserializer_func=None,
                                      num_tests: int = 3) -> Dict[str, Any]:
        """
        Test caching performance by making multiple requests for the same model
        
        Args:
            model_type: Type of model to test
            track_name: Track name (optional)
            car_name: Car name (optional)
            deserializer_func: Deserializer function
            num_tests: Number of test iterations
            
        Returns:
            Performance test results
        """
        print(f"[INFO] Testing cache performance for {model_type} ({num_tests} iterations)")
        
        results = {
            "model_type": model_type,
            "track_name": track_name,
            "car_name": car_name,
            "num_tests": num_tests,
            "test_times": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Get initial cache stats
        initial_stats = self.model_cache.get_stats()
        
        for i in range(num_tests):
            start_time = datetime.now()
            
            try:
                model_instance, metadata = await self._get_cached_model_or_fetch(
                    model_type=model_type,
                    track_name=track_name,
                    car_name=car_name,
                    deserializer_func=deserializer_func
                )
                
                end_time = datetime.now()
                test_time = (end_time - start_time).total_seconds()
                results["test_times"].append(test_time)
                
                print(f"[TEST {i+1}/{num_tests}] Load time: {test_time:.4f}s")
                
            except Exception as e:
                print(f"[TEST {i+1}/{num_tests}] Failed: {str(e)}")
                results["test_times"].append(None)
        
        # Get final cache stats
        final_stats = self.model_cache.get_stats()
        results["cache_hits"] = final_stats["hits"] - initial_stats["hits"]
        results["cache_misses"] = final_stats["misses"] - initial_stats["misses"]
        
        # Calculate performance metrics
        valid_times = [t for t in results["test_times"] if t is not None]
        if valid_times:
            results["avg_time"] = sum(valid_times) / len(valid_times)
            results["min_time"] = min(valid_times)
            results["max_time"] = max(valid_times)
        
        print(f"[RESULTS] Avg: {results.get('avg_time', 0):.4f}s, "
              f"Min: {results.get('min_time', 0):.4f}s, "
              f"Max: {results.get('max_time', 0):.4f}s")
        print(f"[RESULTS] Cache hits: {results['cache_hits']}, misses: {results['cache_misses']}")
        
        return results
    
    def analyze_large_model_cache_usage(self) -> Dict[str, Any]:
        """
        Analyze cache usage specifically for large models and provide optimization recommendations
        
        Returns:
            Analysis results with recommendations
        """
        cache_stats = self.model_cache.get_stats()
        large_model_threshold_mb = 500
        
        analysis = {
            "total_memory_mb": cache_stats["memory_usage_mb"],
            "max_memory_mb": cache_stats["max_memory_mb"],
            "memory_usage_percent": (cache_stats["memory_usage_mb"] / cache_stats["max_memory_mb"]) * 100,
            "large_models": [],
            "small_models": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Analyze individual models
        for entry_info in cache_stats.get("entries", []):
            size_mb = entry_info["size_mb"]
            model_info = {
                "key": entry_info["key"],
                "size_mb": size_mb,
                "access_count": entry_info["access_count"],
                "ttl_remaining_seconds": entry_info.get("ttl_remaining_seconds")
            }
            
            if size_mb >= large_model_threshold_mb:
                analysis["large_models"].append(model_info)
            else:
                analysis["small_models"].append(model_info)
        
        # Generate recommendations
        if analysis["memory_usage_percent"] > 90:
            analysis["recommendations"].append("Memory usage is very high (>90%). Consider increasing max_memory_mb or clearing unused models.")
        
        if len(analysis["large_models"]) > 5:
            analysis["recommendations"].append(f"Many large models cached ({len(analysis['large_models'])}). Consider preloading only essential models.")
        
        if analysis["memory_usage_percent"] > 70 and len(analysis["small_models"]) > 10:
            analysis["recommendations"].append("Consider clearing small models to make room for large models.")
        
        total_large_mb = sum(m["size_mb"] for m in analysis["large_models"])
        if total_large_mb > analysis["max_memory_mb"] * 0.8:
            analysis["recommendations"].append("Large models are consuming >80% of cache memory. Consider using raw data caching strategy.")
        
        return analysis
    
    def optimize_cache_for_large_models(self) -> Dict[str, Any]:
        """
        Optimize cache configuration for handling large models
        
        Returns:
            Optimization results
        """
        print("[INFO] Optimizing cache for large models...")
        
        # Analyze current usage
        analysis = self.analyze_large_model_cache_usage()
        
        optimization_actions = []
        
        # Clear small models if memory is tight
        if analysis["memory_usage_percent"] > 80:
            small_models_cleared = 0
            for model_info in analysis["small_models"]:
                # Try to extract model details from key for invalidation
                key_parts = model_info["key"].split(":")
                if len(key_parts) >= 3:
                    model_type, track_name, car_name = key_parts[0], key_parts[1], key_parts[2]
                    if self.model_cache.invalidate(
                        model_type=model_type,
                        track_name=track_name if track_name != 'any' else None,
                        car_name=car_name if car_name != 'any' else None
                    ):
                        small_models_cleared += 1
            
            if small_models_cleared > 0:
                optimization_actions.append(f"Cleared {small_models_cleared} small models to free memory")
        
        # Configure longer TTLs for large models
        large_model_types = ["imitation_learning", "transformer_expert_action"]
        for model_type in large_model_types:
            if model_type in self._caching_strategies:
                # Ensure raw data caching for large models
                self._caching_strategies[model_type]["cache_raw_data"] = True
                optimization_actions.append(f"Enabled raw data caching for {model_type}")
        
        # Clean up expired entries
        expired_before = len([m for m in analysis["large_models"] + analysis["small_models"] 
                            if m.get("ttl_remaining_seconds", 1) <= 0])
        
        # Force cleanup by accessing cache stats (which triggers cleanup)
        self.model_cache._cleanup_expired()
        
        optimization_actions.append("Cleaned up expired cache entries")
        
        return {
            "success": True,
            "actions_taken": optimization_actions,
            "analysis_before": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    async def emergency_cache_reset_for_large_models(self) -> Dict[str, Any]:
        """
        Emergency cache reset specifically designed for large model issues.
        Use this when models keep re-downloading despite cache configuration.
        
        Returns:
            Reset operation results
        """
        print("[WARNING] Performing emergency cache reset for large models...")
        
        # Get pre-reset stats
        pre_stats = self.model_cache.get_stats()
        
        results = {
            "pre_reset_memory_mb": pre_stats["memory_usage_mb"],
            "pre_reset_models": pre_stats["cache_size"],
            "actions_taken": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Clear all stuck fetch locks
        cleared_locks = await self.clear_stuck_fetch_locks()
        if cleared_locks["cleared_count"] > 0:
            results["actions_taken"].append(f"Cleared {cleared_locks['cleared_count']} stuck fetch locks")
        
        # 2. Clear entire cache
        self.model_cache.clear()
        results["actions_taken"].append("Cleared entire model cache")
        
        # 3. Reset caching strategies with optimal settings for large models
        self._caching_strategies = {
            "imitation_learning": {
                "cache_raw_data": True, 
                "configured_at": datetime.now().isoformat()
            },
            "transformer_expert_action": {
                "cache_raw_data": True, 
                "configured_at": datetime.now().isoformat()
            },
            "corner_identification": {
                "cache_raw_data": True, 
                "configured_at": datetime.now().isoformat()
            },
            "tire_grip_analysis": {
                "cache_raw_data": True, 
                "configured_at": datetime.now().isoformat()
            },
        }
        results["actions_taken"].append("Reset caching strategies to optimal settings for large models")
        
        # 4. Log new configuration
        self._log_cache_configuration()
        results["actions_taken"].append("Logged new cache configuration")
        
        # Get post-reset stats
        post_stats = self.model_cache.get_stats()
        results["post_reset_memory_mb"] = post_stats["memory_usage_mb"]
        results["post_reset_models"] = post_stats["cache_size"]
        results["memory_freed_mb"] = results["pre_reset_memory_mb"] - results["post_reset_memory_mb"]
        
        print("[INFO] Emergency cache reset completed")
        for action in results["actions_taken"]:
            print(f"  ✓ {action}")
        print(f"Memory freed: {results['memory_freed_mb']:.1f}MB")
        
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

        imitation_learning = ExpertImitateLearningService()
        results = imitation_learning.train_ai_model(telemetry_data)
            
        try:
            #save the info to backend
            await backend_service.save_ai_model(
                model_type="imitation_learning",
                track_name=trackName,
                car_name=carName,
                model_data=results,
                metadata={
                    "summary": results.get("summary", {}),
                    "training_timestamp": datetime.now().isoformat()
                },
                is_active=True
            )
        except Exception as error:
            pass
        
        return results
    
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
                try:
                    await self.backend_service.save_ai_model(
                        model_type="corner_identification",
                        track_name=trackName,
                        car_name=carName or "all_cars",
                        model_data=results,
                        metadata={
                            "timestamp": datetime.now().isoformat()
                        },
                        is_active=True
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
            
            # Initialize transformer model
            transformer_model = ExpertActionTransformer()
            
            # Fetch and load the trained model
            transformer_model, model_metadata = await self._get_cached_model_or_fetch(
                model_type="transformer_expert_action",
                track_name=trackName,
                car_name=carName,
                model_subtype="transformer_model_data",
                deserializer_func=transformer_model.deserialize_transformer_model
            )
            
            # Extract context data by running corner and tire grip analysis
            context_data = {}
            
            # Get corner identification features
            try:
                corner_service = CornerIdentificationUnsupervisedService()
                corner_service, corner_metadata = await self._get_cached_model_or_fetch(
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
                if corner_features_list and len(corner_features_list) > 0:
                    # Since we passed a single telemetry record, extract the single result dictionary
                    corner_features = corner_features_list[0]
                    context_data.update(corner_features)
                    print(f"[INFO] Added {len(corner_features)} corner features to context")
                
            except Exception as e:
                print(f"[Error] Corner identification failed: {e}")
            
            # Get tire grip features
            try:
                tire_grip_service = TireGripAnalysisService()
                tire_grip_service, tire_metadata = await self._get_cached_model_or_fetch(
                    model_type="tire_grip_analysis",
                    track_name='generic',
                    car_name='all_cars',
                    model_subtype="tire_grip_model_data",
                    deserializer_func=tire_grip_service.deserialize_tire_grip_model
                )
                
                # Extract tire grip features from telemetry (async; returns List[Dict])
                tire_features_list = await tire_grip_service.extract_tire_grip_features([telemetry_dict])
                if tire_features_list and len(tire_features_list) > 0:
                    # Since we passed a single telemetry record, extract the single result dictionary
                    tire_features = tire_features_list[0]
                    context_data.update(tire_features)
                    print(f"[INFO] Added {len(tire_features)} tire grip features to context")
                
            except Exception as e:
                print(f"[Error] Tire grip analysis failed: {e}")
            
            try:
                expert_service = ExpertImitateLearningService()
                expert_service, imitation_metadata = await self._get_cached_model_or_fetch(
                    model_type="imitation_learning",
                    track_name=trackName,
                    car_name=carName,
                    model_subtype="imitation_model_data",
                    deserializer_func=expert_service.deserialize_imitation_model
                )
                expert_state_list = expert_service.extract_expert_state_for_telemetry([telemetry_dict])
                if expert_state_list and len(expert_state_list) > 0:
                    # Since we passed a single telemetry record, extract the single result dictionary
                    expert_state_features = expert_state_list[0]
                    context_data.update(expert_state_features)
                    print(f"[INFO] Added {len(expert_state_features)} expert state features to context")
            except Exception as e:
                print(f"[Error] Expert service failed: {e}")   
                
            # Use transformer model's predict_human_readable method
            print(f"[INFO] Generating predictions with sequence length: {sequence_length}")
            print(f"[INFO] Context data keys: {list(context_data.keys()) if context_data else 'No context data'}")
            print(f"[INFO] Total context features: {len(context_data)}")
            
            predictions = transformer_model.predict_human_readable(
                current_telemetry=telemetry_dict,
                context_data=context_data if context_data else None,
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

        segment_length = 50  # Default segment length for transformer training
        try:
            # enrich data
            self._print_section_divider("ENRICHING CONTEXTUAL DATA")
            combined_segments = await self.enriched_contextual_data(top_laps_telemetry_list, bottom_laps_telemetry_list, trackName, segment_length=segment_length)
        except Exception as e:
            return {"error": str(e)}
        
        try:
        # train transformer model
            self._print_section_divider("TRAINING TRANSFORMER MODEL")
            transformer_results = await self._train_expert_action_transformer(
                combined_segments=combined_segments,  # combined_segments contain both telemetry and context data
                trackName=trackName,
                fixed_segment_length=segment_length  # Use default segment length
            )
        except Exception as e:
            return {"error": str(e)}
        
        self._print_section_divider("TRANSFORMER LEARNING COMPLETED")
        return {
            "success": True,
            "transformer_training": transformer_results,
            "track_name": trackName
        }

    async def _train_expert_action_transformer(self, 
                                             combined_segments: List[List[Dict[str, Any]]],
                                             trackName: str,
                                             fixed_segment_length: int = 50,
                                             enrichment_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the transformer model to learn non-expert driver progression toward expert performance using fixed-size segments
        
        Args:
            combined_segments: List of fixed-size segments containing combined telemetry and context data
            trackName: Track name for model identification
            fixed_segment_length: Length that all segments must have (default: 50)
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
            
            print(f"[INFO] Starting transformer training with {len(combined_segments)} fixed-size segments")
            print(f"[INFO] Each segment has fixed length: {fixed_segment_length}")
            
            # Validate segments before creating dataset
            validation_result = TelemetryActionDataset.validate_segments(
                combined_segments=combined_segments,
                expected_length=fixed_segment_length
            )
            
            if not validation_result['is_valid']:
                error_details = "\n".join(validation_result['errors'][:5])  # Show first 5 errors
                raise ValueError(f"Segment validation failed:\n{error_details}")
            
            print(f"[INFO] ✓ Validated {validation_result['num_segments']} segments")
            print(f"[INFO] ✓ Valid segments: {validation_result['statistics']['valid_segments']}")
            
            # Create fixed-size segmented dataset
            dataset = TelemetryActionDataset(
                combined_segments=combined_segments,
                fixed_segment_length=fixed_segment_length
            )
            
            # Configure PyTorch optimizations
            use_cuda = torch.cuda.is_available()
            
            if use_cuda:
                # Enable TF32 on Ampere+ for faster matmuls while keeping accuracy
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
            
            print(f"[INFO] Using all {len(dataset)} segments for training")
            print(f"[INFO] Total training samples: {len(dataset) * fixed_segment_length}")
            print(f"[INFO] Device: {'CUDA' if use_cuda else 'CPU'}")
            print(f"[INFO] No validation split - using full dataset for training")
            
            # Get feature dimensions from dataset
            input_feature_names, action_feature_names = dataset.get_feature_names()
            input_features_count = len(input_feature_names)
            
            print(f"[INFO] Dataset info: {input_features_count} combined input features, "
                  f"{len(action_feature_names)} action features")
            print(f"[INFO] Model will output 4 action features: gas, brake, steer_angle, gear")
            print(f"[INFO] Fixed segment length: {fixed_segment_length}")
            
            # Create model with unified input features
            model = ExpertActionTransformer(
                input_features_count=input_features_count,
                d_model=256,
                nhead=8,
                num_layers=20,  # Smaller model for faster training
                sequence_length=fixed_segment_length  # Use fixed segment length
            )
            
            # Create trainer
            device = 'cuda' if use_cuda else 'cpu'
            trainer = ExpertActionTrainer(model, device=device, learning_rate=1e-4)
        
            
            # Train model using the new fixed-size segment approach
            training_history = trainer.train(
                train_dataset=dataset,
                val_dataset=None,  # No validation split for now
                epochs=30,
                patience=10
            )
            
            # Evaluate model on training data
            test_metrics = trainer.evaluate(dataset)
            
            # Serialize model
            serialized_model = model.serialize_model()
            
            # Save to backend
            await backend_service.save_ai_model(
                model_type="transformer_expert_action",
                track_name=trackName,
                car_name='AllCars',
                model_data=serialized_model,
                metadata={
                    "training_history": training_history,
                    "test_metrics": test_metrics,
                    "feature_names": {
                        "input_features": input_feature_names,
                        "action_features": action_feature_names
                    },
                    "training_timestamp": datetime.now().isoformat()
                },
                is_active=True
            )
            
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

    async def enriched_contextual_data(self, top_telemetry_list: List[Dict[str, Any]], bottom_telemetry_list: List[Dict[str, Any]], track_name: str, segment_length: int) -> List[List[Dict[str, Any]]]:
        """
        Extract enriched contextual features from telemetry data using trained models. it adds expert state, corner identification, and tire grip features,
        and helps transformer model to better understand track geometry, physics constraints, extra expert insights to differentiate actions that
        converge towards feature expert state.
        
        This method returns combined segmented data for the unified transformer model.
        
        Args:
            top_telemetry_list: List of expert telemetry record dictionaries (flat list)
            bottom_telemetry_list: List of non-expert telemetry record dictionaries (flat list)
            track_name: Track name for model identification
            
        Returns:
            List[List[Dict[str, Any]]] - List of combined telemetry+context segments
        """


        if not bottom_telemetry_list:
            print("[WARNING] No telemetry data provided for enrichment")
            return []
        
        try:
            # Use all telemetry data for both training enrichment models and feature extraction
            bottom_training_telemetry_list = bottom_telemetry_list.copy()
            top_training_telemetry_list = top_telemetry_list.copy()

            self._print_section_divider("TRAINING IMITATION LEARNING MODEL")
            try:        
                imitation_learning = ExpertImitateLearningService()
                # Train imitation model only on top (expert) telemetry laps
                imitation_result = imitation_learning.train_ai_model(top_training_telemetry_list)
            
                # Save imitation learning model to backend
                await backend_service.save_ai_model(
                    model_type="imitation_learning",
                    track_name=track_name,
                    car_name='AllCars',
                    model_data=imitation_result.get("modelData", {}),
                    metadata=imitation_result.get("metadata", {}),
                    is_active=True
                )
            except Exception as e:
                raise Exception(f"[ERROR] Imitation learning training failed: {str(e)}")

            # Train corner identification model using training data
            self._print_section_divider("Training corner identification model...")
            try:
                from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
                corner_service = CornerIdentificationUnsupervisedService()
                corner_model = await corner_service.learn_track_corner_patterns(top_training_telemetry_list)

                corner_serialized = corner_service.serialize_corner_identification_model(track_name=track_name, car_name="all_cars")
                await self.backend_service.save_ai_model(
                    model_type="corner_identification",
                    track_name=corner_serialized.get("track_name"),
                    car_name='all_cars',
                    model_data=corner_serialized,
                    metadata={
                        "total_corners": corner_serialized.get("total_corners"),
                        "clusters": len(corner_serialized.get("corner_clusters", [])),
                        "serialization_timestamp": corner_serialized.get("serialized_timestamp")
                        },
                        is_active=True
                    )
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
                await self.backend_service.save_ai_model(
                    model_type="tire_grip_analysis",
                    track_name="generic",
                    car_name="all_cars",
                    model_data=tire_service_serialized,
                    metadata={
                        "model_info": tire_service_serialized.get("model_info", {}),
                        "serialization_timestamp": tire_service_serialized.get("serialized_timestamp")
                    },
                    is_active=True
                )
            except Exception as e:
                raise Exception(f"[ERROR] Tire grip analysis training failed: {str(e)}")
            
            # First, combine telemetry with context features
            self._print_section_divider("COMBINING TELEMETRY WITH CONTEXT FEATURES")
            
            # Extract features for all bottom telemetry records
            feature_sources = []
            
            # Extract expert state features for all records
            try:
                imitation_state_features = imitation_learning.extract_expert_state_for_telemetry(bottom_training_telemetry_list)
                print(f"[INFO] Extracted expert state features for {len(imitation_state_features)} records")
                feature_sources.append("expert_state")
            except Exception as e:
                print(f"[WARNING] Failed to extract expert state features: {str(e)}")
                imitation_state_features = []
            
            # Extract corner features for all records
            corner_enriched_data = []
            if corner_model.get("success"):
                try:
                    corner_enriched_data = await corner_service.extract_corner_features_for_telemetry(bottom_training_telemetry_list)
                    feature_sources.append("corner_identification")
                    print(f"[INFO] Extracted corner features for {len(corner_enriched_data)} records")
                except Exception as e:
                    print(f"[WARNING] Failed to extract corner features: {str(e)}")
            
            # Extract tire grip features for all records
            try:
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
                    print(f"[WARNING] Tire grip feature validation skipped: {v_err}")
                    
                feature_sources.append("tire_grip_analysis")
                print(f"[INFO] Extracted tire grip features for {len(grip_enriched_data)} records")
            except Exception as e:
                print(f"[WARNING] Failed to extract tire grip features: {str(e)}")
                grip_enriched_data = []
            
            # Combine telemetry with all context features
            combined_telemetry_data = []
            for i, telemetry_record in enumerate(bottom_training_telemetry_list):
                combined_record = telemetry_record.copy()
                
                # Add expert state features if available
                if i < len(imitation_state_features):
                    combined_record.update(imitation_state_features[i])
                
                # Add corner features if available
                if i < len(corner_enriched_data):
                    combined_record.update(corner_enriched_data[i])
                
                # Add tire grip features if available
                if i < len(grip_enriched_data):
                    combined_record.update(grip_enriched_data[i])
                
                combined_telemetry_data.append(combined_record)
            
            print(f"[INFO] Combined {len(combined_telemetry_data)} telemetry records with context features")
            
            # Now filter the combined data into optimal segments
            self._print_section_divider("FILTERING COMBINED DATA INTO SEGMENTS")
            try:
                # Use the imitation learning service to filter the combined data into segments
                combined_segments = imitation_learning.filter_optimal_telemetry_segments(combined_telemetry_data, segment_length=segment_length)
                print(f"[INFO] Created {len(combined_segments)} combined segments from filtered data")
            except Exception as e:
                raise Exception(f"Failed to filter combined data into segments: {str(e)}")
            
            # Create feature metadata
            sample_combined = {}
            if combined_segments and combined_segments[0]:
                sample_combined = combined_segments[0][0]
            
            grip_sample_enriched = grip_enriched_data[0] if grip_enriched_data else {}
            corner_sample_enriched = corner_enriched_data[0] if corner_enriched_data else {}
            expert_sample = imitation_state_features[0] if imitation_state_features else {}
            
            feature_metadata = {
                'sources': feature_sources,
                'feature_count': len(sample_combined),
                'feature_names': list(sample_combined.keys()),
                'corner_features': list(corner_sample_enriched.keys()),
                'grip_features': list(grip_sample_enriched.keys()),
                'expert_state_features': list(expert_sample.keys()),
                'total_segments': len(combined_segments),
                'total_records': len(combined_telemetry_data),
                'corner_identification_success': corner_model.get("success", False),
                'tire_grip_analysis_success': tire_grip_model.get("success", False)
            }
            
            print(f"[INFO] Feature metadata: {feature_metadata['feature_count']} combined features from {len(feature_sources)} sources")
            print(f"[INFO] Successfully created {len(combined_segments)} segments with combined telemetry and context data")
            
            return combined_segments
            
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