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
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from datetime import datetime
from pathlib import Path
from app.models import AiModelDto, ActiveModelData
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

# Import hybrid data cache service
from .hybrid_data_cache_service import hybrid_data_cache, get_shared_data_cache

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
        
        # Backend service integration
        self.backend_service = backend_service
        
        # Model cache service integration
        self.model_cache = model_cache_service
        
        # Use shared hybrid data cache for large datasets
        self.data_cache = hybrid_data_cache
        print(f"[INFO] Using shared hybrid data cache for large dataset processing")
        print(f"[INFO] Shared cache can be reused across backend_service, full_dataset_ml_service, and imitate_expert_learning_service")
        
        # Add a simple lock mechanism to prevent concurrent fetches of the same model
        self._model_fetch_locks = {}
        self._lock_creation_lock = asyncio.Lock()
        
        # Clear entire cache to ensure we start fresh with model instances only
        self.model_cache.clear()
        print("[INFO] Cleared entire cache on startup - will cache model instances directly")
        
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
        print("Caching Strategy: Model instances cached directly")
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
                                    deserializer_func=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Fetch model from backend and cache the model instance directly.
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            track_name: Track name for the model (optional)
            car_name: Car name for the model (optional)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize model data from backend and return model instance
            
        Returns:
            Tuple of (model_instance, metadata)
        """
        print(f"[DEBUG] Fetching model from backend: {model_type}/{track_name}/{car_name}")
        
        # getCompleteActiveModelData now always returns ActiveModelData or raises exception
        model_response: ActiveModelData = await self.backend_service.getCompleteActiveModelData(
            track_name, car_name, model_type
        )
        # Prepare cache metadata with all available structured data
        cache_metadata = {
            "track_name": model_response.trackName,
            "car_name": model_response.carName,
            "model_type": model_response.modelType,
            "is_active": model_response.isActive,
            "fetched_at": datetime.now().isoformat(),
            "backend_metadata": model_response.metadata,
            "model_subtype": model_subtype
        }
        
        # Deserialize the model instance
        if deserializer_func:
            print(f"[DEBUG] Deserializing model instance: {model_type}")
            model_instance = deserializer_func(model_response.modelData)
            if model_instance is None:
                raise Exception("Deserializer function returned None - must return model instance")
        else:
            raise ValueError("deserializer_func is required to deserialize model data")
        
        # Cache the model instance directly
        print(f"[DEBUG] Caching model instance: {model_type}")
        self.model_cache.put(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            data=model_instance,  # Store model instance directly
            metadata=cache_metadata,
            model_subtype=model_subtype
        )
        
        print(f"[DEBUG] Successfully cached model instance: {model_type}")
        
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
        Caches and retrieves model instances directly.
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            track_name: Track name for the model (optional)
            car_name: Car name for the model (optional)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize model data from backend and return model instance
            
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
                    model_instance, metadata = cached_result
                    print(f"[DEBUG] Retrieved cached model instance for {cache_key}")
                    
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
                    model_instance, metadata = cached_result
                    print(f"[DEBUG] Retrieved cached model instance (double-check) for {cache_key}")
                    
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
                        model_instance, metadata = cached_result
                        
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
                    model_instance, metadata = await self._fetch_and_cache_model(
                        model_type=model_type,
                        track_name=track_name,
                        car_name=car_name,
                        model_subtype=model_subtype,
                        deserializer_func=deserializer_func
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
            analysis["recommendations"].append("Large models are consuming >80% of cache memory. Consider increasing max_memory_mb or clearing unused models.")
        
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
        
        # Models are now cached as instances directly - no additional configuration needed
        optimization_actions.append("Model instances are cached directly for optimal performance")
        
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
        
        # 3. Log new configuration
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
            
            # Fetch and load the trained model using the class method deserializer
            transformer_model, model_metadata = await self._get_cached_model_or_fetch(
                model_type="transformer_expert_action",
                track_name=trackName,
                car_name=carName,
                model_subtype="transformer_model_data",
                deserializer_func=ExpertActionTransformer.deserialize_transformer_model
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
                # Use the single telemetry record as a list for extraction
                corner_features_list = await corner_service.extract_corner_features_for_telemetry([telemetry_dict])
                if corner_features_list and len(corner_features_list) > 0:
                    # Since we passed a single telemetry record, extract the single result dictionary
                    corner_features = corner_features_list[0]
                    print(f"[DEBUG] Corner features extracted: {corner_features}")
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
        
        # Validate dependencies before starting pipeline
        self._print_section_divider("VALIDATING PIPELINE DEPENDENCIES")
        
        # Check PyTorch availability for transformer training
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available - transformer functionality is disabled. Please install PyTorch to use this pipeline.")
        
        # Validate backend service
        if not hasattr(self, 'backend_service') or self.backend_service is None:
            raise ValueError("Backend service is not initialized - cannot proceed with pipeline")
        
        # Validate data cache service
        if not hasattr(self, 'data_cache') or self.data_cache is None:
            raise ValueError("Data cache service is not initialized - cannot proceed with pipeline")
        
        # Validate track name
        if not trackName or not isinstance(trackName, str) or len(trackName.strip()) == 0:
            raise ValueError("Invalid trackName provided - must be a non-empty string")
        
        print(f"[INFO] ✓ All dependencies validated successfully")
        print(f"[INFO] ✓ PyTorch available: {TORCH_AVAILABLE}")
        print(f"[INFO] ✓ Backend service initialized")
        print(f"[INFO] ✓ Data cache service initialized") 
        print(f"[INFO] ✓ Track name validated: '{trackName}'")
        
        self._print_section_divider("STREAMING TELEMETRY DATA FROM BACKEND DIRECTLY TO CACHE")
        
        # Always fetch from backend (no cache check)
        try:
            # Stream sessions directly to cache without loading into memory
            sessions_metadata = await backend_service.get_all_racing_sessions_streaming(
                trackName=trackName
            )
            
            if not sessions_metadata.get("success", False):
                raise Exception(f"Failed to stream sessions: {sessions_metadata.get('message', 'Unknown error')}")
            
            print(f"[INFO] Successfully streamed {sessions_metadata['total_sessions']} sessions directly to cache")
            print(f"[INFO] Estimated {sessions_metadata['summary']['estimated_total_records']} total records")
            
        except Exception as e:
            print(f"[ERROR] Failed to stream sessions from backend: {str(e)}")
            raise Exception(f"Backend streaming failed: {str(e)}") from e

        # Validate that we have sessions from backend streaming
        total_sessions = sessions_metadata.get("total_sessions", 0)
        total_records = sessions_metadata.get("summary", {}).get("estimated_total_records", 0)
        
        if total_sessions == 0:
            raise ValueError("No sessions found")
        if total_records == 0:
            raise ValueError("No telemetry data found")
        
        print(f"[INFO] Total sessions: {total_sessions}")
        print(f"[INFO] Total telemetry records: {total_records}")
        
        self._print_section_divider("LARGE DATASET ASSUMED - USING EFFICIENT PROCESSING")
        
        # Always use efficient processing for large datasets - no fallback
        # sessions_summary data is already cached, process directly from cache
        top_laps_telemetry_list, bottom_laps_cache_key = self.process_large_dataset_efficiently(
            trackName=trackName,
            max_memory_records=100000
        )

        segment_length = 50  # Default segment length for transformer training
        segments_cache_key = None
        transformer_results = None
        
        try:
            # enrich data - now using cache-based approach to avoid memory overflow
            self._print_section_divider("ENRICHING CONTEXTUAL DATA")
            segments_cache_key = await self.enriched_contextual_data(top_laps_telemetry_list, bottom_laps_cache_key, trackName, segment_length=segment_length)
            
            # remove any unused variable to this point to free memory
            del top_laps_telemetry_list
            
            # train transformer model
            self._print_section_divider("TRAINING TRANSFORMER MODEL")
            transformer_results = await self._train_expert_action_transformer(
                segments_cache_key=segments_cache_key,  # Pass cache key instead of segments in memory
                trackName=trackName,
                fixed_segment_length=segment_length  # Use default segment length
            )
            
        finally:
            # Clean up cached data even if processing fails
            if bottom_laps_cache_key:
                try:
                    self.data_cache.clear_cache(bottom_laps_cache_key)
                    print(f"[INFO] Cleaned up cached bottom laps data: {bottom_laps_cache_key}")
                except Exception as e:
                    print(f"[WARNING] Failed to clean up cached bottom laps data: {str(e)}")
            
            if segments_cache_key:
                try:
                    self.data_cache.clear_cache(segments_cache_key)
                    print(f"[INFO] Cleaned up cached segments data: {segments_cache_key}")
                except Exception as e:
                    print(f"[WARNING] Failed to clean up cached segments: {str(e)}")
        
        # Ensure transformer training completed successfully before returning
        if not transformer_results or not transformer_results.get("success"):
            raise Exception("Transformer training failed - no valid results produced")
        
        self._print_section_divider("TRANSFORMER LEARNING COMPLETED")
        return {
            "success": True,
            "transformer_training": transformer_results,
            "track_name": trackName
        }

    def process_sessions_streaming(self, sessions_data: List[Dict[str, Any]], 
                                 chunk_size: int = 5000) -> Iterator[List[Dict[str, Any]]]:
        """
        Process session data in streaming chunks to avoid memory overflow
        
        Args:
            sessions_data: List of session dictionaries
            chunk_size: Number of records per chunk
            
        Yields:
            Chunks of processed telemetry records
        """
        print(f"[INFO] Processing {len(sessions_data)} sessions in streaming chunks of {chunk_size}")
        
        current_chunk = []
        processed_sessions = 0
        
        for session in sessions_data:
            session_data = session.get("data", [])
            if not session_data:
                continue
                
            # Add session records to current chunk
            current_chunk.extend(session_data)
            processed_sessions += 1
            
            # Yield chunk when it reaches the size limit
            while len(current_chunk) >= chunk_size:
                yield current_chunk[:chunk_size]
                current_chunk = current_chunk[chunk_size:]
            
            # Progress logging
            if processed_sessions % 10 == 0:
                print(f"[INFO] Processed {processed_sessions}/{len(sessions_data)} sessions")
        
        # Yield remaining records
        if current_chunk:
            yield current_chunk
        
        print(f"[INFO] Completed streaming processing of {processed_sessions} sessions")

    def process_large_dataset_efficiently(self, trackName: str,
                                        max_memory_records: int = 50000) -> Tuple[List[Dict[str, Any]], str]:
        """
        Process very large cached datasets efficiently using streaming and Dask
        Optimized for datasets that cannot fit in memory - stores bottom laps in cache instead of memory
        
        Args:
            trackName: Track name for data retrieval
            segment_length: Length of segments for processing
            max_memory_records: Maximum records to keep in memory at once (reduced for large datasets)
            
        Returns:
            Tuple of (top_laps_telemetry_list, bottom_laps_cache_key) where cache_key is used to access bottom laps via data_cache
        """
        print(f"[INFO] Processing very large dataset for {trackName} with conservative memory limit {max_memory_records}")
        print(f"[INFO] Using streaming approach with Dask/HDF5 backend")
        
        def process_chunk(chunk_df: pd.DataFrame) -> Dict[str, Any]:
            """Process a single chunk of data with optimized memory usage"""
            try:
                if chunk_df.empty:
                    return {"top_laps": [], "bottom_laps": [], "processed_records": 0}
                
                print(f"[DEBUG] Processing chunk with {len(chunk_df)} records")
                
                feature_processor = FeatureProcessor(chunk_df)
                
                # Clean and filter data
                processed_df = feature_processor.general_cleaning_for_analysis()
                
                if processed_df.empty:
                    return {"top_laps": [], "bottom_laps": [], "processed_records": 0}
                
                # Filter to relevant features early to reduce memory usage
                telemetry_features = TelemetryFeatures()
                relevant_features = telemetry_features.get_features_for_imitate_expert()
                processed_df = feature_processor.filter_features_by_list(processed_df, relevant_features)
                
                if processed_df.empty:
                    return {"top_laps": [], "bottom_laps": [], "processed_records": 0}
                
                # Extract performance laps with strict filtering
                _, lap_df_list = feature_processor._filter_top_performance_laps(processed_df, 1)
                
                if not lap_df_list:
                    return {"top_laps": [], "bottom_laps": [], "processed_records": len(processed_df)}
                
                # More aggressive filtering for very large datasets - take only top 0.5%
                top_laps_df_count = max(1, int(len(lap_df_list) * 0.005))  # Top 0.5% for very large datasets
                top_laps_df = lap_df_list[:top_laps_df_count]
                
                # Use ALL remaining laps as bottom laps (all non-expert laps for training)
                bottom_laps_df = lap_df_list[top_laps_df_count:]
                
                # Convert to records efficiently
                top_records = []
                for lap_df in top_laps_df:
                    top_records.extend(lap_df.to_dict('records'))
                
                bottom_records = []
                for lap_df in bottom_laps_df:
                    bottom_records.extend(lap_df.to_dict('records'))
                
                # Clear DataFrames to free memory
                del processed_df, lap_df_list, top_laps_df, bottom_laps_df
                
                return {
                    "top_laps": top_records,
                    "bottom_laps": bottom_records,
                    "processed_records": len(chunk_df)
                }
                
            except Exception as e:
                print(f"[WARNING] Failed to process chunk of {len(chunk_df) if not chunk_df.empty else 0} records: {e}")
                return {"top_laps": [], "bottom_laps": [], "processed_records": 0}
        
        try:
            print("[INFO] Initiating streaming processing with hybrid cache backend")
            
            # Use hybrid cache streaming processing with conservative chunk size
            chunk_results = self.data_cache.process_large_dataset_streaming(
                track_name=trackName,
                processing_func=process_chunk,
                chunk_size=max_memory_records
            )
            
            if not chunk_results:
                raise ValueError(f"No data chunks returned from cache for {trackName}")
            
            # Aggregate top laps in memory but cache bottom laps to avoid memory overflow
            top_laps_telemetry_list = []
            bottom_laps_cache_key = f"bottom_laps_{trackName}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            total_processed = 0
            chunks_processed = 0
                            
            # Re-iterate through results for processing since generator was consumed
            chunk_results_list = list(chunk_results)
            total_bottom_laps_count = 0
            
            for result in chunk_results_list:
                if isinstance(result, dict) and result:
                    top_laps_telemetry_list.extend(result.get("top_laps", []))
                    total_bottom_laps_count += len(result.get("bottom_laps", []))
                    total_processed += result.get("processed_records", 0)
                    chunks_processed += 1
                    
                    # Progress logging for large datasets
                    if chunks_processed % 10 == 0:
                        print(f"[INFO] Processed {chunks_processed} chunks, {total_processed} total records")
            
            # Cache bottom laps using streaming to avoid memory accumulation
            print(f"[INFO] Caching {total_bottom_laps_count} bottom laps to avoid memory overflow")
            
            def create_bottom_laps_generator():
                """Recreate generator for caching"""
                chunk_idx = 0
                for result in chunk_results_list:
                    if isinstance(result, dict) and result:
                        bottom_laps = result.get("bottom_laps", [])
                        if bottom_laps:
                            yield {
                                "sessionId": f"bottom_laps_chunk_{chunk_idx}",
                                "data": bottom_laps
                            }
                            chunk_idx += 1
            
            # Estimate size for caching decision
            estimated_size_mb = (total_bottom_laps_count * 50) / (1024 * 1024)  # Rough estimate: 50 bytes per record
            
            cache_success = self.data_cache.cache_sessions_streaming(
                track_name=bottom_laps_cache_key,
                sessions_iterator=create_bottom_laps_generator(),
                estimated_size_mb=estimated_size_mb
            )
            
            if not cache_success:
                print(f"[WARNING] Failed to cache bottom laps, this may cause memory issues")
            
            if not top_laps_telemetry_list and total_bottom_laps_count == 0:
                raise ValueError(f"No valid telemetry data extracted from {trackName} dataset")
            
            print(f"[SUCCESS] Processed {chunks_processed} chunks, {total_processed} records total")
            print(f"[SUCCESS] Extracted {len(top_laps_telemetry_list)} expert records, cached {total_bottom_laps_count} training records")
            print(f"[INFO] Bottom laps cached with key: {bottom_laps_cache_key}")
            
            return top_laps_telemetry_list, bottom_laps_cache_key
            
        except Exception as e:
            print(f"[ERROR] Large dataset processing failed: {e}")
            print(f"[ERROR] This indicates an issue with the hybrid cache or data processing pipeline")
            raise Exception(f"Failed to process large dataset for {trackName}: {str(e)}")

    def get_data_cache_info(self) -> Dict[str, Any]:
        """Get information about the shared hybrid data cache"""
        from .hybrid_data_cache_service import get_shared_cache_info
        return get_shared_cache_info()

    def clear_data_cache(self, track_name: Optional[str] = None):
        """Clear hybrid data cache"""
        self.data_cache.clear_cache(track_name)
        print(f"[INFO] Cleared data cache" + (f" for {track_name}" if track_name else ""))

    def print_data_cache_info(self):
        """Print detailed shared data cache information"""
        info = self.get_data_cache_info()
        print("\n" + "="*60)
        print("SHARED HYBRID DATA CACHE INFORMATION")
        print("="*60)
        
        # Print sharing information
        sharing_info = info.get('sharing_info', {})
        if sharing_info.get('is_shared'):
            print("Cache Sharing: ENABLED")
            print(f"Shared across: {', '.join(sharing_info.get('shared_across', []))}")
            print("Benefits:")
            for benefit in sharing_info.get('benefits', []):
                print(f"  • {benefit}")
            print("")
        
        print(f"Memory Cache: {info['memory_cache']['entries']}/{info['memory_cache']['max_entries']} datasets")
        print(f"Dask Enabled: {info['dask_enabled']}")
        if info.get('dask_client'):
            print(f"Dask Client: {info['dask_client']}")
        
        disk_info = info['disk_cache']
        print(f"Disk Cache: {len(disk_info['entries'])} entries, {disk_info['total_size_mb']:.1f}MB")
        print(f"Storage Directory: {disk_info['storage_directory']}")
        
        if disk_info['entries']:
            print("\nCached Datasets:")
            for entry in disk_info['entries'][:5]:  # Show first 5
                print(f"  - {entry['track_name']}: {entry['record_count']} records "
                      f"({entry['size_mb']:.1f}MB, {entry['storage_type']})")
        print("="*60 + "\n")

    async def _train_expert_action_transformer(self, 
                                             segments_cache_key: str,
                                             trackName: str,
                                             fixed_segment_length: int = 50,
                                             enrichment_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the transformer model to learn non-expert driver progression toward expert performance using fixed-size segments
        
        Args:
            segments_cache_key: Cache key to access segments from data cache
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
            
            # Load segments from cache
            print(f"[INFO] Loading segments from cache key: {segments_cache_key}")
            cached_segments_data = self.data_cache.get_cached_sessions(segments_cache_key, max_age_hours=24)
            
            if not cached_segments_data:
                raise ValueError(f"No cached segments found for key: {segments_cache_key}")
            
            # Extract segments from cached data structure
            combined_segments = []
            for session in cached_segments_data.get("sessions", []):
                segment_data = session.get("data", [])
                if segment_data:
                    combined_segments.append(segment_data)
            
            print(f"[INFO] Starting transformer training with {len(combined_segments)} fixed-size segments loaded from cache")
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
            
            # Clear combined_segments to free memory since dataset now has the data
            del combined_segments
            
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
        
            trainer.validate_training_data_quality(dataset)
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

    async def _extract_features_from_chunk_async(self, chunk_data: List[Dict[str, Any]], imitation_learning, corner_service, tire_service, feature_sources: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features from a chunk of telemetry data with proper async handling
        
        Args:
            chunk_data: List of telemetry records
            imitation_learning: Imitation learning service instance
            corner_service: Corner identification service instance  
            tire_service: Tire grip service instance
            feature_sources: List to track active feature sources
            
        Returns:
            List of combined telemetry records with context features
        """
        try:
            if not chunk_data:
                return []
            
            combined_chunk_data = []
            
            # Extract expert state features for chunk
            chunk_imitation_features = []
            try:
                chunk_imitation_features = imitation_learning.extract_expert_state_for_telemetry(chunk_data)
                if "expert_state" not in feature_sources:
                    feature_sources.append("expert_state")
            except Exception as e:
                print(f"[WARNING] Failed to extract expert state features for chunk: {str(e)}")
            
            # Extract corner features for chunk (with proper async)
            chunk_corner_features = []
            try:
                chunk_corner_features = await corner_service.extract_corner_features_for_telemetry(chunk_data)
                if "corner_identification" not in feature_sources:
                    feature_sources.append("corner_identification") 
            except Exception as e:
                print(f"[WARNING] Failed to extract corner features for chunk: {str(e)}")
            
            # Extract tire grip features for chunk (with proper async)
            chunk_grip_features = []
            try:
                chunk_grip_features = await tire_service.extract_tire_grip_features(chunk_data)
                if "tire_grip_analysis" not in feature_sources:
                    feature_sources.append("tire_grip_analysis")
            except Exception as e:
                print(f"[WARNING] Failed to extract tire grip features for chunk: {str(e)}")
            
            # Combine telemetry with all context features for this chunk
            for i, telemetry_record in enumerate(chunk_data):
                combined_record = telemetry_record.copy()
                
                # Add expert state features if available
                if i < len(chunk_imitation_features):
                    combined_record.update(chunk_imitation_features[i])
                
                # Add corner features if available  
                if i < len(chunk_corner_features):
                    combined_record.update(chunk_corner_features[i])
                
                # Add tire grip features if available
                if i < len(chunk_grip_features):
                    combined_record.update(chunk_grip_features[i])
                
                combined_chunk_data.append(combined_record)
            
            return combined_chunk_data
            
        except Exception as e:
            print(f"[WARNING] Failed to process chunk: {str(e)}")
            return []

    async def enriched_contextual_data(self, top_laps_telemetry_list: List[Dict[str, Any]], bottom_laps_cache_key: str, track_name: str, segment_length: int) -> str:
        """
        Extract enriched contextual features from telemetry data using trained models. it adds expert state, corner identification, and tire grip features,
        and helps transformer model to better understand track geometry, physics constraints, extra expert insights to differentiate actions that
        converge towards feature expert state.
        
        This method returns combined segmented data for the unified transformer model.
        IMPORTANT: This method processes cached bottom laps data in streaming chunks to avoid memory overflow.
        
        Args:
            top_laps_telemetry_list: List of expert telemetry record dictionaries (flat list)
            bottom_laps_cache_key: Cache key for accessing bottom laps data via data_cache (avoids memory overflow)
            track_name: Track name for model identification
            segment_length: Length of segments for processing
            
        Returns:
            str - Cache key for accessing the combined telemetry+context segments
        """

        print(f"[INFO] Processing contextual data:")
        print(f"  - Top laps (expert): {len(top_laps_telemetry_list)} records")
        print(f"  - Bottom laps (training): cached with key '{bottom_laps_cache_key}'")
        
        # Validate cached data exists
        cached_bottom_laps = self.data_cache.get_cached_sessions(bottom_laps_cache_key, max_age_hours=24)
        if not cached_bottom_laps:
            print("[ERROR] No cached bottom laps data found")
            raise ValueError("No cached bottom laps data found - unable to proceed with contextual data enrichment")
        
        # Get total count from cached data for logging
        total_bottom_laps_count = cached_bottom_laps.get("summary", {}).get("total_telemetry_records", 0)
        print(f"  - Bottom laps total count: {total_bottom_laps_count} records (cached)")
        
        print(f"[INFO] Using ALL data without sampling since we're processing via cache streaming")
        
        try:
            # Use only expert data for training enrichment models (memory efficient)
            # We'll process bottom laps via streaming later for feature extraction
            top_laps_training_telemetry_list = top_laps_telemetry_list  # Small dataset, safe to reference
            
            print(f"[INFO] Training enrichment models with expert data:")
            print(f"  - Expert samples: {len(top_laps_training_telemetry_list)}")
            print(f"  - Bottom laps will be processed via streaming from cache")

            self._print_section_divider("TRAINING IMITATION LEARNING MODEL")
            try:        
                imitation_learning = ExpertImitateLearningService()
                # Train imitation model only on top (expert) telemetry laps
                imitation_result = imitation_learning.train_ai_model(top_laps_training_telemetry_list)
            
                # Extract only serialized data for backend storage (same fix as in train_imitation_model)
                serialized_data = imitation_learning.serialize_learning_model()
                if not serialized_data:
                    print("[ERROR] No serialized_modelData found in imitation results!")
                    raise Exception("No serialized model data available from imitation learning")
                # Save imitation learning model to backend
                await backend_service.save_ai_model(
                    model_type="imitation_learning",
                    track_name=track_name,
                    car_name='AllCars',
                    model_data=serialized_data,
                    metadata=imitation_result.get("learning_summary", {}),
                    is_active=True
                )
            except Exception as e:
                raise Exception(f"[ERROR] Imitation learning training failed: {str(e)}")

            # Train corner identification model using training data
            self._print_section_divider("Training corner identification model...")
            try:
                from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
                corner_service = CornerIdentificationUnsupervisedService()
                corner_model = await corner_service.learn_track_corner_patterns(top_laps_training_telemetry_list)

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
            
            # Train tire grip analysis model using training data (use expert samples for training)
            self._print_section_divider("Training tire grip analysis model...")
            try:
                # The tire grip service is now heuristic-only: it computes features deterministically from physics telemetry
                from .tire_grip_analysis_service import TireGripAnalysisService
                tire_service = TireGripAnalysisService()
                # Use expert data for training the tire grip model
                tire_grip_model = await tire_service.train_tire_grip_model(top_laps_training_telemetry_list)
                
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
            
            # First, combine telemetry with context features using cache-based streaming
            self._print_section_divider("COMBINING TELEMETRY WITH CONTEXT FEATURES VIA CACHE STREAMING")
            
            # Process feature extraction using cache streaming to avoid memory issues
            chunk_size = 10000  # Process 10k records at a time
            all_combined_telemetry_data = []
            feature_sources = []
            
            print(f"[INFO] Processing feature extraction via cache streaming in chunks of {chunk_size} records")
            
            # Use cache streaming to process all bottom laps data with async feature extraction
            async def process_cached_data_with_async_features():
                """Process cached data with proper async feature extraction"""
                cached_sessions = self.data_cache.get_cached_sessions(bottom_laps_cache_key, max_age_hours=24)
                if not cached_sessions:
                    raise ValueError(f"No cached data found for {bottom_laps_cache_key}")
                
                all_combined_data = []
                processed_chunks = 0
                
                # Process each session chunk from cache
                for session in cached_sessions.get("sessions", []):
                    session_data = session.get("data", [])
                    if not session_data:
                        continue
                    
                    # Process in chunks to avoid memory overflow
                    for i in range(0, len(session_data), chunk_size):
                        chunk_data = session_data[i:i + chunk_size]
                        processed_chunks += 1
                        
                        print(f"[INFO] Processing chunk {processed_chunks}: {len(chunk_data)} records")
                        
                        # Extract features for this chunk (with proper async handling)
                        chunk_combined_data = await self._extract_features_from_chunk_async(
                            chunk_data, imitation_learning, corner_service, tire_service, feature_sources
                        )
                        
                        all_combined_data.extend(chunk_combined_data)
                        
                        # Clear chunk data to free memory
                        del chunk_data, chunk_combined_data
                        
                        if processed_chunks % 10 == 0:
                            print(f"[INFO] Processed {processed_chunks} chunks, {len(all_combined_data)} total records")
                
                return all_combined_data
            
            # Execute the async processing
            all_combined_telemetry_data = await process_cached_data_with_async_features()
            
            print(f"[INFO] Successfully processed {len(all_combined_telemetry_data)} records via cache streaming")
            print(f"[INFO] Active feature sources: {feature_sources}")
            
            print(f"[INFO] Combined {len(all_combined_telemetry_data)} telemetry records with context features")
            
            # Filter the combined data into optimal segments and cache them instead of keeping in memory
            self._print_section_divider("FILTERING COMBINED DATA INTO SEGMENTS AND CACHING")
            try:
                # Use the imitation learning service to filter the combined data into segments
                combined_segments = imitation_learning.filter_optimal_telemetry_segments(all_combined_telemetry_data, segment_length=segment_length)
                print(f"[INFO] Created {len(combined_segments)} combined segments from filtered data")
                
                # Clear all_combined_telemetry_data to free memory since we now have segments
                del all_combined_telemetry_data
                
                # Cache the segments to avoid memory accumulation
                segments_cache_key = f"segments_{track_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                def segments_generator():
                    """Generator that yields segments for caching"""
                    for idx, segment in enumerate(combined_segments):
                        yield {
                            "sessionId": f"segment_{idx}",
                            "data": segment
                        }
                        
                # Estimate size for caching decision
                total_segment_records = sum(len(segment) for segment in combined_segments)
                estimated_size_mb = (total_segment_records * 50) / (1024 * 1024)  # Rough estimate: 50 bytes per record
                
                print(f"[INFO] Caching {len(combined_segments)} segments (~{estimated_size_mb:.1f}MB) to avoid memory accumulation")
                
                cache_success = self.data_cache.cache_sessions_streaming(
                    track_name=segments_cache_key,
                    sessions_iterator=segments_generator(),
                    estimated_size_mb=estimated_size_mb
                )
                
                if not cache_success:
                    print(f"[WARNING] Failed to cache segments, keeping in memory as fallback")
                    # Return the segments directly if caching fails
                    return combined_segments
                
                # Clear segments from memory since they're now cached
                del combined_segments
                
                print(f"[INFO] Successfully cached segments with key: {segments_cache_key}")
                
                # Return cache key instead of segments
                return segments_cache_key
                
            except Exception as e:
                raise Exception(f"Failed to filter combined data into segments: {str(e)}")
            
        except Exception as e:
            raise Exception(f"{self.__dir__} Failed to enrich contextual data: {str(e)}")
    
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")