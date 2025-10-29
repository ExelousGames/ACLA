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
from .telemetry_segment_visualizer import visualize_optimal_segments
import numpy as np
import joblib
import warnings
import base64
import pickle
import io
import asyncio
import time
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator, AsyncIterator
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
from ..models.telemetry_models import (
    TelemetryFeatures,
    FeatureProcessor,
    _safe_float,
)

# Import backend service
from .backend_service import backend_service

# Import model cache service
from .model_cache_service import model_cache_service

# Import hybrid data cache service
from .Training_data_cache_service import training_cache

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
        
        # Use training-optimized data cache for large datasets
        self.data_cache = training_cache
        print(f"[INFO] Using training-optimized data cache for large dataset processing")
        print(f"[INFO] Parquet-only storage optimized for ML training pipelines")
        
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

    async def predict_expert_actions(self, 
                                   telemetry_dict: Dict[str, Any],
                                   trackName: str, 
                                   carName: Optional[str] = 'AllCars',
                                   sequence_length: int = 20) -> Dict[str, Any]:
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
            # Convert telemetry_dict to DataFrame for processing
            import pandas as pd
            telemetry_df = pd.DataFrame([telemetry_dict])
            
            # Process and filter data
            processor = FeatureProcessor(telemetry_df)
            processed_df = processor.general_cleaning_for_analysis()
            processor.add_time_delta()
            processor.flip_y_z_features()
            features = self._imitate_expert_feature_names or TelemetryFeatures().get_features_for_imitate_expert()

            filtered_telemetry_df = processor.filter_features_by_list(processed_df, features)
            
            # Convert back to dict for further processing
            if not filtered_telemetry_df.empty:
                processed_telemetry_dict = filtered_telemetry_df.iloc[0].to_dict()
            else:
                processed_telemetry_dict = telemetry_dict
                    
            # Fetch and load the trained model using the class method deserializer
            transformer_model, model_metadata = await self._get_cached_model_or_fetch(
                model_type="transformer_expert_action",
                track_name=trackName,
                car_name=carName,
                model_subtype="transformer_model_data",
                deserializer_func=ExpertActionTransformer.deserialize_transformer_model
            )
            
            # Enrich processed telemetry with additional model-derived features
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
                tire_features_list = await tire_grip_service.extract_tire_grip_features([processed_telemetry_dict])
                if tire_features_list and len(tire_features_list) > 0:
                    # Since we passed a single telemetry record, extract the single result dictionary
                    tire_features = tire_features_list[0]
                    processed_telemetry_dict.update(tire_features)
                    print(f"[INFO] Added {len(tire_features)} tire grip features to telemetry payload")
                
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
                expert_state_list = expert_service.extract_expert_state_for_telemetry([processed_telemetry_dict])
                if expert_state_list and len(expert_state_list) > 0:
                    # Since we passed a single telemetry record, extract the single result dictionary
                    processed_telemetry_dict.update(expert_state_list[0])
            except Exception as e:
                print(f"[Error] Expert service failed: {e}")   
                
            # Use transformer model's predict_human_readable method
            print(f"[INFO] Generating predictions with sequence length: {sequence_length}")
            print(f"[INFO] Telemetry payload keys: {list(processed_telemetry_dict.keys())}")
            print(f"telemetry data: {processed_telemetry_dict}")
            
            predictions = transformer_model.predict_human_readable(
                current_telemetry=processed_telemetry_dict,
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

        self._print_section_divider("STREAMING TELEMETRY DATA FROM BACKEND DIRECTLY TO CACHE")
        
        # Always fetch from backend (no cache check)
        try:
            # Generate explicit cache key for this dataset
            dataset_cache_key = f"racing_sessions_{trackName}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Stream sessions directly to cache without loading into memory
            sessions_metadata = await backend_service.get_all_racing_sessions_streaming(
                cache_key=dataset_cache_key,
                trackName=trackName
            )
            
            if not sessions_metadata.get("success", False):
                raise Exception(f"Failed to stream sessions: {sessions_metadata.get('message', 'Unknown error')}")
            
            print(f"[INFO] Successfully streamed {sessions_metadata['total_sessions']} sessions directly to cache")
            print(f"[INFO] Processed {sessions_metadata['summary']['data_points_processed']} total records")
            
        except Exception as e:
            print(f"[ERROR] Failed to stream sessions from backend: {str(e)}")
            raise Exception(f"Backend streaming failed: {str(e)}") from e

        # Validate that we have sessions from backend streaming
        total_sessions = sessions_metadata.get("total_sessions", 0)
        total_records = sessions_metadata.get("summary", {}).get("data_points_processed", 0)
        
        if total_sessions == 0:
            raise ValueError("No sessions found")
        if total_records == 0:
            raise ValueError("No telemetry data found")
        
        print(f"[INFO] Total sessions: {total_sessions}")
        print(f"[INFO] Total telemetry records: {total_records}")
        
        self._print_section_divider("LARGE DATASET ASSUMED - USING EFFICIENT PROCESSING")
        
        # Always use efficient processing for large datasets - no fallback
        # sessions_summary data is already cached, process directly from cache
        top_laps_telemetry_list, bottom_laps_cache_key = await self.process_large_dataset_efficiently(
            data_cache_key=dataset_cache_key,
            trackName=trackName,
            max_memory_records=10000,
            telemetry_time_gap_ms=300
        )

        segment_length = 20  # Default segment length for transformer training
        segments_cache_key = None
        transformer_results = {"success": False, "error": "Training not started"}  # Initialize with failure
        
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

    async def process_large_dataset_efficiently(self, data_cache_key: str, trackName: str,
                                        max_memory_records: int = 10000, telemetry_time_gap_ms: int = 100) -> Tuple[List[Dict[str, Any]], str]:
        """
        Streamlined processing of large cached datasets with a bounded memory footprint for bottom laps.

        Args:
            trackName: Track name for data retrieval
            max_memory_records: Maximum number of bottom-lap telemetry records kept in memory before flushing to cache

        Returns:
            Tuple of (top_laps_telemetry_list, bottom_laps_cache_key)
        """
        print(f"[INFO] Processing {trackName} dataset with {max_memory_records} in-memory bottom records")

        top_laps: List[Dict[str, Any]] = []
        bottom_laps_cache_key = f"bottom_laps_{trackName}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        bottom_buffer: List[Dict[str, Any]] = []
        bottom_buffer_records = 0
        total_bottom_laps_cached = 0
        total_processed = 0
        chunk_idx = 0

        features = self._imitate_expert_feature_names or self.telemetry_features.get_features_for_imitate_expert()

        async def flush_bottom_buffer(reason: str) -> None:
            nonlocal bottom_buffer, bottom_buffer_records, total_bottom_laps_cached
            if not bottom_buffer:
                return

            async def buffer_iterator() -> AsyncIterator[Dict[str, Any]]:
                for entry in bottom_buffer:
                    yield entry

            try:
                cache_success = await self.data_cache.cache_chunks_streaming(
                    cache_key=bottom_laps_cache_key,
                    chunks_iterator=buffer_iterator()
                )
            except Exception as cache_error:
                print(f"[WARNING] Exception while flushing bottom laps during {reason}: {cache_error}")
                return

            if cache_success:
                cached_count = len(bottom_buffer)
                print(f"[INFO] Cached {cached_count} bottom laps ({bottom_buffer_records} records) [{reason}] -> {bottom_laps_cache_key}")
                total_bottom_laps_cached += cached_count
                bottom_buffer = []
                bottom_buffer_records = 0
            else:
                print(f"[WARNING] Cache service reported failure while flushing bottom laps during {reason}")

        async def stage_bottom_lap(lap_entry: Dict[str, Any], current_chunk_idx: int) -> None:
            nonlocal bottom_buffer_records

            lap_records = lap_entry.get("records", [])
            record_count = len(lap_records)
            if record_count == 0:
                print(f"[DEBUG] Skipping bottom staging for lap {lap_entry.get('id')} with no records")
                return

            if bottom_buffer_records and bottom_buffer_records + record_count > max_memory_records:
                await flush_bottom_buffer("memory limit reached")

            if bottom_buffer_records and bottom_buffer_records + record_count > max_memory_records:
                print(f"[WARNING] Unable to clear bottom lap buffer; proceeding with {record_count} additional records")

            if record_count > max_memory_records:
                print(
                    f"[WARNING] Single lap {lap_entry['id']} ({record_count} records) exceeds memory limit {max_memory_records}; caching immediately"
                )
                bottom_buffer.append({
                    "chunkId": f"bottom_lap_{lap_entry['id']}_{current_chunk_idx}",
                    "data": lap_records
                })
                bottom_buffer_records = record_count
                await flush_bottom_buffer("single lap overflow")
                return

            bottom_buffer.append({
                "chunkId": f"bottom_lap_{lap_entry['id']}_{current_chunk_idx}",
                "data": lap_records
            })
            bottom_buffer_records += record_count

        def make_lap_entry(
            lap_identifier: str,
            lap_time_ms: Optional[float],
            lap_num: Any,
            lap_records: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            return {
                "id": lap_identifier,
                "lap_time_ms": lap_time_ms if lap_time_ms is not None else float("inf"),
                "lap_num": lap_num,
                "records": lap_records,
            }

        def update_top_laps(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Maintain the five fastest laps; return lap treated as bottom if applicable."""
            if len(top_laps) < 5:
                top_laps.append(candidate)
                print(
                    f"[DEBUG] Added lap {candidate['id']} to top laps ({len(top_laps)}/5, time: {candidate['lap_time_ms']}ms)"
                )
                return None

            slowest_idx = max(range(len(top_laps)), key=lambda idx: top_laps[idx]["lap_time_ms"])
            slowest = top_laps[slowest_idx]
            if candidate["lap_time_ms"] < slowest["lap_time_ms"]:
                top_laps[slowest_idx] = candidate
                print(
                    f"[DEBUG] Replaced slowest lap {slowest['id']} with {candidate['id']} (time: {candidate['lap_time_ms']}ms)"
                )
                return slowest

            return candidate

        session_chunks_iterator = self.data_cache.get_cached_data_chunks(cache_key=data_cache_key)
        print(f"[DEBUG] Created chunk iterator for cache key: {data_cache_key}")

        session_chunks_processed = 0
        for session_chunk_df in session_chunks_iterator:
            session_chunks_processed += 1

            if session_chunk_df is None or session_chunk_df.empty:
                print(f"[DEBUG] Chunk {session_chunks_processed} is empty, skipping")
                continue

            try:
                telemetry_df = session_chunk_df
                print(f"[DEBUG] Processing {len(telemetry_df)} number of telemetry records from the session {session_chunks_processed}")

                processor = FeatureProcessor(telemetry_df)
                processor.general_cleaning_for_analysis()
                processed_df = processor.flip_y_z_features()

                if processed_df.empty:
                    print(f"[DEBUG] Processed DataFrame empty for chunk {session_chunks_processed}, skipping")
                    continue

                lap_structs = processor.split_into_laps(processed_df)
                if not lap_structs:
                    print(f"[DEBUG] No laps identified in chunk {session_chunks_processed}, skipping")
                    continue

                lap_frames: List[pd.DataFrame] = [lap_struct["dataframe"] for lap_struct in lap_structs]
                stripped_lap_frames = processor.strip_dataframe_by_time_gap(lap_frames, telemetry_time_gap_ms)

                laps_processed_in_chunk = 0

                for lap_index, (lap_struct, stripped_df) in enumerate(zip(lap_structs, stripped_lap_frames)):
                    lap_metrics = lap_struct["metrics"]
                    lap_metrics["record_count_after_gap"] = len(stripped_df)

                    filtered_df = processor.filter_features_by_list(stripped_df, features) if not stripped_df.empty else pd.DataFrame()
                    filtered_count = len(filtered_df)

                    if stripped_df.empty:
                        print(
                            f"[DEBUG] Chunk {chunk_idx} lap {lap_index} empty after down-sampling; staging as bottom if data existed"
                        )
                    elif filtered_df.empty:
                        print(
                            f"[DEBUG] Chunk {chunk_idx} lap {lap_index} has no features after filtering; staging to bottom"
                        )

                    lap_records = filtered_df.to_dict("records") if not filtered_df.empty else []

                    lap_identifier = (
                        f"{chunk_idx}_{lap_struct['lap_sequence']}_{lap_metrics.get('lap_time_ms', 'na')}"
                    )

                    candidate_entry = make_lap_entry(
                        lap_identifier=lap_identifier,
                        lap_time_ms=lap_metrics["lap_time_ms"],
                        lap_num=lap_struct["lap_num"],
                        lap_records=lap_records,
                    )

                    if lap_metrics["is_full_valid"] and lap_metrics["lap_time_ms"] is not None and lap_records:
                        laps_processed_in_chunk += 1
                        print(
                            f"[DEBUG] Chunk {chunk_idx}: valid lap {lap_struct['lap_num']} ({filtered_count} records) time {lap_metrics['lap_time_ms']}ms"
                        )
                        bottom_candidate = update_top_laps(candidate_entry)
                        if bottom_candidate:
                            await stage_bottom_lap(bottom_candidate, chunk_idx)
                    else:
                        print(
                            f"[DEBUG] Chunk {chunk_idx}: staging non-top lap {lap_struct['lap_num']} (full={lap_metrics['is_full']}, valid={lap_metrics['is_valid']})"
                        )
                        await stage_bottom_lap(candidate_entry, chunk_idx)

                print(
                    f"[DEBUG] Chunk {chunk_idx}: Processed {laps_processed_in_chunk} full valid laps. Top laps: {len(top_laps)} Bottom buffer records: {bottom_buffer_records}"
                )

                total_processed += len(telemetry_df)
                chunk_idx += 1

            except ValueError as gap_error:
                raise gap_error
            except Exception as error:
                print(f"[WARNING] Chunk processing failed: {error}")
                continue

        await flush_bottom_buffer("final flush")

        print(f"[DEBUG] Finished processing all chunks:")
        print(f"[DEBUG] - Total sessions processed: {session_chunks_processed}")
        print(f"[DEBUG] - Valid sessions processed: {chunk_idx}")
        print(f"[DEBUG] - Total records processed: {total_processed}")
        print(f"[DEBUG] - Top laps found: {len(top_laps)}")
        print(f"[DEBUG] - Bottom laps cached: {total_bottom_laps_cached}")

        if top_laps:
            lap_times = [lap_info["lap_time_ms"] for lap_info in top_laps]
            print(f"[DEBUG] Top lap times: {sorted(lap_times)}")

        if not session_chunks_processed:
            raise ValueError(
                f"No chunks were returned by iterator for track {trackName}. Check if data exists in cache."
            )

        if not chunk_idx:
            raise ValueError(
                f"All {session_chunks_processed} chunks failed processing for track {trackName}. Check data quality."
            )

        if len(top_laps) < 5:
            raise ValueError(
                f"Insufficient top laps found: {len(top_laps)}/5 required. Processed {chunk_idx} valid chunks with {total_processed} records."
            )

        top_laps.sort(key=lambda entry: entry["lap_time_ms"])

        top_laps_telemetry_list: List[Dict[str, Any]] = []
        for lap_info in top_laps:
            top_laps_telemetry_list.extend(lap_info["records"])

        print(f"[SUCCESS] Processed {chunk_idx} chunks, {total_processed} records")
        print(f"[SUCCESS] Selected top 5 laps: {len(top_laps_telemetry_list)} records")
        print(f"[SUCCESS] Cached {total_bottom_laps_cached} bottom laps across {chunk_idx} chunks")

        return top_laps_telemetry_list, bottom_laps_cache_key

    def get_data_cache_info(self) -> Dict[str, Any]:
        """Get information about the training-optimized data cache"""
        return self.data_cache.get_cache_info()

    def clear_data_cache(self, track_name: Optional[str] = None):
        """Clear training-optimized data cache"""
        self.data_cache.clear_cache(track_name)
        print(f"[INFO] Cleared data cache" + (f" for {track_name}" if track_name else ""))

    def print_data_cache_info(self):
        """Print detailed training-optimized cache information"""
        info = self.get_data_cache_info()
        print("\n" + "="*60)
        print("TRAINING-OPTIMIZED DATA CACHE INFORMATION")
        print("="*60)
        
        print(f"Cache Entries: {info.get('cache_entries', 0)}")
        print(f"Total Size: {info.get('total_size_mb', 0):.1f}MB")
        print(f"Storage Format: {info.get('storage_format', 'Parquet with snappy compression')}")
        print(f"Cache Directory: {info.get('cache_directory', 'N/A')}")
        
        print("\nOptimizations:")
        print("  • Parquet-only storage for consistency")
        print("  • Snappy compression for fast I/O")
        print("  • Multi-part files for large datasets")
        print("  • Streaming processing with minimal memory footprint")
        print("  • Zero-copy operations for ML training pipelines")
        print("="*60 + "\n")

    async def _train_expert_action_transformer(self, 
                                             segments_cache_key: str,
                                             trackName: str,
                                             fixed_segment_length: int = 50) -> Dict[str, Any]:
        """
        Train the transformer model to learn non-expert driver progression toward expert performance using fixed-size segments
        
        Args:
            segments_cache_key: Cache key to access enriched segments from data cache (complete key, not track name)
            trackName: Track name for model identification
            fixed_segment_length: Length that all segments must have (default: 50)
            
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
            
            # Create streaming dataset that loads segments on-demand from cache
            print(f"[INFO] Creating streaming dataset from segments cache key: {segments_cache_key}")
            
            # Use streaming dataset that doesn't load all segments into memory
            dataset = TelemetryActionDataset(
                data_cache=self.data_cache,
                segments_cache_key=segments_cache_key,
                fixed_segment_length=fixed_segment_length
            )
            
            print(f"[INFO] ✓ Streaming dataset created with {len(dataset)} segments")
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
            print(f"[INFO] Device: {'CUDA' if use_cuda else 'CPU'}")
            
            # Get feature dimensions from dataset
            input_feature_names, action_feature_names = dataset.get_feature_names()
            input_features_count = len(input_feature_names)
            
            print(f"[INFO] Dataset info: {input_features_count} combined input features, "
                  f"{len(action_feature_names)} action features")
            
            # Create model with unified input features
            model = ExpertActionTransformer(
                total_features_count=input_features_count,
                d_model=256, # Embedding dimension
                nhead=16,  # Number of attention heads
                num_layers=20,  # Smaller model for faster training
                sequence_length=fixed_segment_length  # Use fixed segment length
            )
            
            # Create trainer
            device = 'cuda' if use_cuda else 'cpu'
            print(f"[DEBUG] Creating trainer on device: {device}")
            trainer = ExpertActionTrainer(model, device=device, learning_rate=1e-4)
        
            # Train model using the new fixed-size segment approach
            print(f"[DEBUG] Starting training loop...")
            training_history = trainer.train(
                train_dataset=dataset,
                val_dataset=None,  # No validation split for now
                epochs=30,
                patience=10
            )
            print(f"[DEBUG] Training loop completed successfully")
            
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



    async def _enrich_chunk_with_context(self, chunk_data: List[Dict[str, Any]], 
                                        imitation_learning: ExpertImitateLearningService, tire_service: TireGripAnalysisService) -> List[Dict[str, Any]]:
        """
        Enrich a single chunk with all contextual features
        
        Args:
            chunk_data: List of telemetry records
            imitation_learning: Trained imitation learning service
            tire_service: Trained tire grip service
            
        Returns:
            List of enriched telemetry records
        """
        if not chunk_data:
            return []
        
        # Extract all feature types for the chunk
        chunk_imitation_features = []
        try:
            chunk_imitation_features = imitation_learning.extract_expert_state_for_telemetry(chunk_data)
        except Exception as e:
            raise RuntimeError(f"Failed to extract imitation features: {str(e)}")
        
        chunk_grip_features = []
        try:
            chunk_grip_features = await tire_service.extract_tire_grip_features(chunk_data)
        except Exception as e:
            raise RuntimeError(f"Failed to extract tire grip features: {str(e)}")
        
        # Combine all features into enriched records
        enriched_chunk = []
        for i, telemetry_record in enumerate(chunk_data):
            enriched_record = telemetry_record.copy()
            
            # Add expert state features
            if i < len(chunk_imitation_features):
                enriched_record.update(chunk_imitation_features[i])
            
            # Add tire grip features
            if i < len(chunk_grip_features):
                enriched_record.update(chunk_grip_features[i])
            
            enriched_chunk.append(enriched_record)
        
        return enriched_chunk
    
    async def _cache_segment_batch(self, segments_batch: List[List[Dict[str, Any]]], 
                                 base_cache_key: str, batch_number: int) -> bool:
        """
        Cache a batch of segments to keep memory usage reasonable
        
        Args:
            segments_batch: List of segments to cache
            base_cache_key: Base cache key for segment storage (used directly)
            batch_number: Batch number for unique session IDs
            
        Returns:
            bool - Success status
        """
        try:
            # Store segments as structured chunk - cache service doesn't need to know about segments
            
            async def segments_generator():
                """Generator for caching - store complete chunk with segments intact"""
                # Package segments as a complete chunk - cache service treats this as opaque data
                chunk_data = {
                    "chunkId": f"batch_{batch_number}",  # Use chunkId format expected by cache service
                    "data": segments_batch,  # Store segments as data payload
                    "batch_number": batch_number,
                    "segment_count": len(segments_batch),
                    "total_records": sum(len(segment) for segment in segments_batch)
                }
                
                # Yield the complete chunk
                yield chunk_data
            
            # Estimate size for this batch
            total_records = sum(len(segment) for segment in segments_batch)
            estimated_size_mb = (total_records * 60) / (1024 * 1024)  # 60 bytes per enriched record
            
            # Cache using the proper cache service method with correct parameters
            cache_success = await self.data_cache.cache_chunks_streaming(
                cache_key=base_cache_key,
                chunks_iterator=segments_generator()
            )
            
            if cache_success:
                print(f"[DEBUG] Cached batch {batch_number}: {len(segments_batch)} segments as chunk (~{estimated_size_mb:.1f}MB) to key: {base_cache_key}")
                return True
            else:
                print(f"[ERROR] Failed to cache segment batch {batch_number} as chunk")
                return False
                
        except Exception as e:
            print(f"[ERROR] Exception caching segment batch {batch_number}: {str(e)}")
            return False

    async def enriched_contextual_data(self, top_laps_telemetry_list: List[Dict[str, Any]], bottom_laps_cache_key: str, track_name: str, segment_length: int) -> str:
        """
        Streamlined contextual data enrichment using chunk iterator approach.
        
        1. Train all enrichment models using expert data
        2. Use chunk_iterator to process all bottom laps data  
        3. Enrich each chunk with contextual features
        4. Filter into segments and cache them in reasonable chunks
        
        Args:
            top_laps_telemetry_list: List of expert telemetry records for training models
            bottom_laps_cache_key: Cache key for accessing cached bottom laps data via iterator
            track_name: Track name for model identification
            segment_length: Length of segments for transformer training
            
        Returns:
            str - Cache key for accessing the enriched segments
        """
        # Step 1: Train all enrichment models using expert data
        self._print_section_divider("TRAINING ENRICHMENT MODELS WITH EXPERT DATA")
        
        # Train imitation learning model
        imitation_learning = ExpertImitateLearningService()
        imitation_result = imitation_learning.train_ai_model(top_laps_telemetry_list)
        serialized_data = imitation_learning.serialize_learning_model()
        if not serialized_data:
            raise Exception("No serialized model data available from imitation learning")
        
        await backend_service.save_ai_model(
            model_type="imitation_learning",
            track_name=track_name,
            car_name='AllCars',
            model_data=serialized_data,
            metadata=imitation_result.get("learning_summary", {}),
            is_active=True
        )
        print("[INFO] ✓ Imitation learning model trained and saved")
        
        # Train corner identification model
        from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
        corner_service = CornerIdentificationUnsupervisedService()
        corner_model = await corner_service.learn_track_corner_patterns(top_laps_telemetry_list)
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
        print("[INFO] ✓ Corner identification model trained and saved")
        
        # Train tire grip analysis model using bottom laps (streaming)
        from .tire_grip_analysis_service import TireGripAnalysisService
        tire_service = TireGripAnalysisService()

        print("[INFO] Streaming bottom-lap telemetry to train tire grip model")
        bottom_laps_training_iterator = self.data_cache.get_cached_data_chunks(
            cache_key=bottom_laps_cache_key
        )
        tire_grip_training = await tire_service.train_tire_grip_model_streaming(
            chunk_iterator=bottom_laps_training_iterator,
            max_samples=250_000,
            random_state=42
        )
        tire_service_serialized = tire_service.serialize_tire_grip_model()
        
        await self.backend_service.save_ai_model(
            model_type="tire_grip_analysis",
            track_name="generic",
            car_name="all_cars",
            model_data=tire_service_serialized,
            metadata={
                "training_summary": tire_grip_training,
                "serialization_timestamp": tire_service_serialized.get("serialized_timestamp"),
                "feature_catalog": tire_service.feature_catalog.CONTEXT_FEATURES
            },
            is_active=True
        )
        print("[INFO] ✓ Tire grip analysis model trained on bottom laps and saved")
        
        # Step 2: Process bottom laps via chunk iterator with enrichment
        self._print_section_divider("PROCESSING BOTTOM get_cached_data_chunksLAPS VIA CHUNK ITERATOR")
        
        segments_cache_key = f"enriched_segments_{track_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        lao_chunks_iterator = self.data_cache.get_cached_data_chunks(
            cache_key=bottom_laps_cache_key  # Use the cache key where bottom laps are stored
        )
        
        processed_chunks = 0
        total_segments_cached = 0
        segments_per_cache = 5000  # Number of segments per cache batch
        
        current_segment_batch = []
        cache_batch_number = 0
        
        for lap_chunk_df in lao_chunks_iterator:
            if lap_chunk_df is None or lap_chunk_df.empty:
                continue
            
            processed_chunks += 1
            chunk_data = lap_chunk_df.to_dict('records')
            
            print(f"[INFO] Processing chunk {processed_chunks}: {len(chunk_data)} records")
            
            # Step 3: Enrich chunk with contextual features
            enriched_chunk_data = await self._enrich_chunk_with_context(
                chunk_data, imitation_learning, tire_service
            )
            
            # Step 4: Filter enriched chunk into segments
            chunk_segments = imitation_learning.filter_optimal_telemetry_segments(
                enriched_chunk_data,
                segment_length=segment_length,
                batch_number=cache_batch_number
            )
            
            print(f"[INFO] Chunk {processed_chunks}: Generated {len(chunk_segments)} segments")

            # Visualize this chunk immediately so artifacts reflect chunk-specific context
            if chunk_segments:
                try:
                    visualization_payloads = visualize_optimal_segments(
                        chunk_segments,
                        max_segments=1,
                        analyze_segment_fn=imitation_learning._analyze_segment_improvement,
                        file_name_prefix=f"{segments_cache_key}_chunk_{processed_chunks}",
                        return_base64=False
                    )
                    if visualization_payloads:
                        print(f"[INFO] Generated {len(visualization_payloads)} visualizations for chunk {processed_chunks}")
                except Exception as viz_error:
                    print(f"[WARN] Failed to visualize chunk {processed_chunks}: {viz_error}")
            
            # Add segments to current batch
            current_segment_batch.extend(chunk_segments)
            
            # Cache segments in batches of segments_per_cache size
            while len(current_segment_batch) >= segments_per_cache:
                # Take exactly segments_per_cache segments for caching
                batch_to_cache = current_segment_batch[:segments_per_cache]
                await self._cache_segment_batch(
                    batch_to_cache,
                    segments_cache_key,
                    cache_batch_number
                )
                total_segments_cached += len(batch_to_cache)
                cache_batch_number += 1
                
                # Remove the cached segments from current batch
                current_segment_batch = current_segment_batch[segments_per_cache:]
                
                print(f"[INFO] Cached chunk {cache_batch_number}: {len(batch_to_cache)} segments as single chunk, {total_segments_cached} total segments cached")
            
            # Clear chunk data to free memory
            del chunk_data, enriched_chunk_data, chunk_segments
            
            if processed_chunks % 5 == 0:
                print(f"[INFO] Progress: {processed_chunks} chunks processed, {total_segments_cached} segments cached")
        
        # Cache remaining segments in final batch
        if current_segment_batch:
            await self._cache_segment_batch(
                current_segment_batch,
                segments_cache_key,
                cache_batch_number
            )
            total_segments_cached += len(current_segment_batch)
            current_segment_batch = []
            cache_batch_number += 1
        
        print(f"[SUCCESS] Enrichment completed:")
        print(f"  - Processed {processed_chunks} data chunks from cache")  
        print(f"  - Generated and cached {total_segments_cached} enriched segments")
        print(f"  - Segments grouped into {cache_batch_number} cache chunks ({segments_per_cache} segments per chunk)")
        print(f"  - Cache key for all chunks: {segments_cache_key}")
        
        return segments_cache_key
    
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")