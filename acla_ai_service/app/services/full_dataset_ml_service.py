"""
Scikit-learn Machine Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service provides comprehensive AI model training and prediction capabilities
using your TelemetryFeatures and FeatureProcessor classes.
"""

import json
import base64
import zipfile
import shutil
import pandas as pd
from dataclasses import asdict

from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
from .tire_grip_analysis_service import TireGripAnalysisService
from .imitate_expert_learning_service import ExpertImitateLearningService
from .telemetry_segment_visualizer import (
    visualize_optimal_segments,
    visualize_segment_position_coverage,
)
import numpy as np
import joblib
import warnings
import pickle
import io
import asyncio
import time
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator, AsyncIterator
from datetime import datetime
from pathlib import Path
from app.models import AiModelDto, ActiveModelData

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
from ..models.transformer_model import ExpertActionTransformer

# Import backend service
from .backend_service import backend_service

# Import model cache service
from .model_cache_service import model_cache_service

# Import hybrid data cache service
from .Training_data_cache_service import training_cache

# Prompt dataset builder and local LLM integration
from .telemetry_prompt_dataset_builder import TelemetryPromptDatasetBuilder, PromptBuilderConfig
from .local_llm_service import LocalTelemetryLLM, LocalLLMConfig, GenerationRequest

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

ACTION_SUMMARY_FEATURES = (
    "Physics_gas",
    "Physics_brake",
    "Physics_steer_angle",
    "Physics_speed_kmh",
    "Physics_gear",
)

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
        
        # Prompt dataset builder and LLM configuration
        self.prompt_builder = TelemetryPromptDatasetBuilder()
        self.prompt_builder_config = self.prompt_builder.config
        self.llm_config = LocalLLMConfig()

        self.llm_adapter_directory = self.models_directory / "llm_adapters"
        self.llm_adapter_directory.mkdir(parents=True, exist_ok=True)
        self.llm_dataset_directory = self.models_directory / "llm_datasets"
        self.llm_dataset_directory.mkdir(parents=True, exist_ok=True)

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

    async def _generate_llm_prompt_dataset(
        self,
        segments_cache_key: str,
        *,
        annotations: Optional[Dict[str, Dict[str, Any]]] = None,
        output_path: Optional[Path] = None,
        shuffle: bool = True,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Build the LLM prompt dataset asynchronously from cached segments."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_filename = f"telemetry_segments_{timestamp}.jsonl"
        dataset_path = Path(output_path) if output_path else self.llm_dataset_directory / dataset_filename

        def _build_dataset() -> Tuple[Path, Dict[str, Any]]:
            stats = self.prompt_builder.build_from_cached_segments(
                segments_cache_key=segments_cache_key,
                output_path=dataset_path,
                annotations=annotations,
                shuffle=shuffle,
            )
            dataset_summary = self._summarize_dataset(dataset_path)
            stats_dict = stats.to_dict()
            stats_dict.update(
                {
                    "output_path": str(dataset_path),
                    "total_examples": dataset_summary.get("total_examples", stats_dict.get("windows_kept", 0)),
                    "annotated_examples": dataset_summary.get("annotated_examples", 0),
                    "annotation_ratio": dataset_summary.get("annotation_ratio", 0.0),
                }
            )
            return dataset_path, stats_dict

        dataset_path_result, stats_dict = await asyncio.to_thread(_build_dataset)
        return dataset_path_result, stats_dict

    def _summarize_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Collect lightweight statistics for an existing dataset file."""

        dataset_path = Path(dataset_path)
        summary = {
            "dataset_path": str(dataset_path),
            "total_examples": 0,
            "annotated_examples": 0,
            "annotation_ratio": 0.0,
        }

        if not dataset_path.exists():
            return summary

        total_examples = 0
        annotated_examples = 0
        try:
            with dataset_path.open("r", encoding="utf-8") as jsonl_file:
                for line in jsonl_file:
                    line = line.strip()
                    if not line:
                        continue
                    total_examples += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    metadata = record.get("metadata", {})
                    if metadata.get("annotation_complete"):
                        annotated_examples += 1
        except Exception as error:
            print(f"[WARNING] Failed to summarize dataset {dataset_path}: {error}")
            return summary

        summary["total_examples"] = total_examples
        summary["annotated_examples"] = annotated_examples
        summary["annotation_ratio"] = (annotated_examples / total_examples) if total_examples else 0.0
        return summary

    def _train_local_llm_sync(self, dataset_path: Path) -> Dict[str, Any]:
        """Synchronous helper that fine-tunes the local LLM and returns metrics."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_identifier = (dataset_path.stem or "telemetry").replace(" ", "_")
        adapter_dir = self.llm_adapter_directory / f"{dataset_identifier}_{timestamp}"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        llm = LocalTelemetryLLM(config=self.llm_config)
        metrics = llm.train(
            dataset_path=dataset_path,
            output_dir=adapter_dir,
            eval_dataset_path=None,
        )

        serialized_adapter = self._serialize_adapter_directory(adapter_dir)
        serialized_adapter["adapter_directory_name"] = adapter_dir.name

        return {
            "metrics": metrics,
            "adapter_dir": adapter_dir,
            "serialized_adapter": serialized_adapter,
            "config": asdict(self.llm_config),
        }

    async def _train_local_llm(self, dataset_path: Path) -> Dict[str, Any]:
        """Async wrapper for LLM fine-tuning."""

        return await asyncio.to_thread(self._train_local_llm_sync, dataset_path)

    async def _train_llm_and_persist(
        self,
        dataset_path: Path,
        dataset_stats: Optional[Dict[str, Any]] = None,
        cleanup_dataset_file: bool = True,
    ) -> Dict[str, Any]:
        """Fine-tune the LLM from an existing dataset and persist results."""

        dataset_path = Path(dataset_path)
        dataset_stats = dataset_stats or self._summarize_dataset(dataset_path)

        training_artifacts = await self._train_local_llm(
            dataset_path=dataset_path,
        )

        serialized_adapter = training_artifacts["serialized_adapter"]
        adapter_dir: Path = training_artifacts["adapter_dir"]
        metrics = training_artifacts["metrics"]

        metadata_payload = {
            "training_metrics": metrics,
            "dataset_stats": dataset_stats,
            "adapter_directory": adapter_dir.name,
            "llm_config": training_artifacts["config"],
            "generated_dataset": str(dataset_path),
            "training_timestamp": datetime.now().isoformat(),
        }

        print(
            f"[INFO] LLM fine-tuning complete. Saving adapter '{adapter_dir.name}' to backend"
        )

        await backend_service.save_ai_model(
            model_type="llm_guidance_v1",
            model_data=serialized_adapter,
            metadata=metadata_payload,
            is_active=True,
        )

        if cleanup_dataset_file:
            try:
                dataset_path.unlink(missing_ok=True)
            except Exception as cleanup_error:
                print(f"[WARNING] Failed to delete dataset file {dataset_path}: {cleanup_error}")

        return {
            "success": True,
            "training_metrics": metrics,
            "dataset_stats": dataset_stats,
            "adapter_directory": adapter_dir.name,
            "dataset_path": str(dataset_path),
        }

    def _serialize_adapter_directory(self, adapter_dir: Path) -> Dict[str, Any]:
        """Zip the adapter directory and produce a backend-friendly payload."""

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in adapter_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(adapter_dir)
                    zip_file.write(file_path, arcname=str(arcname))
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "adapter_zip_base64": encoded,
            "created_at": datetime.now().isoformat(),
        }

    def _deserialize_llm_model(self, payload: Dict[str, Any]) -> LocalTelemetryLLM:
        """Restore a fine-tuned LLM instance from backend payload data."""

        encoded = payload.get("adapter_zip_base64")
        adapter_name = payload.get("adapter_directory_name") or f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not encoded:
            raise ValueError("Adapter payload missing 'adapter_zip_base64'")

        target_dir = self.llm_adapter_directory / adapter_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        raw_bytes = base64.b64decode(encoded)
        buffer = io.BytesIO(raw_bytes)
        with zipfile.ZipFile(buffer, mode="r") as zip_file:
            zip_file.extractall(path=target_dir)

        llm = LocalTelemetryLLM(config=self.llm_config)
        llm.load_for_inference(adapter_path=target_dir)
        return llm

    def _format_context_window(self, telemetry_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construct and format a context window for inference from a single telemetry record."""

        context_steps = self.prompt_builder_config.context_steps
        repeated_window = [dict(telemetry_record) for _ in range(max(1, context_steps))]
        return self.prompt_builder.format_timesteps(repeated_window)

    def _summarize_transformer_sequence(
        self,
        sequence_predictions: List[Dict[str, Any]],
        *,
        max_steps: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Summarize transformer outputs for LLM prompting."""

        if not sequence_predictions:
            return []

        limit = max_steps if isinstance(max_steps, int) and max_steps > 0 else len(sequence_predictions)
        summary: List[Dict[str, Any]] = []

        for step_data in sequence_predictions[:limit]:
            if not isinstance(step_data, dict):
                continue

            step_index = step_data.get("step")
            targets = step_data.get("all_targets")
            targets = targets if isinstance(targets, dict) else {}

            actions: Dict[str, Any] = {}
            for feature in ACTION_SUMMARY_FEATURES:
                value = targets.get(feature)
                if value is None:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    actions[feature] = round(float(value), 4)
                else:
                    actions[feature] = value

            if not actions:
                continue

            summary.append(
                {
                    "step": int(step_index) if isinstance(step_index, (int, float)) else len(summary) + 1,
                    "actions": actions,
                }
            )

        return summary

    def _build_llm_user_prompt(
        self,
        *,
        context_timesteps: List[Dict[str, Any]],
        plan_summary: List[Dict[str, Any]],
        user_request: Optional[str] = None,
    ) -> str:
        """Create a commentary-focused user prompt for the LLM."""

        driver_request = (user_request or "Provide expert driving guidance for the upcoming sequence.").strip()
        context_json = json.dumps(context_timesteps, indent=2, ensure_ascii=False)
        plan_json = json.dumps(plan_summary or [], indent=2, ensure_ascii=False)

        return (
            "Task: Translate the numeric driving plan into clear coaching guidance.\n"
            f"Driver request: {driver_request}\n\n"
            "Recent telemetry snapshot (formatted):\n"
            f"{context_json}\n\n"
            "Predicted improvement plan from the ExpertActionTransformer (next steps):\n"
            f"{plan_json}\n\n"
            "Respond with a JSON object containing: \n"
            "- `coaching_summary`: concise natural language guidance for the driver.\n"
            "- Optional `key_focus`: list of bullet strings with the most important adjustments.\n"
            "Do not repeat the raw numbers; convert them into actionable coaching commentary."
        )

    def _parse_llm_output(self, generated_text: str) -> Dict[str, Any]:
        """Parse LLM commentary output, falling back to plain text when needed."""

        text = (generated_text or "").strip()
        if not text:
            return {"coaching_summary": ""}

        candidate = text

        first_brace = candidate.find("{")
        last_brace = candidate.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = candidate[first_brace : last_brace + 1]

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if "coaching_summary" not in parsed and "commentary" in parsed:
                    parsed["coaching_summary"] = parsed.pop("commentary")
                return parsed
        except json.JSONDecodeError:
            pass

        return {"coaching_summary": text}

    def clear_all_cache(self):
        """Clear cached LLM, imitation, corner, and tire model instances."""
        self.model_cache.clear()
        print("[INFO] All cached model_cache entries cleared (corner & tire services are on-demand so no persistent cache to clear)")
    
    async def get_llm_guidance_model(
        self,
        *,
        force_refresh: bool = False,
        model_subtype: str = "llm_adapter_data",
    ) -> Tuple[Optional[LocalTelemetryLLM], Optional[Dict[str, Any]]]:
        """Retrieve the active LLM guidance model if one has been saved."""

        if force_refresh:
            try:
                self.model_cache.invalidate(
                    model_type="llm_guidance_v1",
                    model_subtype=model_subtype,
                )
            except Exception as invalidate_error:
                print(f"[WARNING] Failed to invalidate LLM cache entry: {invalidate_error}")

        try:
            model_instance, metadata = await self._get_cached_model_or_fetch(
                model_type="llm_guidance_v1",
                model_subtype=model_subtype,
                deserializer_func=self._deserialize_llm_model,
            )
            return model_instance, metadata
        except Exception as fetch_error:
            print(f"[WARNING] No active LLM guidance model available: {fetch_error}")
            return None, {"error": str(fetch_error)}


    
    async def _fetch_and_cache_model(
        self,
        model_type: str,
        *,
        model_subtype: str = "complete_model_data",
        deserializer_func=None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Fetch model from backend and cache the model instance directly.
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize model data from backend and return model instance
            
        Returns:
            Tuple of (model_instance, metadata)
        """
        print(f"[DEBUG] Fetching model from backend: {model_type}")
        
        # getCompleteActiveModelData now always returns ActiveModelData or raises exception
        model_response: ActiveModelData = await self.backend_service.getCompleteActiveModelData(
            model_type
        )
        # Prepare cache metadata with all available structured data
        cache_metadata = {
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
            data=model_instance,  # Store model instance directly
            metadata=cache_metadata,
            model_subtype=model_subtype
        )
        
        print(f"[DEBUG] Successfully cached model instance: {model_type}")
        
        return model_instance, cache_metadata
    
    def _cleanup_fetch_lock(self, cache_key: str):
        """
        Clean up fetch lock for a specific cache key
        
        Args:
            cache_key: The cache key used for locking
        """
        if cache_key in self._model_fetch_locks:
            try:
                self._model_fetch_locks[cache_key].set()
                del self._model_fetch_locks[cache_key]
                print(f"[DEBUG] Released fetch lock for {cache_key}")
            except Exception as cleanup_error:
                print(f"[WARNING] Error cleaning up fetch lock: {str(cleanup_error)}")
    
    def _emergency_cleanup_fetch_lock(self, cache_key: str):
        """
        Emergency cleanup of fetch lock with additional safety checks
        
        Args:
            cache_key: The cache key used for locking
        """
        if hasattr(self, '_model_fetch_locks') and cache_key in self._model_fetch_locks:
            try:
                self._model_fetch_locks[cache_key].set()  # Signal any waiting threads
                del self._model_fetch_locks[cache_key]
                print(f"[INFO] Emergency cleanup of fetch lock for {cache_key}")
            except Exception as cleanup_error:
                print(f"[WARNING] Error during emergency lock cleanup: {str(cleanup_error)}")
    
    async def _get_cached_model_or_fetch(
        self,
        model_type: str,
        *,
        model_subtype: str = "complete_model_data",
        deserializer_func=None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Get model from cache or fetch from backend with thread-safe locking.
        Caches and retrieves model instances directly.
        
        Args:
            model_type: Type of model ('imitation_learning', etc.)
            model_subtype: Subtype identifier for the model
            deserializer_func: Function to deserialize model data from backend and return model instance
            
        Returns:
            Tuple of (model_instance, metadata)
        """
        # Generate consistent cache key using the same method as ModelCacheService
        cache_key = self.model_cache._generate_cache_key(
            model_type=model_type,
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
                model_subtype=model_subtype
            )
            
            if cache_info and not cache_info.get('is_expired', True):
                print(f"[DEBUG] Cache hit for {cache_key} - age: {cache_info.get('ttl_remaining_seconds', 'N/A')}s remaining")
                
                # Get the actual cached data
                cached_result = self.model_cache.get(
                    model_type=model_type,
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
                        model_subtype=model_subtype,
                        deserializer_func=deserializer_func
                    )
                    print(f"[INFO] Successfully fetched and cached model for {cache_key}")
                    
                except Exception as fetch_error:
                    print(f"[ERROR] Failed to fetch model {cache_key}: {str(fetch_error)}")
                    raise fetch_error
                finally:
                    # Always signal completion and clean up lock when we're the fetching thread
                    self._cleanup_fetch_lock(cache_key)
            
            # At this point, we should have the model instance
            if model_instance is None:
                raise Exception("Failed to obtain model instance from cache or backend")
            
            return model_instance, metadata
            
        except Exception as e:
            # Clean up any locks that might have been created by this thread
            if is_fetching_thread:
                self._emergency_cleanup_fetch_lock(cache_key)
            
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

    async def predict_expert_actions(
        self,
        telemetry_dict: Dict[str, Any],
        *,
        sequence_length: int = 40,
        track_name: Optional[str] = None,
        car_name: Optional[str] = None,
        user_request: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate numeric action plans via transformer and LLM commentary."""

        try:
            import pandas as pd

            resolved_track = (
                track_name
                or telemetry_dict.get("trackName")
                or telemetry_dict.get("track_name")
                or "generic"
            )
            resolved_car = (
                car_name
                or telemetry_dict.get("carName")
                or telemetry_dict.get("car_name")
                or "all_cars"
            )
            driver_request = (user_request or "").strip()

            telemetry_df = pd.DataFrame([telemetry_dict])

            processor = FeatureProcessor(telemetry_df)
            processed_df = processor.general_cleaning_for_analysis()
            processor.add_time_delta()
            processor.flip_y_z_features()
            features = self._imitate_expert_feature_names or self.telemetry_features.get_features_for_imitate_expert()

            filtered_df = processor.filter_features_by_list(processed_df, features)
            processed_telemetry_dict = (
                filtered_df.iloc[0].to_dict() if not filtered_df.empty else telemetry_dict
            )

            try:
                tire_grip_service = TireGripAnalysisService()
                tire_grip_service, _ = await self._get_cached_model_or_fetch(
                    model_type="tire_grip_analysis",
                    model_subtype="tire_grip_model_data",
                    deserializer_func=tire_grip_service.deserialize_tire_grip_model,
                )

                if getattr(tire_grip_service, "_trained", False):
                    tire_features_list = await tire_grip_service.extract_tire_grip_features(
                        [processed_telemetry_dict]
                    )
                    if tire_features_list:
                        processed_telemetry_dict.update(tire_features_list[0])
            except Exception as enrichment_error:
                print(f"[WARNING] Tire grip enrichment failed: {enrichment_error}")

            try:
                expert_service = ExpertImitateLearningService()
                expert_service, _ = await self._get_cached_model_or_fetch(
                    model_type="imitation_learning",
                    model_subtype="imitation_model_data",
                    deserializer_func=expert_service.deserialize_imitation_model,
                )
                expert_state_list = expert_service.extract_expert_state_for_telemetry(
                    [processed_telemetry_dict]
                )
                if expert_state_list:
                    processed_telemetry_dict.update(expert_state_list[0])
            except Exception as enrichment_error:
                print(f"[WARNING] Imitation learning enrichment failed: {enrichment_error}")

            transformer_model, transformer_metadata = await self._get_cached_model_or_fetch(
                model_type="transformer_expert_action",
                model_subtype="transformer_model_data",
                deserializer_func=ExpertActionTransformer.deserialize_transformer_model,
            )

            try:
                transformer_result = transformer_model.predict_human_readable(
                    current_telemetry=processed_telemetry_dict,
                    sequence_length=sequence_length,
                )
            except Exception as transformer_error:
                raise RuntimeError(f"Transformer prediction failed: {transformer_error}") from transformer_error

            plan_summary = self._summarize_transformer_sequence(
                transformer_result.get("sequence_predictions", []),
                max_steps=sequence_length,
            )

            context_payload = self._format_context_window(processed_telemetry_dict)

            llm_model, llm_metadata = await self._get_cached_model_or_fetch(
                model_type="llm_guidance_v1",
                model_subtype="llm_adapter_data",
                deserializer_func=self._deserialize_llm_model,
            )

            user_prompt = self._build_llm_user_prompt(
                context_timesteps=context_payload,
                plan_summary=plan_summary,
                user_request=driver_request,
            )
            system_prompt = self.prompt_builder_config.system_prompt

            generation_request = GenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.llm_config.generation_max_new_tokens,
                temperature=self.llm_config.generation_temperature,
                top_p=self.llm_config.generation_top_p,
                do_sample=self.llm_config.generation_do_sample,
            )

            output_text = llm_model.generate(generation_request)
            commentary_payload = self._parse_llm_output(output_text)

            sequence_predictions = transformer_result.get("sequence_predictions", [])

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "track_name": resolved_track,
                "car_name": resolved_car,
                "user_request": driver_request,
                "sequence_predictions": sequence_predictions,
                "preprocessed_telemetry": processed_telemetry_dict,
                "context_window": context_payload,
                "plan_summary": plan_summary,
                "commentary": commentary_payload,
                "transformer": {
                    "metadata": transformer_metadata,
                    "prediction": transformer_result,
                },
                "llm": {
                    "metadata": llm_metadata,
                    "raw_output": output_text,
                    "commentary": commentary_payload,
                },
                "prompt": {
                    "system": system_prompt,
                    "user": user_prompt,
                },
            }

        except Exception as error:
            error_msg = f"Failed to generate expert guidance: {error}"
            print(f"[ERROR] {error_msg}")
            return {
                "status": "error",
                "error_message": error_msg,
                "error_type": type(error).__name__,
            }

    async def StartImitateExpertPipeline(
        self,
        trackName: str,
        *,
        pipeline_mode: str = "full",
        dataset_path: Optional[Path] = None,
        annotation_map: Optional[Dict[str, Dict[str, Any]]] = None,
        cleanup_dataset_file: Optional[bool] = None,
        shuffle_dataset: bool = True,
    ):

        """Execute the telemetry pipeline with configurable annotation/training phases."""

        mode = (pipeline_mode or "full").lower()

        if mode == "train-only":
            if dataset_path is None:
                raise ValueError("dataset_path is required when pipeline_mode='train-only'")

            dataset_path = Path(dataset_path)
            dataset_stats = self._summarize_dataset(dataset_path)
            training_results = await self._train_llm_and_persist(
                dataset_path=dataset_path,
                dataset_stats=dataset_stats,
                cleanup_dataset_file=cleanup_dataset_file if cleanup_dataset_file is not None else False,
            )

            return {
                "success": training_results.get("success", False),
                "llm_training": training_results,
                "track_name": trackName,
                "mode": mode,
                "dataset_path": str(dataset_path),
                "dataset_stats": dataset_stats,
            }

        training_required = mode == "full"
        cleanup_after_training = cleanup_dataset_file if cleanup_dataset_file is not None else training_required
        dataset_path_result: Optional[Path] = None
        dataset_stats: Optional[Dict[str, Any]] = None
        annotation_map = annotation_map or {}
        
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
        top_laps_telemetry_list, sessions_cache_key = await self.process_lap_sessions_efficiently(
            processed_session_data_cache_key=dataset_cache_key,
            max_memory_records=10000,
            telemetry_time_gap_ms=200
        )

        segment_length = 500  # Nominal segment length used for context window sizing
        segments_cache_key = None
        training_results: Optional[Dict[str, Any]] = None

        try:
            # enrich data - now using cache-based approach to avoid memory overflow
            self._print_section_divider("ENRICHING CONTEXTUAL DATA")
            segments_cache_key = await self.enriched_contextual_data(top_laps_telemetry_list, sessions_cache_key, segment_length=segment_length)
            
            # remove any unused variable to this point to free memory
            del top_laps_telemetry_list
            
            # generate dataset for LLM guidance
            dataset_path_result, dataset_stats = await self._generate_llm_prompt_dataset(
                segments_cache_key=segments_cache_key,
                annotations=annotation_map,
                shuffle=shuffle_dataset,
            )
            print(
                f"[INFO] LLM dataset ready at {dataset_path_result} (mode={mode})."
            )

            if training_required:
                self._print_section_divider("TRAINING LOCAL LLM GUIDANCE MODEL")
                training_results = await self._train_llm_and_persist(
                    dataset_path=dataset_path_result,
                    dataset_stats=dataset_stats,
                    cleanup_dataset_file=cleanup_after_training,
                )
            
        finally:
            # Clean up cached data even if processing fails
            if sessions_cache_key:
                try:
                    self.data_cache.clear_cache(sessions_cache_key)
                    print(f"[INFO] Cleaned up cached session data: {sessions_cache_key}")
                except Exception as e:
                    print(f"[WARNING] Failed to clean up session cache: {str(e)}")
            
            if segments_cache_key:
                try:
                    self.data_cache.clear_cache(segments_cache_key)
                    print(f"[INFO] Cleaned up cached segments data: {segments_cache_key}")
                except Exception as e:
                    print(f"[WARNING] Failed to clean up cached segments: {str(e)}")

        if mode == "annotate":
            self._print_section_divider("DATASET READY FOR ANNOTATION")
            return {
                "success": True,
                "mode": mode,
                "track_name": trackName,
                "dataset_path": str(dataset_path_result) if dataset_path_result else None,
                "dataset_stats": dataset_stats,
            }

        if training_required:
            if not training_results or not training_results.get("success"):
                raise Exception("LLM training failed - no valid results produced")

            self._print_section_divider("LLM TRAINING COMPLETED")
            return {
                "success": True,
                "llm_training": training_results,
                "track_name": trackName,
                "mode": mode,
                "dataset_path": str(dataset_path_result) if dataset_path_result else None,
                "dataset_stats": dataset_stats,
            }

        raise ValueError(f"Unsupported pipeline mode: {pipeline_mode}")

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

    async def process_lap_sessions_efficiently(
        self,
        processed_session_data_cache_key: str,
        max_memory_records: int = 10000,
        telemetry_time_gap_ms: int = 100,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Streamlined processing of large cached datasets with a bounded memory footprint while caching
        full session telemetry for downstream training.

        Args:
            data_cache_key: Cache key that stores streamed sessions from the backend
            max_memory_records: Maximum number of session telemetry records kept in memory before flushing to cache
            telemetry_time_gap_ms: Maximum allowed gap (in milliseconds) between telemetry points when stripping laps

        Returns:
            Tuple of (top_laps_telemetry_list, sessions_cache_key)
        """
        print(
            f"[INFO] Processing cached dataset '{processed_session_data_cache_key}' with {max_memory_records} in-memory session records"
        )

        top_laps: List[Dict[str, Any]] = []
        sessions_cache_key = f"{processed_session_data_cache_key}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_buffer: List[Dict[str, Any]] = []
        session_buffer_records = 0
        total_sessions_cached = 0
        total_processed = 0
        chunk_idx = 0

        features = self._imitate_expert_feature_names or self.telemetry_features.get_features_for_imitate_expert()

        async def flush_session_buffer(reason: str) -> None:
            nonlocal session_buffer, session_buffer_records, total_sessions_cached
            if not session_buffer:
                return

            async def buffer_iterator() -> AsyncIterator[Dict[str, Any]]:
                for entry in session_buffer:
                    yield entry

            try:
                cache_success = await self.data_cache.cache_chunks_streaming(
                    cache_key=sessions_cache_key,
                    chunks_iterator=buffer_iterator()
                )
            except Exception as cache_error:
                print(f"[WARNING] Exception while flushing sessions during {reason}: {cache_error}")
                return

            if cache_success:
                cached_count = len(session_buffer)
                print(f"[INFO] Cached {cached_count} session chunks ({session_buffer_records} records) [{reason}] -> {sessions_cache_key}")
                total_sessions_cached += cached_count
                session_buffer = []
                session_buffer_records = 0
            else:
                print(f"[WARNING] Cache service reported failure while flushing sessions during {reason}")

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

        def update_top_laps(candidate: Dict[str, Any]) -> None:
            """Maintain the five fastest laps for expert reference."""
            if len(top_laps) < 5:
                top_laps.append(candidate)
                print(
                    f"[DEBUG] Added lap {candidate['id']} to top laps ({len(top_laps)}/5, time: {candidate['lap_time_ms']}ms)"
                )
                return

            slowest_idx = max(range(len(top_laps)), key=lambda idx: top_laps[idx]["lap_time_ms"])
            slowest = top_laps[slowest_idx]
            if candidate["lap_time_ms"] < slowest["lap_time_ms"]:
                top_laps[slowest_idx] = candidate
                print(
                    f"[DEBUG] Replaced slowest lap {slowest['id']} with {candidate['id']} (time: {candidate['lap_time_ms']}ms)"
                )

        session_chunks_iterator = self.data_cache.get_cached_data_chunks(cache_key=processed_session_data_cache_key)
        print(f"[DEBUG] Created chunk iterator for cache key: {processed_session_data_cache_key}")

        session_chunks_processed = 0
        for session_chunk_df in session_chunks_iterator:
            session_chunks_processed += 1

            if session_chunk_df is None or session_chunk_df.empty:
                print(f"[DEBUG] Chunk {session_chunks_processed} is empty, skipping")
                continue

            try:
                telemetry_df = session_chunk_df
                print(f"[DEBUG] Processing {len(telemetry_df)} telemetry records from chunk {session_chunks_processed}")

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
                session_records: List[Dict[str, Any]] = []

                for lap_index, (lap_struct, stripped_df) in enumerate(zip(lap_structs, stripped_lap_frames)):
                    lap_metrics = lap_struct["metrics"]
                    lap_metrics["record_count_after_gap"] = len(stripped_df)

                    filtered_df = processor.filter_features_by_list(stripped_df, features) if not stripped_df.empty else pd.DataFrame()
                    filtered_count = len(filtered_df)

                    if stripped_df.empty:
                        print(
                            f"[DEBUG] Chunk {chunk_idx} lap {lap_index} empty after down-sampling"
                        )
                    elif filtered_df.empty:
                        print(
                            f"[DEBUG] Chunk {chunk_idx} lap {lap_index} has no features after filtering"
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
                        update_top_laps(candidate_entry)
                    else:
                        print(
                            f"[DEBUG] Chunk {chunk_idx}: including non-top lap {lap_struct['lap_num']} (full={lap_metrics['is_full']}, valid={lap_metrics['is_valid']})"
                        )

                    if lap_records:
                        session_records.extend(lap_records)

                print(
                    f"[DEBUG] Chunk {chunk_idx}: Processed {laps_processed_in_chunk} full valid laps. Top laps: {len(top_laps)} Session buffer records: {session_buffer_records + len(session_records)}"
                )

                if session_records:
                    session_entry = {
                        "chunkId": f"session_{chunk_idx}_{session_chunks_processed}",
                        "data": session_records,
                        "metadata": {
                            "chunk_index": chunk_idx,
                            "session_index": session_chunks_processed,
                            "record_count": len(session_records),
                        },
                    }
                    session_buffer.append(session_entry)
                    session_buffer_records += len(session_records)

                    if session_buffer_records >= max_memory_records:
                        await flush_session_buffer("memory limit reached")

                total_processed += len(telemetry_df)
                chunk_idx += 1

            except ValueError as gap_error:
                raise gap_error
            except Exception as error:
                print(f"[WARNING] Chunk processing failed: {error}")
                continue

        await flush_session_buffer("final flush")

        print(f"[DEBUG] Finished processing all chunks:")
        print(f"[DEBUG] - Total chunks processed: {session_chunks_processed}")
        print(f"[DEBUG] - Valid chunks processed: {chunk_idx}")
        print(f"[DEBUG] - Total records processed: {total_processed}")
        print(f"[DEBUG] - Top laps found: {len(top_laps)}")
        print(f"[DEBUG] - Session chunks cached: {total_sessions_cached}")

        if top_laps:
            lap_times = [lap_info["lap_time_ms"] for lap_info in top_laps]
            print(f"[DEBUG] Top lap times: {sorted(lap_times)}")

        if not session_chunks_processed:
            raise ValueError(
                f"No chunks were returned by iterator for cache key {processed_session_data_cache_key}. Check if data exists in cache."
            )

        if not chunk_idx:
            raise ValueError(
                f"All {session_chunks_processed} chunks failed processing for cache key {processed_session_data_cache_key}. Check data quality."
            )

        if len(top_laps) < 5:
            raise ValueError(
                f"Insufficient top laps found: {len(top_laps)}/5 required. Processed {chunk_idx} valid chunks with {total_processed} records."
            )

        if total_sessions_cached == 0 and session_buffer_records == 0:
            raise ValueError("No session data cached for transformer training")

        top_laps.sort(key=lambda entry: entry["lap_time_ms"])

        top_laps_telemetry_list: List[Dict[str, Any]] = []
        for lap_info in top_laps:
            top_laps_telemetry_list.extend(lap_info["records"])

        if session_buffer_records > 0:
            print(f"[WARN] Session buffer retains {session_buffer_records} records that were not flushed to cache")

        print(f"[SUCCESS] Processed {chunk_idx} chunks, {total_processed} records")
        print(f"[SUCCESS] Selected top 5 laps: {len(top_laps_telemetry_list)} records")
        print(f"[SUCCESS] Cached {total_sessions_cached} session batches across {chunk_idx} chunks")

        return top_laps_telemetry_list, sessions_cache_key

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

    async def _train_llm_guidance_model(
        self,
        segments_cache_key: str,
    ) -> Dict[str, Any]:
        """Fine-tune the local telemetry LLM using cached segments and persist the adapter."""

        try:
            dataset_path, dataset_stats = await self._generate_llm_prompt_dataset(
                segments_cache_key=segments_cache_key,
            )
            print(
                f"[INFO] Generated LLM prompt dataset at {dataset_path} with stats: {json.dumps(dataset_stats, indent=2)}"
            )

            training_results = await self._train_llm_and_persist(
                dataset_path=dataset_path,
                dataset_stats=dataset_stats,
                cleanup_dataset_file=True,
            )

            return training_results

        except Exception as error:
            print(f"[ERROR] LLM guidance training failed: {error}")
            return {
                "success": False,
                "error": str(error),
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
            if not getattr(tire_service, "_trained", False):
                raise RuntimeError("Tire grip service must be trained before extracting features")
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

    async def enriched_contextual_data(self, top_laps_telemetry_list: List[Dict[str, Any]], sessions_cache_key: str, segment_length: int) -> str:
        """
        Streamlined contextual data enrichment using chunk iterator approach.
        
        1. Train all enrichment models using expert data
        2. Use chunk_iterator to process cached session data  
        3. Enrich each chunk with contextual features
        4. Filter into segments and cache them in reasonable chunks
        
        Args:
            top_laps_telemetry_list: List of expert telemetry records for training models
            sessions_cache_key: Cache key for accessing cached session data via iterator
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
            model_data=serialized_data,
            metadata=imitation_result.get("learning_summary", {}),
            is_active=True
        )
        print("[INFO] ✓ Imitation learning model trained and saved")
        
        # Train corner identification model
        from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
        corner_service = CornerIdentificationUnsupervisedService()
        corner_model = await corner_service.learn_track_corner_patterns(top_laps_telemetry_list)
        corner_serialized = corner_service.serialize_corner_identification_model()
        
        await self.backend_service.save_ai_model(
            model_type="corner_identification",
            model_data=corner_serialized,
            metadata={
                "total_corners": corner_serialized.get("total_corners"),
                "clusters": len(corner_serialized.get("corner_clusters", [])),
                "serialization_timestamp": corner_serialized.get("serialized_timestamp")
            },
            is_active=True
        )
        print("[INFO] ✓ Corner identification model trained and saved")
        
        # Train tire grip analysis model using full sessions (streaming)
        from .tire_grip_analysis_service import TireGripAnalysisService
        tire_service = TireGripAnalysisService()

        print("[INFO] Streaming cached session telemetry to train tire grip model")
        session_training_iterator = self.data_cache.get_cached_data_chunks(
            cache_key=sessions_cache_key
        )
        tire_grip_training = await tire_service.train_tire_grip_model_streaming(
            chunk_iterator=session_training_iterator
        )
        if not tire_grip_training.get("success", False):
            raise RuntimeError(
                "Tire grip training yielded no safe samples; cannot proceed with contextual enrichment"
            )
        tire_service_serialized = tire_service.serialize_tire_grip_model()
        
        await self.backend_service.save_ai_model(
            model_type="tire_grip_analysis",
            model_data=tire_service_serialized,
            metadata={
                "training_summary": tire_grip_training,
                "serialization_timestamp": tire_service_serialized.get("serialized_timestamp"),
                "feature_catalog": tire_service.feature_catalog.CONTEXT_FEATURES
            },
            is_active=True
        )
        print("[INFO] ✓ Tire grip analysis model trained on sessions and saved")
        
        # Step 2: Process cached sessions via chunk iterator with enrichment
        self._print_section_divider("PROCESSING SESSION DATA VIA CHUNK ITERATOR")
        
        segments_cache_key = f"enriched_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_chunks_iterator = self.data_cache.get_cached_data_chunks(
            cache_key=sessions_cache_key  # Use the cache key where session data is stored
        )
        
        processed_chunks = 0
        total_segments_cached = 0
        segments_per_cache = 5000  # Number of segments per cache batch
        
        current_segment_batch = []
        cache_batch_number = 0

        coverage_histogram_bins = np.linspace(0.0, 1.0, num=101)
        coverage_histogram_counts = np.zeros(len(coverage_histogram_bins) - 1, dtype=np.float64)
        coverage_sample_count = 0
        
        for session_chunk_df in session_chunks_iterator:
            if session_chunk_df is None or session_chunk_df.empty:
                continue
            
            processed_chunks += 1
            chunk_data = session_chunk_df.to_dict('records')
            
            print(f"[INFO] Processing chunk {processed_chunks}: {len(chunk_data)} records")
            
            # Step 3: Enrich chunk with contextual features
            enriched_chunk_data = await self._enrich_chunk_with_context(
                chunk_data, imitation_learning, tire_service
            )
            
            # Step 4: Filter enriched chunk into segments
            chunk_segments = imitation_learning.filter_optimal_telemetry_segments(
                enriched_chunk_data,
                max_segment_length=segment_length,
            )
            
            print(f"[INFO] Chunk {processed_chunks}: Generated {len(chunk_segments)} segments")

            if chunk_segments:
                chunk_positions: List[float] = []
                for segment in chunk_segments:
                    for record in segment:
                        position_value = record.get("Graphics_normalized_car_position")
                        if position_value is None:
                            continue
                        try:
                            normalized_position = float(position_value)
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(normalized_position):
                            continue
                        clamped_position = min(1.0, max(0.0, normalized_position))
                        chunk_positions.append(clamped_position)

                if chunk_positions:
                    chunk_positions_np = np.asarray(chunk_positions, dtype=float)
                    chunk_counts, _ = np.histogram(
                        chunk_positions_np,
                        bins=coverage_histogram_bins,
                        range=(0.0, 1.0),
                    )
                    coverage_histogram_counts += chunk_counts
                    coverage_sample_count += int(chunk_positions_np.size)

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

        try:
            if coverage_sample_count > 0:
                coverage_payload = visualize_segment_position_coverage(
                    histogram_counts=coverage_histogram_counts,
                    bin_edges=coverage_histogram_bins,
                    total_points=coverage_sample_count,
                    file_name_prefix=f"{segments_cache_key}_coverage",
                    return_base64=False,
                )
                saved_path = coverage_payload.get("saved_path")
                if saved_path:
                    print(f"[INFO] Saved normalized position coverage visualization to {saved_path}")
                else:
                    print("[INFO] Generated normalized position coverage visualization")
            else:
                print("[WARN] No normalized car position samples found; skipping coverage visualization")
        except Exception as coverage_error:
            print(f"[WARN] Failed to generate normalized position coverage visualization: {coverage_error}")
        
        return segments_cache_key
    
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")