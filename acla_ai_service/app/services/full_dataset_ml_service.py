"""
Scikit-learn Machine Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service provides comprehensive AI model training and prediction capabilities
using your TelemetryFeatures and FeatureProcessor classes.
"""

import json
import base64
import logging
from dataclasses import dataclass
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from .corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
from .tire_grip_analysis_service import TireGripAnalysisService
from .imitate_expert_learning_service import ExpertImitateLearningService
from .segment_classifier_service import segment_classifier
from .telemetry_segment_visualizer import (
    visualize_optimal_segments,
    visualize_segment_position_coverage,
)
import numpy as np
import joblib
import warnings
import pickle
import asyncio
import time
import traceback
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator, AsyncIterator
from datetime import datetime
from pathlib import Path
from app.models import AiModelDto, ActiveModelData
from app.models.segment_models import PredictedSegment, AnnotatedSegment

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
# Transformer model utilities
from .transformer_model import prepare_and_train_coach_transformer_model

# Import backend service
from .backend_service import backend_service

# Import model cache service
from .model_cache_service import model_cache_service

# Import hybrid data cache service
from .zarr_telemetry_store import get_shared_zarr_store
from app.config.pipeline_config import PipelineConfig

# Prompt dataset builder and local LLM integration
from .local_llm_service import LocalTelemetryLLM, LocalLLMConfig, GenerationRequest
from .llm.telemetry_llm_orchestrator import TelemetryLLMOrchestrator

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class Full_dataset_TelemetryMLService:
    """
    Machine Learning Service for AC Competizione Telemetry Analysis
    """ 
    
    def __init__(self, models_directory: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the ML service
        
        Args:
            models_directory: Directory to save/load trained models.
                            If None, defaults to 'models' in the project root.
        """
        # Resolve to an absolute path so downstream tooling operates on a single location
        if models_directory:
            self.models_directory = Path(models_directory).resolve()
        else:
            # Default to project_root/models
            # This file is in app/services/full_dataset_ml_service.py
            # Project root is ../../..
            self.models_directory = Path(__file__).resolve().parents[2] / "models"

        self.models_directory.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.telemetry_features = TelemetryFeatures()
        self.trained_models = {}
        self.scalers = {}
        self.label_encoders = {}
        # Cache frequently used feature lists to avoid recreating TelemetryFeatures each prediction
        try:
            self._imitate_expert_feature_names = self.telemetry_features.get_features_for_learning_expert()
        except Exception:
            self._imitate_expert_feature_names = []
        
        # Backend service integration
        self.backend_service = backend_service
        
        # Model cache service integration
        self.model_cache_service = model_cache_service
        
        # Use shared Zarr-backed telemetry store for large datasets
        self.telemetry_store = get_shared_zarr_store()
        self.logger.info(
            "Using Zarr-based telemetry store for large dataset processing (store: %s)",
            self.telemetry_store.store_dir,
        )
        
        # LLM configuration
        self.llm_config = LocalLLMConfig(
                                        offload_folder=self.models_directory / "llm_offload",
                                        offload_state_dict=True,
                                        base_model=self.models_directory / "Mistral-7B-Instruct-v0.2",
                                        tokenizer_name=self.models_directory / "Mistral-7B-Instruct-v0.2",
                                        max_seq_length=32000,    
                                        max_memory={0: "8GiB", "cpu": "12GiB"},
                                        load_in_4bit= True,       
                                       )
        
        self.llm_adapter_directory = self.models_directory / "llm_adapters"
        self.llm_adapter_directory.mkdir(parents=True, exist_ok=True)
        self.llm_dataset_directory = self.models_directory / "llm_datasets"
        self.llm_dataset_directory.mkdir(parents=True, exist_ok=True)

        self.llm_orchestrator = TelemetryLLMOrchestrator(
            llm_config=self.llm_config,
            adapter_directory=self.llm_adapter_directory,
            dataset_directory=self.llm_dataset_directory,
        )

        # Centralize cache key usage for coordinated cleanup
        self.cache_config = PipelineConfig()

        # Add a simple lock mechanism to prevent concurrent fetches of the same model
        self._model_fetch_locks = {}
        self._lock_creation_lock = asyncio.Lock()
        
        # Clear entire cache to ensure we start fresh with model instances only
        self.model_cache_service.clear()
        self.logger.info("Cleared entire cache on startup - will cache model instances directly")
        
        # Log cache configuration on startup
        self._log_cache_configuration()
    
    def _log_cache_configuration(self):
        """Log cache configuration details"""
        _ = self.model_cache_service.get_stats()
        details = [
            "",
            "=" * 60,
            "CACHE CONFIGURATION",
            "=" * 60,
            f"Max Cache Size: {self.model_cache_service.max_cache_size} models",
            (
                f"Max Memory: {self.model_cache_service.max_memory_mb}MB"
                f" ({self.model_cache_service.max_memory_mb/1024:.1f}GB)"
            ),
            f"Environment: {self.model_cache_service.environment}",
            (
                f"Default TTL: {self.model_cache_service.default_ttl_seconds}s"
                f" ({self.model_cache_service.default_ttl_seconds/3600:.1f}h)"
            ),
            (
                "Large Model Priority: "
                f"{self.model_cache_service.config.get('performance', {}).get('large_model_priority', False)}"
            ),
            "Caching Strategy: Model instances cached directly",
            "=" * 60,
            "",
        ]
        self.logger.info("\n".join(details))

    # Dataset generation is now handled by TelemetryLLMOrchestrator providers.

    def _format_context_window(self, telemetry_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construct and format a context window for inference from a single telemetry record."""

        context_steps = 40  # Default context window size
        repeated_window = [dict(telemetry_record) for _ in range(max(1, context_steps))]
        return repeated_window

    def _build_llm_user_prompt(
        self,
        *,
        context_timesteps: List[Dict[str, Any]],
        future_timesteps: Optional[List[Dict[str, Any]]] = None,
        segment_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a segment-purpose prompt for the LLM."""

        context_json = json.dumps(context_timesteps or [], indent=2, ensure_ascii=False)
        future_json = json.dumps(future_timesteps or [], indent=2, ensure_ascii=False)

        metadata_section = ""
        if segment_metadata:
            metadata_json = json.dumps(segment_metadata, indent=2, ensure_ascii=False)
            metadata_section = f"Segment metadata:\n{metadata_json}\n\n"

        return (
            "Task: Provide a concise coaching explanation that captures the purpose of this telemetry segment.\n"
            f"{metadata_section}"
            "Telemetry context (ordered timesteps):\n"
            f"{context_json}\n\n"
            "Continuation of the segment (if available):\n"
            f"{future_json}\n\n"
            "Respond with a JSON object containing:\n"
            "- `coaching_summary`: 2-3 sentences describing the segment focus or key coaching insight.\n"
            "- Optional `key_focus`: list of short bullet strings for the most important adjustments or observations."
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
        self.model_cache_service.clear()
        self.llm_orchestrator.clear_llm_cache()
        print("[INFO] All cached model_cache entries cleared (corner & tire services are on-demand so no persistent cache to clear)")
    
    async def get_llm_guidance_model(
        self,
        *,
        force_refresh: bool = False,
        model_subtype: str = "llm_adapter_data",
        provider: str = "local",
        model_id: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Retrieve the active LLM guidance model if one has been saved."""

        return await self.llm_orchestrator.get_llm_for_inference(
            force_refresh=force_refresh,
            model_subtype=model_subtype,
            provider=provider,
            model_id=model_id,
        )

    def _cleanup_fetch_lock(self, cache_key: str):
        """Clean up fetch lock for a specific cache key"""
        if cache_key in self._model_fetch_locks:
            try:
                self._model_fetch_locks[cache_key].set()
                del self._model_fetch_locks[cache_key]
                print(f"[DEBUG] Released fetch lock for {cache_key}")
            except Exception as cleanup_error:
                print(f"[WARNING] Error cleaning up fetch lock: {str(cleanup_error)}")

    # Dataset generation is now handled by TelemetryLLMOrchestrator providers.
    
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
        deserializer_func=None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Get model from cache or fetch from backend with thread-safe locking."""

        cache_key = self.model_cache_service.build_cache_key(
            model_type=model_type,
            model_subtype=model_subtype,
        )

        model_instance: Optional[Any] = None
        metadata: Dict[str, Any] = {}
        is_fetching_thread = False
        fetch_event: Optional[asyncio.Event] = None

        try:
            cached_result = self.model_cache_service.get(
                model_type=model_type,
                model_subtype=model_subtype,
            )
            if cached_result:
                model_instance, metadata = cached_result
                if model_instance is not None:
                    return model_instance, metadata

            async with self._lock_creation_lock:
                cached_result = self.model_cache_service.get(
                    model_type=model_type,
                    model_subtype=model_subtype,
                )
                if cached_result:
                    model_instance, metadata = cached_result
                    if model_instance is not None:
                        return model_instance, metadata

                if cache_key not in self._model_fetch_locks:
                    self._model_fetch_locks[cache_key] = asyncio.Event()
                    is_fetching_thread = True
                else:
                    fetch_event = self._model_fetch_locks[cache_key]

            if not is_fetching_thread and fetch_event is not None:
                try:
                    await fetch_event.wait()
                    cached_result = self.model_cache_service.get(
                        model_type=model_type,
                        model_subtype=model_subtype,
                    )
                    if cached_result:
                        model_instance, metadata = cached_result
                        if model_instance is not None:
                            return model_instance, metadata

                    print(f"[WARNING] Expected cached model not found after waiting for {cache_key}")
                    is_fetching_thread = True
                except Exception as wait_error:
                    print(f"[WARNING] Error while waiting for {cache_key}: {wait_error}")
                    is_fetching_thread = True

            if is_fetching_thread:
                model_instance, metadata = await self._fetch_and_cache_model(
                    model_type=model_type,
                    model_subtype=model_subtype,
                    deserializer_func=deserializer_func,
                )

            if model_instance is None:
                raise RuntimeError(f"Failed to obtain model instance for {cache_key}")

            return model_instance, metadata

        except Exception as error:
            if is_fetching_thread:
                self._emergency_cleanup_fetch_lock(cache_key)
            raise error
        finally:
            if is_fetching_thread:
                self._cleanup_fetch_lock(cache_key)

    def _print_section_divider(self, title: str, width: int = 80):
        """Print a large console divider to visually separate log sections."""

        normalized_width = max(width, len(title) + 4)
        divider = "=" * normalized_width
        print(f"\n{divider}\n{title.center(normalized_width)}\n{divider}")

    def get_cache_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive cache debugging information."""

        cache_stats = self.model_cache_service.get_stats()
        lock_status = self.get_fetch_locks_status()
        llm_cache = self.llm_orchestrator.get_cache_debug_info()

        return {
            "cache_stats": cache_stats,
            "fetch_locks": lock_status,
            "llm_cache": llm_cache,
            "timestamp": datetime.now().isoformat(),
        }

    def print_cache_debug_info(self):
        """Print cache debugging information to console."""

        debug_info = self.get_cache_debug_info()

        print("\n" + "=" * 60)
        print("CACHE DEBUG INFORMATION")
        print("=" * 60)

        cache_stats = debug_info["cache_stats"]
        print(f"Cache Size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        print(f"Memory Usage: {cache_stats['memory_usage_mb']:.2f}/{cache_stats['max_memory_mb']} MB")
        print(f"Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
        print(f"Evictions: {cache_stats['evictions']}, Cleanups: {cache_stats['cleanups']}")

        print(f"\nActive Fetch Locks: {debug_info['fetch_locks']['lock_count']}")
        for lock_key in debug_info['fetch_locks']['active_locks']:
            print(f"  - {lock_key}")

        print("\nCached Models:")
        for entry in cache_stats.get("entries", []):
            print(
                f"  - {entry['key']} ({entry['size_mb']:.2f} MB, accessed {entry['access_count']} times)"
            )
            ttl_remaining = entry.get("ttl_remaining_seconds")
            if ttl_remaining:
                print(f"    TTL remaining: {ttl_remaining:.0f}s")

        print("=" * 60 + "\n")
        self.llm_orchestrator.print_cache_debug_info()
    
    async def clear_stuck_fetch_locks(self, max_age_minutes: int = 10) -> Dict[str, Any]:
        """
        Clear fetch locks that might be stuck (emergency cleanup)
        
        Args:
            max_age_minutes: Age threshold for considering locks as stuck
            
        Returns:
            Dictionary with cleanup results
        """
        cleared_locks = []
        
        llm_result = await self.llm_orchestrator.clear_stuck_fetch_locks(max_age_minutes=max_age_minutes)

        try:
            async with self._lock_creation_lock:
                for cache_key in list(self._model_fetch_locks.keys()):
                    try:
                        self._model_fetch_locks[cache_key].set()
                        del self._model_fetch_locks[cache_key]
                        cleared_locks.append(cache_key)
                    except Exception:
                        pass

            return {
                "success": True,
                "cleared_locks": cleared_locks,
                "cleared_count": len(cleared_locks),
                "timestamp": datetime.now().isoformat(),
                "llm": llm_result,
            }

        except Exception as error:
            return {
                "success": False,
                "error": str(error),
                "cleared_locks": cleared_locks,
                "cleared_count": len(cleared_locks),
                "timestamp": datetime.now().isoformat(),
                "llm": llm_result,
            }

    async def predict_expert_actions(
        self,
        telemetry_dict: Dict[str, Any],
        *,
        sequence_length: int = 40,
        user_request: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate segment-purpose guidance using the LLM without requiring the transformer."""

        try:
            import pandas as pd

            driver_request = (user_request or "").strip()

            telemetry_df = pd.DataFrame([telemetry_dict])

            processor = FeatureProcessor(telemetry_df)
            processed_df = processor.general_cleaning_for_analysis()
            processor.add_time_delta()
            processor.flip_y_z_features()
            features = self._imitate_expert_feature_names or self.telemetry_features.get_features_for_learning_expert()

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
                self.logger.warning(
                    "Tire grip enrichment failed: %s",
                    enrichment_error,
                )

            try:
                expert_service = ExpertImitateLearningService(logger=self.logger)
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
                self.logger.warning(
                    "Imitation learning enrichment failed: %s",
                    enrichment_error,
                )

            context_payload = self._format_context_window(processed_telemetry_dict)

            future_payload: List[Dict[str, Any]] = []
            segment_metadata: Dict[str, Any] = {
                "sequence_length_hint": sequence_length,
            }
            if driver_request:
                segment_metadata["user_request"] = driver_request

            telemetry_highlights = (
                "Physics_speed_kmh",
                "Physics_rpm",
                "Physics_gear",
                "Physics_brake",
                "Physics_gas",
                "Physics_steer_angle",
                "Graphics_delta_lap_time",
            )
            for feature in telemetry_highlights:
                value = processed_telemetry_dict.get(feature)
                if value is None:
                    continue
                try:
                    segment_metadata[feature] = _safe_float(value)
                except Exception:
                    segment_metadata[feature] = value

            llm_model, llm_metadata = await self.llm_orchestrator.get_llm_for_inference()
            if llm_model is None:
                raise RuntimeError("LLM guidance model is not available")

            user_prompt = self._build_llm_user_prompt(
                context_timesteps=context_payload,
                future_timesteps=future_payload,
                segment_metadata=segment_metadata,
            )
            system_prompt = "You are a professional racing coach. Analyze the telemetry data provided and offer concise, actionable advice to improve lap times."

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

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "user_request": driver_request,
                "sequence_predictions": [],
                "preprocessed_telemetry": processed_telemetry_dict,
                "context_window": context_payload,
                "future_window": future_payload,
                "segment_metadata": segment_metadata,
                "commentary": commentary_payload,
                "transformer": {
                    "metadata": None,
                    "prediction": None,
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

    async def prepare_training_data(
        self,
        top_laps_count: int = 5,
    ):
        """Stream telemetry, build a segment-purpose dataset, and fine-tune the LLM."""

        self._print_section_divider("STREAMING TELEMETRY DATA FROM BACKEND DIRECTLY TO CACHE")

        cache_config = self.cache_config

        # Stream sessions directly into cache for downstream processing
        dataset_cache_key = cache_config.session_data_cache_key
        processed_sessions_cache_key = cache_config.processed_session_data_cache_key
        enriched_sessions_cache_key = cache_config.enriched_sessions_cache_key
        segments_cache_key = cache_config.segments_cache_key
        try:
            sessions_metadata = await backend_service.get_all_racing_sessions_streaming(
                cache_key=dataset_cache_key,
                cleanup_cache=cache_config.session_cleanup,
            )
        except Exception as streaming_error:
            raise RuntimeError(f"Backend streaming failed: {streaming_error}") from streaming_error

        if not sessions_metadata.get("success"):
            raise RuntimeError(sessions_metadata.get("message") or "Backend streaming request failed")

        self._print_section_divider("LARGE DATASET ASSUMED - USING EFFICIENT PROCESSING")

        # Check if top laps are already cached
        top_laps_cache_key = cache_config.top_laps_cache_key
        top_laps_available = False
        
        if self.telemetry_store.has_cached_data(top_laps_cache_key) and self.telemetry_store.has_cached_data(processed_sessions_cache_key):
            if cache_config.processed_session_cleanup:
                print(f"[INFO] Cleaning up existing top laps cache: {top_laps_cache_key}")
                self.telemetry_store.clear_cache(top_laps_cache_key)
                print(f"[INFO] Cleaning up processed sessions cache: {processed_sessions_cache_key}")
                self.telemetry_store.clear_cache(processed_sessions_cache_key)
                print(f"[INFO] Cleaning up enriched sessions cache: {enriched_sessions_cache_key}")
                self.telemetry_store.clear_cache(enriched_sessions_cache_key)
                # Force regeneration after cleanup
                top_laps_available = False
            else:
                top_laps_available = True
                print(f"[INFO] Using existing top laps from cache: {top_laps_cache_key}")
        
        # Process sessions if top laps not available from cache
        if not top_laps_available:
            await self.process_lap_sessions_efficiently(
                session_data_cache_key=dataset_cache_key,
                telemetry_time_gap_ms=500,
                processed_sessions_cache_key=processed_sessions_cache_key,
                top_laps_count=top_laps_count,
            )

        self._print_section_divider("ENRICHING CONTEXTUAL DATA")
        max_segment_length = 20

        try:
            # Only run enrichment if segments not cached or cleanup requested

            print("[INFO] Enriching telemetry sessions with segment data...")
            enriched_sessions_cache_key = await self.enriched_contextual_data(
                processed_sessions_cache_key,
            )
            
            return {
                "success": True,
                "max_segment_length": max_segment_length
            }

        except RuntimeError as runtime_error:
            print(f"[ERROR] Pipeline error: {runtime_error}")
            return {
                "success": False,
                "error": str(runtime_error),
            }
        
    async def run_transformer_guidance_training(
        self,
        annotation_cache_key: str,
        processed_sessions_cache_key: str,
        max_segment_length: int = 20,
        *,
        shuffle_dataset: bool = True,
    ) -> Dict[str, Any]:
            
        transformer_training: Optional[Dict[str, Any]] = None
        llm_training: Optional[Dict[str, Any]] = None
        generated_datasets = []
    
        try:
            print(f"[INFO] Filtering annotations from {annotation_cache_key}...")
            filtered_segments = []
            
            # The store returns chunks which are lists of segments (dicts)
            chunk_iterator = self.telemetry_store.get_cached_data_chunks(annotation_cache_key)
            
            total_segments_checked = 0
            
            for chunk in chunk_iterator:
                # Iterate over segments in the chunk
                if isinstance(chunk, list):
                    for seg_dict in chunk:
                        total_segments_checked += 1
                        labels = seg_dict.get("labels", [])
                        # Check for "EA" or "RM"
                        if any(l in ["EA", "RM"] for l in labels):
                            filtered_segments.append(seg_dict)
                elif isinstance(chunk, dict):
                     total_segments_checked += 1
                     labels = chunk.get("labels", [])
                     if any(l in ["EA", "RM"] for l in labels):
                            filtered_segments.append(chunk)

            print(f"[INFO] Found {len(filtered_segments)} annotated segments with EA/RM labels (scanned {total_segments_checked}).")

            if not filtered_segments:
                raise RuntimeError(f"No segments found with EA or RM labels in {annotation_cache_key}")

            # Use a temporary key for the filtered training dataset
            segments_cache_key = f"temp_training_segments_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Save the filtered segments as one chunk
            await self.telemetry_store.cache_chunks_streaming(segments_cache_key, [filtered_segments])
            print(f"[INFO] Cached filtered segments to {segments_cache_key}")

            # Train and save transformer model
            transformer_training = await prepare_and_train_coach_transformer_model(
                data_cache=self.telemetry_store,
                segments_cache_key=segments_cache_key,
                segment_length_hint=max_segment_length,
            )

            if not transformer_training.get("success"):
                raise RuntimeError(transformer_training.get("error") or "Transformer training failed")

            await backend_service.save_ai_model(
                model_type="transformer_expert_action",
                model_data=transformer_training["serialized_model"],
                metadata={
                    "training_history": transformer_training["training_history"],
                    "test_metrics": transformer_training["test_metrics"],
                    "training_timestamp": datetime.now().isoformat()
                },
                is_active=True
            )
            print("[INFO] ✓ Transformer model trained and saved")
            
        except RuntimeError as runtime_error:
            print(f"[ERROR] Pipeline error: {runtime_error}")
            return {
                "success": False,
                "error": str(runtime_error),
            }
        except Exception as training_error:
            print(f"[ERROR] Unexpected error: {training_error}")
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Transformer training / LLM dataset generation failed: {training_error}",
            }
        
        result_payload = {
            "success": True,
            "datasets": generated_datasets,
            "transformer_training": transformer_training,
            "llm_dataset_generation": llm_training,
        }

        self._print_section_divider("DATASET GENERATION COMPLETED")
        return result_payload

    async def process_lap_sessions_efficiently(
        self,
        session_data_cache_key: str,
        *,
        telemetry_time_gap_ms: int = 100,
        processed_sessions_cache_key: Optional[str] = None,
        top_laps_count: int = 5,
    ) -> None:
        """
        Streamlined processing of large cached datasets with a bounded memory footprint while caching
        full session telemetry for downstream training.

        Args:
            data_cache_key: Cache key that stores streamed sessions from the backend
            max_memory_records: Maximum number of session telemetry records kept in memory before flushing to cache
            telemetry_time_gap_ms: Maximum allowed gap (in milliseconds) between telemetry points when stripping laps
            processed_sessions_cache_key: Optional override for cache key storing processed session data
            top_laps_count: Number of fastest laps to keep for expert reference (default: 5)
        """
        print(
            f"[INFO] Processing cached dataset '{session_data_cache_key}' with immediate caching"
        )

        top_laps: Dict[str, List[Dict[str, Any]]] = {}
        total_sessions_cached = 0
        total_processed = 0
        chunk_idx = 0

        features = self._imitate_expert_feature_names or self.telemetry_features.get_features_for_learning_expert()

        def update_top_laps(candidate: Dict[str, Any]) -> None:
            """Maintain the fastest laps for expert reference per track."""
            records = candidate.get("records", [])
            if not records:
                return
            print(
                f"[DEBUG] Evaluating lap {candidate.get('id', 'unknown')} with {len(records)} telemetry records"
            )
            
            # Get track name from first record
            track_name = records[0].get("Static_track", "unknown_track")
            
            if track_name not in top_laps:
                top_laps[track_name] = []
            
            track_top_laps = top_laps[track_name]

            if len(track_top_laps) < top_laps_count:
                track_top_laps.append(candidate)
                print(
                    f"[DEBUG] Added lap {candidate['id']} to top laps for {track_name} ({len(track_top_laps)}/{top_laps_count}, time: {candidate['lap_time_ms']}ms)"
                )
                return

            slowest_idx = max(range(len(track_top_laps)), key=lambda idx: track_top_laps[idx]["lap_time_ms"])
            slowest = track_top_laps[slowest_idx]
            if candidate["lap_time_ms"] < slowest["lap_time_ms"]:
                track_top_laps[slowest_idx] = candidate
                print(
                    f"[DEBUG] Replaced slowest lap {slowest['id']} with {candidate['id']} for {track_name} (time: {candidate['lap_time_ms']}ms)"
                )

        session_chunks_iterator = self.telemetry_store.get_cached_data_chunks(cache_key=session_data_cache_key, include_ids=True)
        print(f"[DEBUG] Created chunk iterator for cache key: {session_data_cache_key}")

        session_chunks_processed = 0

        async def process_chunk_result(result_tuple):
            nonlocal chunk_idx, total_sessions_cached
            c_idx, laps_processed, candidates, records, error, chunk_id = result_tuple
            
            if error:
                print(f"[WARNING] Chunk {c_idx} processing failed: {error}")
                return

            chunk_idx += 1
            
            # Update top laps
            for candidate in candidates:
                update_top_laps(candidate)
                
            # Save immediately to telemetry_store
            if records:
                async def single_chunk_generator():
                    yield (records, chunk_id)

                try:
                    cache_success = await self.telemetry_store.cache_chunks_streaming(
                        cache_key=processed_sessions_cache_key,
                        chunks_iterator=single_chunk_generator()
                    )
                    if cache_success:
                        total_sessions_cached += 1
                        print(f"[INFO] Immediately cached chunk {c_idx} (ID: {chunk_id}) ({len(records)} records)")
                    else:
                        print(f"[WARNING] Failed to cache chunk {c_idx}")
                except Exception as cache_error:
                    print(f"[ERROR] Exception caching chunk {c_idx}: {cache_error}")
                    
            print(
                f"[DEBUG] Chunk {c_idx}: Processed {laps_processed} full valid laps. Top laps: {len(top_laps)}"
            )
        
        # Use ProcessPoolExecutor for parallel processing with bounded concurrency
        with ProcessPoolExecutor(max_workers=4) as executor:
            pending_aws = set()
            
            # Submit tasks
            for chunk_tuple in session_chunks_iterator:
                session_chunk_df, chunk_id = chunk_tuple
                session_chunks_processed += 1
                
                future = executor.submit(
                    self._process_single_chunk,
                    session_chunk_df,
                    chunk_id,
                    features,
                    telemetry_time_gap_ms
                )
                # Wrap in asyncio future to await it
                aw = asyncio.wrap_future(future)
                pending_aws.add(aw)

                # Limit concurrency to avoid loading all chunks into memory
                if len(pending_aws) >= 4:
                    done, pending_aws = await asyncio.wait(pending_aws, return_when=asyncio.FIRST_COMPLETED)
                    for aw_done in done:
                        try:
                            result = await aw_done
                            await process_chunk_result(result)
                        except Exception as e:
                            print(f"[ERROR] Error retrieving chunk result: {e}")

            # Process remaining tasks
            while pending_aws:
                done, pending_aws = await asyncio.wait(pending_aws, return_when=asyncio.FIRST_COMPLETED)
                for aw_done in done:
                    try:
                        result = await aw_done
                        await process_chunk_result(result)
                    except Exception as e:
                        print(f"[ERROR] Error retrieving chunk result: {e}")

        print(f"[DEBUG] Finished processing all chunks:")
        print(f"[DEBUG] - Total chunks processed: {session_chunks_processed}")
        print(f"[DEBUG] - Valid chunks processed: {chunk_idx}")
        print(f"[DEBUG] - Total records processed: {total_processed}")
        print(f"[DEBUG] - Tracks found: {len(top_laps)}")
        print(f"[DEBUG] - Session chunks cached: {total_sessions_cached}")

        if top_laps:
            for track_name, laps in top_laps.items():
                lap_times = [lap_info["lap_time_ms"] for lap_info in laps]
                print(f"[DEBUG] Top lap times for {track_name}: {sorted(lap_times)}")

        if not session_chunks_processed:
            raise ValueError(
                f"No chunks were returned by iterator for cache key {session_data_cache_key}. Check if data exists in cache."
            )

        if not chunk_idx:
            raise ValueError(
                f"All {session_chunks_processed} chunks failed processing for cache key {session_data_cache_key}. Check data quality."
            )

        if not top_laps:
             raise ValueError(
                f"No top laps found. Processed {chunk_idx} valid chunks."
            )

        if total_sessions_cached == 0:
            raise ValueError("No session data cached for transformer training")

        print(f"[SUCCESS] Processed {chunk_idx} chunks")
        print(f"[SUCCESS] Selected top laps from {len(top_laps)} tracks")
        print(f"[SUCCESS] Cached {total_sessions_cached} session batches across {chunk_idx} chunks")

        # Cache top laps telemetry list for downstream use
        top_laps_cache_key = self.cache_config.top_laps_cache_key
        try:
            async def top_laps_generator():
                # Yield each track's top laps as a separate chunk
                for track_name, track_laps in top_laps.items():
                    if len(track_laps) < top_laps_count:
                        print(f"[WARNING] Insufficient top laps found for {track_name}: {len(track_laps)}/{top_laps_count}")
                    
                    track_laps.sort(key=lambda entry: entry["lap_time_ms"])

                    track_records = []
                    for lap_info in track_laps:
                        track_records.append(lap_info["records"])
                    
                    if track_records:
                        print(f"[DEBUG] Yielding top laps chunk for {track_name} ({len(track_records)} laps)")
                        yield (track_records, track_name)
            
            cache_success = await self.telemetry_store.cache_chunks_streaming(
                cache_key=top_laps_cache_key,
                chunks_iterator=top_laps_generator()
            )
            
            if cache_success:
                print(f"[SUCCESS] Cached top lap telemetry records to {top_laps_cache_key} (separate chunks per track)")
            else:
                print(f"[WARNING] Failed to cache top laps to {top_laps_cache_key}")
        except Exception as cache_error:
            print(f"[WARNING] Error caching top laps: {cache_error}")

    @staticmethod
    def _process_single_chunk(
        chunk_data: Any,
        chunk_idx: Union[int, str],
        features: List[str],
        telemetry_time_gap_ms: int
    ) -> Tuple[int, int, List[Dict[str, Any]], List[Dict[str, Any]], Optional[Exception], Optional[str]]:
        """
        Process a single chunk of telemetry data in a separate thread.
        """
        chunk_id = str(chunk_idx)
        actual_data = chunk_data

        # Downstream consumers should rely on chunk_idx passed from the iterator
        # The chunk payload now contains only the data, not the _id wrapper
        
        print(f"[DEBUG] Thread started processing chunk {chunk_idx} (ID: {chunk_id}) ({len(actual_data)} records)")
        laps_processed_in_chunk = 0
        candidates: List[Dict[str, Any]] = []
        session_records: List[Dict[str, Any]] = []

        try:
            session_chunk_df = pd.DataFrame(actual_data)
            telemetry_df = session_chunk_df
            
            # Basic validation
            if telemetry_df.empty:
                return chunk_idx, 0, [], [], None, chunk_id

            processor = FeatureProcessor(telemetry_df)
            processor.general_cleaning_for_analysis()
            processed_df = processor.flip_y_z_features()

            if processed_df.empty:
                return chunk_idx, 0, [], [], None, chunk_id

            # Strip by time gap on the whole session chunk
            stripped_session_df = processor.strip_dataframe_by_time_gap(processed_df, telemetry_time_gap_ms)

            if stripped_session_df.empty:
                return chunk_idx, 0, [], [], None, chunk_id

            # Filter features on the whole session chunk
            filtered_session_df = processor.filter_features_by_list(stripped_session_df, features)
            
            if filtered_session_df.empty:
                 return chunk_idx, 0, [], [], None, chunk_id

            # Use filtered_session_df for session_records
            session_records = filtered_session_df.to_dict("records")

            # Split into laps using the filtered session
            lap_structs = processor.split_into_laps(filtered_session_df)
            if not lap_structs:
                return chunk_idx, 0, [], session_records, None, chunk_id

            for i, lap_struct in enumerate(lap_structs):
                lap_df = lap_struct["dataframe"]
                lap_time_ms = lap_struct["lap_time_ms"]

                # Check for valid lap (all rows must have Graphics_is_valid_lap == 1)
                if "Graphics_is_valid_lap" in lap_df.columns:
                    if not (lap_df["Graphics_is_valid_lap"] == 1).all():
                        continue

                # Check for irregular time gaps
                if "Graphics_current_time" in lap_df.columns and len(lap_df) > 2:
                    time_diffs = lap_df["Graphics_current_time"].diff().dropna()
                    valid_diffs = time_diffs[time_diffs > 0]
                    
                    if not valid_diffs.empty:
                        mean_diff = valid_diffs.mean()
                        std_diff = valid_diffs.std()
                        
                        if std_diff > 0:
                            # Check if standard deviation is a significant percentage of the mean (e.g. > 50%)
                            # This indicates irregular gaps far from the majority
                            if std_diff > (mean_diff * 0.5):
                                continue

                lap_records = lap_df.to_dict("records") if not lap_df.empty else []

                lap_identifier = (
                    f"{chunk_id}_{i}_{lap_time_ms or 'na'}"
                )

                candidate_entry = {
                    "id": lap_identifier,
                    "lap_time_ms": lap_time_ms if lap_time_ms is not None else float("inf"),
                    "lap_num": lap_struct["lap_num"],
                    "records": lap_records,
                }

                # Include all laps with valid time and records
                if lap_time_ms is not None and lap_records:
                    laps_processed_in_chunk += 1
                    candidates.append(candidate_entry)
            
            return chunk_idx, laps_processed_in_chunk, candidates, session_records, None, chunk_id

        except Exception as e:
            return chunk_idx, 0, [], [], e, chunk_id

    async def _enrich_sessions_with_context(self, chunk_data: List[Dict[str, Any]], 
                                        imitation_learning: ExpertImitateLearningService, tire_service: TireGripAnalysisService) -> List[Dict[str, Any]]:
        """
        Enrich a single chunk with all contextual features
        
        Args:
            chunk_data: List of telemetry records
            imitation_learning: Trained imitation learning service
            tire_service: Trained tire grip service
            
        Returns:
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
        chunk_grip_features = await tire_service.extract_tire_grip_features(chunk_data)
        
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
                """Generator yielding entire batch as a single chunk"""
                # Yield the entire batch as one chunk, not individual segments
                yield segments_batch
            
            # Estimate size for this batch
            total_records = sum(len(segment) for segment in segments_batch)
            estimated_size_mb = (total_records * 60) / (1024 * 1024)  # 60 bytes per enriched record
            
            # Cache raw segments directly
            cache_success = await self.telemetry_store.cache_chunks_streaming(
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

    async def get_cached_all_top_laps_in_one_list(
        self,
        *,
        top_laps_cache_key: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve cached top laps telemetry list for downstream use.
        
        Args:
            top_laps_cache_key: Optional override for the cache key
            
        Returns:
            List of laps, where each lap is a list of telemetry records
            
        Raises:
            ValueError: If top laps cache is not available
        """
        cache_key = top_laps_cache_key or self.cache_config.top_laps_cache_key
        
        try:
            if not self.telemetry_store.has_cached_data(cache_key):
                raise ValueError(f"No cached top laps found at key: {cache_key}")
            
            # Get the cached data chunks
            chunks_iterator = self.telemetry_store.get_cached_data_chunks(cache_key=cache_key)
            
            all_top_laps = []
            chunk_count = 0
            
            for chunk in chunks_iterator:
                # Handle both raw list and (data, id) tuple formats
                if isinstance(chunk, tuple):
                    chunk = chunk[0]

                if isinstance(chunk, list):
                    all_top_laps.extend(chunk)
                    chunk_count += 1
            
            if chunk_count > 0:
                self.logger.info(
                    "Retrieved %d top laps from %d cache chunks",
                    len(all_top_laps),
                    chunk_count,
                )
                return all_top_laps
            
            raise ValueError(f"Cached data at {cache_key} has unexpected format or is empty")
            
        except Exception as error:
            self.logger.error("Failed to retrieve cached top laps: %s", error)
            raise

    async def enriched_contextual_data(
        self,
        sessions_cache_key: str
    ) -> Tuple[str, Any]:
        """
        Streamlined contextual data enrichment using chunk iterator approach.
        
        1. Train all enrichment models using expert data
        2. Use chunk_iterator to process cached session data  
        3. Enrich each chunk with contextual features
        4. Cache enriched data
        
        Args:
            sessions_cache_key: Cache key for accessing cached session data via iterator
            
        Returns:
            Tuple[str, Any] - (enriched_sessions_cache_key, imitation_learning_service)
        """

        # Step 1: Train all enrichment models using expert data
        self._print_section_divider("TRAINING ENRICHMENT MODELS WITH EXPERT DATA")
        
        # Train imitation learning model
        imitation_learning = ExpertImitateLearningService(logger=self.logger,debug=True)
        
        # Use cached top laps for imitation learning (per track)
        top_laps_cache_key = self.cache_config.top_laps_cache_key
        imitation_result = await imitation_learning.train_ai_model(top_laps_cache_key)

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
        
        # Retrieve top laps for corner identification
        print(f"[INFO] Retrieving top laps for corner identification from {top_laps_cache_key}")
        top_laps_list = await self.get_cached_all_top_laps_in_one_list(top_laps_cache_key=top_laps_cache_key)
        
        corner_model = await corner_service.learn_track_corner_patterns(top_laps_list)
        corner_serialized = corner_service.serialize_corner_identification_model()
        
        # Free memory
        del top_laps_list
        
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
        session_training_iterator = self.telemetry_store.get_cached_data_chunks(
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
        
        enriched_sessions_cache_key = self.cache_config.enriched_sessions_cache_key
        
        session_chunks_iterator = self.telemetry_store.get_cached_data_chunks(
            cache_key=sessions_cache_key,  # Use the cache key where session data is stored
            include_ids=True
        )
        
        processed_chunks = 0
        
        for chunk_tuple in session_chunks_iterator:
            session_chunk_df, chunk_id = chunk_tuple
            # Convert raw payload to DataFrame (iterator yields list/dict, not DataFrame)
            session_chunk_df = pd.DataFrame(session_chunk_df)
            
            if session_chunk_df is None or session_chunk_df.empty:
                continue
            
            processed_chunks += 1
            chunk_data = session_chunk_df.to_dict('records')
            
            print(f"[INFO] Processing chunk {processed_chunks}: {len(chunk_data)} records")
            
            # Step 3: Enrich sessions with contextual features
            enriched_chunk_data = await self._enrich_sessions_with_context(
                chunk_data, imitation_learning, tire_service
            )
            
            # Cache enriched chunk
            async def enriched_chunk_generator():
                yield (enriched_chunk_data, chunk_id)
                
            await self.telemetry_store.cache_chunks_streaming(
                cache_key=enriched_sessions_cache_key,
                chunks_iterator=enriched_chunk_generator()
            )
            
            # Clear chunk data to free memory
            del chunk_data, enriched_chunk_data
            
            if processed_chunks % 5 == 0:
                print(f"[INFO] Progress: {processed_chunks} chunks enriched and cached")
        
        print(f"[SUCCESS] Enrichment completed:")
        print(f"  - Processed {processed_chunks} data chunks from cache")  
        print(f"  - Enriched data cached to: {enriched_sessions_cache_key}")
        
        return enriched_sessions_cache_key, imitation_learning

    async def process_and_cache_segments(
        self,
        enriched_sessions_cache_key: str,
        segments_cache_key: str,
        max_segment_length: int,
    ) -> str:
        """
        Process enriched sessions: filter into segments, visualize, and cache.
        
        Args:
            enriched_sessions_cache_key: Cache key for enriched session data
            segments_cache_key: Cache key for storing segments
            max_segment_length: Maximum length for segments
            
        Returns:
            str - Cache key for the segments
        """
        self._print_section_divider("PROCESSING ENRICHED DATA INTO SEGMENTS")
        
        session_chunks_iterator = self.telemetry_store.get_cached_data_chunks(
            cache_key=enriched_sessions_cache_key,
            include_ids=True
        )
        
        processed_chunks = 0
        
        for chunk_tuple in session_chunks_iterator:
            session_chunk_df, chunk_id = chunk_tuple
            session_chunk_df = pd.DataFrame(session_chunk_df)
            if session_chunk_df is None or session_chunk_df.empty:
                continue
                
            processed_chunks += 1
            
            print(f"[INFO] Processing enriched chunk {processed_chunks}: {len(session_chunk_df)} records")

            # Filter using segment_classifier.scan_session
            # scan_session now caches segments directly to the store for memory efficiency
            await segment_classifier.scan_telemetry_data(
                dataframe=session_chunk_df, 
                window_size=max_segment_length,
                # chunk_id=chunk_id # Assuming scan_telemetry_data can take chunk_id if needed, but user didn't ask to change segment_classifier
            )
            
            if processed_chunks % 5 == 0:
                print(f"[INFO] Progress: {processed_chunks} chunks processed")

        print(f"[SUCCESS] Segment generation completed:")
        print(f"  - Processed {processed_chunks} enriched chunks")
        print(f"  - Segments cached to: {segments_cache_key}")

        # Visualization logic
        self._print_section_divider("GENERATING VISUALIZATIONS FROM CACHED SEGMENTS")
        
        coverage_histogram_bins = np.linspace(0.0, 1.0, num=101)
        coverage_histogram_counts = np.zeros(len(coverage_histogram_bins) - 1, dtype=np.float64)
        coverage_sample_count = 0
        
        segments_to_visualize = []
        max_viz_segments = 5;
        
        # Stream back the cached segments
        cached_segments_iterator = self.telemetry_store.get_cached_data_chunks(segments_cache_key)
        
        processed_viz_chunks = 0
        for chunk_segments in cached_segments_iterator:
            if not chunk_segments:
                continue
                
            # chunk_segments is List[Dict] (PredictedSegment dicts)
            
            # Collect samples for detailed visualization (just take from the first few chunks)
            if len(segments_to_visualize) < max_viz_segments:
                remaining = max_viz_segments - len(segments_to_visualize)
                for seg_dict in chunk_segments[:remaining]:
                    # Use PredictedSegment to ensure structure
                    pred_seg = PredictedSegment(**seg_dict)
                    # Visualizer likely expects list of records
                    segments_to_visualize.append(pred_seg.telemetry_data)

            # Accumulate coverage data
            chunk_positions = []
            for segment_item in chunk_segments:
                telemetry_data = segment_item.get("telemetry_data", [])
                
                for record in telemetry_data:
                    position_value = record.get("Graphics_normalized_car_position")
                    if position_value is not None:
                        try:
                            val = float(position_value)
                            if not np.isnan(val):
                                chunk_positions.append(min(1.0, max(0.0, val)))
                        except (TypeError, ValueError):
                            pass
            
            if chunk_positions:
                chunk_positions_np = np.asarray(chunk_positions, dtype=float)
                chunk_counts, _ = np.histogram(
                    chunk_positions_np,
                    bins=coverage_histogram_bins,
                    range=(0.0, 1.0),
                )
                coverage_histogram_counts += chunk_counts
                coverage_sample_count += int(chunk_positions_np.size)
                
            processed_viz_chunks += 1
            if processed_viz_chunks % 10 == 0:
                 print(f"[INFO] Processed {processed_viz_chunks} cached chunks for visualization...")

        # Generate visualizations
        if segments_to_visualize:
             try:
                visualization_payloads = visualize_optimal_segments(
                    segments_to_visualize,
                    max_segments=len(segments_to_visualize),
                    file_name_prefix=f"{segments_cache_key}_sample",
                    return_base64=False
                )
                if visualization_payloads:
                    print(f"[INFO] Generated {len(visualization_payloads)} sample segment visualizations")
             except Exception as viz_error:
                print(f"[WARN] Failed to visualize sample segments: {viz_error}")

        # Coverage Viz
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
                print("[WARN] No normalized car position samples found; skipping coverage visualization")
        except Exception as coverage_error:
            print(f"[WARN] Failed to generate normalized position coverage visualization: {coverage_error}")

        return segments_cache_key
    
if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")