"""
Training pipeline stages:

- ``cleaning``: stream raw sessions and cache cleaned telemetry.
- ``selection``: select top laps from the cleaned telemetry.
- ``enrich``: build enrichment models (top-lap reference, tire grip) and enrich sessions
  with contextual features; segment enriched data.
- ``training``: filter annotated segments and train the coach transformer.

``prepare_training_data`` runs these preparation stages in order or individually and is
the entry point used by ``scripts/open_pipeline_management.py``.
``run_transformer_guidance_training`` is re-exported here for convenience.
"""

from typing import Any, Dict, List, Optional

from app.integrations.backend.client import backend_service as default_backend_service
from training.pipelines.training.config import TrainingPipelineConfig
from training.pipelines.training.pipeline.cleaning import (
    print_section_divider,
    process_lap_sessions_efficiently,
    process_single_chunk,
)
from training.pipelines.training.pipeline.enrich import (
    cache_segment_batch,
    enrich_sessions_with_context,
    enriched_contextual_data,
    get_cached_all_top_laps_in_one_list,
    process_and_cache_segments,
)
from training.pipelines.training.pipeline.selection import select_top_laps
from training.pipelines.training.pipeline.training import run_transformer_guidance_training

__all__ = [
    "prepare_training_data",
    "run_transformer_guidance_training",
    "print_section_divider",
    "process_lap_sessions_efficiently",
    "process_single_chunk",
    "select_top_laps",
    "enrich_sessions_with_context",
    "enriched_contextual_data",
    "get_cached_all_top_laps_in_one_list",
    "cache_segment_batch",
    "process_and_cache_segments",
]


async def prepare_training_data(
    *,
    telemetry_store,
    config: Optional[TrainingPipelineConfig] = None,
    cache_config: Optional[TrainingPipelineConfig] = None,
    backend_service=None,
    top_lap_reference_feature_names: List[str],
    stage: str = "all",
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Run cleanup, top-lap selection, and expert enrichment in order."""

    pipeline_config = config or cache_config
    if pipeline_config is None:
        raise ValueError("prepare_training_data requires a TrainingPipelineConfig")
    if stage not in ("all", "cleanup", "top_laps", "enrich"):
        raise ValueError(f"Unknown data preparation stage: {stage}")
    if max_workers < 1:
        raise ValueError("Workers must be at least 1")

    backend = backend_service or default_backend_service

    dataset_cache_key = pipeline_config.session_data_cache_key
    processed_sessions_cache_key = pipeline_config.processed_session_data_cache_key
    enriched_sessions_cache_key = pipeline_config.enriched_sessions_cache_key
    segments_cache_key = pipeline_config.segments_cache_key
    top_laps_cache_key = pipeline_config.top_laps_cache_key

    def clear_caches(*keys: str) -> None:
        for key in keys:
            if telemetry_store.has_cached_data(key):
                print(f"[INFO] Clearing stale preparation cache: {key}")
                telemetry_store.clear_cache(key)

    def require_cache(key: str, instruction: str) -> None:
        if not telemetry_store.has_cached_data(key):
            raise RuntimeError(instruction)

    try:
        if stage in ("all", "cleanup"):
            print_section_divider("DOWNLOADING AND CLEANING TELEMETRY")
            sessions_metadata = await backend.get_all_racing_sessions_streaming(
                cache_key=dataset_cache_key,
                cleanup_cache=pipeline_config.session_cleanup,
                data_cache=telemetry_store,
            )
            if not sessions_metadata.get("success"):
                raise RuntimeError(sessions_metadata.get("message") or "Backend streaming request failed")

            if pipeline_config.processed_session_cleanup or not telemetry_store.has_cached_data(processed_sessions_cache_key):
                clear_caches(processed_sessions_cache_key, top_laps_cache_key,
                             enriched_sessions_cache_key, segments_cache_key)
                await process_lap_sessions_efficiently(
                    session_data_cache_key=dataset_cache_key,
                    telemetry_store=telemetry_store,
                    top_lap_reference_feature_names=top_lap_reference_feature_names,
                    telemetry_time_gap_ms=500,
                    processed_sessions_cache_key=processed_sessions_cache_key,
                    max_workers=max_workers,
                )

        if stage in ("all", "top_laps", "enrich"):
            require_cache(processed_sessions_cache_key, "Run data cleanup before selecting top laps or adding expert features.")

        if stage in ("all", "top_laps"):
            print_section_divider("SELECTING TOP LAPS FROM CLEANED TELEMETRY")
            if pipeline_config.top_laps_cleanup or not telemetry_store.has_cached_data(top_laps_cache_key):
                clear_caches(top_laps_cache_key, enriched_sessions_cache_key, segments_cache_key)
                await select_top_laps(
                    processed_sessions_cache_key,
                    telemetry_store=telemetry_store,
                    cache_config=pipeline_config,
                    max_workers=max_workers,
                )

        if stage in ("all", "enrich"):
            require_cache(top_laps_cache_key, "Select top laps before adding expert features.")
            print_section_divider("ADDING EXPERT FEATURES TO CLEANED TELEMETRY")
            if pipeline_config.segment_cleanup:
                clear_caches(enriched_sessions_cache_key, segments_cache_key)
            await enriched_contextual_data(
                processed_sessions_cache_key,
                telemetry_store=telemetry_store,
                cache_config=pipeline_config,
                backend_service=backend,
                max_workers=max_workers,
            )

        return {
            "success": True,
            "max_segment_length": 20,
        }

    except RuntimeError as runtime_error:
        print(f"[ERROR] Pipeline error: {runtime_error}")
        return {
            "success": False,
            "error": str(runtime_error),
        }
