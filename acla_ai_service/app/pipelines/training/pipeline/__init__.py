"""
Training pipeline split into three stages:

- ``cleaning``: stream raw sessions, filter laps, select top laps per track.
- ``enrich``: train enrichment models (imitation, tire grip) and enrich sessions
  with contextual features; segment enriched data.
- ``training``: filter annotated segments and train the coach transformer.

``prepare_training_data`` orchestrates ``cleaning`` followed by ``enrich`` and is
the entry point used by ``scripts/run_full_pipeline.py``.
``run_transformer_guidance_training`` is re-exported here for convenience.
"""

from typing import Any, Dict, List

from app.integrations.backend.client import backend_service as default_backend_service
from app.pipelines.training.pipeline.cleaning import (
    print_section_divider,
    process_lap_sessions_efficiently,
    process_single_chunk,
)
from app.pipelines.training.pipeline.enrich import (
    cache_segment_batch,
    enrich_sessions_with_context,
    enriched_contextual_data,
    get_cached_all_top_laps_in_one_list,
    process_and_cache_segments,
)
from app.pipelines.training.pipeline.training import run_transformer_guidance_training

__all__ = [
    "prepare_training_data",
    "run_transformer_guidance_training",
    "print_section_divider",
    "process_lap_sessions_efficiently",
    "process_single_chunk",
    "enrich_sessions_with_context",
    "enriched_contextual_data",
    "get_cached_all_top_laps_in_one_list",
    "cache_segment_batch",
    "process_and_cache_segments",
]


async def prepare_training_data(
    *,
    telemetry_store,
    cache_config,
    backend_service=None,
    imitate_expert_feature_names: List[str],
    top_laps_count: int = 5,
) -> Dict[str, Any]:
    """Stream telemetry, build a segment-purpose dataset, and fine-tune the LLM."""

    backend = backend_service or default_backend_service

    print_section_divider("STREAMING TELEMETRY DATA FROM BACKEND DIRECTLY TO CACHE")

    dataset_cache_key = cache_config.session_data_cache_key
    processed_sessions_cache_key = cache_config.processed_session_data_cache_key
    enriched_sessions_cache_key = cache_config.enriched_sessions_cache_key
    segments_cache_key = cache_config.segments_cache_key
    try:
        sessions_metadata = await backend.get_all_racing_sessions_streaming(
            cache_key=dataset_cache_key,
            cleanup_cache=cache_config.session_cleanup,
        )
    except Exception as streaming_error:
        raise RuntimeError(f"Backend streaming failed: {streaming_error}") from streaming_error

    if not sessions_metadata.get("success"):
        raise RuntimeError(sessions_metadata.get("message") or "Backend streaming request failed")

    print_section_divider("LARGE DATASET ASSUMED - USING EFFICIENT PROCESSING")

    top_laps_cache_key = cache_config.top_laps_cache_key
    top_laps_available = False

    if cache_config.top_laps_cleanup:
        if telemetry_store.has_cached_data(top_laps_cache_key):
            print(f"[INFO] Cleaning up existing top laps cache: {top_laps_cache_key}")
            telemetry_store.clear_cache(top_laps_cache_key)
        top_laps_available = False
    elif telemetry_store.has_cached_data(top_laps_cache_key) and telemetry_store.has_cached_data(processed_sessions_cache_key):
        top_laps_available = True
        print(f"[INFO] Using existing top laps from cache: {top_laps_cache_key}")

    if not top_laps_available:
        if cache_config.processed_session_cleanup and telemetry_store.has_cached_data(processed_sessions_cache_key):
            print(f"[INFO] Cleaning up existing processed sessions cache: {processed_sessions_cache_key}")
            telemetry_store.clear_cache(processed_sessions_cache_key)

        await process_lap_sessions_efficiently(
            session_data_cache_key=dataset_cache_key,
            telemetry_store=telemetry_store,
            cache_config=cache_config,
            imitate_expert_feature_names=imitate_expert_feature_names,
            telemetry_time_gap_ms=500,
            processed_sessions_cache_key=processed_sessions_cache_key,
            top_laps_count=top_laps_count,
        )

    print_section_divider("ENRICHING CONTEXTUAL DATA")
    max_segment_length = 20

    try:
        if cache_config.segment_cleanup:
            if telemetry_store.has_cached_data(enriched_sessions_cache_key):
                print(f"[INFO] Cleaning up existing enriched sessions cache: {enriched_sessions_cache_key}")
                telemetry_store.clear_cache(enriched_sessions_cache_key)
            if telemetry_store.has_cached_data(segments_cache_key):
                print(f"[INFO] Cleaning up existing segments cache: {segments_cache_key}")
                telemetry_store.clear_cache(segments_cache_key)

        print("[INFO] Enriching telemetry sessions with segment data...")
        enriched_sessions_cache_key = await enriched_contextual_data(
            processed_sessions_cache_key,
            telemetry_store=telemetry_store,
            cache_config=cache_config,
            backend_service=backend,
        )

        return {
            "success": True,
            "max_segment_length": max_segment_length,
        }

    except RuntimeError as runtime_error:
        print(f"[ERROR] Pipeline error: {runtime_error}")
        return {
            "success": False,
            "error": str(runtime_error),
        }
