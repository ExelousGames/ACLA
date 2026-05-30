"""
Enrichment stage of the training pipeline.

Trains the imitation-learning and tire-grip enrichment models on the cleaned
top laps, then streams the cached session chunks through both models to produce
the enriched dataset used by segmentation and transformer training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.domain.segment import PredictedSegment
from app.features.tire_grip import TireGripAnalysisService
from app.integrations.backend.client import backend_service as default_backend_service
from app.ml.imitation.model import NoExpertLapError
from app.ml.imitation.service import ExpertImitateLearningService
from app.ml.segment_classifier.service import segment_classifier
from app.pipelines.inference.visualizer import (
    visualize_optimal_segments,
    visualize_segment_position_coverage,
)
from app.pipelines.training.pipeline.cleaning import print_section_divider


async def enrich_sessions_with_context(
    chunk_data: List[Dict[str, Any]],
    imitation_learning: ExpertImitateLearningService,
    tire_service: TireGripAnalysisService,
) -> List[Dict[str, Any]]:
    """Enrich a single chunk with all contextual features."""
    if not chunk_data:
        return []

    chunk_imitation_features = []
    try:
        chunk_imitation_features = imitation_learning.extract_expert_state_for_telemetry(chunk_data)
    except NoExpertLapError as e:
        print(
            f"[WARN] No expert lap for ({e.track}, {e.car}); "
            f"skipping session ({len(chunk_data)} records)"
        )
        return []
    except Exception as e:
        raise RuntimeError(f"Failed to extract imitation features: {str(e)}")

    chunk_grip_features = await tire_service.extract_tire_grip_features(chunk_data)

    enriched_chunk = []
    for i, telemetry_record in enumerate(chunk_data):
        enriched_record = telemetry_record.copy()

        if i < len(chunk_imitation_features):
            enriched_record.update(chunk_imitation_features[i])

        if i < len(chunk_grip_features):
            enriched_record.update(chunk_grip_features[i])

        enriched_chunk.append(enriched_record)

    return enriched_chunk


async def cache_segment_batch(
    segments_batch: List[List[Dict[str, Any]]],
    base_cache_key: str,
    batch_number: int,
    *,
    telemetry_store,
) -> bool:
    """Cache a batch of segments to keep memory usage reasonable."""
    try:
        async def segments_generator():
            yield segments_batch

        total_records = sum(len(segment) for segment in segments_batch)
        estimated_size_mb = (total_records * 60) / (1024 * 1024)

        cache_success = await telemetry_store.cache_chunks_streaming(
            cache_key=base_cache_key,
            chunks_iterator=segments_generator(),
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
    *,
    telemetry_store,
    cache_config,
    logger: Optional[logging.Logger] = None,
    top_laps_cache_key: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    """Retrieve cached top laps telemetry list for downstream use."""
    log = logger or logging.getLogger(__name__)
    cache_key = top_laps_cache_key or cache_config.top_laps_cache_key

    try:
        if not telemetry_store.has_cached_data(cache_key):
            raise ValueError(f"No cached top laps found at key: {cache_key}")

        chunks_iterator = telemetry_store.get_cached_data_chunks(cache_key=cache_key)

        all_top_laps = []
        chunk_count = 0

        for chunk in chunks_iterator:
            if isinstance(chunk, tuple):
                chunk = chunk[0]

            if isinstance(chunk, list):
                all_top_laps.extend(chunk)
                chunk_count += 1

        if chunk_count > 0:
            log.info(
                "Retrieved %d top laps from %d cache chunks",
                len(all_top_laps),
                chunk_count,
            )
            return all_top_laps

        raise ValueError(f"Cached data at {cache_key} has unexpected format or is empty")

    except Exception as error:
        log.error("Failed to retrieve cached top laps: %s", error)
        raise


async def enriched_contextual_data(
    sessions_cache_key: str,
    *,
    telemetry_store,
    cache_config,
    backend_service=None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Any]:
    """
    Streamlined contextual data enrichment using chunk iterator approach.

    1. Train all enrichment models using expert data
    2. Use chunk_iterator to process cached session data
    3. Enrich each chunk with contextual features
    4. Cache enriched data
    """
    backend = backend_service or default_backend_service
    log = logger or logging.getLogger(__name__)

    print_section_divider("TRAINING ENRICHMENT MODELS WITH EXPERT DATA")

    imitation_learning = ExpertImitateLearningService(logger=log, debug=True)

    top_laps_cache_key = cache_config.top_laps_cache_key
    imitation_result = await imitation_learning.train_ai_model(top_laps_cache_key)

    serialized_data = imitation_learning.serialize_learning_model()
    if not serialized_data:
        raise Exception("No serialized model data available from imitation learning")

    await backend.save_ai_model(
        model_type="imitation_learning",
        model_data=serialized_data,
        metadata=imitation_result.get("learning_summary", {}),
        is_active=True,
    )
    print("[INFO] ✓ Imitation learning model trained and saved")

    print(f"[INFO] Retrieving top laps for corner identification from {top_laps_cache_key}")
    top_laps_list = await get_cached_all_top_laps_in_one_list(
        telemetry_store=telemetry_store,
        cache_config=cache_config,
        logger=log,
        top_laps_cache_key=top_laps_cache_key,
    )

    del top_laps_list

    tire_service = TireGripAnalysisService()

    print("[INFO] Streaming cached session telemetry to train tire grip model")
    session_training_iterator = telemetry_store.get_cached_data_chunks(
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

    await backend.save_ai_model(
        model_type="tire_grip_analysis",
        model_data=tire_service_serialized,
        metadata={
            "training_summary": tire_grip_training,
            "serialization_timestamp": tire_service_serialized.get("serialized_timestamp"),
            "feature_catalog": tire_service.feature_catalog.CONTEXT_FEATURES,
        },
        is_active=True,
    )
    print("[INFO] ✓ Tire grip analysis model trained on sessions and saved")

    print_section_divider("PROCESSING SESSION DATA VIA CHUNK ITERATOR")

    enriched_sessions_cache_key = cache_config.enriched_sessions_cache_key

    session_chunks_iterator = telemetry_store.get_cached_data_chunks(
        cache_key=sessions_cache_key,
        include_ids=True,
    )

    processed_chunks = 0

    for chunk_tuple in session_chunks_iterator:
        session_chunk_df, chunk_id = chunk_tuple
        session_chunk_df = pd.DataFrame(session_chunk_df)

        if session_chunk_df is None or session_chunk_df.empty:
            continue

        processed_chunks += 1
        chunk_data = session_chunk_df.to_dict('records')

        print(f"[INFO] Processing chunk {processed_chunks}: {len(chunk_data)} records")

        enriched_chunk_data = await enrich_sessions_with_context(
            chunk_data, imitation_learning, tire_service
        )

        if not enriched_chunk_data:
            print(f"[INFO] Chunk {processed_chunks} skipped (no expert reference); not cached")
            del chunk_data
            continue

        async def enriched_chunk_generator():
            yield (enriched_chunk_data, chunk_id)

        await telemetry_store.cache_chunks_streaming(
            cache_key=enriched_sessions_cache_key,
            chunks_iterator=enriched_chunk_generator(),
        )

        del chunk_data, enriched_chunk_data

        if processed_chunks % 5 == 0:
            print(f"[INFO] Progress: {processed_chunks} chunks enriched and cached")

    print(f"[SUCCESS] Enrichment completed:")
    print(f"  - Processed {processed_chunks} data chunks from cache")
    print(f"  - Enriched data cached to: {enriched_sessions_cache_key}")

    return enriched_sessions_cache_key, imitation_learning


async def process_and_cache_segments(
    enriched_sessions_cache_key: str,
    segments_cache_key: str,
    max_segment_length: int,
    *,
    telemetry_store,
) -> str:
    """Process enriched sessions: filter into segments, visualize, and cache."""
    print_section_divider("PROCESSING ENRICHED DATA INTO SEGMENTS")

    session_chunks_iterator = telemetry_store.get_cached_data_chunks(
        cache_key=enriched_sessions_cache_key,
        include_ids=True,
    )

    processed_chunks = 0

    for chunk_tuple in session_chunks_iterator:
        session_chunk_df, chunk_id = chunk_tuple
        session_chunk_df = pd.DataFrame(session_chunk_df)
        if session_chunk_df is None or session_chunk_df.empty:
            continue

        processed_chunks += 1

        print(f"[INFO] Processing enriched chunk {processed_chunks}: {len(session_chunk_df)} records")

        await segment_classifier.scan_telemetry_data(
            dataframe=session_chunk_df,
            window_size=max_segment_length,
        )

        if processed_chunks % 5 == 0:
            print(f"[INFO] Progress: {processed_chunks} chunks processed")

    print(f"[SUCCESS] Segment generation completed:")
    print(f"  - Processed {processed_chunks} enriched chunks")
    print(f"  - Segments cached to: {segments_cache_key}")

    print_section_divider("GENERATING VISUALIZATIONS FROM CACHED SEGMENTS")

    coverage_histogram_bins = np.linspace(0.0, 1.0, num=101)
    coverage_histogram_counts = np.zeros(len(coverage_histogram_bins) - 1, dtype=np.float64)
    coverage_sample_count = 0

    segments_to_visualize = []
    max_viz_segments = 5

    cached_segments_iterator = telemetry_store.get_cached_data_chunks(segments_cache_key)

    processed_viz_chunks = 0
    for chunk_segments in cached_segments_iterator:
        if not chunk_segments:
            continue

        if len(segments_to_visualize) < max_viz_segments:
            remaining = max_viz_segments - len(segments_to_visualize)
            for seg_dict in chunk_segments[:remaining]:
                pred_seg = PredictedSegment(**seg_dict)
                segments_to_visualize.append(pred_seg.telemetry_data)

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

    if segments_to_visualize:
        try:
            visualization_payloads = visualize_optimal_segments(
                segments_to_visualize,
                max_segments=len(segments_to_visualize),
                file_name_prefix=f"{segments_cache_key}_sample",
                return_base64=False,
            )
            if visualization_payloads:
                print(f"[INFO] Generated {len(visualization_payloads)} sample segment visualizations")
        except Exception as viz_error:
            print(f"[WARN] Failed to visualize sample segments: {viz_error}")

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
