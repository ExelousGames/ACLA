"""
Cleaning stage of the training pipeline.

Streams raw session telemetry, filters / cleans per-chunk via FeatureProcessor,
selects the top laps per track, and caches the processed chunks for downstream
enrichment.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from app.domain.telemetry import FeatureProcessor


def print_section_divider(title: str, width: int = 80) -> None:
    """Print a large console divider to visually separate log sections."""
    normalized_width = max(width, len(title) + 4)
    divider = "=" * normalized_width
    print(f"\n{divider}\n{title.center(normalized_width)}\n{divider}")


async def process_lap_sessions_efficiently(
    session_data_cache_key: str,
    *,
    telemetry_store,
    cache_config,
    imitate_expert_feature_names: List[str],
    telemetry_time_gap_ms: int = 100,
    processed_sessions_cache_key: Optional[str] = None,
    top_laps_count: int = 5,
) -> None:
    """
    Streamlined processing of large cached datasets with a bounded memory footprint while caching
    full session telemetry for downstream training.
    """
    print(
        f"[INFO] Processing cached dataset '{session_data_cache_key}' with immediate caching"
    )

    # Keyed by (track, car, avg_grip_int). Stores the fastest lap(s) per combo.
    top_laps: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    total_sessions_cached = 0
    total_processed = 0
    chunk_idx = 0

    features = imitate_expert_feature_names

    def update_top_laps(candidate: Dict[str, Any]) -> None:
        records = candidate.get("records", [])
        if not records:
            return
        print(
            f"[DEBUG] Evaluating lap {candidate.get('id', 'unknown')} with {len(records)} telemetry records"
        )

        track_name = records[0].get("Static_track", "unknown_track")
        car_name = candidate.get("car", "unknown_car")
        avg_grip_int = candidate.get("avg_grip_int", 2)
        key = (track_name, car_name, avg_grip_int)

        if key not in top_laps:
            top_laps[key] = []

        bucket = top_laps[key]

        if len(bucket) < top_laps_count:
            bucket.append(candidate)
            print(
                f"[DEBUG] Added lap {candidate['id']} to top laps for {key} ({len(bucket)}/{top_laps_count}, time: {candidate['lap_time_ms']}ms)"
            )
            return

        slowest_idx = max(range(len(bucket)), key=lambda idx: bucket[idx]["lap_time_ms"])
        slowest = bucket[slowest_idx]
        if candidate["lap_time_ms"] < slowest["lap_time_ms"]:
            bucket[slowest_idx] = candidate
            print(
                f"[DEBUG] Replaced slowest lap {slowest['id']} with {candidate['id']} for {key} (time: {candidate['lap_time_ms']}ms)"
            )

    session_chunks_iterator = telemetry_store.get_cached_data_chunks(cache_key=session_data_cache_key, include_ids=True)
    print(f"[DEBUG] Created chunk iterator for cache key: {session_data_cache_key}")

    session_chunks_processed = 0

    async def process_chunk_result(result_tuple):
        nonlocal chunk_idx, total_sessions_cached
        c_idx, laps_processed, candidates, records, error, chunk_id = result_tuple

        if error:
            print(f"[WARNING] Chunk {c_idx} processing failed: {error}")
            return

        chunk_idx += 1

        for candidate in candidates:
            update_top_laps(candidate)

        if records:
            async def single_chunk_generator():
                yield (records, chunk_id)

            try:
                cache_success = await telemetry_store.cache_chunks_streaming(
                    cache_key=processed_sessions_cache_key,
                    chunks_iterator=single_chunk_generator(),
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

    with ProcessPoolExecutor(max_workers=4) as executor:
        pending_aws = set()

        for chunk_tuple in session_chunks_iterator:
            session_chunk_df, chunk_id = chunk_tuple
            session_chunks_processed += 1

            future = executor.submit(
                process_single_chunk,
                session_chunk_df,
                chunk_id,
                features,
                telemetry_time_gap_ms,
            )
            aw = asyncio.wrap_future(future)
            pending_aws.add(aw)

            if len(pending_aws) >= 4:
                done, pending_aws = await asyncio.wait(pending_aws, return_when=asyncio.FIRST_COMPLETED)
                for aw_done in done:
                    try:
                        result = await aw_done
                        await process_chunk_result(result)
                    except Exception as e:
                        print(f"[ERROR] Error retrieving chunk result: {e}")

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
        for key, laps in top_laps.items():
            lap_times = [lap_info["lap_time_ms"] for lap_info in laps]
            print(f"[DEBUG] Top lap times for {key}: {sorted(lap_times)}")

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
    print(f"[SUCCESS] Selected top laps from {len(top_laps)} (track, car, grip) buckets")
    print(f"[SUCCESS] Cached {total_sessions_cached} session batches across {chunk_idx} chunks")

    top_laps_cache_key = cache_config.top_laps_cache_key
    try:
        async def top_laps_generator():
            for key, bucket_laps in top_laps.items():
                track_name, car_name, avg_grip_int = key
                if len(bucket_laps) < top_laps_count:
                    print(f"[WARNING] Insufficient top laps found for {key}: {len(bucket_laps)}/{top_laps_count}")

                bucket_laps.sort(key=lambda entry: entry["lap_time_ms"])

                bucket_records = []
                for lap_info in bucket_laps:
                    bucket_records.append(lap_info["records"])

                if bucket_records:
                    chunk_id = f"{track_name}|{car_name}|grip{avg_grip_int}"
                    print(f"[DEBUG] Yielding top laps chunk for {key} ({len(bucket_records)} laps)")
                    yield (bucket_records, chunk_id)

        cache_success = await telemetry_store.cache_chunks_streaming(
            cache_key=top_laps_cache_key,
            chunks_iterator=top_laps_generator(),
        )

        if cache_success:
            print(f"[SUCCESS] Cached top lap telemetry records to {top_laps_cache_key} (separate chunks per track)")
        else:
            print(f"[WARNING] Failed to cache top laps to {top_laps_cache_key}")
    except Exception as cache_error:
        print(f"[WARNING] Error caching top laps: {cache_error}")


def process_single_chunk(
    chunk_data: Any,
    chunk_idx: Union[int, str],
    features: List[str],
    telemetry_time_gap_ms: int,
) -> Tuple[int, int, List[Dict[str, Any]], List[Dict[str, Any]], Optional[Exception], Optional[str]]:
    """Process a single chunk of telemetry data in a separate process."""
    chunk_id = str(chunk_idx)
    actual_data = chunk_data

    print(f"[DEBUG] Thread started processing chunk {chunk_idx} (ID: {chunk_id}) ({len(actual_data)} records)")
    laps_processed_in_chunk = 0
    candidates: List[Dict[str, Any]] = []
    session_records: List[Dict[str, Any]] = []

    try:
        session_chunk_df = pd.DataFrame(actual_data)
        telemetry_df = session_chunk_df

        if telemetry_df.empty:
            return chunk_idx, 0, [], [], None, chunk_id

        processor = FeatureProcessor(telemetry_df)
        processor.general_cleaning_for_analysis()
        processed_df = processor.flip_y_z_features()

        if processed_df.empty:
            return chunk_idx, 0, [], [], None, chunk_id

        stripped_session_df = processor.strip_dataframe_by_time_gap(processed_df, telemetry_time_gap_ms)

        if stripped_session_df.empty:
            return chunk_idx, 0, [], [], None, chunk_id

        filtered_session_df = processor.filter_features_by_list(stripped_session_df, features)

        if filtered_session_df.empty:
            return chunk_idx, 0, [], [], None, chunk_id

        session_records = filtered_session_df.to_dict("records")

        lap_structs = processor.split_into_laps(filtered_session_df)
        if not lap_structs:
            return chunk_idx, 0, [], session_records, None, chunk_id

        for i, lap_struct in enumerate(lap_structs):
            lap_df = lap_struct["dataframe"]
            lap_time_ms = lap_struct["lap_time_ms"]

            if "Graphics_is_valid_lap" in lap_df.columns:
                if not (lap_df["Graphics_is_valid_lap"] == 1).all():
                    continue

            if "Graphics_current_time" in lap_df.columns and len(lap_df) > 2:
                time_diffs = lap_df["Graphics_current_time"].diff().dropna()
                valid_diffs = time_diffs[time_diffs > 0]

                if not valid_diffs.empty:
                    mean_diff = valid_diffs.mean()
                    std_diff = valid_diffs.std()

                    if std_diff > 0:
                        if std_diff > (mean_diff * 0.5):
                            continue

            lap_records = lap_df.to_dict("records") if not lap_df.empty else []

            lap_identifier = (
                f"{chunk_id}_{i}_{lap_time_ms or 'na'}"
            )

            if "Static_car_model" in lap_df.columns and not lap_df["Static_car_model"].empty:
                car_name = str(lap_df["Static_car_model"].iloc[0])
            else:
                car_name = "unknown_car"

            if "Graphics_track_grip_status" in lap_df.columns:
                grip_mean = pd.to_numeric(lap_df["Graphics_track_grip_status"], errors="coerce").mean()
                if pd.isna(grip_mean):
                    avg_grip_int = 2
                else:
                    avg_grip_int = max(0, min(6, int(round(float(grip_mean)))))
            else:
                avg_grip_int = 2

            candidate_entry = {
                "id": lap_identifier,
                "lap_time_ms": lap_time_ms if lap_time_ms is not None else float("inf"),
                "lap_num": lap_struct["lap_num"],
                "records": lap_records,
                "car": car_name,
                "avg_grip_int": avg_grip_int,
            }

            if lap_time_ms is not None and lap_records:
                laps_processed_in_chunk += 1
                candidates.append(candidate_entry)

        return chunk_idx, laps_processed_in_chunk, candidates, session_records, None, chunk_id

    except Exception as e:
        return chunk_idx, 0, [], [], e, chunk_id
