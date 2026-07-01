"""
Cleaning stage of the training pipeline.

Streams raw session telemetry, filters / cleans per-session via FeatureProcessor,
selects the top laps per track, and caches the processed chunks for downstream
enrichment.
"""

import asyncio
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.shared.telemetry import FeatureProcessor
from app.pipelines.manifest.protection import (
    is_protected_chunk_id,
    session_id_from_chunk_id,
)

PLAYER_POSITION_COLUMNS = (
    "Graphics_player_pos_x",
    "Graphics_player_pos_y",
    "Graphics_player_pos_z",
)
CAR_POSITION_RE = re.compile(r"^Car_(\d+)_pos_([xyz])$")
MAX_ABS_TRACK_COORDINATE_M = 100_000.0
MAX_PLAYER_POSITION_JUMP_M = 500.0
MAX_PLAYER_POSITION_SPEED_MPS = 250.0
MAX_OPPONENT_POSITION_JUMP_M = 1_000.0
MAX_OPPONENT_DISTANCE_FROM_PLAYER_M = 20_000.0
TopLapKey = Tuple[str, str, int]
TopLapMap = Dict[TopLapKey, Dict[str, Any]]


def _records_from_chunk_payload(chunk_data: Any) -> List[Dict[str, Any]]:
    if isinstance(chunk_data, list):
        return chunk_data
    if isinstance(chunk_data, dict) and isinstance(chunk_data.get("data"), list):
        return chunk_data["data"]
    return []


def _iter_session_payloads(
    chunk_tuples: Iterable[Tuple[Any, Any]],
    protected_session_ids: Optional[Iterable[Any]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> Iterable[Tuple[List[Dict[str, Any]], str, int]]:
    """Assemble raw GridFS chunks back into session-sized payloads."""
    current_session_id: Optional[str] = None
    current_records: List[Dict[str, Any]] = []
    current_chunk_count = 0

    def bump(name: str) -> None:
        if stats is not None:
            stats[name] = stats.get(name, 0) + 1

    for chunk_data, chunk_id in chunk_tuples:
        bump("raw_chunks_seen")
        session_id = session_id_from_chunk_id(chunk_id)

        if current_session_id is not None and session_id != current_session_id:
            yield current_records, current_session_id, current_chunk_count
            current_records = []
            current_chunk_count = 0

        if is_protected_chunk_id(chunk_id, protected_session_ids):
            bump("protected_chunks_skipped")
            current_session_id = None
            continue

        current_session_id = session_id
        current_records.extend(_records_from_chunk_payload(chunk_data))
        current_chunk_count += 1

    if current_session_id is not None:
        yield current_records, current_session_id, current_chunk_count


def print_section_divider(title: str, width: int = 80) -> None:
    """Print a large console divider to visually separate log sections."""
    normalized_width = max(width, len(title) + 4)
    divider = "=" * normalized_width
    print(f"\n{divider}\n{title.center(normalized_width)}\n{divider}")


def _coordinate_invalid_mask(
    df: pd.DataFrame,
    columns: List[str],
    *,
    max_abs_coordinate: float = MAX_ABS_TRACK_COORDINATE_M,
) -> pd.Series:
    invalid = pd.Series(False, index=df.index)
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        invalid |= (
            (~np.isfinite(values.to_numpy(dtype=float)))
            | (values.abs() > max_abs_coordinate)
        )
    return invalid


def _isolated_position_spike_mask(
    df: pd.DataFrame,
    columns: List[str],
    *,
    max_jump_m: float,
    max_speed_mps: Optional[float] = None,
) -> pd.Series:
    if len(df) < 3 or len(columns) < 2:
        return pd.Series(False, index=df.index)

    coords = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_rows = np.isfinite(coords).all(axis=1)

    prev_jump = np.zeros(len(df), dtype=bool)
    next_jump = np.zeros(len(df), dtype=bool)

    prev_delta = coords[1:] - coords[:-1]
    prev_distance = np.linalg.norm(prev_delta, axis=1)
    finite_prev = finite_rows[1:] & finite_rows[:-1]
    prev_jump[1:] = finite_prev & (prev_distance > max_jump_m)
    next_jump[:-1] = prev_jump[1:]

    if max_speed_mps is not None and "Graphics_current_time" in df.columns:
        time_values = pd.to_numeric(
            df["Graphics_current_time"],
            errors="coerce",
        ).to_numpy(dtype=float)
        time_delta_s = (time_values[1:] - time_values[:-1]) / 1000.0
        valid_time_delta = np.isfinite(time_delta_s) & (time_delta_s > 0.0)

        speed_bad = np.zeros(len(df), dtype=bool)
        speeds = np.divide(
            prev_distance,
            time_delta_s,
            out=np.zeros_like(prev_distance),
            where=valid_time_delta,
        )
        speed_bad[1:] = finite_prev & valid_time_delta & (speeds > max_speed_mps)

        speed_bad_next = np.zeros(len(df), dtype=bool)
        speed_bad_next[:-1] = speed_bad[1:]
        prev_jump |= speed_bad
        next_jump |= speed_bad_next

    return pd.Series(prev_jump & next_jump, index=df.index)


def _car_position_slots(df: pd.DataFrame) -> Dict[int, List[str]]:
    slots: Dict[int, List[str]] = {}
    for col in df.columns:
        match = CAR_POSITION_RE.match(str(col))
        if not match:
            continue
        slot = int(match.group(1))
        slots.setdefault(slot, []).append(col)
    return slots


def _clean_position_anomalies(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Remove impossible player samples and clear impossible opponent slots."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    cleaned = df.copy()
    player_columns = [col for col in PLAYER_POSITION_COLUMNS if col in cleaned.columns]
    car_slots = _car_position_slots(cleaned)
    position_columns = player_columns + [
        col for columns in car_slots.values() for col in columns
    ]

    for col in position_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    label = f" ({context})" if context else ""

    if player_columns:
        player_invalid = _coordinate_invalid_mask(cleaned, player_columns)
        player_spikes = _isolated_position_spike_mask(
            cleaned,
            player_columns,
            max_jump_m=MAX_PLAYER_POSITION_JUMP_M,
            max_speed_mps=MAX_PLAYER_POSITION_SPEED_MPS,
        )
        rows_to_drop = player_invalid | player_spikes
        if rows_to_drop.any():
            count = int(rows_to_drop.sum())
            print(
                "[WARNING] Removed "
                f"{count} telemetry rows with invalid/anomalous player positions{label}"
            )
            cleaned = cleaned.loc[~rows_to_drop].copy()

    if cleaned.empty:
        return cleaned.reset_index(drop=True)

    player_xy_available = {
        "Graphics_player_pos_x",
        "Graphics_player_pos_y",
    }.issubset(cleaned.columns)
    opponent_samples_cleared = 0

    for slot, columns in car_slots.items():
        available_columns = [col for col in columns if col in cleaned.columns]
        if not available_columns:
            continue

        bad_mask = _coordinate_invalid_mask(cleaned, available_columns)
        bad_mask |= _isolated_position_spike_mask(
            cleaned,
            available_columns,
            max_jump_m=MAX_OPPONENT_POSITION_JUMP_M,
        )

        x_col = f"Car_{slot}_pos_x"
        y_col = f"Car_{slot}_pos_y"
        if player_xy_available and x_col in cleaned.columns and y_col in cleaned.columns:
            ox = pd.to_numeric(cleaned[x_col], errors="coerce").to_numpy(dtype=float)
            oy = pd.to_numeric(cleaned[y_col], errors="coerce").to_numpy(dtype=float)
            px = pd.to_numeric(
                cleaned["Graphics_player_pos_x"],
                errors="coerce",
            ).to_numpy(dtype=float)
            py = pd.to_numeric(
                cleaned["Graphics_player_pos_y"],
                errors="coerce",
            ).to_numpy(dtype=float)
            active = (
                ((ox != 0.0) | (oy != 0.0))
                & np.isfinite(ox)
                & np.isfinite(oy)
            )
            player_finite = np.isfinite(px) & np.isfinite(py)
            distance = np.sqrt((ox - px) ** 2 + (oy - py) ** 2)
            bad_mask |= pd.Series(
                active
                & player_finite
                & np.isfinite(distance)
                & (distance > MAX_OPPONENT_DISTANCE_FROM_PLAYER_M),
                index=cleaned.index,
            )

        if bad_mask.any():
            row_count = int(bad_mask.sum())
            opponent_samples_cleared += row_count
            cleaned.loc[bad_mask, available_columns] = 0.0

    if opponent_samples_cleared:
        print(
            "[WARNING] Cleared "
            f"{opponent_samples_cleared} anomalous opponent position samples{label}"
        )

    return cleaned.reset_index(drop=True)


def _update_top_lap(
    top_laps: TopLapMap,
    candidate: Dict[str, Any],
) -> Tuple[Optional[TopLapKey], Optional[Dict[str, Any]], bool]:
    records = candidate.get("records", [])
    if not records:
        return None, None, False

    key = (
        records[0].get("Static_track", "unknown_track"),
        candidate.get("car", "unknown_car"),
        candidate.get("avg_grip_int", 2),
    )
    current_top_lap = top_laps.get(key)
    if (
        current_top_lap is None
        or candidate["lap_time_ms"] < current_top_lap["lap_time_ms"]
    ):
        top_laps[key] = candidate
        return key, current_top_lap, True

    return key, None, False


async def process_lap_sessions_efficiently(
    session_data_cache_key: str,
    *,
    telemetry_store,
    cache_config,
    imitate_expert_feature_names: List[str],
    telemetry_time_gap_ms: int = 100,
    processed_sessions_cache_key: Optional[str] = None,
    protected_session_ids: Optional[Iterable[Any]] = None,
) -> None:
    """
    Streamlined processing of large cached datasets with a bounded memory footprint while caching
    full session telemetry for downstream training.
    """
    print(
        f"[INFO] Processing cached dataset '{session_data_cache_key}' with immediate caching"
    )

    # Keyed by (track, car, avg_grip_int). Stores the fastest lap per combo.
    top_laps: TopLapMap = {}
    total_sessions_cached = 0
    total_processed = 0
    chunk_idx = 0

    features = imitate_expert_feature_names

    def update_top_laps(candidate: Dict[str, Any]) -> None:
        records = candidate.get("records", [])
        print(
            f"[DEBUG] Evaluating lap {candidate.get('id', 'unknown')} with {len(records)} telemetry records"
        )

        key, replaced_lap, updated = _update_top_lap(top_laps, candidate)
        if key is None:
            return

        if updated and replaced_lap is None:
            print(
                f"[DEBUG] Added top lap {candidate['id']} for {key} "
                f"(time: {candidate['lap_time_ms']}ms)"
            )
            return

        if replaced_lap is not None:
            print(
                f"[DEBUG] Replaced top lap {replaced_lap['id']} with "
                f"{candidate['id']} for {key} (time: {candidate['lap_time_ms']}ms)"
            )

    session_chunks_iterator = telemetry_store.get_cached_data_chunks(cache_key=session_data_cache_key, include_ids=True)
    print(f"[DEBUG] Created chunk iterator for cache key: {session_data_cache_key}")

    session_chunks_processed = 0
    chunk_stats: Dict[str, int] = {}

    async def process_chunk_result(result_tuple):
        nonlocal chunk_idx, total_processed, total_sessions_cached
        c_idx, laps_processed, candidates, records, error, chunk_id = result_tuple

        if error:
            print(f"[WARNING] Chunk {c_idx} processing failed: {error}")
            return

        chunk_idx += 1
        total_processed += len(records)

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

        for session_records, session_id, session_chunk_count in _iter_session_payloads(
            session_chunks_iterator,
            protected_session_ids,
            chunk_stats,
        ):
            session_chunks_processed += session_chunk_count

            future = executor.submit(
                process_single_chunk,
                session_records,
                session_id,
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
    print(f"[DEBUG] - Protected chunks skipped: {chunk_stats.get('protected_chunks_skipped', 0)}")
    print(f"[DEBUG] - Valid sessions processed: {chunk_idx}")
    print(f"[DEBUG] - Total records processed: {total_processed}")
    print(f"[DEBUG] - Tracks found: {len(top_laps)}")
    print(f"[DEBUG] - Session chunks cached: {total_sessions_cached}")

    if top_laps:
        for key, lap_info in top_laps.items():
            print(f"[DEBUG] Top lap time for {key}: {lap_info['lap_time_ms']}")

    if not session_chunks_processed:
        if chunk_stats.get("protected_chunks_skipped", 0):
            print("[INFO] No unprotected chunks were available for processing.")
            return
        raise ValueError(
            f"No chunks were returned by iterator for cache key {session_data_cache_key}. Check if data exists in cache."
        )

    if not chunk_idx:
        raise ValueError(
            f"All {session_chunks_processed} chunks failed processing for cache key {session_data_cache_key}. Check data quality."
        )

    if not top_laps:
        raise ValueError(
            f"No top laps found. Processed {chunk_idx} valid sessions from {session_chunks_processed} raw chunks."
        )

    if total_sessions_cached == 0:
        raise ValueError("No session data cached for transformer training")

    print(f"[SUCCESS] Processed {chunk_idx} sessions from {session_chunks_processed} raw chunks")
    print(f"[SUCCESS] Selected top laps from {len(top_laps)} (track, car, grip) buckets")
    print(f"[SUCCESS] Cached {total_sessions_cached} session batches across {chunk_idx} sessions")

    top_laps_cache_key = cache_config.top_laps_cache_key
    try:
        async def top_laps_generator():
            for key, lap_info in top_laps.items():
                track_name, car_name, avg_grip_int = key
                bucket_records = [lap_info["records"]]

                if bucket_records:
                    chunk_id = f"{track_name}|{car_name}|grip{avg_grip_int}"
                    print(f"[DEBUG] Yielding top lap chunk for {key}")
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
) -> Tuple[Union[int, str], int, List[Dict[str, Any]], List[Dict[str, Any]], Optional[Exception], Optional[str]]:
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
        processed_df = _clean_position_anomalies(
            processed_df,
            context=f"chunk {chunk_id}",
        )
        processor.df = processed_df

        if processed_df.empty:
            return chunk_idx, 0, [], [], None, chunk_id

        stripped_session_df = processor.strip_dataframe_by_time_gap(processed_df, telemetry_time_gap_ms)
        stripped_session_df = _clean_position_anomalies(
            stripped_session_df,
            context=f"chunk {chunk_id} downsampled",
        )

        if stripped_session_df.empty:
            return chunk_idx, 0, [], [], None, chunk_id

        filtered_session_df = processor.filter_features_by_list(stripped_session_df, features)
        filtered_session_df = filtered_session_df.reset_index(drop=True)

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
