"""Select reference laps from the persisted cleaned session telemetry."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from app.shared.telemetry import FeatureProcessor
from app.top_laps.model import _resolve_game
from training.pipelines.training.pipeline.parallel import map_chunks_in_processes

TopLapKey = Tuple[str, str, str, int]
TopLapMap = Dict[TopLapKey, Dict[str, Any]]


def _has_complete_lap_coverage(lap_df: pd.DataFrame) -> bool:
    """Require forward coverage in telemetry order, including the lap boundaries."""
    positions = lap_df.get("Graphics_normalized_car_position")
    if positions is None or len(positions) < 2:
        return False
    if not positions.between(0.0, 1.0).all():
        return False
    if positions.iloc[0] > 0.1 or positions.iloc[-1] < 0.9:
        return False

    # Endpoint extrema alone also describe a backwards start/finish crossing.
    # Require samples at most 10% of a lap apart, without reverse travel;
    # tolerate the six-decimal rounding used by telemetry cleaning.
    position_diffs = positions.diff().iloc[1:]
    return bool(position_diffs.between(-1e-6, 0.1 + 1e-6).all())


def _update_top_lap(
    top_laps: TopLapMap,
    candidate: Dict[str, Any],
) -> Tuple[Optional[TopLapKey], Optional[Dict[str, Any]], bool]:
    records = candidate.get("records", [])
    if not records:
        return None, None, False

    key = (
        _resolve_game(None, records[0]),
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


def _iter_lap_candidates(
    records: List[Dict[str, Any]], chunk_id: str,
) -> Iterable[Dict[str, Any]]:
    if not records:
        return
    processor = FeatureProcessor(pd.DataFrame(records))
    lap_structs = processor.split_into_laps()
    for i, lap_struct in enumerate(lap_structs):
        lap_df = lap_struct["dataframe"]
        lap_time_ms = lap_struct["lap_time_ms"]

        if not _has_complete_lap_coverage(lap_df):
            continue

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
            yield candidate_entry


def _select_chunk_top_laps(chunk) -> TopLapMap:
    records, chunk_id = chunk
    top_laps: TopLapMap = {}
    for candidate in _iter_lap_candidates(records, str(chunk_id)):
        _update_top_lap(top_laps, candidate)
    return top_laps


async def select_top_laps(
    processed_sessions_cache_key: str,
    *,
    telemetry_store,
    cache_config,
    max_workers: int = 4,
) -> None:
    """Cache the fastest valid lap per (game, track, car, grip) bucket."""
    top_laps: TopLapMap = {}
    print(f"[INFO] Selecting top laps with {max_workers} worker(s)")
    chunks = telemetry_store.get_cached_data_chunks(
        cache_key=processed_sessions_cache_key, include_ids=True,
    )
    async for chunk_top_laps in map_chunks_in_processes(
        _select_chunk_top_laps, chunks, max_workers=max_workers,
    ):
        for candidate in chunk_top_laps.values():
            _update_top_lap(top_laps, candidate)

    if not top_laps:
        raise ValueError("No top laps found in the cleaned sessions")

    async def top_laps_generator():
        for (game, track, car, grip), lap in top_laps.items():
            yield ([lap["records"]], f"{game}|{track}|{car}|grip{grip}")

    if not await telemetry_store.cache_chunks_streaming(
        cache_key=cache_config.top_laps_cache_key,
        chunks_iterator=top_laps_generator(),
    ):
        raise RuntimeError("Failed to cache selected top laps")
    print(f"[SUCCESS] Selected top laps from {len(top_laps)} (game, track, car, grip) buckets")
