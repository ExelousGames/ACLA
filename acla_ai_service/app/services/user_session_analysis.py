from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app.domain.circuit_sections import CIRCUIT_SECTION_RANGES
from app.domain.labels import LABEL_MAPPING, LABEL_NAME_TO_ID
from app.integrations.backend.client import backend_service
from app.ml.segment_classifier.service import segment_classifier


NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"
WINDOW_ROWS = 2000
WINDOW_OVERLAP_ROWS = 100


def _track_id(raw: Any) -> str:
    value = str(raw or "").strip()
    if value in LABEL_MAPPING:
        return value
    mapped = LABEL_NAME_TO_ID.get(value)
    if mapped:
        return mapped
    return value.lower().replace(" ", "_").replace("-", "_")


def _has_measured_sections(track_id: str) -> bool:
    prefix = f"{track_id}"
    return any(section_id.startswith(prefix) for section_id in CIRCUIT_SECTION_RANGES)


def _position_in_range(position: float, section_range: Tuple[float, float]) -> bool:
    lo, hi = section_range
    position = position % 1.0
    if hi >= lo:
        return lo <= position <= hi
    return position >= lo or position <= hi


def _section_for_rows(rows: Sequence[Dict[str, Any]], track_id: str) -> Optional[str]:
    positions: List[float] = []
    for row in rows:
        try:
            value = float(row.get(NORMALIZED_POSITION_COLUMN))
        except (TypeError, ValueError):
            continue
        positions.append(value % 1.0)

    if not positions:
        return None

    position = sorted(positions)[len(positions) // 2]
    candidates = [
        (section_id, section_range)
        for section_id, section_range in CIRCUIT_SECTION_RANGES.items()
        if section_id.startswith(track_id)
    ]
    for section_id, section_range in candidates:
        if _position_in_range(position, section_range):
            return section_id
    return None


def _ensure_track(summary: Dict[str, Any], track_id: str, track_name: str) -> Dict[str, Any]:
    tracks = summary.setdefault("tracks", {})
    return tracks.setdefault(
        track_id,
        {
            "trackName": track_name,
            "sessionsAnalyzed": 0,
            "sessionsSkipped": 0,
            "totalTelemetryRows": 0,
            "cars": {},
            "sections": {},
        },
    )


def _ensure_section(track_summary: Dict[str, Any], section_id: str) -> Dict[str, Any]:
    sections = track_summary.setdefault("sections", {})
    return sections.setdefault(
        section_id,
        {
            "sectionName": LABEL_MAPPING.get(section_id, section_id),
            "expertLevelTurns": 0,
            "mistakes": 0,
            "practiceMistakes": 0,
            "racingMistakes": 0,
            "labelCounts": {},
        },
    )


def _increment_label(section_summary: Dict[str, Any], label: str) -> None:
    section_summary["labelCounts"][label] = section_summary["labelCounts"].get(label, 0) + 1
    if label == "EA":
        section_summary["expertLevelTurns"] += 1
    if label.startswith("MSP") or label.startswith("MSR"):
        section_summary["mistakes"] += 1
    if label.startswith("MSP"):
        section_summary["practiceMistakes"] += 1
    if label.startswith("MSR"):
        section_summary["racingMistakes"] += 1


def _scan_window(
    summary: Dict[str, Any],
    session_meta: Dict[str, Any],
    rows: List[Dict[str, Any]],
    base_index: int,
    emit_end_index: Optional[int],
    seen: set,
) -> None:
    if not rows:
        return

    track_id = _track_id(session_meta.get("map"))
    track_summary = _ensure_track(summary, track_id, str(session_meta.get("map") or track_id))
    df = pd.DataFrame(rows)
    predicted_segments = segment_classifier.scan_telemetry_data(df)
    session_id = str(session_meta.get("sessionId") or "")

    for segment in predicted_segments:
        start = int(segment.start_index or 0)
        end = int(segment.end_index or start)
        if end <= start:
            continue

        global_start = base_index + start
        global_end = base_index + end
        midpoint = (global_start + global_end) / 2
        if emit_end_index is not None and midpoint >= emit_end_index:
            continue

        segment_rows = rows[start:end]
        section_id = _section_for_rows(segment_rows, track_id)
        if not section_id:
            continue

        labels = sorted(str(label) for label in segment.labels)
        dedupe_key = (session_id, global_start, global_end, tuple(labels))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        section_summary = _ensure_section(track_summary, section_id)
        for label in labels:
            _increment_label(section_summary, label)


async def analyze_user_sessions(user_id: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "classifier": "segment_classifier",
        "sessionsAnalyzed": 0,
        "sessionsSkipped": 0,
        "totalTelemetryRows": 0,
        "tracks": {},
    }

    init_response = await backend_service.get_user_analysis_sessions(user_id)
    sessions = init_response.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []

    for session_meta in sessions:
        track_id = _track_id(session_meta.get("map"))
        track_name = str(session_meta.get("map") or track_id)
        track_summary = _ensure_track(summary, track_id, track_name)

        if not _has_measured_sections(track_id):
            summary["sessionsSkipped"] += 1
            track_summary["sessionsSkipped"] += 1
            continue

        summary["sessionsAnalyzed"] += 1
        track_summary["sessionsAnalyzed"] += 1

        car_name = str(session_meta.get("car_name") or "unknown")
        track_summary["cars"][car_name] = track_summary["cars"].get(car_name, 0) + 1

        buffer: List[Dict[str, Any]] = []
        buffer_start = 0
        seen = set()

        async for chunk_rows in backend_service.iter_user_analysis_chunks(user_id, session_meta):
            summary["totalTelemetryRows"] += len(chunk_rows)
            track_summary["totalTelemetryRows"] += len(chunk_rows)
            buffer.extend(chunk_rows)

            advance = max(1, WINDOW_ROWS - WINDOW_OVERLAP_ROWS)
            while len(buffer) >= WINDOW_ROWS:
                window = buffer[:WINDOW_ROWS]
                emit_end = buffer_start + advance
                _scan_window(summary, session_meta, window, buffer_start, emit_end, seen)
                buffer = buffer[advance:]
                buffer_start += advance

        if buffer:
            _scan_window(summary, session_meta, buffer, buffer_start, None, seen)

    return summary


__all__ = [
    "analyze_user_sessions",
    "_section_for_rows",
    "_track_id",
]
