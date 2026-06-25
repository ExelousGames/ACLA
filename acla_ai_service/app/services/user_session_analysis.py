from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app.shared.circuit_sections import CIRCUIT_SECTION_RANGES
from app.shared.labels import LABEL_CATEGORIES, LABEL_MAPPING, LABEL_NAME_TO_ID
from app.integrations.backend.client import backend_service
from app.ml.segment_classifier.service import segment_classifier


NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"
WINDOW_ROWS = 2000
WINDOW_OVERLAP_ROWS = 100
logger = logging.getLogger(__name__)


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
            "sessionsFailed": 0,
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


def _parent_label_id(label_id: str) -> Optional[str]:
    for parent_id, child_ids in LABEL_CATEGORIES.items():
        if label_id in child_ids:
            return parent_id
    return None


def _label_name(label_id: str) -> str:
    return LABEL_MAPPING.get(label_id, label_id)


def _child_segment_kind(label_id: str) -> str:
    if label_id == "EA" or label_id.startswith("O") or label_id.startswith("OD"):
        return "strength"
    if label_id.startswith("MSP") or label_id.startswith("MSR"):
        return "needs_work"
    if label_id.startswith("RM"):
        return "recovery"
    return "info"


def _section_score(section_summary: Dict[str, Any]) -> int:
    return int(section_summary.get("expertLevelTurns", 0)) - int(section_summary.get("mistakes", 0))


def _child_segments_for_section(section_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    child_segments = []
    for label_id, count in sorted(
        section_summary.get("labelCounts", {}).items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    ):
        parent_label_id = _parent_label_id(label_id)
        child_segments.append(
            {
                "childSegmentId": label_id,
                "childSegmentName": _label_name(label_id),
                "labelId": label_id,
                "labelName": _label_name(label_id),
                "parentLabelId": parent_label_id,
                "parentLabelName": _label_name(parent_label_id) if parent_label_id else None,
                "count": int(count),
                "kind": _child_segment_kind(label_id),
            }
        )
    return child_segments


def _build_parent_segments(track_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    parent_segments = []
    for section_id, section_summary in track_summary.get("sections", {}).items():
        child_segments = _child_segments_for_section(section_summary)
        if not child_segments:
            continue

        parent_segments.append(
            {
                "parentSegmentId": section_id,
                "parentSegmentName": section_summary.get("sectionName") or _label_name(section_id),
                "sectionId": section_id,
                "sectionName": section_summary.get("sectionName") or _label_name(section_id),
                "expertLevelTurns": int(section_summary.get("expertLevelTurns", 0)),
                "mistakes": int(section_summary.get("mistakes", 0)),
                "practiceMistakes": int(section_summary.get("practiceMistakes", 0)),
                "racingMistakes": int(section_summary.get("racingMistakes", 0)),
                "score": _section_score(section_summary),
                "childSegments": child_segments,
            }
        )

    return sorted(
        parent_segments,
        key=lambda segment: (
            -int(segment["mistakes"] + segment["expertLevelTurns"]),
            str(segment["parentSegmentName"]),
        ),
    )


def _ranked_child_segments(
    parent_segments: List[Dict[str, Any]],
    kinds: Sequence[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    ranked = []
    kind_set = set(kinds)
    for parent_segment in parent_segments:
        for child_segment in parent_segment.get("childSegments", []):
            if child_segment.get("kind") not in kind_set:
                continue
            ranked.append(
                {
                    "parentSegmentId": parent_segment["parentSegmentId"],
                    "parentSegmentName": parent_segment["parentSegmentName"],
                    "childSegmentId": child_segment["childSegmentId"],
                    "childSegmentName": child_segment["childSegmentName"],
                    "count": child_segment["count"],
                    "kind": child_segment["kind"],
                }
            )

    return sorted(
        ranked,
        key=lambda item: (-int(item["count"]), str(item["parentSegmentName"]), str(item["childSegmentName"])),
    )[:limit]


def _finalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    for track_summary in summary.get("tracks", {}).values():
        parent_segments = _build_parent_segments(track_summary)
        track_summary["parentSegments"] = parent_segments
        track_summary["strengths"] = _ranked_child_segments(parent_segments, ["strength"])
        track_summary["improvementAreas"] = _ranked_child_segments(parent_segments, ["needs_work", "recovery"])
        track_summary["trackOverview"] = {
            "parentSegmentCount": len(parent_segments),
            "strengthCount": sum(1 for segment in parent_segments for child in segment["childSegments"] if child["kind"] == "strength"),
            "needsWorkCount": sum(1 for segment in parent_segments for child in segment["childSegments"] if child["kind"] == "needs_work"),
            "recoveryCount": sum(1 for segment in parent_segments for child in segment["childSegments"] if child["kind"] == "recovery"),
        }
    return summary


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


async def analyze_user_sessions(user_id: str, session_limit: int = 10) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "classifier": "segment_classifier",
        "sessionsAnalyzed": 0,
        "sessionsSkipped": 0,
        "sessionsFailed": 0,
        "totalTelemetryRows": 0,
        "errors": [],
        "tracks": {},
    }

    session_limit = max(1, min(int(session_limit or 10), 10))
    init_response = await backend_service.get_user_analysis_sessions(user_id, session_limit)
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

        car_name = str(session_meta.get("car_name") or "unknown")
        track_summary["cars"][car_name] = track_summary["cars"].get(car_name, 0) + 1

        buffer: List[Dict[str, Any]] = []
        buffer_start = 0
        seen = set()

        try:
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
        except Exception as exc:
            session_id = str(session_meta.get("sessionId") or "")
            message = str(exc)
            logger.exception("User session analysis failed for session %s", session_id)
            summary["sessionsFailed"] += 1
            track_summary["sessionsFailed"] += 1
            summary["errors"].append(
                {
                    "sessionId": session_id,
                    "trackId": track_id,
                    "message": message,
                }
            )
            continue

        summary["sessionsAnalyzed"] += 1
        track_summary["sessionsAnalyzed"] += 1

    return _finalize_summary(summary)


__all__ = [
    "analyze_user_sessions",
    "_section_for_rows",
    "_track_id",
]
