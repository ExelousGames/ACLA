"""Deterministic telemetry-to-label evaluation.

The annotation catalog owns the selection policy.  This module only turns a
telemetry range into stable, scalar facts and evaluates the catalog's
``selection_requirements`` predicate tree.  Missing facts fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.internal_knowledge_base import skills
from app.internal_knowledge_base.label_lookup import get_label
from app.local_annotation_agent.workflow.label_evidence import (
    LabelEvidence,
    resolve_label_evidence,
)
from app.local_annotation_agent.workflow.results import AnnotationResult, LapAnnotationResult
from app.shared.labels import LABEL_MAPPING


SUPPORTED_OPERATORS = frozenset({
    "eq", "neq", "in", "not_in", "lt", "lte", "gt", "gte",
    "between", "contains", "exists",
})
KNOWN_FACTS = frozenset({
    "altitude.apex.trend", "altitude.entry.trend", "altitude.exit.trend",
    "balance.oversteer", "balance.understeer",
    "brake.application_end_relation", "brake.application_onset_relation",
    "brake.hold_length_relation", "brake.peak_ratio", "brake.peak_relation",
    "brake.release_end_relation", "brake.release_onset_relation", "brake.similarity",
    "controls.expert_overlap_count", "controls.overlap_count",
    "gear.downshift_relation", "gear.exit_relation", "gear.upshift_relation",
    "grip.over_limit", "grip.sustained_low",
    "opponent.confidence_level", "opponent.drew_alongside", "opponent.gap_shrank",
    "opponent.outcome", "opponent.side_swap",
    "phase.apex", "phase.entry", "phase.exit",
    "section.name", "section.overlap_names",
    "segment.corner_shape_key", "segment.shape_key",
    "speed.expert_faster", "speed.gap_closing", "speed.gap_peak_abs_kmh",
    "throttle.application_end_relation", "throttle.application_onset_relation",
    "throttle.release_end_relation", "throttle.release_onset_relation",
    "throttle.similarity", "time_gap.direction", "time_gap.end_ms",
    "time_gap.ending_direction", "time_gap.has_significant_rise",
    "time_gap.has_spike", "time_gap.middle_has_new_significant_rise",
    "time_gap.flattening_at_end", "time_gap.overall_gap", "time_gap.slope_shape",
    "time_gap.starting_direction",
    "time_gap.total_change_abs_ms",
    "trajectory.converging", "trajectory.peak_abs_offset_m", "trajectory.position",
    "turn.apex_relation", "turn.exit_relation", "turn.in_relation",
})
_MISSING = object()
_ALIGN_TOLERANCE = 2
_TELEMETRY_SMOOTHING_WINDOW = 3


@dataclass
class RequirementEvaluation:
    matched: bool
    branch: Optional[int] = None
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    fact_ids: List[str] = field(default_factory=list)


@dataclass
class LabelEvaluation:
    labels: List[str]
    evaluations: Dict[str, RequirementEvaluation]
    conflicts: List[Tuple[str, str]] = field(default_factory=list)


class FactSet(dict):
    """Scalar facts plus the telemetry ranges that produced each fact."""

    def __init__(
        self,
        values: Optional[Mapping[str, Any]] = None,
        *,
        evidence: Optional[Mapping[str, Sequence[Tuple[int, int]]]] = None,
        phases: Optional[Mapping[str, Sequence[Tuple[int, int]]]] = None,
    ) -> None:
        super().__init__(values or {})
        self.evidence = {
            key: [(int(start), int(end)) for start, end in ranges]
            for key, ranges in (evidence or {}).items()
        }
        self.phases = {
            key: [(int(start), int(end)) for start, end in ranges]
            for key, ranges in (phases or {}).items()
        }


def _point_range(*points: Optional[int]) -> List[Tuple[int, int]]:
    finite = [int(point) for point in points if point is not None]
    return [(min(finite), max(finite))] if finite else []


def _mask_ranges(mask: np.ndarray, index: np.ndarray) -> List[Tuple[int, int]]:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return []
    return [
        (int(index[int(run[0])]), int(index[int(run[-1])]))
        for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        if len(run)
    ]


def _decreasing_magnitude_ranges(
    values: np.ndarray, index: np.ndarray,
) -> List[Tuple[int, int]]:
    if len(values) < 2 or len(values) != len(index):
        return []
    transitions = (
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & (np.abs(values[1:]) < np.abs(values[:-1]))
    )
    positions = np.flatnonzero(transitions)
    return [
        (int(index[int(run[0])]), int(index[int(run[-1]) + 1]))
        for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        if len(run)
    ]


def _attachment_content(value: Any) -> Dict[str, Any]:
    content = getattr(value, "content", None)
    return content if isinstance(content, dict) else {}


def _smooth_telemetry(values: np.ndarray) -> np.ndarray:
    """Suppress single-sample noise with a centered three-sample median."""
    if len(values) < 2:
        return values
    edge = _TELEMETRY_SMOOTHING_WINDOW // 2
    padded = np.pad(values, edge, mode="edge")
    return (
        pd.Series(padded)
        .rolling(_TELEMETRY_SMOOTHING_WINDOW, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)[edge:-edge]
    )


def _series(df: pd.DataFrame, *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(values)):
                return _smooth_telemetry(values)
    return None


def _raw_speed_delta(segment: pd.DataFrame) -> Optional[np.ndarray]:
    player = _series(segment, "Physics_speed_kmh")
    expert = _series(segment, "expert_optimal_speed")
    if player is None or expert is None or len(player) != len(expert):
        return None
    return expert - player


def _raw_trajectory_offset(segment: pd.DataFrame) -> Optional[np.ndarray]:
    from app.shared.annotation_agent_tools import calculate_trajectory_offset

    return calculate_trajectory_offset(segment)


def _raw_slip_balance(segment: pd.DataFrame) -> Optional[np.ndarray]:
    front_left = _series(segment, "Physics_slip_angle_front_left")
    front_right = _series(segment, "Physics_slip_angle_front_right")
    rear_left = _series(segment, "Physics_slip_angle_rear_left")
    rear_right = _series(segment, "Physics_slip_angle_rear_right")
    if any(value is None for value in (front_left, front_right, rear_left, rear_right)):
        return None
    return (np.abs(rear_left) + np.abs(rear_right)) / 2.0 - (
        np.abs(front_left) + np.abs(front_right)
    ) / 2.0


def _raw_push_to_limit(segment: pd.DataFrame) -> Optional[np.ndarray]:
    from app.shared.tire_grip_features import SlipEnvelopeConfig

    angle_names = (
        "Physics_slip_angle_front_left", "Physics_slip_angle_front_right",
        "Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right",
    )
    ratio_names = (
        "Physics_slip_ratio_front_left", "Physics_slip_ratio_front_right",
        "Physics_slip_ratio_rear_left", "Physics_slip_ratio_rear_right",
    )
    angles = [_series(segment, name) for name in angle_names]
    ratios = [_series(segment, name) for name in ratio_names]
    if any(value is None for value in [*angles, *ratios]):
        return None
    config = SlipEnvelopeConfig()
    lateral = np.maximum.reduce([np.abs(value) for value in angles])
    longitudinal = np.maximum.reduce([np.abs(value) for value in ratios])
    normalized_lateral = lateral / max(config.front_slip_limit, config.rear_slip_limit)
    normalized_longitudinal = longitudinal / max(
        config.front_longitudinal_slip_limit, config.rear_longitudinal_slip_limit,
    )
    return np.sqrt(
        (config.slip_angle_weight * normalized_lateral) ** 2
        + (config.slip_ratio_weight * normalized_longitudinal) ** 2
    )


def _first(mask: np.ndarray, index: np.ndarray) -> Optional[int]:
    positions = np.flatnonzero(mask)
    return int(index[int(positions[0])]) if len(positions) else None


def _relation(player: Optional[int], expert: Optional[int]) -> Optional[str]:
    if player is None or expert is None:
        return None
    delta = int(player) - int(expert)
    if abs(delta) <= _ALIGN_TOLERANCE:
        return "aligned"
    return "earlier" if delta < 0 else "later"


def _input_landmarks(values: Optional[np.ndarray], index: np.ndarray) -> Dict[str, Any]:
    if values is None or len(values) != len(index):
        return {}
    finite = np.where(np.isfinite(values), values, 0.0)
    peak_pos = int(np.argmax(finite))
    peak = float(finite[peak_pos])
    active = finite >= max(0.05, peak * 0.10)
    high = finite >= max(0.10, peak * 0.90)
    application_onset = _first(active, index)
    application_end = _first(high, index)
    release_onset = None
    release_end = None
    after_peak = np.arange(len(finite)) > peak_pos
    candidates = np.flatnonzero(after_peak & (finite < max(0.10, peak * 0.90)))
    if len(candidates):
        release_onset = int(index[int(candidates[0])])
    candidates = np.flatnonzero(after_peak & (finite <= 0.05))
    if len(candidates):
        release_end = int(index[int(candidates[0])])
    hold_length = None
    if application_end is not None and release_onset is not None:
        hold_length = max(0, release_onset - application_end)
    return {
        "application_onset": application_onset,
        "application_end": application_end,
        "release_onset": release_onset,
        "release_end": release_end,
        "peak": peak,
        "peak_iloc": int(index[peak_pos]),
        "hold_length": hold_length,
        "active_fraction": float(np.mean(active)),
    }


def _add_input_facts(
    facts: Dict[str, Any], prefix: str, player: Dict[str, Any], expert: Dict[str, Any],
) -> None:
    for key in ("application_onset", "application_end", "release_onset", "release_end"):
        facts[f"{prefix}.{key}_relation"] = _relation(player.get(key), expert.get(key))
        facts[f"{prefix}.player_{key}_iloc"] = player.get(key)
        facts[f"{prefix}.expert_{key}_iloc"] = expert.get(key)
    p_peak, e_peak = player.get("peak"), expert.get("peak")
    if p_peak is not None and e_peak is not None:
        facts[f"{prefix}.peak_difference"] = float(p_peak) - float(e_peak)
        facts[f"{prefix}.peak_ratio"] = (
            float(p_peak) / float(e_peak) if abs(float(e_peak)) > 1e-9 else None
        )
        facts[f"{prefix}.peak_relation"] = (
            "aligned" if abs(float(p_peak) - float(e_peak)) <= 0.10
            else "higher" if float(p_peak) > float(e_peak) else "lower"
        )
    p_hold, e_hold = player.get("hold_length"), expert.get("hold_length")
    if p_hold is not None and e_hold is not None:
        facts[f"{prefix}.hold_length_difference"] = int(p_hold) - int(e_hold)
        facts[f"{prefix}.hold_length_relation"] = (
            "aligned" if abs(int(p_hold) - int(e_hold)) <= _ALIGN_TOLERANCE
            else "longer" if int(p_hold) > int(e_hold) else "shorter"
        )


def _slope_facts(
    df: pd.DataFrame, start: int, end: int,
    evidence: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> Dict[str, Any]:
    from app.shared.annotation_agent_tools import run_pipeline_query

    payload, error = run_pipeline_query(
        df, "compute_slope",
        {"range": [start, end], "column": "expert_time_difference"},
    )
    if error or not isinstance(payload.get("extra"), dict):
        return {}
    extra = payload["extra"]
    samples = payload.get("samples") or []
    values = [s.get("value") for s in samples if isinstance(s, dict)]
    delta = extra.get("delta_value")
    runs = [
        run for run in extra.get("point_trend_runs") or []
        if isinstance(run, dict)
        and run.get("start_iloc") is not None
        and run.get("end_iloc") is not None
    ]
    ranges_by_direction = {
        direction: [
            [int(run["start_iloc"]), int(run["end_iloc"])]
            for run in runs if run.get("direction") == direction
        ]
        for direction in ("rising", "falling", "flat")
    }
    local_significant_rises = [
        run for run in runs
        if run.get("direction") == "rising"
        and run.get("is_label_significant") is True
    ]
    significant_rise_ranges = [
        [int(run["start_iloc"]), int(run["end_iloc"])]
        for run in local_significant_rises
    ]
    total_change_direction = extra.get("total_change_direction")
    has_significant_rise = bool(
        total_change_direction == "rising"
        and extra.get("total_change_is_label_significant") is True
    )
    if evidence is not None:
        for direction, ranges in ranges_by_direction.items():
            if ranges:
                evidence[f"time_gap.{direction}_ranges"] = [
                    tuple(value) for value in ranges
                ]
        if significant_rise_ranges:
            localized_rises = [tuple(value) for value in significant_rise_ranges]
            evidence["time_gap.significant_rise_ranges"] = localized_rises
        if total_change_direction is not None:
            evidence["time_gap.direction"] = [(int(start), int(end))]
        if has_significant_rise:
            evidence["time_gap.has_significant_rise"] = [(int(start), int(end))]
        evidence["time_gap.total_change_abs_ms"] = [(int(start), int(end))]
        evidence["time_gap.overall_gap"] = [(int(start), int(end))]
    spike_runs = [
        run for run in runs[:-1]
        if run.get("direction") == "rising"
        and run.get("is_label_significant") is True
    ]
    has_spike = bool(spike_runs)
    section_length = max(int(end) - int(start), 1)
    middle_start = int(start) + section_length / 3.0
    middle_end = int(end) - section_length / 3.0
    middle_significant_rises = [
        run for run in local_significant_rises
        if middle_start <= float(run["start_iloc"]) < middle_end
    ]
    middle_has_new_significant_rise = bool(middle_significant_rises)
    if evidence is not None:
        if middle_significant_rises:
            evidence["time_gap.middle_has_new_significant_rise"] = [
                (int(run["start_iloc"]), int(run["end_iloc"]))
                for run in middle_significant_rises
            ]
        if spike_runs:
            evidence["time_gap.has_spike"] = [
                (int(run["start_iloc"]), int(run["end_iloc"]))
                for run in spike_runs
            ]
    start_direction = (
        runs[0].get("direction") if runs and isinstance(runs[0], dict) else None
    )
    end_direction = (
        runs[-1].get("direction") if runs and isinstance(runs[-1], dict) else None
    )
    flattening_at_end = False
    previous_end_slope = extra.get("previous_end_slope")
    end_slope = extra.get("end_slope")
    if previous_end_slope is not None and end_slope is not None:
        try:
            previous_end_slope = float(previous_end_slope)
            end_slope = float(end_slope)
            flattening_at_end = (
                previous_end_slope > 0 and end_slope < previous_end_slope
            )
        except (TypeError, ValueError):
            pass
    return {
        "time_gap.total_change_ms": delta,
        "time_gap.total_change_abs_ms": abs(float(delta)) if delta is not None else None,
        "time_gap.direction": total_change_direction,
        "time_gap.overall_gap": abs(float(delta)) if delta is not None else None,
        "time_gap.slope_shape": extra.get("slope_shape"),
        "time_gap.starting_direction": start_direction,
        "time_gap.ending_direction": end_direction,
        "time_gap.flattening_at_end": flattening_at_end,
        "time_gap.has_significant_rise": has_significant_rise,
        "time_gap.rising_ranges": ranges_by_direction["rising"],
        "time_gap.falling_ranges": ranges_by_direction["falling"],
        "time_gap.flat_ranges": ranges_by_direction["flat"],
        "time_gap.significant_rise_ranges": significant_rise_ranges,
        "time_gap.middle_has_new_significant_rise": middle_has_new_significant_rise,
        "time_gap.has_spike": has_spike,
        "time_gap.start_ms": values[0] if values else None,
        "time_gap.end_ms": values[-1] if values else None,
    }


def _shape_facts(
    df: pd.DataFrame, start: int, end: int,
    evidence: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    phases: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> Tuple[Dict[str, Any], List[Tuple[int, int]]]:
    from app.shared.annotation_agent_tools import measure_segment_shape

    try:
        content = _attachment_content(measure_segment_shape(df, start, end))
    except Exception:
        return {}, []
    facts: Dict[str, Any] = {}
    base = content.get("base_segment_shape") or {}
    refinement = content.get("corner_shape_refinement") or {}
    facts["segment.shape_key"] = base.get("shape_key")
    facts["segment.corner_shape_key"] = refinement.get("shape_key")
    if evidence is not None:
        evidence["segment.shape_key"] = [(int(start), int(end))]
        evidence["segment.corner_shape_key"] = [(int(start), int(end))]
    phase_ranges: List[Tuple[int, int]] = []
    phase_ranges_by_name: Dict[str, List[Tuple[int, int]]] = {}
    for phase in content.get("phases") or []:
        if not isinstance(phase, dict):
            continue
        entry, apex, exit_ = phase.get("entry"), phase.get("apex"), phase.get("exit")
        if all(isinstance(v, int) for v in (entry, apex, exit_)):
            named_ranges = {
                "entry": (entry, apex),
                "apex": (max(entry, apex - 2), min(exit_, apex + 2)),
                "exit": (apex, exit_),
            }
            phase_ranges.extend(named_ranges.values())
            for name, value in named_ranges.items():
                phase_ranges_by_name.setdefault(name, []).append(value)
                if phases is not None:
                    phases.setdefault(name, []).append(value)
    altitude = content.get("altitude") or {}
    for phase_name in ("entry", "apex", "exit"):
        phase_evidence = list(phase_ranges_by_name.get(phase_name) or [])
        if phase_evidence:
            facts[f"phase.{phase_name}"] = True
            if evidence is not None:
                evidence[f"phase.{phase_name}"] = phase_evidence
        summary = altitude.get(phase_name) or {}
        facts[f"altitude.{phase_name}.trend"] = summary.get("trend")
        facts[f"altitude.{phase_name}.slope_angle_degrees"] = summary.get("slope_angle_degrees")
        if evidence is not None and phase_evidence:
            evidence[f"altitude.{phase_name}.trend"] = phase_evidence
    return facts, phase_ranges


def _opponent_facts(
    df: pd.DataFrame, start: int, end: int,
    evidence: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> Dict[str, Any]:
    from app.shared.annotation_agent_tools import (
        classify_opponent_interaction,
        query_opponent_trajectory,
    )

    try:
        content = _attachment_content(classify_opponent_interaction(df, start, end))
    except Exception:
        return {}
    facts = {
        "opponent.data_available": content.get("data_available"),
        "opponent.outcome": content.get("outcome"),
        "opponent.role": content.get("role"),
        "opponent.confidence_level": content.get("confidence_level"),
        "opponent.started_ahead": None,
        "opponent.driver_ended_ahead": None,
        "opponent.gap_shrank": None,
        "opponent.drew_alongside": None,
        "opponent.side_swap": None,
    }
    if evidence is not None and content.get("outcome") not in (None, "no_data"):
        evidence["opponent.outcome"] = [(int(start), int(end))]
    candidates = content.get("candidates") or []
    primary = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    entry = primary.get("entry_signed_long_gap_m")
    exit_ = primary.get("exit_signed_long_gap_m")
    if entry is not None:
        facts["opponent.started_ahead"] = float(entry) > 0
    if exit_ is not None:
        facts["opponent.driver_ended_ahead"] = float(exit_) < 0
    if entry is not None and exit_ is not None:
        facts["opponent.gap_shrank"] = abs(float(exit_)) < abs(float(entry))
    facts["opponent.drew_alongside"] = int(primary.get("side_by_side_iloc_count") or 0) > 0
    slot = content.get("targeted_car_slot")
    if slot is not None:
        try:
            trajectory = _attachment_content(
                query_opponent_trajectory(df, start, end, int(slot), n_samples=7)
            )
        except Exception:
            trajectory = {}
        lateral = [
            sample.get("lateral_offset_m")
            for sample in trajectory.get("samples") or []
            if isinstance(sample, dict) and sample.get("lateral_offset_m") is not None
        ]
        samples = [
            sample for sample in trajectory.get("samples") or []
            if isinstance(sample, dict) and sample.get("iloc") is not None
        ]
        if evidence is not None and samples:
            sample_ilocs = [int(sample["iloc"]) for sample in samples]
            evidence["opponent.gap_shrank"] = _point_range(
                sample_ilocs[0], sample_ilocs[-1],
            )
            alongside = [
                int(sample["iloc"])
                for sample in samples
                if sample.get("signed_long_gap_m") is not None
                and sample.get("lateral_offset_m") is not None
                and abs(float(sample["signed_long_gap_m"])) <= 6.0
                and abs(float(sample["lateral_offset_m"])) >= 1.25
            ]
            if alongside:
                evidence["opponent.drew_alongside"] = _point_range(*alongside)
            if lateral and min(lateral) < 0 < max(lateral):
                swap_ilocs = [
                    int(sample["iloc"])
                    for sample in samples
                    if sample.get("lateral_offset_m") is not None
                ]
                evidence["opponent.side_swap"] = _point_range(*swap_ilocs)
        if lateral:
            facts["opponent.side_swap"] = min(lateral) < 0 < max(lateral)
    return facts


def calculate_facts(
    df: pd.DataFrame, start: int, end: int, *, section_id: str = "",
) -> Tuple[FactSet, List[Tuple[int, int]]]:
    """Calculate normalized facts and reusable phase windows for one range."""
    segment = df.loc[(df.index >= int(start)) & (df.index <= int(end))]
    evidence: Dict[str, List[Tuple[int, int]]] = {}
    phases: Dict[str, List[Tuple[int, int]]] = {}
    facts: Dict[str, Any] = {
        "section.id": section_id,
        "section.name": LABEL_MAPPING.get(section_id),
        "section.overlap_names": [LABEL_MAPPING[section_id]] if section_id in LABEL_MAPPING else [],
    }
    facts.update(_slope_facts(df, start, end, evidence))
    shape, phase_ranges = _shape_facts(df, start, end, evidence, phases)
    facts.update(shape)
    facts.update(_opponent_facts(df, start, end, evidence))

    index = segment.index.to_numpy(dtype=int)
    brake = _input_landmarks(_series(segment, "Physics_brake"), index)
    expert_brake = _input_landmarks(_series(segment, "expert_optimal_brake"), index)
    throttle = _input_landmarks(_series(segment, "Physics_gas"), index)
    expert_throttle = _input_landmarks(_series(segment, "expert_optimal_throttle"), index)
    _add_input_facts(facts, "brake", brake, expert_brake)
    _add_input_facts(facts, "throttle", throttle, expert_throttle)
    for prefix, player, expert in (
        ("brake", brake, expert_brake), ("throttle", throttle, expert_throttle),
    ):
        for key in ("application_onset", "application_end", "release_onset", "release_end"):
            fact = f"{prefix}.{key}_relation"
            if fact in facts:
                evidence[fact] = _point_range(player.get(key), expert.get(key))
        for fact in (f"{prefix}.peak_relation", f"{prefix}.peak_ratio"):
            if fact in facts:
                evidence[fact] = _point_range(
                    player.get("peak_iloc"), expert.get("peak_iloc"),
                )
        hold_points = (
            player.get("application_end"), player.get("release_onset"),
            expert.get("application_end"), expert.get("release_onset"),
        )
        if f"{prefix}.hold_length_relation" in facts:
            evidence[f"{prefix}.hold_length_relation"] = _point_range(*hold_points)

    player_brake = _series(segment, "Physics_brake")
    player_throttle = _series(segment, "Physics_gas")
    expert_b = _series(segment, "expert_optimal_brake")
    expert_t = _series(segment, "expert_optimal_throttle")
    if player_brake is not None and expert_b is not None:
        facts["brake.similarity"] = float(np.mean(np.isclose(player_brake, expert_b, atol=0.02, equal_nan=False)))
        evidence["brake.similarity"] = [(int(start), int(end))]
    if player_throttle is not None and expert_t is not None:
        facts["throttle.similarity"] = float(np.mean(np.isclose(player_throttle, expert_t, atol=0.02, equal_nan=False)))
        evidence["throttle.similarity"] = [(int(start), int(end))]
    if player_brake is not None and player_throttle is not None:
        overlap = (player_brake > 0.05) & (player_throttle > 0.05)
        facts["controls.overlap_count"] = int(np.sum(overlap))
        facts["controls.overlap_fraction"] = float(np.mean(overlap))
        if expert_b is not None and expert_t is not None:
            expert_overlap = (expert_b > 0.05) & (expert_t > 0.05)
            facts["controls.expert_overlap_count"] = int(np.sum(expert_overlap))
            evidence["controls.expert_overlap_count"] = (
                _mask_ranges(expert_overlap, index) or [(int(start), int(end))]
            )
        evidence["controls.overlap_count"] = _mask_ranges(overlap, index)

    speed_delta = _raw_speed_delta(segment)
    if speed_delta is not None:
        finite = speed_delta[np.isfinite(speed_delta)]
        if len(finite):
            facts["speed.gap_peak_abs_kmh"] = float(np.max(np.abs(finite)))
            facts["speed.expert_faster"] = float(np.nanmedian(finite)) > 0
            facts["speed.gap_closing"] = abs(float(finite[-1])) < abs(float(finite[0]))
            finite_mask = np.isfinite(speed_delta)
            peak_pos = int(np.nanargmax(np.abs(speed_delta)))
            evidence["speed.gap_peak_abs_kmh"] = _point_range(index[peak_pos])
            evidence["speed.expert_faster"] = _mask_ranges(
                finite_mask & (speed_delta > 0), index,
            )
            evidence["speed.gap_closing"] = _decreasing_magnitude_ranges(
                speed_delta, index,
            )

    trajectory = _raw_trajectory_offset(segment)
    if trajectory is not None:
        finite = trajectory[np.isfinite(trajectory)]
        if len(finite):
            facts["trajectory.start_offset_m"] = float(finite[0])
            facts["trajectory.end_offset_m"] = float(finite[-1])
            facts["trajectory.peak_abs_offset_m"] = float(np.max(np.abs(finite)))
            facts["trajectory.converging"] = abs(float(finite[-1])) < abs(float(finite[0]))
            median = float(np.nanmedian(finite))
            facts["trajectory.position"] = "aligned" if abs(median) <= 0.5 else "wider" if median > 0 else "tighter"
            finite_mask = np.isfinite(trajectory)
            peak_pos = int(np.nanargmax(np.abs(trajectory)))
            evidence["trajectory.peak_abs_offset_m"] = _point_range(index[peak_pos])
            evidence["trajectory.converging"] = _decreasing_magnitude_ranges(
                trajectory, index,
            )
            position_mask = (
                np.abs(trajectory) <= 0.5 if facts["trajectory.position"] == "aligned"
                else trajectory > 0.5 if facts["trajectory.position"] == "wider"
                else trajectory < -0.5
            )
            evidence["trajectory.position"] = _mask_ranges(
                finite_mask & position_mask, index,
            )

    player_steer = _series(segment, "Physics_steer_angle")
    expert_steer = _series(segment, "expert_optimal_steering")
    if player_steer is not None and expert_steer is not None:
        def _steer_marks(values: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int]]:
            absolute = np.abs(values)
            peak_pos = int(np.nanargmax(absolute))
            threshold = max(0.02, float(absolute[peak_pos]) * 0.10)
            onset_positions = np.flatnonzero(absolute >= threshold)
            exit_positions = np.flatnonzero((np.arange(len(values)) > peak_pos) & (absolute < threshold))
            return (
                int(index[int(onset_positions[0])]) if len(onset_positions) else None,
                int(index[peak_pos]),
                int(index[int(exit_positions[0])]) if len(exit_positions) else None,
            )
        p_turn, p_apex, p_exit = _steer_marks(player_steer)
        e_turn, e_apex, e_exit = _steer_marks(expert_steer)
        facts["turn.in_relation"] = _relation(p_turn, e_turn)
        facts["turn.apex_relation"] = _relation(p_apex, e_apex)
        facts["turn.exit_relation"] = _relation(p_exit, e_exit)
        evidence["turn.in_relation"] = _point_range(p_turn, e_turn)
        evidence["turn.apex_relation"] = _point_range(p_apex, e_apex)
        evidence["turn.exit_relation"] = _point_range(p_exit, e_exit)

    balance = _raw_slip_balance(segment)
    if balance is not None:
        facts["balance.oversteer"] = bool(np.nanmax(balance) > 0.02)
        facts["balance.understeer"] = bool(np.nanmin(balance) < -0.02)
        evidence["balance.oversteer"] = _mask_ranges(balance > 0.02, index)
        evidence["balance.understeer"] = _mask_ranges(balance < -0.02, index)
    push = _raw_push_to_limit(segment)
    if push is not None:
        facts["grip.max"] = float(np.nanmax(push))
        facts["grip.min"] = float(np.nanmin(push))
        facts["grip.over_limit"] = bool(np.nanmax(push) > 1.0)
        facts["grip.sustained_low"] = bool(np.mean(push < 0.8) >= 0.5)
        evidence["grip.over_limit"] = _mask_ranges(push > 1.0, index)
        evidence["grip.sustained_low"] = _mask_ranges(push < 0.8, index)

    player_gear = _series(segment, "Physics_gear")
    expert_gear = _series(segment, "expert_optimal_gear")
    if player_gear is not None and expert_gear is not None:
        facts["gear.exit_relation"] = (
            "lower" if player_gear[-1] < expert_gear[-1]
            else "higher" if player_gear[-1] > expert_gear[-1] else "aligned"
        )
        evidence["gear.exit_relation"] = _point_range(index[-1])
        p_changes = np.flatnonzero(np.diff(player_gear) != 0)
        e_changes = np.flatnonzero(np.diff(expert_gear) != 0)
        if len(p_changes) and len(e_changes):
            p_i, e_i = int(index[p_changes[0] + 1]), int(index[e_changes[0] + 1])
            direction = "up" if player_gear[p_changes[0] + 1] > player_gear[p_changes[0]] else "down"
            facts[f"gear.{direction}shift_relation"] = _relation(p_i, e_i)
            evidence[f"gear.{direction}shift_relation"] = _point_range(p_i, e_i)
    filtered = {key: value for key, value in facts.items() if value is not None}
    return FactSet(filtered, evidence=evidence, phases=phases), phase_ranges


def _compare(actual: Any, operator: str, expected: Any = None) -> bool:
    if operator not in SUPPORTED_OPERATORS:
        return False
    if operator == "exists":
        return actual is not _MISSING and (bool(actual is not None) == bool(expected))
    if actual is _MISSING or actual is None:
        return False
    try:
        if operator == "eq": return actual == expected
        if operator == "neq": return actual != expected
        if operator == "in": return actual in expected
        if operator == "not_in": return actual not in expected
        if operator == "lt": return actual < expected
        if operator == "lte": return actual <= expected
        if operator == "gt": return actual > expected
        if operator == "gte": return actual >= expected
        if operator == "between": return expected[0] <= actual <= expected[1]
        if operator == "contains": return expected in actual
    except (TypeError, ValueError, IndexError):
        return False
    return False


def evaluate_requirements(requirements: Mapping[str, Any], facts: Mapping[str, Any]) -> RequirementEvaluation:
    if requirements.get("enabled") is False:
        return RequirementEvaluation(False, failed=["label disabled"])
    branches = requirements.get("any_of")
    if not isinstance(branches, list) or not branches:
        return RequirementEvaluation(False, failed=["no valid requirement branches"])
    closest_branch: Optional[RequirementEvaluation] = None
    for branch_index, branch in enumerate(branches):
        predicates = branch.get("all_of") if isinstance(branch, dict) else None
        if not isinstance(predicates, list) or not predicates:
            continue
        passed: List[str] = []
        failed: List[str] = []
        fact_ids: List[str] = []
        for predicate in predicates:
            if not isinstance(predicate, dict):
                failed.append("invalid predicate")
                continue
            fact = str(predicate.get("fact") or "")
            fact_ids.append(fact)
            operator = str(predicate.get("operator") or "")
            expected = predicate.get("value")
            actual = facts.get(fact, _MISSING)
            value = "unavailable" if actual is _MISSING else repr(actual)
            text = f"{fact}: {value}"
            (passed if _compare(actual, operator, expected) else failed).append(text)
        if not failed:
            return RequirementEvaluation(True, branch_index, passed, [], fact_ids)
        candidate = RequirementEvaluation(
            False, branch_index, passed, failed, fact_ids,
        )
        if closest_branch is None or (len(failed), -len(passed)) < (
            len(closest_branch.failed), -len(closest_branch.passed)
        ):
            closest_branch = candidate
    return closest_branch or RequirementEvaluation(False, failed=["facts unavailable"])


def validate_catalog() -> List[str]:
    """Return structural errors in deterministic label requirements."""
    errors: List[str] = []
    main_labels = {doc["id"]: doc for doc in skills.iter("lap_annotation.labels")}
    non_main_labels = {
        doc["id"]: doc for doc in skills.iter("sub_label_annotation.labels")
    }
    duplicate_ids = set(main_labels) & set(non_main_labels)
    if duplicate_ids:
        errors.append(f"label IDs exist in both catalogs: {sorted(duplicate_ids)}")
    labels = {**main_labels, **non_main_labels}
    lap_requirements = skills.get("lap_annotation.selection_requirements", {})
    sub_requirements = skills.get("sub_label_annotation.selection_requirements", {})
    range_policies = skills.get("sub_label_annotation.range_policies", {})
    if any(doc.get("type") != "main" for doc in main_labels.values()):
        errors.append("lap label catalog contains non-main labels")
    if any(doc.get("type") == "main" for doc in non_main_labels.values()):
        errors.append("sub-label catalog contains main labels")
    if (
        not isinstance(lap_requirements, dict)
        or set(lap_requirements) != set(main_labels)
    ):
        errors.append("lap requirement IDs do not exactly match main label IDs")
    if (
        not isinstance(sub_requirements, dict)
        or set(sub_requirements) != set(non_main_labels)
    ):
        errors.append("sub-label requirement IDs do not exactly match non-main label IDs")
    active_sub_ids = {
        label_id for label_id, doc in non_main_labels.items()
        if doc.get("type") == "sub"
        and isinstance(sub_requirements, dict)
        and (sub_requirements.get(label_id) or {}).get("enabled") is not False
    }
    if not isinstance(range_policies, dict) or set(range_policies) != active_sub_ids:
        errors.append("range policy IDs do not exactly match active sub-label IDs")
    for label_id, policy in (range_policies.items() if isinstance(range_policies, dict) else []):
        if not isinstance(policy, dict):
            errors.append(f"{label_id}: invalid range policy")
            continue
        invalid_phases = set(policy.get("phases") or []) - _VALID_RANGE_PHASES
        if invalid_phases:
            errors.append(f"{label_id}: invalid range phases {sorted(invalid_phases)}")
        for predicate in policy.get("supporting_evidence") or []:
            fact = predicate.get("fact") if isinstance(predicate, dict) else None
            operator = predicate.get("operator") if isinstance(predicate, dict) else None
            if fact not in KNOWN_FACTS:
                errors.append(f"{label_id}: unknown supporting fact {fact!r}")
            if operator not in SUPPORTED_OPERATORS:
                errors.append(f"{label_id}: unknown supporting operator {operator!r}")
    for label_id, doc in labels.items():
        requirements = _requirements_for(label_id, get_label(label_id) or doc)
        enabled = requirements.get("enabled") is not False
        branches = requirements.get("any_of")
        if enabled and (not isinstance(branches, list) or not branches):
            errors.append(f"{label_id}: active label has no any_of branches")
            continue
        for branch_index, branch in enumerate(branches or []):
            predicates = branch.get("all_of") if isinstance(branch, dict) else None
            if enabled and (not isinstance(predicates, list) or not predicates):
                errors.append(f"{label_id}: branch {branch_index} has no all_of predicates")
                continue
            for predicate in predicates or []:
                fact = predicate.get("fact") if isinstance(predicate, dict) else None
                operator = predicate.get("operator") if isinstance(predicate, dict) else None
                if fact not in KNOWN_FACTS:
                    errors.append(f"{label_id}: unknown fact {fact!r}")
                if operator not in SUPPORTED_OPERATORS:
                    errors.append(f"{label_id}: unknown operator {operator!r}")
        parent = doc.get("parent")
        if parent is not None and parent not in labels:
            errors.append(f"{label_id}: unknown parent {parent!r}")
        for other in doc.get("exclusive_with") or []:
            if other not in labels:
                errors.append(f"{label_id}: unknown exclusive label {other!r}")
    return errors


def _requirements_for(label_id: str, doc: Mapping[str, Any]) -> Dict[str, Any]:
    requirements = doc.get("selection_requirements")
    if isinstance(requirements, dict):
        return dict(requirements)
    ref = doc.get("selection_requirements_ref")
    if isinstance(ref, str):
        value = skills.get(ref, {})
        if isinstance(value, dict):
            return dict(value)
    return {}


def evaluate_labels(label_ids: Iterable[str], facts: Mapping[str, Any]) -> LabelEvaluation:
    evaluations: Dict[str, RequirementEvaluation] = {}
    matched: List[str] = []
    docs: Dict[str, Dict[str, Any]] = {}
    for label_id in label_ids:
        doc = get_label(label_id)
        if not doc:
            continue
        docs[label_id] = doc
        evaluation = evaluate_requirements(_requirements_for(label_id, doc), facts)
        evaluations[label_id] = evaluation
        if evaluation.matched:
            matched.append(label_id)
    conflicts: List[Tuple[str, str]] = []
    suppressed: set[str] = set()
    matched_set = set(matched)
    for label_id in matched:
        for other in docs[label_id].get("exclusive_with") or []:
            if other in matched_set:
                pair = tuple(sorted((label_id, other)))
                if pair not in conflicts:
                    conflicts.append(pair)
                suppressed.update(pair)
    return LabelEvaluation([label for label in matched if label not in suppressed], evaluations, conflicts)


def _reason(label_id: str, evaluation: RequirementEvaluation, start: int, end: int) -> str:
    details = [
        f"{label_id} selected for iloc range [{int(start)}, {int(end)}]",
        *(f"Passed — {fact}" for fact in evaluation.passed),
    ]
    return "; ".join(details)


def _resolve_circuit_sections(
    df: pd.DataFrame, circuit_id: str, section_id: str, start: int, end: int,
    opponent_interaction: Optional[dict],
) -> Tuple[str, List[str]]:
    primary_id = (
        section_id
        if section_id in LABEL_MAPPING and section_id != "interaction_window"
        else None
    )
    context_ids = []
    if isinstance(opponent_interaction, dict):
        for context in opponent_interaction.get("section_context") or []:
            candidate = context.get("circuit_section_id") if isinstance(context, dict) else None
            if candidate in LABEL_MAPPING and candidate not in context_ids:
                context_ids.append(candidate)
    try:
        from app.shared.annotation_agent_tools import locate_circuit_section
        content = _attachment_content(locate_circuit_section(df, circuit_id, start, end))
    except Exception:
        content = {}
    best = content.get("best_match") or {}
    candidate = best.get("label_id") if isinstance(best, dict) else None
    matches = content.get("top_matches") or []
    overlap_ids = [
        str(match["label_id"])
        for match in matches
        if isinstance(match, dict) and match.get("label_id") in LABEL_MAPPING
    ]
    candidate_ids = [*([primary_id] if primary_id else []), *context_ids, *overlap_ids]
    candidate_ids = [
        value for index, value in enumerate(candidate_ids)
        if value not in candidate_ids[:index]
    ]
    if primary_id:
        return primary_id, candidate_ids
    if len(context_ids) == 1:
        return context_ids[0], candidate_ids
    if candidate in LABEL_MAPPING:
        return str(candidate), candidate_ids or [str(candidate)]
    if candidate_ids:
        return candidate_ids[0], candidate_ids
    return section_id, []


def _resolve_circuit_section(
    df: pd.DataFrame, circuit_id: str, section_id: str, start: int, end: int,
    opponent_interaction: Optional[dict],
) -> str:
    resolved, _ = _resolve_circuit_sections(
        df, circuit_id, section_id, start, end, opponent_interaction,
    )
    return resolved


def _is_far_from_expert_in_pit(facts: Mapping[str, Any]) -> bool:
    overlap_names = facts.get("section.overlap_names") or []
    offset = facts.get("trajectory.peak_abs_offset_m")
    return "Pit" in overlap_names and isinstance(offset, (int, float)) and offset >= 10.0


def calculate_lap_annotation(
    df: pd.DataFrame, *, lap_start: int, lap_end: int, section_id: str,
    section_start: int, section_end: int, circuit_id: str,
    section_split_basis: Optional[str] = None,
    opponent_interaction: Optional[dict] = None,
) -> LapAnnotationResult:
    session = "racing" if opponent_interaction or "interaction" in str(section_split_basis or "") else "practice"
    eligible = skills.get(f"lap_annotation.behavior_parent_label_ids.eligible_by_session.{session}", [])
    eligible = [label for label in eligible if label in skills.get("lap_annotation.labels", {})]
    resolved_section_id, overlap_section_ids = _resolve_circuit_sections(
        df, circuit_id, section_id, section_start, section_end, opponent_interaction,
    )
    facts, _ = calculate_facts(
        df, section_start, section_end, section_id=resolved_section_id,
    )
    facts["section.overlap_names"] = [
        LABEL_MAPPING[candidate]
        for candidate in overlap_section_ids
        if candidate in LABEL_MAPPING
    ]
    if _is_far_from_expert_in_pit(facts):
        eligible = [label for label in eligible if label != "RM"]
    evaluated = evaluate_labels(eligible, facts)
    segment_types = evaluate_labels([f"ST{i}" for i in range(1, 12)], facts)
    behavior = evaluated.labels
    if "PS" in behavior:
        resolved_section_id = next(
            (
                candidate for candidate in overlap_section_ids
                if LABEL_MAPPING.get(candidate) == "Pit"
            ),
            resolved_section_id,
        )
    child_ids = [
        doc["id"] for doc in skills.iter("sub_label_annotation.labels")
        if doc.get("type") == "sub" and doc.get("parent") in set(behavior)
    ]
    children = evaluate_labels(child_ids, facts)
    resolved_children: List[Tuple[str, RequirementEvaluation, LabelEvidence]] = []
    for label in children.labels:
        evidence = _label_evidence(
            label, children.evaluations[label], facts, section_start, section_end,
        )
        if evidence is not None and evidence.required_covers(section_start, section_end):
            resolved_children.append((label, children.evaluations[label], evidence))
    label_ids = [
        circuit_id, resolved_section_id, *behavior,
        *(label for label, _, _ in resolved_children),
        *segment_types.labels,
    ] if behavior else []
    notes = [
        _reason(label, evaluated.evaluations[label], section_start, section_end)
        for label in behavior
    ]
    notes.extend(
        "; ".join([
            _reason(label, evaluation, *evidence.annotation_range),
            *evidence.supporting_reasons,
        ])
        for label, evaluation, evidence in resolved_children
    )
    rejected = [
        {
            "value": label_id,
            "reason": "; ".join([
                *(f"Passed — {fact}" for fact in evaluation.passed),
                *(f"Failed — {fact}" for fact in evaluation.failed),
            ]),
        }
        for label_id, evaluation in evaluated.evaluations.items()
        if not evaluation.matched or any(label_id in pair for pair in evaluated.conflicts)
    ]
    rejected.extend(
        {
            "value": " / ".join(pair),
            "label_ids": list(pair),
            "reason": "exclusive deterministic matches",
        }
        for pair in [*children.conflicts, *segment_types.conflicts]
    )
    return LapAnnotationResult(
        section_id=resolved_section_id,
        start_index=int(section_start),
        end_index=int(section_end),
        label_ids=[label for i, label in enumerate(label_ids) if label and label not in label_ids[:i]],
        reasoning="\n".join(notes) or "No behavior label satisfied a complete requirement branch.",
        submitted=True,
        rejected_proposals=rejected,
        transcript="deterministic label evaluation",
        tool_calls=0,
    )


def _candidate_ranges(
    df: pd.DataFrame, start: int, end: int, phase_ranges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = list(phase_ranges)
    segment = df.loc[(df.index >= start) & (df.index <= end)]
    for names in (
        ("Physics_brake",), ("Physics_gas",),
        ("Physics_steer_angle",), ("Physics_gear",),
    ):
        values = _series(segment, *names)
        if values is None or len(values) < 3:
            continue
        finite = np.where(np.isfinite(values), values, 0.0)
        changes = np.abs(np.diff(finite))
        if not np.any(changes > 0):
            continue
        threshold = max(float(np.nanpercentile(changes, 75)), 1e-6)
        last_pos = -10
        for pos in np.flatnonzero(changes >= threshold):
            if int(pos) - last_pos < 4:
                continue
            last_pos = int(pos)
            lo = max(start, int(segment.index[max(0, int(pos) - 2)]))
            hi = min(end, int(segment.index[min(len(segment) - 1, int(pos) + 3)]))
            ranges.append((lo, hi))
    cleaned: List[Tuple[int, int]] = []
    for lo, hi in ranges:
        lo, hi = max(start, int(lo)), min(end, int(hi))
        if lo >= hi or (lo == start and hi == end):
            continue
        if (lo, hi) not in cleaned:
            cleaned.append((lo, hi))
    return cleaned


_RANGE_CONTEXT_FACTS = frozenset({
    "opponent.confidence_level", "opponent.outcome",
})
_RANGE_CONTEXT_PREFIXES = ("phase.", "section.", "segment.")
_VALID_RANGE_PHASES = frozenset({"entry", "apex", "exit"})


def _range_policy(label_id: str) -> Mapping[str, Any]:
    value = skills.get(f"sub_label_annotation.range_policies.{label_id}", {})
    return value if isinstance(value, dict) else {}


def _fact_ranges(facts: Mapping[str, Any], fact_ids: Iterable[str]) -> List[Tuple[int, int]]:
    if not isinstance(facts, FactSet):
        return []
    ranges: List[Tuple[int, int]] = []
    for fact_id in fact_ids:
        if fact_id in _RANGE_CONTEXT_FACTS or fact_id.startswith(_RANGE_CONTEXT_PREFIXES):
            continue
        ranges.extend(facts.evidence.get(fact_id) or [])
    return ranges


def _supporting_facts(
    policy: Mapping[str, Any], facts: Mapping[str, Any],
) -> Tuple[List[str], List[str]]:
    matched_ids: List[str] = []
    reasons: List[str] = []
    for predicate in policy.get("supporting_evidence") or []:
        if not isinstance(predicate, dict):
            continue
        fact = str(predicate.get("fact") or "")
        actual = facts.get(fact, _MISSING)
        localized = isinstance(facts, FactSet) and bool(facts.evidence.get(fact))
        if localized and _compare(
            actual, str(predicate.get("operator") or ""), predicate.get("value"),
        ):
            matched_ids.append(fact)
            reasons.append(f"Supporting — {fact}: {actual!r}")
    return matched_ids, reasons


def _label_evidence(
    label_id: str, evaluation: RequirementEvaluation, facts: Mapping[str, Any],
    parent_start: int, parent_end: int,
    supporting_facts: Optional[Mapping[str, Any]] = None,
) -> Optional[LabelEvidence]:
    policy = _range_policy(label_id)
    required_ids = [
        fact_id for fact_id in evaluation.fact_ids
        if fact_id not in _RANGE_CONTEXT_FACTS
        and not fact_id.startswith(_RANGE_CONTEXT_PREFIXES)
    ]
    if not isinstance(facts, FactSet) or not required_ids or any(
        not facts.evidence.get(fact_id) for fact_id in required_ids
    ):
        return None
    phase_names = policy.get("phases") or []
    allowed_phase_ranges = (
        [
            value for phase_name in phase_names
            for value in facts.phases.get(str(phase_name), [])
        ]
        if phase_names else None
    )
    support_source = supporting_facts or facts
    support_ids, support_reasons = _supporting_facts(policy, support_source)
    return resolve_label_evidence(
        required_ranges=_fact_ranges(facts, required_ids),
        parent_range=(int(parent_start), int(parent_end)),
        allowed_phase_ranges=allowed_phase_ranges,
        supporting_ranges=_fact_ranges(support_source, support_ids),
        supporting_reasons=support_reasons,
    )


def calculate_detailed_annotation(
    df: pd.DataFrame, *, parent_start: int, parent_end: int,
    parent_main_labels: Sequence[str], existing_children: Sequence[dict] = (),
) -> AnnotationResult:
    parent_facts, phase_ranges = calculate_facts(df, parent_start, parent_end)
    candidates = _candidate_ranges(df, parent_start, parent_end, phase_ranges)
    existing = {
        (int(child.get("start_index", -1)), int(child.get("end_index", -1)), label)
        for child in existing_children
        for label in child.get("labels", [])
    }
    parent_children = [
        doc["id"] for doc in skills.iter("sub_label_annotation.labels")
        if doc.get("type") == "sub" and doc.get("parent") in set(parent_main_labels)
    ]
    segment_types = [f"ST{i}" for i in range(1, 21)]
    annotations: List[dict] = []
    conflicts: List[Tuple[str, str]] = []
    for start, end in candidates:
        facts, _ = calculate_facts(df, start, end)
        if isinstance(facts, FactSet) and isinstance(parent_facts, FactSet):
            for phase_name in ("entry", "apex", "exit"):
                facts.pop(f"phase.{phase_name}", None)
                facts.evidence.pop(f"phase.{phase_name}", None)
                facts.phases.pop(phase_name, None)
            for phase_name, ranges in parent_facts.phases.items():
                inherited = [
                    (max(start, phase_start), min(end, phase_end))
                    for phase_start, phase_end in ranges
                    if max(start, phase_start) <= min(end, phase_end)
                ]
                if inherited:
                    facts.phases[phase_name] = inherited
                    facts[f"phase.{phase_name}"] = True
                    facts.evidence[f"phase.{phase_name}"] = list(inherited)
        for key, value in parent_facts.items():
            if key.startswith("opponent."):
                if key in {
                    "opponent.outcome", "opponent.confidence_level",
                    "opponent.started_ahead", "opponent.driver_ended_ahead",
                }:
                    facts[key] = value
                    if isinstance(facts, FactSet) and isinstance(parent_facts, FactSet):
                        if parent_facts.evidence.get(key):
                            facts.evidence[key] = list(parent_facts.evidence[key])
                else:
                    facts.setdefault(key, value)
        evaluated = evaluate_labels([*parent_children, *segment_types], facts)
        conflicts.extend(evaluated.conflicts)
        for label_id in evaluated.labels:
            label_doc = get_label(label_id) or {}
            support_reasons: List[str] = []
            if label_doc.get("type") == "sub":
                evidence = _label_evidence(
                    label_id, evaluated.evaluations[label_id], facts,
                    parent_start, parent_end, parent_facts,
                )
                if evidence is None:
                    continue
                label_start, label_end = evidence.annotation_range
                support_reasons = list(evidence.supporting_reasons)
            else:
                label_start, label_end = start, end
            key = (label_start, label_end, label_id)
            if key in existing or any(
                (a["start_index"], a["end_index"], a["label_id"]) == key
                for a in annotations
            ):
                continue
            annotations.append({
                "label_id": label_id,
                "start_index": label_start,
                "end_index": label_end,
                "reasoning": "; ".join([
                    _reason(label_id, evaluated.evaluations[label_id], label_start, label_end),
                    *support_reasons,
                ]),
            })
    labels = list(dict.fromkeys(a["label_id"] for a in annotations))
    summary = f"Deterministically selected {len(annotations)} label proposal(s)."
    if conflicts:
        summary += f" Suppressed {len(set(conflicts))} exclusive conflict(s)."
    return AnnotationResult(
        final_labels=labels,
        final_reasoning=summary,
        accepted=True,
        iterations=1,
        messages=[],
        label_annotations=annotations,
    )


__all__ = [
    "KNOWN_FACTS", "SUPPORTED_OPERATORS", "calculate_facts", "calculate_lap_annotation",
    "calculate_detailed_annotation", "evaluate_labels", "evaluate_requirements",
    "validate_catalog",
]
