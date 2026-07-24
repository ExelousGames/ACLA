"""Telemetry input and fact strategies for deterministic annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.local_annotation_agent.workflow.deterministic_engine import (
    FactDefinition,
    FactRegistry,
    InclusiveRange,
    InputDefinition,
    InputRegistry,
    MISSING,
    ResolvedInput,
)
from app.shared.labels import LABEL_MAPPING


SMOOTHING_WINDOW = 3
SLOPE_ANGLE_DEGREES = 5.0


def smooth_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    telemetry = df.copy()
    edge = SMOOTHING_WINDOW // 2
    for name in telemetry.select_dtypes(include=[np.number]).columns:
        values = telemetry[name].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        padded = np.pad(values, edge, mode="edge")
        telemetry[name] = (
            pd.Series(padded)
            .rolling(SMOOTHING_WINDOW, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)[edge:-edge]
        )
    return telemetry


def _series(df: pd.DataFrame, *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(values)):
                return values
    return None


@dataclass
class EvaluationContext:
    telemetry: pd.DataFrame
    section_id: str = ""
    overlap_section_ids: Tuple[str, ...] = ()
    _input_cache: Dict[Tuple[int, int, str], Optional[ResolvedInput]] = field(default_factory=dict)
    _fact_cache: Dict[Tuple[Any, ...], Any] = field(default_factory=dict)
    _analysis_cache: Dict[Tuple[Any, ...], Any] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, *, section_id: str = "",
        overlap_section_ids: Sequence[str] = (),
    ) -> "EvaluationContext":
        return cls(
            smooth_telemetry(df), section_id,
            tuple(str(value) for value in overlap_section_ids),
        )

    def segment(self, range_: InclusiveRange) -> pd.DataFrame:
        return self.telemetry.loc[
            (self.telemetry.index >= range_.start)
            & (self.telemetry.index <= range_.end)
        ]

    def memo(self, key: Tuple[Any, ...], calculate: Callable[[], Any]) -> Any:
        if key not in self._analysis_cache:
            self._analysis_cache[key] = calculate()
        return self._analysis_cache[key]

    def resolve_input(
        self, tag: str, scope: InclusiveRange, registry: InputRegistry,
    ) -> Optional[ResolvedInput]:
        key = (scope.start, scope.end, tag)
        if key not in self._input_cache:
            definition = registry.get(tag)
            self._input_cache[key] = (
                definition.resolve(self, scope) if definition is not None else None
            )
        return self._input_cache[key]

    def calculate_fact(
        self, name: str, definition: FactDefinition,
        inputs: Sequence[ResolvedInput],
    ) -> Any:
        key = (
            name,
            *((value.kind, value.value) for value in inputs),
        )
        if key not in self._fact_cache:
            try:
                self._fact_cache[key] = definition.calculate(self, inputs)
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
                self._fact_cache[key] = MISSING
        return self._fact_cache[key]


def _range_input(tag: str, range_: InclusiveRange) -> ResolvedInput:
    return ResolvedInput(tag, "range", range_, range_)


def _iloc_input(tag: str, value: int) -> ResolvedInput:
    point = InclusiveRange(value, value)
    return ResolvedInput(tag, "iloc", int(value), point)


def _scope_resolver(tag: str) -> Callable[[EvaluationContext, InclusiveRange], ResolvedInput]:
    return lambda _context, scope: _range_input(tag, scope)


def _shape_analysis(context: EvaluationContext, range_: InclusiveRange) -> Mapping[str, Any]:
    def calculate() -> Mapping[str, Any]:
        from app.shared.annotation_agent_tools import measure_segment_shape

        try:
            return measure_segment_shape(context.telemetry, range_.start, range_.end) or {}
        except Exception:
            return {}
    return context.memo(("shape", range_.start, range_.end), calculate)


def _phase_resolver(
    tag: str, phase_name: str,
) -> Callable[[EvaluationContext, InclusiveRange], Optional[ResolvedInput]]:
    def resolve(context: EvaluationContext, scope: InclusiveRange) -> Optional[ResolvedInput]:
        ranges = []
        for phase in _shape_analysis(context, scope).get("phases") or []:
            if not isinstance(phase, Mapping):
                continue
            entry, apex, exit_ = phase.get("entry"), phase.get("apex"), phase.get("exit")
            if not all(isinstance(value, int) for value in (entry, apex, exit_)):
                continue
            named = {
                "entry": InclusiveRange(entry, apex),
                "apex": InclusiveRange(max(entry, apex - 2), min(exit_, apex + 2)),
                "exit": InclusiveRange(apex, exit_),
            }
            ranges.append(named[phase_name])
        envelope = InclusiveRange.envelope(ranges)
        return _range_input(tag, envelope) if envelope is not None else None
    return resolve


def _first(mask: np.ndarray, index: np.ndarray) -> Optional[int]:
    positions = np.flatnonzero(mask)
    return int(index[int(positions[0])]) if len(positions) else None


def _landmarks(values: Optional[np.ndarray], index: np.ndarray) -> Dict[str, Any]:
    if values is None or len(values) != len(index):
        return {}
    finite = np.where(np.isfinite(values), values, 0.0)
    peak_position = int(np.argmax(finite))
    peak = float(finite[peak_position])
    active = finite >= max(0.05, peak * 0.10)
    high = finite >= max(0.10, peak * 0.90)
    application_onset = _first(active, index)
    application_end = _first(high, index)
    after_peak = np.arange(len(finite)) > peak_position
    release_onset_positions = np.flatnonzero(
        after_peak & (finite < max(0.10, peak * 0.90))
    )
    release_end_positions = np.flatnonzero(after_peak & (finite <= 0.05))
    release_onset = (
        int(index[int(release_onset_positions[0])])
        if len(release_onset_positions) else None
    )
    release_end = (
        int(index[int(release_end_positions[0])])
        if len(release_end_positions) else None
    )
    hold_length = (
        max(0, release_onset - application_end)
        if application_end is not None and release_onset is not None else None
    )
    return {
        "application_onset": application_onset,
        "application_end": application_end,
        "release_onset": release_onset,
        "release_end": release_end,
        "peak": peak,
        "peak_iloc": int(index[peak_position]),
        "hold_length": hold_length,
    }


CONTROL_COLUMNS = {
    ("player", "brake"): ("Physics_brake",),
    ("expert", "brake"): ("expert_optimal_brake",),
    ("player", "throttle"): ("Physics_gas",),
    ("expert", "throttle"): ("expert_optimal_throttle",),
}


def _control_landmarks(
    context: EvaluationContext, scope: InclusiveRange, driver: str, control: str,
) -> Mapping[str, Any]:
    def calculate() -> Mapping[str, Any]:
        segment = context.segment(scope)
        return _landmarks(
            _series(segment, *CONTROL_COLUMNS[(driver, control)]),
            segment.index.to_numpy(dtype=int),
        )
    return context.memo(
        ("landmarks", scope.start, scope.end, driver, control), calculate,
    )


def _control_iloc_resolver(
    tag: str, driver: str, control: str, landmark: str,
) -> Callable[[EvaluationContext, InclusiveRange], Optional[ResolvedInput]]:
    def resolve(context: EvaluationContext, scope: InclusiveRange) -> Optional[ResolvedInput]:
        value = _control_landmarks(context, scope, driver, control).get(landmark)
        return _iloc_input(tag, value) if value is not None else None
    return resolve


def _brake_comparison_range_resolver(
    tag: str,
) -> Callable[[EvaluationContext, InclusiveRange], Optional[ResolvedInput]]:
    def resolve(context: EvaluationContext, scope: InclusiveRange) -> Optional[ResolvedInput]:
        player = _control_landmarks(context, scope, "player", "brake")
        expert = _control_landmarks(context, scope, "expert", "brake")
        onsets = [player.get("application_onset"), expert.get("application_onset")]
        ends = [player.get("release_end"), expert.get("release_end")]
        if not all(isinstance(value, int) for value in (*onsets, *ends)):
            return None
        range_ = InclusiveRange(min(onsets), max(ends))
        return _range_input(tag, range_)
    return resolve


def _steering_landmarks(
    context: EvaluationContext, scope: InclusiveRange, driver: str,
) -> Mapping[str, Optional[int]]:
    def calculate() -> Mapping[str, Optional[int]]:
        segment = context.segment(scope)
        names = ("Physics_steer_angle",) if driver == "player" else ("expert_optimal_steering",)
        values = _series(segment, *names)
        if values is None:
            return {}
        index = segment.index.to_numpy(dtype=int)
        absolute = np.abs(values)
        peak_position = int(np.nanargmax(absolute))
        threshold = max(0.02, float(absolute[peak_position]) * 0.10)
        onset = np.flatnonzero(absolute >= threshold)
        exit_ = np.flatnonzero(
            (np.arange(len(values)) > peak_position) & (absolute < threshold)
        )
        return {
            "turn_in": int(index[int(onset[0])]) if len(onset) else None,
            "apex": int(index[peak_position]),
            "turn_exit": int(index[int(exit_[0])]) if len(exit_) else None,
        }
    return context.memo(("steering", scope.start, scope.end, driver), calculate)


def _steering_iloc_resolver(
    tag: str, driver: str, landmark: str,
) -> Callable[[EvaluationContext, InclusiveRange], Optional[ResolvedInput]]:
    def resolve(context: EvaluationContext, scope: InclusiveRange) -> Optional[ResolvedInput]:
        value = _steering_landmarks(context, scope, driver).get(landmark)
        return _iloc_input(tag, value) if value is not None else None
    return resolve


def _expert_shift_range_resolver(
    tag: str, direction: str,
) -> Callable[[EvaluationContext, InclusiveRange], Optional[ResolvedInput]]:
    def resolve(context: EvaluationContext, scope: InclusiveRange) -> Optional[ResolvedInput]:
        segment = context.segment(scope)
        gears = _series(segment, "expert_optimal_gear")
        if gears is None:
            return None
        differences = np.diff(gears)
        matches = differences > 0 if direction == "up" else differences < 0
        positions = np.flatnonzero(matches)
        if not len(positions):
            return None
        start = int(positions[0])
        end = start
        while end + 1 < len(matches) and bool(matches[end + 1]):
            end += 1
        index = segment.index.to_numpy(dtype=int)
        range_ = InclusiveRange(int(index[start]), int(index[end + 1]))
        return _range_input(tag, range_)
    return resolve


def build_input_registry() -> InputRegistry:
    definitions: Dict[str, InputDefinition] = {}
    for tag in (
        "section_range", "segment_range", "opponent_interaction_range",
        "control_range",
    ):
        definitions[tag] = InputDefinition("range", _scope_resolver(tag))
    definitions.update({
        "speed_comparison_range": InputDefinition(
            "range", _scope_resolver("speed_comparison_range"),
        ),
        "trajectory_comparison_range": InputDefinition(
            "range", _scope_resolver("trajectory_comparison_range"),
        ),
        "brake_comparison_range": InputDefinition(
            "range", _brake_comparison_range_resolver("brake_comparison_range"),
        ),
        "expert_upshift_range": InputDefinition(
            "range",
            _expert_shift_range_resolver("expert_upshift_range", "up"),
        ),
        "expert_downshift_range": InputDefinition(
            "range",
            _expert_shift_range_resolver("expert_downshift_range", "down"),
        ),
        "corner_entry_range": InputDefinition("range", _phase_resolver("corner_entry_range", "entry")),
        "corner_apex_range": InputDefinition("range", _phase_resolver("corner_apex_range", "apex")),
        "corner_exit_range": InputDefinition("range", _phase_resolver("corner_exit_range", "exit")),
    })
    for driver in ("player", "expert"):
        for control in ("brake", "throttle"):
            for landmark in (
                "application_onset", "application_end", "release_onset", "release_end",
            ):
                tag = f"{driver}_{control}_{landmark}_iloc"
                definitions[tag] = InputDefinition(
                    "iloc", _control_iloc_resolver(tag, driver, control, landmark),
                )
        for landmark in ("turn_in", "apex", "turn_exit"):
            tag = f"{driver}_{landmark}_iloc"
            definitions[tag] = InputDefinition(
                "iloc", _steering_iloc_resolver(tag, driver, landmark),
            )
    return InputRegistry(definitions)


def _range(inputs: Sequence[ResolvedInput]) -> InclusiveRange:
    value = inputs[0].value
    if not isinstance(value, InclusiveRange):
        raise TypeError("range input required")
    return value


def _relation(player: Optional[int], expert: Optional[int]) -> Any:
    if player is None or expert is None:
        return MISSING
    delta = int(player) - int(expert)
    if delta == 0:
        return "aligned"
    return "earlier" if delta < 0 else "later"


def _compare_ilocs(_context: EvaluationContext, inputs: Sequence[ResolvedInput]) -> Any:
    return _relation(int(inputs[0].value), int(inputs[1].value))


def _compare_shift_timing(
    context: EvaluationContext, range_: InclusiveRange, direction: str,
) -> Any:
    segment = context.segment(range_)
    player = _series(segment, "Physics_gear")
    expert = _series(segment, "expert_optimal_gear")
    if player is None or expert is None or len(player) < 2 or len(expert) < 2:
        return MISSING
    if not np.all(np.isfinite(player)) or not np.all(np.isfinite(expert)):
        return MISSING

    sign = 1.0 if direction == "up" else -1.0
    expert_progress = sign * float(expert[-1] - expert[0])
    player_progress = sign * float(player[-1] - player[0])
    player_differences = sign * np.diff(player)
    if (
        expert_progress <= 0.0
        or player_progress < 0.0
        or np.any(player_differences < 0.0)
    ):
        return MISSING

    start_gap = sign * float(player[0] - expert[0])
    end_gap = sign * float(player[-1] - expert[-1])
    starts_aligned = bool(np.isclose(start_gap, 0.0))
    ends_aligned = bool(np.isclose(end_gap, 0.0))
    if start_gap > 0.0 and ends_aligned:
        return "earlier"
    if starts_aligned and end_gap < 0.0:
        return "later"
    if starts_aligned and ends_aligned:
        return "aligned"
    return MISSING


def _trajectory(context: EvaluationContext, range_: InclusiveRange) -> Optional[np.ndarray]:
    def calculate() -> Optional[np.ndarray]:
        from app.shared.annotation_agent_tools import calculate_trajectory_offset

        return calculate_trajectory_offset(context.segment(range_))
    return context.memo(("trajectory", range_.start, range_.end), calculate)


def _speed_delta(context: EvaluationContext, range_: InclusiveRange) -> Optional[np.ndarray]:
    def calculate() -> Optional[np.ndarray]:
        segment = context.segment(range_)
        player = _series(segment, "Physics_speed_kmh")
        expert = _series(segment, "expert_optimal_speed")
        return expert - player if player is not None and expert is not None else None
    return context.memo(("speed_delta", range_.start, range_.end), calculate)


def _finite(values: Optional[np.ndarray]) -> np.ndarray:
    return values[np.isfinite(values)] if values is not None else np.array([])


def _trajectory_position(context: EvaluationContext, range_: InclusiveRange) -> Any:
    from app.shared.annotation_agent_tools import (
        TRAJECTORY_ALIGNMENT_TOLERANCE_METERS,
    )

    values = _finite(_trajectory(context, range_))
    if not len(values):
        return MISSING
    median = float(np.nanmedian(values))
    if abs(median) <= TRAJECTORY_ALIGNMENT_TOLERANCE_METERS:
        return "aligned"
    return "wider" if median > 0 else "tighter"


def _time_analysis(context: EvaluationContext, range_: InclusiveRange) -> Mapping[str, Any]:
    def calculate() -> Mapping[str, Any]:
        segment = context.segment(range_)
        values = _series(segment, "expert_time_difference")
        finite = _finite(values)
        if len(finite) < 2:
            return {}
        runs, accelerating_rises = _time_slope_runs(context.telemetry, range_)
        return {
            "delta_value": round(float(finite[-1] - finite[0]), 2),
            "starting_direction": runs[0].get("direction") if runs else None,
            "ending_direction": runs[-1].get("direction") if runs else None,
            "middle_has_rise": bool(accelerating_rises),
        }
    return context.memo(("time", range_.start, range_.end), calculate)


def _time_slope_direction(
    previous_angle: float, current_angle: float, flattening: bool,
) -> str:
    angle_change = current_angle - previous_angle
    moves_toward_zero = (
        previous_angle * current_angle >= 0.0
        and abs(current_angle) < abs(previous_angle)
        and abs(angle_change) >= SLOPE_ANGLE_DEGREES
    )
    if moves_toward_zero:
        return "flattening"
    if flattening:
        if angle_change >= SLOPE_ANGLE_DEGREES:
            return "rising"
        if angle_change <= -SLOPE_ANGLE_DEGREES:
            return "falling"
        return "flattening"
    if current_angle >= SLOPE_ANGLE_DEGREES:
        return "rising"
    if current_angle <= -SLOPE_ANGLE_DEGREES:
        return "falling"
    return "flat"


def _time_slope_runs(
    df: pd.DataFrame, range_: InclusiveRange,
) -> Tuple[list[Dict[str, Any]], list[Tuple[int, int]]]:
    segment = df.loc[
        (df.index >= range_.start) & (df.index <= range_.end)
    ]
    if "expert_time_difference" not in segment.columns:
        return [], []
    values = pd.to_numeric(
        segment["expert_time_difference"], errors="coerce",
    ).to_numpy(dtype=float)
    ilocs = segment.index.to_numpy(dtype=float)
    if len(values) < 2:
        return [], []
    iloc_deltas = np.diff(ilocs)
    value_deltas = np.diff(values)
    valid = (
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & np.isfinite(iloc_deltas)
        & (iloc_deltas == 1.0)
        & np.isfinite(value_deltas)
    )
    if not np.any(valid):
        return [], []
    finite_values = values[np.isfinite(values)]
    value_span = float(np.max(finite_values) - np.min(finite_values))
    iloc_span = float(np.max(ilocs) - np.min(ilocs))
    normalized_slopes = (
        (value_deltas[valid] / iloc_deltas[valid]) * (iloc_span / value_span)
        if value_span > 0.0 and iloc_span > 0.0
        else np.zeros(int(np.sum(valid)), dtype=float)
    )
    angles = np.degrees(np.arctan(normalized_slopes))
    step_starts = ilocs[:-1][valid]
    step_ends = ilocs[1:][valid]
    step_start_values = values[:-1][valid]
    step_end_values = values[1:][valid]
    runs: list[Dict[str, Any]] = []
    accelerating_rises: list[Tuple[int, int]] = []
    previous_angle: Optional[float] = None
    previous_end_iloc: Optional[int] = None
    flattening = False
    for angle, start_iloc, end_iloc, start_value, end_value in zip(
        angles, step_starts, step_ends, step_start_values, step_end_values,
    ):
        if previous_end_iloc is not None and int(start_iloc) != previous_end_iloc:
            previous_angle = None
            flattening = False
        if (
            previous_angle is not None
            and float(angle) > 0.0
            and float(angle) - previous_angle >= SLOPE_ANGLE_DEGREES
        ):
            accelerating_rises.append((int(start_iloc), int(end_iloc)))
        direction = _time_slope_direction(
            previous_angle if previous_angle is not None else 0.0,
            float(angle), flattening,
        )
        flattening = direction == "flattening"
        step = {
            "start_iloc": int(start_iloc),
            "end_iloc": int(end_iloc),
            "start_value": float(start_value),
            "end_value": float(end_value),
            "direction": direction,
        }
        if (
            runs
            and runs[-1]["direction"] == direction
            and runs[-1]["end_iloc"] == step["start_iloc"]
        ):
            runs[-1]["end_iloc"] = step["end_iloc"]
            runs[-1]["end_value"] = step["end_value"]
        else:
            runs.append(step)
        previous_angle = float(angle)
        previous_end_iloc = int(end_iloc)
    return runs, accelerating_rises


def _opponent(context: EvaluationContext, range_: InclusiveRange) -> Mapping[str, Any]:
    def calculate() -> Mapping[str, Any]:
        from app.shared.annotation_agent_tools import (
            classify_opponent_interaction,
            query_opponent_trajectory,
        )

        try:
            content = classify_opponent_interaction(
                context.telemetry, range_.start, range_.end,
            ) or {}
        except Exception:
            return {}
        facts: Dict[str, Any] = {
            "outcome": content.get("outcome"),
            "confidence_level": content.get("confidence_level"),
        }
        candidates = content.get("candidates") or []
        primary = candidates[0] if candidates and isinstance(candidates[0], Mapping) else {}
        entry = primary.get("entry_signed_long_gap_m")
        exit_ = primary.get("exit_signed_long_gap_m")
        facts["gap_shrank"] = (
            abs(float(exit_)) < abs(float(entry))
            if entry is not None and exit_ is not None else None
        )
        facts["drew_alongside"] = int(primary.get("side_by_side_iloc_count") or 0) > 0
        facts["side_swap"] = None
        slot = content.get("targeted_car_slot")
        if slot is not None:
            try:
                trajectory = query_opponent_trajectory(
                    context.telemetry, range_.start, range_.end, int(slot), n_samples=7,
                )
            except Exception:
                trajectory = {}
            lateral = [
                float(sample["lateral_offset_m"])
                for sample in trajectory.get("samples") or []
                if isinstance(sample, Mapping) and sample.get("lateral_offset_m") is not None
            ]
            if lateral:
                facts["side_swap"] = min(lateral) < 0 < max(lateral)
        return facts
    return context.memo(("opponent", range_.start, range_.end), calculate)


def _slip_balance(context: EvaluationContext, range_: InclusiveRange) -> Optional[np.ndarray]:
    def calculate() -> Optional[np.ndarray]:
        segment = context.segment(range_)
        values = [
            _series(segment, name) for name in (
                "Physics_slip_angle_front_left", "Physics_slip_angle_front_right",
                "Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right",
            )
        ]
        if any(value is None for value in values):
            return None
        front_left, front_right, rear_left, rear_right = values
        return (np.abs(rear_left) + np.abs(rear_right)) / 2.0 - (
            np.abs(front_left) + np.abs(front_right)
        ) / 2.0
    return context.memo(("slip_balance", range_.start, range_.end), calculate)


def _push_to_limit(context: EvaluationContext, range_: InclusiveRange) -> Optional[np.ndarray]:
    def calculate() -> Optional[np.ndarray]:
        from app.shared.tire_grip_features import SlipEnvelopeConfig

        segment = context.segment(range_)
        angles = [_series(segment, name) for name in (
            "Physics_slip_angle_front_left", "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right",
        )]
        ratios = [_series(segment, name) for name in (
            "Physics_slip_ratio_front_left", "Physics_slip_ratio_front_right",
            "Physics_slip_ratio_rear_left", "Physics_slip_ratio_rear_right",
        )]
        if any(value is None for value in [*angles, *ratios]):
            return None
        config = SlipEnvelopeConfig()
        lateral = np.maximum.reduce([np.abs(value) for value in angles])
        longitudinal = np.maximum.reduce([np.abs(value) for value in ratios])
        return np.sqrt(
            (
                config.slip_angle_weight * lateral
                / max(config.front_slip_limit, config.rear_slip_limit)
            ) ** 2
            + (
                config.slip_ratio_weight * longitudinal
                / max(
                    config.front_longitudinal_slip_limit,
                    config.rear_longitudinal_slip_limit,
                )
            ) ** 2
        )
    return context.memo(("push_to_limit", range_.start, range_.end), calculate)


def _control_similarity(
    context: EvaluationContext, range_: InclusiveRange, control: str,
) -> Any:
    segment = context.segment(range_)
    player = _series(segment, *CONTROL_COLUMNS[("player", control)])
    expert = _series(segment, *CONTROL_COLUMNS[("expert", control)])
    if player is None or expert is None:
        return MISSING
    return float(np.mean(np.isclose(player, expert, atol=0.02, equal_nan=False)))


def _control_overlap(
    context: EvaluationContext, range_: InclusiveRange, driver: str,
) -> Any:
    segment = context.segment(range_)
    brake = _series(segment, *CONTROL_COLUMNS[(driver, "brake")])
    throttle = _series(segment, *CONTROL_COLUMNS[(driver, "throttle")])
    if brake is None or throttle is None:
        return MISSING
    return int(np.sum((brake > 0.05) & (throttle > 0.05)))


def _control_comparison(
    context: EvaluationContext, range_: InclusiveRange, control: str, metric: str,
) -> Any:
    player = _control_landmarks(context, range_, "player", control)
    expert = _control_landmarks(context, range_, "expert", control)
    first, second = player.get(metric), expert.get(metric)
    if first is None or second is None:
        return MISSING
    if metric == "peak":
        return (
            "aligned" if abs(float(first) - float(second)) <= 0.10
            else "higher" if float(first) > float(second) else "lower"
        )
    if metric == "hold_length":
        return (
            "aligned" if int(first) == int(second)
            else "longer" if int(first) > int(second) else "shorter"
        )
    raise ValueError(metric)


def _fact_range(
    calculate: Callable[[EvaluationContext, InclusiveRange], Any],
) -> Callable[[EvaluationContext, Sequence[ResolvedInput]], Any]:
    return lambda context, inputs: calculate(context, _range(inputs))


def _mapping_fact(
    analysis: Callable[[EvaluationContext, InclusiveRange], Mapping[str, Any]], key: str,
) -> Callable[[EvaluationContext, Sequence[ResolvedInput]], Any]:
    return _fact_range(lambda context, range_: analysis(context, range_).get(key, MISSING))


def build_fact_registry() -> FactRegistry:
    range_kind = ("range",)
    iloc_pair = ("iloc", "iloc")
    definitions: Dict[str, FactDefinition] = {
        "compare_ilocs": FactDefinition(iloc_pair, _compare_ilocs),
        "compare_upshift_timing": FactDefinition(
            range_kind,
            _fact_range(lambda c, r: _compare_shift_timing(c, r, "up")),
        ),
        "compare_downshift_timing": FactDefinition(
            range_kind,
            _fact_range(lambda c, r: _compare_shift_timing(c, r, "down")),
        ),
        "find_phase_presence": FactDefinition(range_kind, lambda _context, _inputs: True),
        "find_section_overlap_names": FactDefinition(range_kind, lambda context, _inputs: [
            LABEL_MAPPING[value] for value in context.overlap_section_ids if value in LABEL_MAPPING
        ]),
        "find_total_time_change": FactDefinition(range_kind, _mapping_fact(_time_analysis, "delta_value")),
        "find_starting_time_direction": FactDefinition(range_kind, _mapping_fact(_time_analysis, "starting_direction")),
        "find_ending_time_direction": FactDefinition(range_kind, _mapping_fact(_time_analysis, "ending_direction")),
        "find_middle_time_rise": FactDefinition(range_kind, _mapping_fact(_time_analysis, "middle_has_rise")),
        "find_brake_similarity": FactDefinition(range_kind, _fact_range(lambda c, r: _control_similarity(c, r, "brake"))),
        "find_throttle_similarity": FactDefinition(range_kind, _fact_range(lambda c, r: _control_similarity(c, r, "throttle"))),
        "find_brake_peak_ratio": FactDefinition(range_kind, _fact_range(lambda c, r: (
            float(_control_landmarks(c, r, "player", "brake")["peak"])
            / float(_control_landmarks(c, r, "expert", "brake")["peak"])
        ))),
        "compare_brake_peaks": FactDefinition(range_kind, _fact_range(lambda c, r: _control_comparison(c, r, "brake", "peak"))),
        "compare_brake_holds": FactDefinition(range_kind, _fact_range(lambda c, r: _control_comparison(c, r, "brake", "hold_length"))),
        "count_control_overlap": FactDefinition(range_kind, _fact_range(lambda c, r: _control_overlap(c, r, "player"))),
        "count_expert_control_overlap": FactDefinition(range_kind, _fact_range(lambda c, r: _control_overlap(c, r, "expert"))),
        "find_speed_expert_faster": FactDefinition(range_kind, _fact_range(lambda c, r: (
            float(np.nanmedian(_finite(_speed_delta(c, r)))) > 0
            if len(_finite(_speed_delta(c, r))) else MISSING
        ))),
        "find_speed_peak_gap": FactDefinition(range_kind, _fact_range(lambda c, r: (
            float(np.max(np.abs(_finite(_speed_delta(c, r)))))
            if len(_finite(_speed_delta(c, r))) else MISSING
        ))),
        "find_speed_gap_closing": FactDefinition(range_kind, _fact_range(lambda c, r: (
            abs(float(_finite(_speed_delta(c, r))[-1])) < abs(float(_finite(_speed_delta(c, r))[0]))
            if len(_finite(_speed_delta(c, r))) else MISSING
        ))),
        "find_trajectory_peak_offset": FactDefinition(range_kind, _fact_range(lambda c, r: (
            float(np.max(np.abs(_finite(_trajectory(c, r)))))
            if len(_finite(_trajectory(c, r))) else MISSING
        ))),
        "find_trajectory_convergence": FactDefinition(range_kind, _fact_range(lambda c, r: (
            abs(float(_finite(_trajectory(c, r))[-1])) < abs(float(_finite(_trajectory(c, r))[0]))
            if len(_finite(_trajectory(c, r))) else MISSING
        ))),
        "find_trajectory_position": FactDefinition(
            range_kind, _fact_range(_trajectory_position),
        ),
        "find_oversteer": FactDefinition(range_kind, _fact_range(lambda c, r: (
            bool(np.nanmax(_slip_balance(c, r)) > 0.02) if _slip_balance(c, r) is not None else MISSING
        ))),
        "find_understeer": FactDefinition(range_kind, _fact_range(lambda c, r: (
            bool(np.nanmin(_slip_balance(c, r)) < -0.02) if _slip_balance(c, r) is not None else MISSING
        ))),
        "find_grip_over_limit": FactDefinition(range_kind, _fact_range(lambda c, r: (
            bool(np.nanmax(_push_to_limit(c, r)) > 1.0) if _push_to_limit(c, r) is not None else MISSING
        ))),
        "find_sustained_low_grip": FactDefinition(range_kind, _fact_range(lambda c, r: (
            bool(np.mean(_push_to_limit(c, r) < 0.8) >= 0.5) if _push_to_limit(c, r) is not None else MISSING
        ))),
        "compare_exit_gear": FactDefinition(range_kind, _fact_range(lambda c, r: _compare_exit_gear(c, r))),
        "find_entry_altitude_trend": FactDefinition(range_kind, _fact_range(lambda c, r: _altitude(c, r, "entry"))),
        "find_apex_altitude_trend": FactDefinition(range_kind, _fact_range(lambda c, r: _altitude(c, r, "apex"))),
        "find_exit_altitude_trend": FactDefinition(range_kind, _fact_range(lambda c, r: _altitude(c, r, "exit"))),
    }
    for key in ("outcome", "confidence_level", "gap_shrank", "drew_alongside", "side_swap"):
        definitions[f"find_opponent_{key}"] = FactDefinition(
            range_kind, _mapping_fact(_opponent, key),
        )
    definitions["classify_segment_shape"] = FactDefinition(
        range_kind, _fact_range(lambda c, r: (_shape_analysis(c, r).get("base_segment_shape") or {}).get("shape_key", MISSING)),
    )
    definitions["classify_corner_shape"] = FactDefinition(
        range_kind, _fact_range(lambda c, r: (_shape_analysis(c, r).get("corner_shape_refinement") or {}).get("shape_key", MISSING)),
    )
    return FactRegistry(definitions)


def _compare_exit_gear(context: EvaluationContext, range_: InclusiveRange) -> Any:
    segment = context.segment(range_)
    player = _series(segment, "Physics_gear")
    expert = _series(segment, "expert_optimal_gear")
    if player is None or expert is None:
        return MISSING
    return "lower" if player[-1] < expert[-1] else "higher" if player[-1] > expert[-1] else "aligned"


def _altitude(context: EvaluationContext, range_: InclusiveRange, phase: str) -> Any:
    del phase
    segment = context.segment(range_)
    if "expert_optimal_player_pos_z" in segment.columns:
        altitude = "expert_optimal_player_pos_z"
        x_column, y_column = (
            "expert_optimal_player_pos_x", "expert_optimal_player_pos_y",
        )
    elif "Graphics_player_pos_z" in segment.columns:
        altitude = "Graphics_player_pos_z"
        x_column, y_column = "Graphics_player_pos_x", "Graphics_player_pos_y"
    else:
        return MISSING
    if x_column not in segment.columns or y_column not in segment.columns:
        return MISSING
    z = pd.to_numeric(segment[altitude], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(segment[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(segment[y_column], errors="coerce").to_numpy(dtype=float)
    finite = np.flatnonzero(np.isfinite(z) & np.isfinite(x) & np.isfinite(y))
    if len(finite) < 2:
        return MISSING
    first, last = int(finite[0]), int(finite[-1])
    distances = np.hypot(
        np.diff(x[first:last + 1]), np.diff(y[first:last + 1]),
    )
    horizontal = float(np.sum(distances[np.isfinite(distances)]))
    if horizontal <= 0.0:
        return MISSING
    angle = float(np.degrees(np.arctan2(float(z[last] - z[first]), horizontal)))
    return "uphill" if angle > 3.0 else "downhill" if angle < -3.0 else "level"


INPUT_REGISTRY = build_input_registry()
FACT_REGISTRY = build_fact_registry()


__all__ = [
    "EvaluationContext", "FACT_REGISTRY", "INPUT_REGISTRY", "build_fact_registry",
    "build_input_registry", "smooth_telemetry",
]
