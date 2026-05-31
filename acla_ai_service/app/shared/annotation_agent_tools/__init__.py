"""
Telemetry capability tools the agent box exposes to its sub-agents.

Provides two categories of tool that the agent nodes can invoke:

1. **Graph generation** — per-graph DataFrame builders + matplotlib
   renderers (feature plots, trajectory plots) catalogued in
   ``AGENT_GRAPH_DEFINITIONS``.
2. **Deterministic queries** — the ``PIPELINE_QUERY_DEFINITIONS`` catalog
   of structured math operations (threshold crossings, extrema, slopes,
   onset ordering) the zoom executor runs against the graph tables to
   extract exact ilocs / values for the synthesizer to cite.

Together they let the vision-capable LLM see visual evidence while the
executor produces verifiable numerical readings off the DataFrame.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from PIL import Image

from app.internal_knowledge_base import skills

LOGGER = logging.getLogger(__name__)

AGENT_GRAPH_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "throttle",
        "title": "Throttle Application - ",
        "description": "Expert vs player throttle traces.",
    },
    {
        "id": "brake",
        "title": "Brake Application - ",
        "description": "Expert vs player brake traces.",
    },
    {
        "id": "time_delta",
        "title": "Time Difference to Expert",
        "description": "Instantaneous time delta vs expert.",
    },
    {
        "id": "speed_delta",
        "title": "Speed Difference (Expert - Player)",
        "description": "Speed difference between expert and player.",
    },
    {
        "id": "speed",
        "title": "Speed Trace: Expert vs Player",
        "description": "Expert vs player speed traces.",
    },
    {
        "id": "push_limit",
        "title": "Driver Push/Limit",
        "description": "Driver push-to-limit metric.",
    },
    {
        "id": "trajectory_detailed",
        "title": "Detailed Trajectory",
        "description": (
            "Close-up trajectory. Green = player, blue dashed = expert. "
            "Expert-anchored phase markers per detected arc: yellow circle "
            "= entry, red star = apex, green triangle = exit. Chicanes / "
            "esses show numbered apexes (#1, #2, …) — one set of markers "
            "per arc."
        ),
    },
    {
        "id": "trajectory_gas_brake",
        "title": "Gas/Brake Trajectory",
        "description": (
            "Player trajectory coloured by throttle/brake balance "
            "(green = full gas, red = full brake, yellow = coasting). "
            "Mirrors the Gas/Brake colour mode in the human annotation track map."
        ),
    },
    {
        "id": "trajectory_balance",
        "title": "Oversteer/Understeer Slip Balance",
        "description": (
            "Line plot over segment index of (mean |rear slip| − mean |front slip|). "
            "Positive (red shading above zero) = oversteer (rear-slip dominant); "
            "negative (blue shading below zero) = understeer (front-slip dominant). "
            "Zero = balanced front/rear slip."
        ),
    },
    {
        "id": "trajectory_offset",
        "title": "Trajectory Offset (signed)",
        "description": (
            "Signed perpendicular offset between player and expert lines over "
            "segment index. Positive y = player wider than expert (toward "
            "outside of corner); negative y = tighter (toward inside). "
            "Expert-anchored entry/apex/exit markers placed on the offset trace."
        ),
    },
    {
        "id": "gear",
        "title": "Gear Selection: Expert vs Player",
        "description": "Expert vs player gear traces (integer steps).",
    },
]

# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def _plot_to_image(fig) -> Image.Image:
    """Convert a matplotlib figure to a PIL Image (PNG in memory)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return img


def _create_feature_plot(
    table: pd.DataFrame,
    title: str,
) -> Optional[Image.Image]:
    """Line plot for every column in ``table`` — one labelled trace each."""
    if table.empty or len(table.columns) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in table.columns:
        ax.plot(table.index, table[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.legend()
    ax.grid(True)

    return _plot_to_image(fig)


def _make_colored_line_collection(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    linewidth: float = 3,
) -> LineCollection:
    """Build a LineCollection whose segments are coloured by *values*."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    # One colour value per segment — use the left endpoint's value
    lc.set_array(values[:-1])
    ax.add_collection(lc)
    return lc


# ---------------------------------------------------------------------------
# Expert-anchored corner phase detection (compute_expert_phases tool)
#
# Phases are derived only from EXPERT telemetry — the player can stop or
# drive erratically mid-corner, so player-derived phases are unreliable.
# The same iloc identifies the same telemetry sample on both lines.
#
# Algorithm (track-data-free): smooth the expert (x, y) trace, compute
# signed parametric curvature κ = (x'·y'' − y'·x'') / (x'² + y'²)^(3/2),
# and split the segment into arcs where |κ| exceeds a noise threshold and
# sign(κ) is constant. Each arc yields one (entry, apex, exit) — chicanes
# and esses naturally produce multiple arcs of opposite sign. Apex is the
# argmax of |κ| within the arc (geometric, robust on flat-speed sweepers
# and trail-braked corners where min-speed and apex disagree).
# ---------------------------------------------------------------------------


_KAPPA_FLOOR = 1e-3   # absolute κ below this is treated as noise (matches existing usage in detailed_track_map.py)
_KAPPA_FRAC = 0.20    # arc threshold = max(_KAPPA_FLOOR, _KAPPA_FRAC * max|κ|)


def _odd(n: int) -> int:
    """Return *n* clipped to odd so a centred convolution window is symmetric."""
    return n if n % 2 == 1 else n + 1


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average via numpy convolution.

    Uses ``mode='same'`` so output length matches input. Edge samples are
    biased by zero-padding inside ``np.convolve`` — callers should mask
    the first/last ``window // 2`` samples when picking peaks.
    """
    if window <= 1 or arr.size < window:
        return arr.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr.astype(float), kernel, mode="same")


def _smoothed_expert_kinematics(
    segment: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Smooth the expert (x, y) trace and return parametric kinematics.

    Shared by ``_detect_expert_phases`` (which derives entry / apex / exit
    ilocs from the curvature peaks) and ``_create_trajectory_offset_plot``
    (which needs the unit tangent for cross-track error and the signed κ
    for the wider/tighter sign flip). Keeping a single helper guarantees
    the smoothing window matches across the marker positions and the
    offset trace.

    Returns ``(x_s, y_s, dx, dy, kappa, window)`` or ``None`` if the
    segment is too short, missing required columns, or all-NaN.
    """
    n = len(segment)
    if n < 8:
        return None
    if "expert_optimal_player_pos_x" not in segment.columns or \
       "expert_optimal_player_pos_y" not in segment.columns:
        return None

    x = segment["expert_optimal_player_pos_x"].to_numpy(dtype=float)
    y = segment["expert_optimal_player_pos_y"].to_numpy(dtype=float)
    if not np.isfinite(x).any() or not np.isfinite(y).any():
        return None
    x = np.where(np.isfinite(x), x, np.interp(np.arange(n), np.where(np.isfinite(x))[0], x[np.isfinite(x)])) if np.isnan(x).any() else x
    y = np.where(np.isfinite(y), y, np.interp(np.arange(n), np.where(np.isfinite(y))[0], y[np.isfinite(y)])) if np.isnan(y).any() else y

    window = _odd(max(5, n // 30))
    x_s = _moving_average(x, window)
    y_s = _moving_average(y, window)

    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    kappa = np.zeros_like(denom, dtype=float)
    np.divide(dx * ddy - dy * ddx, denom, out=kappa, where=denom > 1e-9)
    kappa = _moving_average(kappa, window)

    return x_s, y_s, dx, dy, kappa, window


def _detect_expert_phases(
    segment: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int]:
    """Detect corner phases from the expert position trace alone.

    Returns ``(phases, smoothing_window)`` where ``phases`` is a list of
    one dict per detected arc (empty for non-corner segments). Each dict
    holds ilocs **relative to the segment start** (the public
    ``compute_expert_phases`` tool shifts these into the parent frame
    before exposing them via ``PipelineAttachment``) plus auxiliary
    fields the VLM can use to reason about trail-braking and turn
    direction:

        {
            "entry": int, "apex": int, "exit": int,
            "direction": "left" | "right",
            "kappa_peak": float,                 # signed κ at apex
            "min_speed_iloc": int,
            "peak_steer_iloc": int,
            "apex_speed_disagreement": int,      # |apex - min_speed_iloc|
        }

    See module-level comment for the algorithm.
    """
    kin = _smoothed_expert_kinematics(segment)
    if kin is None:
        return [], 0
    _x_s, _y_s, _dx, _dy, kappa, window = kin
    n = len(segment)

    # Mask edge samples — convolution edge bias makes them unreliable for peaks.
    edge = window // 2
    mask = np.zeros(n, dtype=bool)
    mask[edge: n - edge] = True

    abs_k = np.abs(kappa)
    abs_k_masked = np.where(mask, abs_k, 0.0)
    peak = float(abs_k_masked.max()) if abs_k_masked.size else 0.0
    if peak < _KAPPA_FLOOR * 2.0:
        return [], window  # whole segment is below the noise floor → no corner

    threshold = max(_KAPPA_FLOOR, _KAPPA_FRAC * peak)
    above = (abs_k_masked > threshold)
    sign_k = np.sign(kappa)

    # Optional speed/steer for auxiliary cross-validation fields.
    speed = (
        segment["expert_optimal_speed"].to_numpy(dtype=float)
        if "expert_optimal_speed" in segment.columns else None
    )
    steer = (
        segment["expert_optimal_steering"].to_numpy(dtype=float)
        if "expert_optimal_steering" in segment.columns else None
    )

    min_arc_len = max(5, window)
    phases: List[Dict[str, Any]] = []

    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        s = sign_k[i]
        j = i
        while j < n and above[j] and sign_k[j] == s:
            j += 1
        # Arc is [i, j)
        arc_len = j - i
        arc_peak = float(abs_k[i:j].max()) if arc_len > 0 else 0.0
        if arc_len >= min_arc_len and arc_peak >= 2.0 * _KAPPA_FLOOR:
            apex_local = i + int(np.argmax(abs_k[i:j]))
            phase: Dict[str, Any] = {
                "entry": int(i),
                "apex": int(apex_local),
                "exit": int(j - 1),
                "direction": "left" if kappa[apex_local] > 0 else "right",
                "kappa_peak": float(kappa[apex_local]),
            }
            if speed is not None and np.isfinite(speed[i:j]).any():
                ms = i + int(np.nanargmin(speed[i:j]))
                phase["min_speed_iloc"] = int(ms)
                phase["apex_speed_disagreement"] = int(abs(apex_local - ms))
            if steer is not None and np.isfinite(steer[i:j]).any():
                phase["peak_steer_iloc"] = int(i + int(np.nanargmax(np.abs(steer[i:j]))))
            phases.append(phase)
        i = j

    return phases, window


def compute_expert_phases(
    df: pd.DataFrame, start_index: int, end_index: int,
):
    """Tool — per-arc entry / apex / exit ilocs from the expert position trace.

    Computes signed parametric curvature κ on the smoothed expert (x, y)
    trace and segments the parent slice into arcs where |κ| exceeds a
    noise threshold with constant sign. Each arc produces one
    (entry, apex, exit) tuple — chicanes / esses naturally yield multiple
    arcs of opposite ``direction``. Apex is ``argmax(|κ|)`` within the
    arc, robust on flat-speed sweepers and trail-braked corners where
    minimum speed precedes the geometric pinch.

    Returns a ``phase_indices`` attachment with shape::

        {
            "phases": [
                { "entry", "apex", "exit", "direction",
                  "kappa_peak", "min_speed_iloc", "peak_steer_iloc",
                  "apex_speed_disagreement" },
                ...
            ],
            "smoothing_window": int,
        }

    All ilocs are absolute parent-frame indices in
    ``[start_index, end_index)`` — matching the feature-plot x-axis and
    the synthesizer prompt's index range. ``phases`` is empty when no
    arc clears the curvature threshold (pure straight, or segment too
    short / missing position columns).
    """
    from app.local_annotation_agent.evaluators import PipelineAttachment

    start = int(start_index)
    segment = df.iloc[start: int(end_index)]
    phases, window = _detect_expert_phases(segment)

    iloc_fields = ("entry", "apex", "exit", "min_speed_iloc", "peak_steer_iloc")
    shifted_phases = [
        {
            k: (int(v) + start if k in iloc_fields else v)
            for k, v in phase.items()
        }
        for phase in phases
    ]

    return PipelineAttachment(
        name="phase_indices",
        kind="structured",
        label="Phase Indices (expert-anchored)",
        content=_round_floats({"phases": shifted_phases, "smoothing_window": int(window)}),
    )


def _player_heading(seg_player_x: np.ndarray, seg_player_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-iloc unit heading vector for the player.

    Smooths the player x/y trace with a centred 5-sample window (or
    smaller on short ranges) and takes the gradient. Sign convention:
    ``heading = (hx, hy)``; the left-perpendicular is ``(-hy, hx)``.
    """
    n_rows = len(seg_player_x)
    window = min(5, n_rows)
    if window % 2 == 0:
        window = max(1, window - 1)
    sx = _moving_average(seg_player_x, window)
    sy = _moving_average(seg_player_y, window)
    dx = np.gradient(sx)
    dy = np.gradient(sy)
    norm = np.sqrt(dx * dx + dy * dy)
    norm_safe = np.where(norm > 1e-6, norm, 1e-6)
    return dx / norm_safe, dy / norm_safe


def _cumulative_path_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative arclength along a 2D polyline."""
    if x.size == 0:
        return np.array([], dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.sqrt(dx * dx + dy * dy)
    return np.concatenate(([0.0], np.cumsum(seg_len)))


def _project_points_to_reference_path(
    point_x: np.ndarray,
    point_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto a reference polyline and return ``(s, d, idx)``.

    ``s`` is arclength progress along the reference path. ``d`` is signed
    lateral offset using the local tangent's left-normal. This is a small
    Frenet-like projection over the current telemetry window; it avoids the
    corner fragility of comparing cars only in the player's instantaneous
    heading frame.
    """
    n_points = int(point_x.size)
    s_out = np.full(n_points, np.nan, dtype=float)
    d_out = np.full(n_points, np.nan, dtype=float)
    idx_out = np.full(n_points, -1, dtype=int)
    if n_points == 0 or ref_x.size < 2:
        return s_out, d_out, idx_out

    ref_s = _cumulative_path_distance(ref_x, ref_y)
    for i in range(n_points):
        px = float(point_x[i])
        py = float(point_y[i])
        if not (np.isfinite(px) and np.isfinite(py)):
            continue

        vx = ref_x[1:] - ref_x[:-1]
        vy = ref_y[1:] - ref_y[:-1]
        wx = px - ref_x[:-1]
        wy = py - ref_y[:-1]
        seg_len2 = vx * vx + vy * vy
        t = np.divide(
            wx * vx + wy * vy,
            seg_len2,
            out=np.zeros_like(seg_len2),
            where=seg_len2 > 1e-9,
        )
        t = np.clip(t, 0.0, 1.0)
        proj_x = ref_x[:-1] + t * vx
        proj_y = ref_y[:-1] + t * vy
        dist2 = (px - proj_x) ** 2 + (py - proj_y) ** 2
        seg_idx = int(np.nanargmin(dist2))
        seg_len = float(np.sqrt(max(seg_len2[seg_idx], 0.0)))
        s_out[i] = float(ref_s[seg_idx] + t[seg_idx] * seg_len)
        cross = vx[seg_idx] * (py - proj_y[seg_idx]) - vy[seg_idx] * (px - proj_x[seg_idx])
        sign = 1.0 if cross >= 0.0 else -1.0
        d_out[i] = sign * float(np.sqrt(dist2[seg_idx]))
        idx_out[i] = seg_idx

    return s_out, d_out, idx_out


def _relative_position_frame(
    seg: pd.DataFrame,
    player_x: np.ndarray,
    player_y: np.ndarray,
    opponent_x: np.ndarray,
    opponent_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return opponent-player long/lateral gaps plus player s/d.

    Prefer expert-path projection when expert position columns are available.
    Fallback uses the player's instantaneous heading, preserving behaviour for
    datasets without expert traces.
    """
    has_expert = (
        "expert_optimal_player_pos_x" in seg.columns
        and "expert_optimal_player_pos_y" in seg.columns
    )
    if has_expert:
        kin = _smoothed_expert_kinematics(seg)
        if kin is not None:
            ref_x, ref_y, _dx, _dy, _kappa, _w = kin
            player_s, player_d, _ = _project_points_to_reference_path(
                player_x, player_y, ref_x, ref_y,
            )
            opponent_s, opponent_d, _ = _project_points_to_reference_path(
                opponent_x, opponent_y, ref_x, ref_y,
            )
            long_gap = opponent_s - player_s
            lateral_gap = opponent_d - player_d
            return long_gap, lateral_gap, player_s, player_d, "expert_path_projection"

    hx, hy = _player_heading(player_x, player_y)
    vx = opponent_x - player_x
    vy = opponent_y - player_y
    long_gap = vx * hx + vy * hy
    lateral_gap = vx * (-hy) + vy * hx
    player_s = _cumulative_path_distance(player_x, player_y)
    player_d = np.zeros_like(player_s)
    return long_gap, lateral_gap, player_s, player_d, "player_heading_projection"


def _active_opponent_mask(
    seg: pd.DataFrame,
    slot: int,
    opponent_x: np.ndarray,
    opponent_y: np.ndarray,
    player_x: np.ndarray,
    player_y: np.ndarray,
    *,
    same_car_tolerance_m: float = 0.25,
) -> np.ndarray:
    """Active mask for a true opponent slot, excluding the player's own slot.

    ``Car_{1..MAX_CARS}`` is a flattening of the raw car-coordinate array, not
    an opponents-only table. The player's car usually remains in one of those
    slots, while ``Graphics_player_pos_*`` stores the same coordinates again.
    If we do not filter that slot, zero-distance self samples look like a
    close opponent interaction.
    """
    active = (
        ((opponent_x != 0.0) | (opponent_y != 0.0))
        & np.isfinite(opponent_x)
        & np.isfinite(opponent_y)
    )
    if not active.any() or player_x.size != opponent_x.size:
        return active

    distance_to_player = np.sqrt((opponent_x - player_x) ** 2 + (opponent_y - player_y) ** 2)
    finite_distance = active & np.isfinite(distance_to_player)
    if finite_distance.any():
        same_car_fraction = float((distance_to_player[finite_distance] <= same_car_tolerance_m).mean())
        if same_car_fraction >= 0.95:
            return np.zeros_like(active, dtype=bool)
    return active


def find_nearest_opponent(
    df: pd.DataFrame, start_index: int, end_index: int,
    max_candidates: int = 3,
    side_by_side_max_distance_m: float = 8.0,
    min_active_fraction: float = 0.3,
):
    """Tool — identify the most relevant opponent(s) inside an iloc range.

    Reads ``Graphics_player_pos_{x,y}``, ``expert_optimal_player_pos_{x,y}``
    when available, and ``Car_{1..MAX_CARS}_pos_{x,y}`` over
    ``[start_index, end_index)``. Empty opponent slots (where both x and y
    are exactly ``0.0`` — the flattening default in ``telemetry.py``) are
    skipped per row. For each slot with active data, computes per-iloc 2D
    distance, signed longitudinal gap, and lateral offset. The signed gap
    prefers projection onto the expert path (positive ⇒ opponent further
    along the expert trace); without expert positions it falls back to the
    player's instantaneous heading frame.

    Slots whose active-iloc fraction is below ``min_active_fraction``
    are dropped. Remaining slots are ranked by minimum 2D distance; the
    top ``max_candidates`` are returned as ``candidates``. This is a
    supporting-detail tool; use ``classify_opponent_interaction`` for the
    role-aware primary slot and O / OD / MSR gate.

    Produces an ``opponent_context`` attachment::

        {
            "range": [start_index, end_index],
            "data_available": bool,
            "n_active_slots": int,
            "candidates": [
                {
                    "slot": int,                       # 1..MAX_CARS
                    "min_distance_m": float,
                    "min_distance_iloc": int,
                    "entry_distance_m": float,
                    "exit_distance_m": float,
                    "entry_signed_long_gap_m": float,  # + ⇒ opp ahead at start
                    "exit_signed_long_gap_m": float,   # + ⇒ opp ahead at end
                    "min_lateral_offset_m": float,
                    "min_lateral_offset_iloc": int,
                    "side_by_side_iloc_count": int,
                    "active_iloc_fraction": float,
                    "coordinate_frame": str,
                    "passed_by_player": bool,          # entry +, exit −
                    "got_passed_by_opponent": bool,    # entry −, exit +
                },
                ...
            ],
        }

    Empty ``candidates`` with ``data_available: False`` means the
    required position columns are absent. Empty ``candidates`` with
    ``data_available: True`` means no opponent was close enough / active
    enough in the range.
    """
    from app.local_annotation_agent.evaluators import PipelineAttachment
    from app.domain.telemetry import MAX_CARS

    def _attach(content: Dict[str, Any]) -> "PipelineAttachment":
        return PipelineAttachment(
            name="opponent_context",
            kind="structured",
            label="Opponent Context (nearest cars in range)",
            content=_round_floats(content),
        )

    s, e = int(start_index), int(end_index)
    base_payload: Dict[str, Any] = {
        "range": [s, e],
        "data_available": False,
        "n_active_slots": 0,
        "candidates": [],
    }

    if "Graphics_player_pos_x" not in df.columns or "Graphics_player_pos_y" not in df.columns:
        base_payload["message"] = (
            "Player position columns (Graphics_player_pos_x/y) missing — "
            "cannot compute opponent context."
        )
        return _attach(base_payload)

    seg = df.iloc[s:e]
    n_rows = len(seg)
    if n_rows < 2:
        base_payload["message"] = "Range too short for opponent context (need ≥ 2 rows)."
        return _attach(base_payload)

    player_x = seg["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = seg["Graphics_player_pos_y"].to_numpy(dtype=float)
    if not (np.isfinite(player_x).any() and np.isfinite(player_y).any()):
        base_payload["message"] = "Player position trace is all NaN/inf."
        return _attach(base_payload)

    candidates_raw: List[Dict[str, Any]] = []
    n_active_slots = 0

    for slot in range(1, MAX_CARS + 1):
        col_x = f"Car_{slot}_pos_x"
        col_y = f"Car_{slot}_pos_y"
        if col_x not in df.columns or col_y not in df.columns:
            continue
        ox = seg[col_x].to_numpy(dtype=float)
        oy = seg[col_y].to_numpy(dtype=float)
        active_mask = _active_opponent_mask(seg, slot, ox, oy, player_x, player_y)
        active_count = int(active_mask.sum())
        if active_count == 0:
            continue
        n_active_slots += 1
        active_fraction = active_count / n_rows
        if active_fraction < min_active_fraction:
            continue

        vx = np.where(active_mask, ox - player_x, np.nan)
        vy = np.where(active_mask, oy - player_y, np.nan)
        distance = np.sqrt(vx * vx + vy * vy)
        signed_long, lateral_signed, _player_s, _player_d, frame_name = _relative_position_frame(
            seg, player_x, player_y, ox, oy,
        )
        signed_long = np.where(active_mask, signed_long, np.nan)
        lateral_signed = np.where(active_mask, lateral_signed, np.nan)
        lateral_abs = np.abs(lateral_signed)

        finite_dist = np.isfinite(distance)
        if not finite_dist.any():
            continue

        min_d_local = int(np.nanargmin(distance))
        min_lat_local = int(np.nanargmin(lateral_abs))

        active_ilocs = np.where(active_mask)[0]
        entry_idx = int(active_ilocs[0])
        exit_idx = int(active_ilocs[-1])
        entry_long = float(signed_long[entry_idx])
        exit_long = float(signed_long[exit_idx])

        side_by_side = int(
            ((distance <= side_by_side_max_distance_m) & finite_dist).sum()
        )

        candidates_raw.append({
            "slot": int(slot),
            "min_distance_m": float(distance[min_d_local]),
            "min_distance_iloc": s + min_d_local,
            "entry_distance_m": float(distance[entry_idx]),
            "exit_distance_m": float(distance[exit_idx]),
            "entry_signed_long_gap_m": entry_long,
            "exit_signed_long_gap_m": exit_long,
            "min_lateral_offset_m": float(lateral_abs[min_lat_local]),
            "min_lateral_offset_iloc": s + min_lat_local,
            "side_by_side_iloc_count": side_by_side,
            "active_iloc_fraction": float(active_fraction),
            "coordinate_frame": frame_name,
            "passed_by_player": bool(entry_long > 0 and exit_long < 0),
            "got_passed_by_opponent": bool(entry_long < 0 and exit_long > 0),
        })

    candidates_raw.sort(key=lambda c: c["min_distance_m"])
    candidates = candidates_raw[:max_candidates]

    return _attach({
        "range": [s, e],
        "data_available": True,
        "n_active_slots": n_active_slots,
        "candidates": candidates,
    })


def query_opponent_trajectory(
    df: pd.DataFrame, start_index: int, end_index: int,
    slot: int,
    n_samples: int = 5,
):
    """Tool — sample one opponent's relative trajectory at N evenly-spaced ilocs.

    For opponent ``slot`` (``1..MAX_CARS``), reads
    ``Car_{slot}_pos_{x,y}`` + ``Graphics_player_pos_{x,y}`` over
    ``[start_index, end_index)`` and returns ``n_samples`` evenly-spaced
    snapshots of:

      * ``distance_m``           — 2D Euclidean distance
      * ``signed_long_gap_m``    — projection along player heading
                                    (+ ⇒ opp ahead)
      * ``lateral_offset_m``     — signed perpendicular projection
                                    (+ ⇒ opp on player's left of heading,
                                    − ⇒ right)

    Use after ``find_nearest_opponent`` has named a candidate slot, to
    inspect HOW the relationship evolved through the range — e.g. gap
    closing steadily on a straight (slipstream), a step-change at apex
    (switchback rotation), or lateral offset crossing zero
    (line-cross during a pass).

    Produces an ``opponent_trajectory`` attachment::

        {
            "range": [start_index, end_index],
            "slot": int,
            "data_available": bool,
            "samples": [
                {
                    "iloc": int,
                    "distance_m": float | None,
                    "signed_long_gap_m": float | None,
                    "lateral_offset_m": float | None,
                    "note": str | None,
                },
                ...
            ],
        }
    """
    from app.local_annotation_agent.evaluators import PipelineAttachment

    slot_int = int(slot)

    def _attach(content: Dict[str, Any]) -> "PipelineAttachment":
        return PipelineAttachment(
            name="opponent_trajectory",
            kind="structured",
            label=f"Opponent Trajectory (slot {slot_int})",
            content=_round_floats(content),
        )

    s, e = int(start_index), int(end_index)
    n_samples = max(2, int(n_samples))
    base: Dict[str, Any] = {
        "range": [s, e],
        "slot": slot_int,
        "data_available": False,
        "samples": [],
    }

    col_x = f"Car_{slot_int}_pos_x"
    col_y = f"Car_{slot_int}_pos_y"
    required = (col_x, col_y, "Graphics_player_pos_x", "Graphics_player_pos_y")
    missing = [c for c in required if c not in df.columns]
    if missing:
        base["message"] = f"required columns missing: {missing}"
        return _attach(base)

    seg = df.iloc[s:e]
    n_rows = len(seg)
    if n_rows < 2:
        base["message"] = "range too short (need ≥ 2 rows)"
        return _attach(base)

    player_x = seg["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = seg["Graphics_player_pos_y"].to_numpy(dtype=float)
    ox = seg[col_x].to_numpy(dtype=float)
    oy = seg[col_y].to_numpy(dtype=float)
    hx, hy = _player_heading(player_x, player_y)

    if n_samples >= n_rows:
        sample_locals = list(range(n_rows))
    else:
        sample_locals = [int(v) for v in np.linspace(0, n_rows - 1, n_samples, dtype=int)]

    samples: List[Dict[str, Any]] = []
    for local in sample_locals:
        opp_x = ox[local]
        opp_y = oy[local]
        is_empty = (opp_x == 0.0) and (opp_y == 0.0)
        if is_empty or not np.isfinite(opp_x) or not np.isfinite(opp_y):
            samples.append({
                "iloc": s + local,
                "distance_m": None,
                "signed_long_gap_m": None,
                "lateral_offset_m": None,
                "note": "opponent slot empty at this iloc",
            })
            continue
        samples.append({
            "iloc": s + local,
            "distance_m": float(np.sqrt((opp_x - player_x[local]) ** 2 + (opp_y - player_y[local]) ** 2)),
            "signed_long_gap_m": float(signed_long_all[local]),
            "lateral_offset_m": float(lateral_all[local]),
        })

    return _attach({
        "range": [s, e],
        "slot": slot_int,
        "data_available": True,
        "coordinate_frame": frame_name,
        "samples": samples,
    })


def classify_opponent_interaction(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    close_distance_m: float = 12.0,
    side_by_side_distance_m: float = 8.0,
    longitudinal_window_m: float = 18.0,
    pass_margin_m: float = 1.5,
    min_role_gain_m: float = 4.0,
    min_active_fraction: float = 0.3,
    max_candidates: int = 5,
):
    """Tool — deterministic O / OD / MSR interaction classifier.

    Computes opponent-relative position math over ``[start_index, end_index)``
    and returns a verdict the LLM can use as the gate for opponent-aware
    labels:

      * ``pass_completed`` -> O
      * ``held_defense`` -> OD
      * ``failed_attack`` or ``broken_defense`` -> MSR

    Signed longitudinal and lateral gaps are computed in an expert-path
    projection frame when expert positions are available; otherwise the
    classifier falls back to the player's instantaneous heading frame. The
    classifier intentionally reads only positional relationship and outcome.
    Player trace technique (late brake, inside cover, switchback, defensive
    lift) still comes from the normal graphs / queries.
    """
    from app.local_annotation_agent.evaluators import PipelineAttachment
    from app.domain.telemetry import MAX_CARS

    def _attach(content: Dict[str, Any]) -> "PipelineAttachment":
        return PipelineAttachment(
            name="opponent_interaction_classification",
            kind="structured",
            label="Opponent Interaction Classification (O / OD / MSR gate)",
            content=_round_floats(content),
        )

    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def _confidence(
        *,
        outcome: str,
        min_distance: float,
        side_count: int,
        n_rows: int,
        gap_delta: float,
    ) -> float:
        close_score = 0.0
        if min_distance <= side_by_side_distance_m:
            close_score = 0.18
        elif min_distance <= close_distance_m:
            close_score = 0.12
        side_score = min(0.15, (side_count / max(1, n_rows)) * 0.8)
        gain_score = min(0.15, abs(gap_delta) / 30.0 * 0.15)
        base = {
            "pass_completed": 0.66,
            "broken_defense": 0.66,
            "failed_attack": 0.48,
            "held_defense": 0.48,
            "side_by_side": 0.34,
            "incidental": 0.22,
        }.get(outcome, 0.0)
        return _clamp01(base + close_score + side_score + gain_score)

    def _confidence_level(confidence: float) -> str:
        if confidence >= 0.82:
            return "high"
        if confidence >= 0.68:
            return "medium"
        if confidence >= 0.50:
            return "low"
        return "weak"

    s, e = int(start_index), int(end_index)
    base_payload: Dict[str, Any] = {
        "range": [s, e],
        "data_available": False,
        "role": "unknown",
        "outcome": "no_data",
        "recommended_label": None,
        "confidence": 0.0,
        "confidence_level": "weak",
        "primary_slot_for_role": None,
        "gates": {"O": False, "OD": False, "MSR": False},
        "label_gates": {"O": False, "OD": False, "MSR": False},
        "candidates": [],
        "confidence_policy": {
            "high": "strong deterministic evidence; use as label gate when player-trace evidence agrees",
            "medium": "usable deterministic evidence; cite supporting graph/query evidence",
            "low": "weak label evidence; refine range or inspect opponent trajectory before labeling",
            "weak": "do not label from classifier alone",
        },
    }

    required = {"Graphics_player_pos_x", "Graphics_player_pos_y"}
    if not required.issubset(df.columns):
        base_payload["message"] = (
            "Player position columns (Graphics_player_pos_x/y) missing — "
            "cannot classify opponent interaction."
        )
        return _attach(base_payload)

    seg = df.iloc[s:e]
    n_rows = len(seg)
    if n_rows < 2:
        base_payload["message"] = "Range too short for opponent classification (need >= 2 rows)."
        return _attach(base_payload)

    player_x = seg["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = seg["Graphics_player_pos_y"].to_numpy(dtype=float)
    if not (np.isfinite(player_x).any() and np.isfinite(player_y).any()):
        base_payload["message"] = "Player position trace is all NaN/inf."
        return _attach(base_payload)

    candidates: List[Dict[str, Any]] = []
    n_active_slots = 0

    for slot in range(1, MAX_CARS + 1):
        col_x = f"Car_{slot}_pos_x"
        col_y = f"Car_{slot}_pos_y"
        if col_x not in df.columns or col_y not in df.columns:
            continue

        ox = seg[col_x].to_numpy(dtype=float)
        oy = seg[col_y].to_numpy(dtype=float)
        active_mask = _active_opponent_mask(seg, slot, ox, oy, player_x, player_y)
        active_count = int(active_mask.sum())
        if active_count == 0:
            continue
        n_active_slots += 1
        active_fraction = active_count / n_rows
        if active_fraction < min_active_fraction:
            continue

        vx = np.where(active_mask, ox - player_x, np.nan)
        vy = np.where(active_mask, oy - player_y, np.nan)
        distance = np.sqrt(vx * vx + vy * vy)
        signed_long, lateral_signed, player_s, player_d, frame_name = _relative_position_frame(
            seg, player_x, player_y, ox, oy,
        )
        signed_long = np.where(active_mask, signed_long, np.nan)
        lateral_signed = np.where(active_mask, lateral_signed, np.nan)
        lateral_abs = np.abs(lateral_signed)
        finite = active_mask & np.isfinite(distance) & np.isfinite(signed_long) & np.isfinite(lateral_abs)
        if not finite.any():
            continue

        active_ilocs = np.where(finite)[0]
        entry_idx = int(active_ilocs[0])
        exit_idx = int(active_ilocs[-1])
        entry_long = float(signed_long[entry_idx])
        exit_long = float(signed_long[exit_idx])
        gap_delta = exit_long - entry_long

        min_d_local = int(np.nanargmin(distance))
        min_lat_local = int(np.nanargmin(lateral_abs))
        min_abs_long_local = int(np.nanargmin(np.abs(signed_long)))

        broad_side_by_side = (
            (distance <= side_by_side_distance_m)
            | ((lateral_abs <= side_by_side_distance_m) & (np.abs(signed_long) <= longitudinal_window_m))
        ) & finite
        side_count = int(broad_side_by_side.sum())
        close_enough = bool((float(distance[min_d_local]) <= close_distance_m) or side_count > 0)

        passed_by_player = bool(entry_long > pass_margin_m and exit_long < -pass_margin_m)
        got_passed_by_opponent = bool(entry_long < -pass_margin_m and exit_long > pass_margin_m)
        attack_pressure = bool(
            entry_long > pass_margin_m
            and close_enough
            and not passed_by_player
            and not got_passed_by_opponent
            and (
                gap_delta <= -min_role_gain_m
                or side_count > 0
                or abs(float(signed_long[min_abs_long_local])) <= longitudinal_window_m
            )
        )
        defense_pressure = bool(
            close_enough
            and not got_passed_by_opponent
            and exit_long <= pass_margin_m
            and (
                entry_long < -pass_margin_m
                or abs(entry_long) <= longitudinal_window_m
                or side_count > 0
            )
            and (
                gap_delta >= min_role_gain_m
                or side_count > 0
                or abs(float(signed_long[min_abs_long_local])) <= longitudinal_window_m
            )
        )

        if passed_by_player:
            role = "attack"
            outcome = "pass_completed"
            recommended = "O"
            reason = "opponent starts ahead and ends behind the player"
        elif got_passed_by_opponent:
            role = "defense"
            outcome = "broken_defense"
            recommended = "MSR"
            reason = "opponent starts behind and ends ahead of the player"
        elif attack_pressure:
            role = "attack"
            outcome = "failed_attack"
            recommended = "MSR"
            reason = "opponent remained ahead, but the player closed or went side-by-side"
        elif defense_pressure:
            role = "defense"
            outcome = "held_defense"
            recommended = "OD"
            reason = "opponent threatened from behind/alongside but did not get ahead by exit"
        elif close_enough and side_count > 0:
            role = "side_by_side"
            outcome = "side_by_side"
            recommended = None
            reason = "cars were close/alongside without a clear attack or defense outcome"
        elif close_enough:
            role = "incidental"
            outcome = "incidental"
            recommended = None
            reason = "nearby car did not create a clear position-change or pressure pattern"
        else:
            role = "none"
            outcome = "no_close_interaction"
            recommended = None
            reason = "opponent was not close enough to gate O / OD / MSR"

        confidence = _confidence(
            outcome=outcome,
            min_distance=float(distance[min_d_local]),
            side_count=side_count,
            n_rows=n_rows,
            gap_delta=gap_delta,
        )
        candidates.append({
            "slot": int(slot),
            "role": role,
            "outcome": outcome,
            "recommended_label": recommended,
            "confidence": confidence,
            "confidence_level": _confidence_level(confidence),
            "reason": reason,
            "min_distance_m": float(distance[min_d_local]),
            "min_distance_iloc": s + min_d_local,
            "entry_distance_m": float(distance[entry_idx]),
            "exit_distance_m": float(distance[exit_idx]),
            "entry_signed_long_gap_m": entry_long,
            "exit_signed_long_gap_m": exit_long,
            "gap_delta_m": gap_delta,
            "min_abs_signed_long_gap_m": float(abs(signed_long[min_abs_long_local])),
            "player_progress_m_at_entry": float(player_s[entry_idx]),
            "player_progress_m_at_exit": float(player_s[exit_idx]),
            "player_lateral_offset_m_at_entry": float(player_d[entry_idx]),
            "player_lateral_offset_m_at_exit": float(player_d[exit_idx]),
            "min_lateral_offset_m": float(lateral_abs[min_lat_local]),
            "min_lateral_offset_iloc": s + min_lat_local,
            "side_by_side_iloc_count": side_count,
            "active_iloc_fraction": float(active_fraction),
            "coordinate_frame": frame_name,
            "passed_by_player": passed_by_player,
            "got_passed_by_opponent": got_passed_by_opponent,
        })

    priority = {
        "pass_completed": 6,
        "broken_defense": 6,
        "failed_attack": 5,
        "held_defense": 5,
        "side_by_side": 3,
        "incidental": 2,
        "no_close_interaction": 1,
    }
    candidates.sort(
        key=lambda c: (
            priority.get(str(c["outcome"]), 0),
            float(c["confidence"]),
            -float(c["min_distance_m"]),
        ),
        reverse=True,
    )
    top = candidates[0] if candidates else None

    if top is None:
        return _attach({
            **base_payload,
            "data_available": True,
            "n_active_slots": n_active_slots,
            "outcome": "no_close_interaction",
            "message": "No active opponent slot met the active-fraction threshold.",
        })

    confidence_ok = top["confidence_level"] in {"high", "medium"}
    return _attach({
        "range": [s, e],
        "data_available": True,
        "n_active_slots": n_active_slots,
        "role": top["role"],
        "outcome": top["outcome"],
        "recommended_label": top["recommended_label"],
        "confidence": top["confidence"],
        "confidence_level": top["confidence_level"],
        "primary_slot_for_role": top["slot"],
        "coordinate_frame": top["coordinate_frame"],
        "reason": top["reason"],
        "confidence_policy": base_payload["confidence_policy"],
        "gates": {
            "O": top["outcome"] == "pass_completed",
            "OD": top["outcome"] == "held_defense",
            "MSR": top["outcome"] in {"failed_attack", "broken_defense"},
        },
        "label_gates": {
            "O": top["outcome"] == "pass_completed" and confidence_ok,
            "OD": top["outcome"] == "held_defense" and confidence_ok,
            "MSR": top["outcome"] in {"failed_attack", "broken_defense"} and confidence_ok,
        },
        "candidates": candidates[:max_candidates],
    })


NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"


def _lap_boundary_offsets(
    pos: np.ndarray,
    completed_laps: Optional[np.ndarray] = None,
) -> List[int]:
    """Offsets where a picked range crosses into a new lap."""
    boundaries: List[int] = []
    for offset in range(1, int(pos.size)):
        if completed_laps is not None:
            prev_lap = completed_laps[offset - 1]
            cur_lap = completed_laps[offset]
            if np.isfinite(prev_lap) and np.isfinite(cur_lap) and cur_lap != prev_lap:
                boundaries.append(offset)
                continue
        prev_p = pos[offset - 1]
        cur_p = pos[offset]
        if np.isfinite(prev_p) and np.isfinite(cur_p) and (prev_p - cur_p) > 0.5:
            boundaries.append(offset)
    return boundaries


def _split_interaction_windows_at_boundaries(
    windows: List[Dict[str, Any]],
    *,
    boundaries: List[int],
    start_index: int,
    end_index: int,
) -> List[Dict[str, Any]]:
    """Split interaction windows at lap boundaries without changing event math."""
    if not windows or not boundaries:
        return windows

    cuts = [int(start_index), *sorted(int(b) for b in boundaries), int(end_index)]
    split: List[Dict[str, Any]] = []
    for window in windows:
        window_start = int(window["start_index"])
        window_end = int(window["end_index"])
        for cut_start, cut_end in zip(cuts, cuts[1:]):
            clip_start = max(window_start, cut_start)
            clip_end = min(window_end, cut_end)
            if clip_end <= clip_start:
                continue
            clipped = dict(window)
            clipped["start_index"] = int(clip_start)
            clipped["end_index"] = int(clip_end)
            clipped["source_window_range"] = [window_start, window_end]
            split.append(clipped)
    return split


def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive ``(start, end)`` runs where mask is true."""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(mask[j]):
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _merge_close_ranges(
    ranges: List[Tuple[int, int]],
    *,
    max_gap: int,
) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged: List[Tuple[int, int]] = [ranges[0]]
    for s, e in ranges[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def _detect_opponent_interaction_windows(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    close_distance_m: float = 12.0,
    side_by_side_distance_m: float = 8.0,
    longitudinal_window_m: float = 18.0,
    min_window_ilocs: int = 3,
    context_padding_ilocs: int = 8,
    merge_gap_ilocs: int = 10,
    min_active_fraction: float = 0.3,
) -> List[Dict[str, Any]]:
    """Detect opponent engagement windows for O / OD / MSR rough cuts.

    Circuit-section ranges are track geometry; overtakes and defenses are
    event geometry. This helper finds short windows where another active car
    is close enough to affect the player's line or position, then pads them
    so ``find_nearest_opponent`` can see entry/min/exit context in one range.
    """
    from app.domain.telemetry import MAX_CARS

    s, e = int(start_index), int(end_index)
    required = {"Graphics_player_pos_x", "Graphics_player_pos_y"}
    if e - s < 2 or not required.issubset(df.columns):
        return []

    seg = df.iloc[s:e]
    n_rows = len(seg)
    if n_rows < 2:
        return []

    player_x = seg["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = seg["Graphics_player_pos_y"].to_numpy(dtype=float)
    if not (np.isfinite(player_x).any() and np.isfinite(player_y).any()):
        return []

    windows: List[Dict[str, Any]] = []

    for slot in range(1, MAX_CARS + 1):
        col_x = f"Car_{slot}_pos_x"
        col_y = f"Car_{slot}_pos_y"
        if col_x not in df.columns or col_y not in df.columns:
            continue

        ox = seg[col_x].to_numpy(dtype=float)
        oy = seg[col_y].to_numpy(dtype=float)
        active_mask = _active_opponent_mask(seg, slot, ox, oy, player_x, player_y)
        active_count = int(active_mask.sum())
        if active_count == 0 or (active_count / n_rows) < min_active_fraction:
            continue

        vx = np.where(active_mask, ox - player_x, np.nan)
        vy = np.where(active_mask, oy - player_y, np.nan)
        distance = np.sqrt(vx * vx + vy * vy)
        signed_long, lateral_signed, _player_s, _player_d, frame_name = _relative_position_frame(
            seg, player_x, player_y, ox, oy,
        )
        signed_long = np.where(active_mask, signed_long, np.nan)
        lateral_signed = np.where(active_mask, lateral_signed, np.nan)
        lateral_abs = np.abs(lateral_signed)
        finite = active_mask & np.isfinite(distance) & np.isfinite(signed_long) & np.isfinite(lateral_abs)
        if not finite.any():
            continue

        side_by_side = (
            (distance <= side_by_side_distance_m)
            | ((lateral_abs <= side_by_side_distance_m) & (np.abs(signed_long) <= longitudinal_window_m))
        )
        close = (distance <= close_distance_m) | side_by_side
        close = close & finite

        runs = _merge_close_ranges(
            _contiguous_true_runs(close),
            max_gap=int(merge_gap_ilocs),
        )
        for local_start, local_end in runs:
            padded_start = max(0, local_start - int(context_padding_ilocs))
            padded_end = min(n_rows, local_end + int(context_padding_ilocs))
            if padded_end - padded_start < int(min_window_ilocs):
                continue

            window_slice = slice(padded_start, padded_end)
            finite_window = finite[window_slice]
            if not finite_window.any():
                continue
            local_indices = np.where(finite_window)[0] + padded_start
            entry_idx = int(local_indices[0])
            exit_idx = int(local_indices[-1])
            min_d_idx = padded_start + int(np.nanargmin(distance[window_slice]))
            min_lat_idx = padded_start + int(np.nanargmin(lateral_abs[window_slice]))

            side_count = int(((distance[window_slice] <= side_by_side_distance_m) & finite_window).sum())
            entry_long = float(signed_long[entry_idx])
            exit_long = float(signed_long[exit_idx])
            windows.append({
                "start_index": int(s + padded_start),
                "end_index": int(s + padded_end),
                "slot": int(slot),
                "min_distance_m": float(distance[min_d_idx]),
                "min_distance_iloc": int(s + min_d_idx),
                "entry_signed_long_gap_m": entry_long,
                "exit_signed_long_gap_m": exit_long,
                "min_lateral_offset_m": float(lateral_abs[min_lat_idx]),
                "side_by_side_iloc_count": side_count,
                "coordinate_frame": frame_name,
                "passed_by_player": bool(entry_long > 0 and exit_long < 0),
                "got_passed_by_opponent": bool(entry_long < 0 and exit_long > 0),
            })

    # Merge overlapping windows, keeping the closest-slot summary as the
    # representative hint. The full per-slot details remain available through
    # find_nearest_opponent once the agent inspects the merged range.
    windows.sort(key=lambda w: (int(w["start_index"]), int(w["end_index"])))
    merged_windows: List[Dict[str, Any]] = []
    for window in windows:
        if not merged_windows or int(window["start_index"]) > int(merged_windows[-1]["end_index"]) + merge_gap_ilocs:
            merged_windows.append(dict(window))
            continue
        current = merged_windows[-1]
        current["start_index"] = min(int(current["start_index"]), int(window["start_index"]))
        current["end_index"] = max(int(current["end_index"]), int(window["end_index"]))
        current["slots"] = sorted(set(current.get("slots", [current["slot"]]) + [window["slot"]]))
        if float(window["min_distance_m"]) < float(current["min_distance_m"]):
            for key in (
                "slot", "min_distance_m", "min_distance_iloc",
                "entry_signed_long_gap_m", "exit_signed_long_gap_m",
                "min_lateral_offset_m", "side_by_side_iloc_count",
                "coordinate_frame", "passed_by_player", "got_passed_by_opponent",
            ):
                current[key] = window[key]
        else:
            current["passed_by_player"] = bool(current.get("passed_by_player")) or bool(window.get("passed_by_player"))
            current["got_passed_by_opponent"] = bool(current.get("got_passed_by_opponent")) or bool(window.get("got_passed_by_opponent"))

    return merged_windows


def _has_active_opponent_data(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> bool:
    """True when the slice contains player + at least one other car slot.

    ``Car_{1..MAX_CARS}`` includes the player's own car, so one active slot is
    still a solo/practice slice. Two or more active slots means the session has
    opponent telemetry available.
    """
    from app.domain.telemetry import MAX_CARS

    seg = df.iloc[int(start_index): int(end_index)]
    if seg.empty:
        return False

    active_slots = 0
    for slot in range(1, MAX_CARS + 1):
        col_x = f"Car_{slot}_pos_x"
        col_y = f"Car_{slot}_pos_y"
        if col_x not in df.columns or col_y not in df.columns:
            continue
        ox = seg[col_x].to_numpy(dtype=float)
        oy = seg[col_y].to_numpy(dtype=float)
        active = ((ox != 0.0) | (oy != 0.0)) & np.isfinite(ox) & np.isfinite(oy)
        if bool(active.any()):
            active_slots += 1
            if active_slots >= 2:
                return True
    return False


def split_lap_by_circuit_sections(
    df: pd.DataFrame, start_index: int, end_index: int,
    circuit_id: Optional[str] = None,
    include_interaction_windows: bool = True,
    interactions_only_when_opponents: bool = True,
):
    """Tool — partition a lap-shaped range into annotation sub-ranges.

    Walks ``Graphics_normalized_car_position`` sample-by-sample across
    ``[start_index, end_index)`` and assigns every iloc to the
    ``circuit_section`` whose ``normalized_position_range`` contains its
    position fraction. Consecutive ilocs that land in the same section are
    grouped into one sub-range.

    When ``include_interaction_windows`` is true and opponent samples are
    present, the function switches to opponent-interaction mode by default:
    it returns close engagement windows for overtake offence / defence
    annotation. Those windows are event-shaped and are not split by circuit
    section; overlapping sections are attached only as context. Normal
    circuit-section work units are returned only for solo sessions. This
    keeps opponent sessions from producing EA / MSP / RM practice-driving
    sections. Wrap-around sections (``range_end < range_start``) and lap
    roll-over (a sample where position resets 1.0 → 0.0) are handled.

    Parameters
    ----------
    df : pandas.DataFrame
        Session telemetry; must contain ``Graphics_normalized_car_position``.
    start_index, end_index : int
        Inclusive-exclusive parent slice. The caller picks any contiguous
        range — the function does not assume a full lap.
    circuit_id : str, optional
        Restrict matching to sections whose ``parent`` equals this circuit
        (e.g. ``"brands_hatch"``). When ``None``, every ``circuit_section``
        with a filled range competes.

    Returns
    -------
    PipelineAttachment ``split_lap_sections`` whose ``content`` is::

        {
            "circuit_id": <str | None>,
            "range": [start_index, end_index],
            "opponent_session": bool,
            "split_mode": "circuit_sections" | "opponent_interactions_only",
            "segments": [
                {
                    "start_index": int,
                    "end_index": int,
                    "circuit_section_id": str,  # interaction_window for racing events
                    "circuit_section_name": str,
                    "normalized_position_range": [float, float] | null,
                    "coverage_fraction": 0.0..1.0,
                    "split_basis": "circuit_section" |
                                   "opponent_interaction",
                    "opponent_interaction": {...} | null,
                },
                ...
            ],
            "interaction_windows": [...],
            "unmatched_ilocs": int,   # samples that hit no defined section
        }

    Segments are ordered by ``start_index``. ``coverage_fraction`` is the
    share of the segment's iloc span that fell inside the matched section's
    range — informative when a section's range is narrow and the player
    crossed in/out at the boundary.
    """
    from app.local_annotation_agent.evaluators import PipelineAttachment
    from app.internal_knowledge_base.label_lookup import find_labels

    s, e = int(start_index), int(end_index)

    def _attach(content: Dict[str, Any]) -> "PipelineAttachment":
        return PipelineAttachment(
            name="split_lap_sections",
            kind="structured",
            label="Lap Split by Circuit Section",
            content=_round_floats(content, ndigits=4),
        )

    if NORMALIZED_POSITION_COLUMN not in df.columns:
        return _attach({
            "error": f"column '{NORMALIZED_POSITION_COLUMN}' missing from telemetry",
            "circuit_id": circuit_id,
            "range": [s, e],
            "segments": [],
            "unmatched_ilocs": 0,
        })

    pos = df.iloc[s:e][NORMALIZED_POSITION_COLUMN].to_numpy(dtype=float)
    if pos.size == 0:
        return _attach({
            "error": "empty slice",
            "circuit_id": circuit_id,
            "range": [s, e],
            "segments": [],
            "unmatched_ilocs": 0,
        })

    completed_laps: Optional[np.ndarray] = None
    if "Graphics_completed_lap" in df.columns:
        completed_laps = df.iloc[s:e]["Graphics_completed_lap"].to_numpy(dtype=float)
    lap_boundary_ilocs = [
        int(s + offset)
        for offset in _lap_boundary_offsets(pos, completed_laps)
    ]
    lap_boundary_set = set(lap_boundary_ilocs)

    interaction_windows: List[Dict[str, Any]] = []
    opponent_session = False
    if include_interaction_windows:
        opponent_session = _has_active_opponent_data(df, s, e)
        interaction_windows = _detect_opponent_interaction_windows(df, s, e)
        interaction_windows = _split_interaction_windows_at_boundaries(
            interaction_windows,
            boundaries=lap_boundary_ilocs,
            start_index=s,
            end_index=e,
        )

    section_filter: Dict[str, Any] = {"type": "circuit_section"}
    if circuit_id is not None:
        section_filter["parent"] = circuit_id
    candidates: List[Dict[str, Any]] = []
    for entry in find_labels(**section_filter):
        rng = entry.get("normalized_position_range")
        if rng is None:
            continue
        candidates.append({
            "id": entry["id"],
            "name": entry["name"],
            "lo": float(rng[0]),
            "hi": float(rng[1]),
        })

    def _interaction_segments_from_windows(
        windows: List[Dict[str, Any]],
        anchors: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        interaction_segments: List[Dict[str, Any]] = []
        for window in windows:
            window_start = int(window["start_index"])
            window_end = int(window["end_index"])
            section_context: List[Dict[str, Any]] = []
            for segment in anchors:
                seg_s = int(segment["start_index"])
                seg_e = int(segment["end_index"])
                overlap_start = max(window_start, seg_s, s)
                overlap_end = min(window_end, seg_e, e)
                if overlap_end <= overlap_start:
                    continue
                section_context.append({
                    "circuit_section_id": segment["circuit_section_id"],
                    "circuit_section_name": segment["circuit_section_name"],
                    "range": [int(overlap_start), int(overlap_end)],
                    "normalized_position_range": segment["normalized_position_range"],
                })

            interaction_segments.append({
                "start_index": int(max(s, window_start)),
                "end_index": int(min(e, window_end)),
                "circuit_section_id": "interaction_window",
                "circuit_section_name": "Racing interaction",
                "normalized_position_range": None,
                "coverage_fraction": 1.0,
                "split_basis": "opponent_interaction",
                "opponent_interaction": {
                    "windows": [window],
                    "section_context": section_context,
                    "reason": (
                        "opponent session: close overtake offence / defence "
                        "engagement emitted as an event-shaped window; "
                        "circuit sections are metadata only"
                    ),
                },
            })
        return interaction_segments

    if not candidates:
        if opponent_session and interactions_only_when_opponents and interaction_windows:
            return _attach({
                "circuit_id": circuit_id,
                "range": [s, e],
                "opponent_session": True,
                "split_mode": "opponent_interactions_only",
                "segments": _interaction_segments_from_windows(interaction_windows, []),
                "interaction_windows": interaction_windows,
                "unmatched_ilocs": int(pos.size),
                "warning": (
                    "no measured circuit_section ranges matched; emitted "
                    "racing interaction work units anchored to the circuit"
                ),
            })
        return _attach({
            "error": (
                f"no circuit_section with a filled normalized_position_range "
                f"matches circuit_id={circuit_id!r}"
            ),
            "circuit_id": circuit_id,
            "range": [s, e],
            "segments": [],
            "unmatched_ilocs": 0,
        })

    def _section_for(p: float) -> Optional[Dict[str, Any]]:
        if not np.isfinite(p):
            return None
        # Wrap p into [0, 1) defensively — some sessions emit values that drift
        # slightly past the boundary at the start/finish line.
        p = p - np.floor(p)
        for c in candidates:
            lo, hi = c["lo"], c["hi"]
            if hi >= lo:
                if lo <= p <= hi:
                    return c
            else:
                # Wrap section: [lo, 1.0] ∪ [0.0, hi]
                if p >= lo or p <= hi:
                    return c
        return None

    segments: List[Dict[str, Any]] = []
    unmatched = 0
    cur_section: Optional[Dict[str, Any]] = None
    cur_start_iloc = s
    matched_in_run = 0
    def _is_lap_boundary(offset: int) -> bool:
        return int(s + offset) in lap_boundary_set

    def _close_run(end_iloc_exclusive: int) -> None:
        nonlocal cur_section, cur_start_iloc, matched_in_run
        if cur_section is not None and end_iloc_exclusive > cur_start_iloc:
            length = end_iloc_exclusive - cur_start_iloc
            segments.append({
                "start_index": int(cur_start_iloc),
                "end_index": int(end_iloc_exclusive),
                "circuit_section_id": cur_section["id"],
                "circuit_section_name": cur_section["name"],
                "normalized_position_range": [cur_section["lo"], cur_section["hi"]],
                "coverage_fraction": float(matched_in_run) / float(length) if length else 0.0,
                "split_basis": "circuit_section",
                "opponent_interaction": None,
            })
        cur_section = None
        cur_start_iloc = end_iloc_exclusive
        matched_in_run = 0

    for offset, p in enumerate(pos):
        iloc = s + offset
        section = _section_for(p)
        lap_boundary = _is_lap_boundary(offset)
        if section is None:
            unmatched += 1
            # Keep the run open — the player may have a noisy sample.
            if cur_section is None:
                cur_start_iloc = iloc + 1
            continue
        if lap_boundary and cur_section is not None:
            _close_run(iloc)
        if cur_section is None:
            cur_section = section
            cur_start_iloc = iloc
            matched_in_run = 1
            continue
        if section["id"] == cur_section["id"]:
            matched_in_run += 1
            continue
        # Section changed — close the prior run, start a new one.
        _close_run(iloc)
        cur_section = section
        cur_start_iloc = iloc
        matched_in_run = 1
    _close_run(e)

    if include_interaction_windows:
        if opponent_session and interactions_only_when_opponents:
            segments = _interaction_segments_from_windows(interaction_windows, segments)
        elif interaction_windows and segments:
            assignments: Dict[int, List[Dict[str, Any]]] = {}
            for window in interaction_windows:
                best_idx = None
                best_score = (-1, -1)
                closest_iloc = int(window.get("min_distance_iloc", window["start_index"]))
                for idx, segment in enumerate(segments):
                    seg_s = int(segment["start_index"])
                    seg_e = int(segment["end_index"])
                    overlap = min(int(window["end_index"]), seg_e) - max(int(window["start_index"]), seg_s)
                    if overlap <= 0:
                        continue
                    contains_closest = 1 if seg_s <= closest_iloc < seg_e else 0
                    score = (contains_closest, overlap)
                    if score > best_score:
                        best_idx = idx
                        best_score = score
                if best_idx is not None:
                    assignments.setdefault(best_idx, []).append(window)

            for idx, overlaps in assignments.items():
                segment = segments[idx]
                expanded_start = min(int(segment["start_index"]), *(int(w["start_index"]) for w in overlaps))
                expanded_end = max(int(segment["end_index"]), *(int(w["end_index"]) for w in overlaps))
                segment["start_index"] = int(max(s, expanded_start))
                segment["end_index"] = int(min(e, expanded_end))
                segment["split_basis"] = "circuit_section+opponent_interaction"
                segment["opponent_interaction"] = {
                    "windows": overlaps,
                    "reason": (
                        "section window expanded around close opponent "
                        "engagement so O / OD / MSR labels see complete "
                        "entry-to-exit context"
                    ),
                }

    return _attach({
        "circuit_id": circuit_id,
        "range": [s, e],
        "opponent_session": bool(opponent_session),
        "split_mode": (
            "opponent_interactions_only"
            if opponent_session and include_interaction_windows and interactions_only_when_opponents
            else "circuit_sections"
        ),
        "segments": segments,
        "interaction_windows": interaction_windows,
        "unmatched_ilocs": int(unmatched),
    })


def locate_circuit_section(
    df: pd.DataFrame, start_index: int, end_index: int,
):
    """Tool — identify which named ``circuit_section`` the segment overlaps.

    Reads ``Graphics_normalized_car_position`` over ``[start_index, end_index)``
    and compares the segment's [min, max] fraction against every
    ``circuit_section`` label whose ``normalized_position_range`` is filled
    in. Returns the top matches ranked by overlap fraction (overlap / segment
    span). Handles wrap-around sections where ``range_end < range_start``
    (e.g. the pit straight that crosses the start/finish line).

    Returns a ``circuit_section_match`` attachment with shape::

        {
            "segment_position_range": [seg_min, seg_max],
            "top_matches": [
                {"label_id", "name", "description",
                 "section_range": [start, end],
                 "overlap_fraction": 0.0..1.0},
                ...
            ],
            "is_ambiguous": <bool>,
            "best_match": <top entry, or None when ambiguous / empty>,
        }

    ``top_matches`` is empty when the column is missing, all values are
    non-finite, or no circuit_section in the catalog has its range filled
    in yet. ``best_match`` is set ONLY when the leading entry's
    ``overlap_fraction`` clears the runner-up by ``AMBIGUOUS_MARGIN``;
    otherwise ``is_ambiguous`` is true and the caller must disambiguate
    using a second telemetry signal (pit-limiter speed, lateral offset,
    brake pattern, etc.).
    """
    from app.local_annotation_agent.evaluators import PipelineAttachment
    from app.internal_knowledge_base.label_lookup import find_labels

    s, e = int(start_index), int(end_index)

    def _attach(content: Dict[str, Any]) -> "PipelineAttachment":
        return PipelineAttachment(
            name="circuit_section_match",
            kind="structured",
            label="Circuit Section Match",
            content=_round_floats(content, ndigits=4),
        )

    if NORMALIZED_POSITION_COLUMN not in df.columns:
        return _attach({
            "error": f"column '{NORMALIZED_POSITION_COLUMN}' missing from telemetry",
            "top_matches": [],
            "is_ambiguous": False,
            "best_match": None,
        })

    pos = df.iloc[s:e][NORMALIZED_POSITION_COLUMN].to_numpy(dtype=float)
    pos = pos[np.isfinite(pos)]
    if pos.size == 0:
        return _attach({
            "error": "no finite values in normalized position over the segment",
            "top_matches": [],
            "is_ambiguous": False,
            "best_match": None,
        })

    seg_lo, seg_hi = float(pos.min()), float(pos.max())
    seg_span = max(seg_hi - seg_lo, 1e-6)

    matches: List[Dict[str, Any]] = []
    for entry in find_labels(type="circuit_section"):
        rng = entry.get("normalized_position_range")
        if rng is None:
            continue
        r_lo, r_hi = rng
        if r_hi >= r_lo:
            overlap = max(0.0, min(seg_hi, r_hi) - max(seg_lo, r_lo))
        else:
            # Section wraps across the lap boundary: [r_lo, 1.0] ∪ [0.0, r_hi]
            overlap = (
                max(0.0, min(seg_hi, 1.0) - max(seg_lo, r_lo))
                + max(0.0, min(seg_hi, r_hi) - max(seg_lo, 0.0))
            )
        if overlap <= 0:
            continue
        matches.append({
            "label_id": entry["id"],
            "name": entry["name"],
            "description": entry.get("description", ""),
            "section_range": [r_lo, r_hi],
            "overlap_fraction": overlap / seg_span,
        })

    matches.sort(key=lambda m: m["overlap_fraction"], reverse=True)
    top = matches[:3]

    # Pit lanes and adjacent straights frequently share a normalized_position
    # range (e.g. brands_hatch1 and brands_hatch17 both at [0.94, 1]), so the
    # leader can tie the runner-up at the same overlap_fraction. Refuse to
    # name a single best_match in that case — force the caller to pick.
    AMBIGUOUS_MARGIN = 0.05
    is_ambiguous = (
        len(top) >= 2
        and (top[0]["overlap_fraction"] - top[1]["overlap_fraction"]) < AMBIGUOUS_MARGIN
    )
    best = None if is_ambiguous or not top else top[0]
    return _attach({
        "segment_position_range": [seg_lo, seg_hi],
        "top_matches": top,
        "is_ambiguous": is_ambiguous,
        "best_match": best,
    })


STATIC_TRACK_COLUMN = "Static_track"


def _canonicalise_track_name(raw: Any) -> Optional[str]:
    """Normalise a Static_track value to a catalog circuit id."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return text.lower().replace(" ", "_").replace("-", "_")


def get_circuit_id(df: pd.DataFrame):
    """Tool — derive the lap's circuit id from the ``Static_track`` column.

    Reads ``df[Static_track].iloc[0]`` and canonicalises it (lowercase,
    spaces / hyphens to underscores) so it matches catalog circuit IDs
    like ``brands_hatch`` / ``silverstone``. Returns a ``circuit_id``
    attachment with::

        {
            "circuit_id": <str | None>,
            "circuit_name": <str | None>,   # display name from CIRCUIT_NAMES
            "raw_track_name": <str | None>,
            "known": <bool>,    # True when canonical id matches a catalog circuit
        }

    Returns ``circuit_id=None`` when the column is missing, the dataframe
    is empty, or the value canonicalises to an empty string. ``known`` is
    informational only — the agent decides whether to proceed when an
    unknown circuit slips through.
    """
    from app.local_annotation_agent.evaluators import PipelineAttachment
    from app.domain.circuits import CIRCUIT_NAMES

    def _attach(content: Dict[str, Any]) -> "PipelineAttachment":
        return PipelineAttachment(
            name="circuit_id",
            kind="structured",
            label="Circuit ID (from Static_track)",
            content=content,
        )

    if STATIC_TRACK_COLUMN not in df.columns or df.empty:
        return _attach({
            "circuit_id": None,
            "circuit_name": None,
            "raw_track_name": None,
            "known": False,
            "error": f"column '{STATIC_TRACK_COLUMN}' missing or dataframe empty",
        })

    raw = df[STATIC_TRACK_COLUMN].iloc[0]
    canon = _canonicalise_track_name(raw)
    return _attach({
        "circuit_id": canon,
        "circuit_name": CIRCUIT_NAMES.get(canon) if canon else None,
        "raw_track_name": None if raw is None else str(raw),
        "known": canon in CIRCUIT_NAMES if canon else False,
    })


# Catalog of pre-compute tools, mirroring AGENT_GRAPH_DEFINITIONS. The
# planner enumerates this in its prompt; the step solver dispatches by id.
PIPELINE_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "compute_expert_phases",
        "label": "Phase Indices (per-arc entry / apex / exit)",
        "description": (
            "Detects expert-anchored corner phases from the expert "
            "position-trace curvature. Produces a 'phase_indices' "
            "attachment with a 'phases' list — one entry per arc, so "
            "chicanes / esses produce multiple entries of opposite "
            "direction. Empty list on straights. Use on any step that "
            "requests a trajectory graph or reasons about corner phases."
        ),
        "callable": compute_expert_phases,
    },
    {
        "id": "locate_circuit_section",
        "label": "Circuit Section Match (named corner / straight)",
        "description": (
            "Reads `Graphics_normalized_car_position` over the segment "
            "and matches it against every `circuit_section` label's "
            "`normalized_position_range`. Produces a "
            "'circuit_section_match' attachment with 'top_matches' "
            "(ranked by overlap fraction), an 'is_ambiguous' flag, and "
            "a 'best_match' that is non-null ONLY when the leader clears "
            "the runner-up by a clear margin. When 'is_ambiguous' is true "
            "(e.g. pit lane and the adjacent straight share a normalized "
            "position range), enumerate 'top_matches' and disambiguate "
            "with a second signal (pit-limiter speed, persistent lateral "
            "offset, brake pattern). Use whenever you need to label which "
            "named corner / straight the segment is on — never guess from "
            "telemetry shape alone."
        ),
        "callable": locate_circuit_section,
    },
    {
        "id": "split_lap_by_circuit_sections",
        "label": "Split lap range into per-section sub-ranges",
        "description": (
            "Walks `Graphics_normalized_car_position` across the parent "
            "range and partitions it into one sub-range per "
            "`circuit_section` whose `normalized_position_range` contains "
            "the sample. Close opponent engagements are assigned to the "
            "best matching section, which expands to include the padded "
            "interaction window so "
            "overtake / defense / racing-mistake labels are not clipped at "
            "corner boundaries. Produces a 'split_lap_sections' attachment with "
            "an ordered `segments` list (`start_index`, `end_index`, "
            "`circuit_section_id`, `coverage_fraction`, `split_basis`). Used by the "
            "lap-to-segment excerpter to compute the rough split that "
            "feeds the per-section annotation agent."
        ),
        "callable": split_lap_by_circuit_sections,
    },
    {
        "id": "find_nearest_opponent",
        "label": "Nearest opponent(s) in range (multi-car positional context)",
        "description": (
            "Scans `Car_{1..60}_pos_{x,y}` against `Graphics_player_pos_"
            "{x,y}` over the iloc range, filters empty slots (x=y=0.0), "
            "and ranks the active opponents by minimum 2D distance to "
            "the player. Produces an 'opponent_context' attachment whose "
            "`candidates` list carries per-slot summary: minimum / entry "
            "/ exit distance, signed longitudinal gap at entry & exit "
            "(+ ⇒ opponent ahead in player heading), minimum lateral "
            "offset, side-by-side iloc count, and `passed_by_player` / "
            "`got_passed_by_opponent` flags (signed-gap sign flips). Use "
            "after `classify_opponent_interaction` when you need "
            "supporting primary-slot details. Empty `candidates` with "
            "`data_available: true` is evidence that no car was close "
            "enough to interact with."
        ),
        "callable": find_nearest_opponent,
    },
    {
        "id": "classify_opponent_interaction",
        "label": "Classify opponent interaction outcome (O / OD / MSR gate)",
        "description": (
            "Deterministically classifies the opponent-relative position "
            "pattern over the iloc range, preferring projection onto "
            "the expert trajectory for signed longitudinal/lateral gaps "
            "and falling back to player heading when expert positions are "
            "missing. Returns a structured "
            "'opponent_interaction_classification' attachment with "
            "`role` (attack / defense / side_by_side / incidental), "
            "`outcome` (pass_completed, held_defense, failed_attack, "
            "broken_defense, etc.), `recommended_label` (O / OD / MSR / "
            "null), numeric `confidence`, readable `confidence_level`, "
            "primary opponent slot, per-slot evidence, raw outcome `gates`, "
            "and confidence-aware `label_gates`. Use this as the "
            "mathematical source of truth for O / OD / MSR eligibility "
            "before choosing labels."
        ),
        "callable": classify_opponent_interaction,
    },
    {
        "id": "query_opponent_trajectory",
        "label": "Per-iloc trajectory samples for one opponent slot",
        "description": (
            "Given a specific opponent slot (1..60) returned by "
            "`find_nearest_opponent`, samples `n_samples` evenly-spaced "
            "ilocs across the range and returns each iloc's 2D distance, "
            "signed longitudinal gap (+ ⇒ opp ahead), and signed lateral "
            "offset (+ ⇒ opp on player's left of heading). Use to see "
            "HOW the relationship evolved: gap closing smoothly on a "
            "straight (slipstream), step change at apex (switchback), or "
            "lateral offset crossing zero (line-cross during a pass)."
        ),
        "callable": query_opponent_trajectory,
    },
]


def get_pipeline_tool(tool_id: str) -> Optional[Dict[str, Any]]:
    """Lookup helper — returns the tool definition or None."""
    return next(
        (t for t in PIPELINE_TOOL_DEFINITIONS if t["id"] == tool_id),
        None,
    )


# ---------------------------------------------------------------------------
# Pipeline queries — VLM-picked, parameter-driven detectors that resolve to
# exact ilocs/values on the real DataFrame. The zoom agent's VLM call
# classifies which query fits the planner's question, picks parameters, and
# the query runs deterministic math — no pixel-reading.
# ---------------------------------------------------------------------------

def _resolve_column(column: str, segment: pd.DataFrame) -> Optional[np.ndarray]:
    if not column or column not in segment.columns:
        return None
    return segment[column].to_numpy(dtype=float)


def _query_find_extremum(
    df: pd.DataFrame, start_index: int, end_index: int,
    column: str, kind: str,
) -> Optional[Dict[str, Any]]:
    """iloc of the global min or max of <column> in the range."""
    segment = df.loc[int(start_index): int(end_index)]
    arr = _resolve_column(column, segment)
    if arr is None or len(arr) == 0:
        return None
    finite = np.where(np.isfinite(arr), arr, np.nan)
    if np.all(np.isnan(finite)):
        return None
    if kind == "max":
        local = int(np.nanargmax(finite))
    elif kind == "min":
        local = int(np.nanargmin(finite))
    else:
        return None
    return {"iloc": int(start_index) + local, "value": float(finite[local])}


def _query_find_first_match(
    df: pd.DataFrame, start_index: int, end_index: int,
    column: str, op: str, value: float,
) -> Optional[Dict[str, Any]]:
    """First iloc in the range where <column> <op> <value> holds.

    ``op`` is one of ``"equal"``, ``"greater_than_or_equal"``,
    ``"less_than_or_equal"``, ``"greater_than"``, ``"less_than"``. Useful
    for discrete-valued columns (gear, integer states) where threshold
    crossings don't make sense.
    """
    segment = df.loc[int(start_index): int(end_index)]
    arr = _resolve_column(column, segment)
    if arr is None or len(arr) == 0:
        return None
    v = float(value)
    if op == "equal":
        mask = arr == v
    elif op == "greater_than_or_equal":
        mask = arr >= v
    elif op == "less_than_or_equal":
        mask = arr <= v
    elif op == "greater_than":
        mask = arr > v
    elif op == "less_than":
        mask = arr < v
    else:
        raise ValueError(
            f"unknown op {op!r} — must be one of: equal, "
            f"greater_than_or_equal, less_than_or_equal, "
            f"greater_than, less_than"
        )
    idxs = np.where(mask & np.isfinite(arr))[0]
    if len(idxs) == 0:
        return None
    local = int(idxs[0])
    return {"iloc": int(start_index) + local, "value": float(arr[local])}


def _query_read_values_at_indices(
    df: pd.DataFrame, start_index: int, end_index: int,
    column: str, indices: List[int],
) -> Optional[Dict[str, Any]]:
    """Read <column> at each parent-frame iloc in <indices>.

    Each requested iloc resolves to one entry in ``samples`` —
    ``{iloc, value}``, with ``value=None`` for out-of-range or NaN samples
    and a ``note`` explaining why. Returns ``None`` only when the column
    itself is missing.
    """
    if not isinstance(indices, (list, tuple)) or not indices:
        return None
    segment = df.loc[int(start_index): int(end_index)]
    arr = _resolve_column(column, segment)
    if arr is None:
        return None
    samples: List[Dict[str, Any]] = []
    for raw_idx in indices:
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        local = idx - int(start_index)
        if local < 0 or local >= len(arr):
            samples.append({"iloc": idx, "value": None, "note": "out of zoom range"})
            continue
        v = arr[local]
        if not np.isfinite(v):
            samples.append({"iloc": idx, "value": None, "note": "NaN / missing"})
        else:
            samples.append({"iloc": idx, "value": float(v)})
    if not samples:
        return None
    return {"samples": samples}


def _query_compute_slope(
    df: pd.DataFrame, start_index: int, end_index: int,
    column: str,
) -> Optional[Dict[str, Any]]:
    """Slope of <column> across the zoom range.

    Returns ``{iloc, value, samples, extra}`` where ``value`` is the slope
    ``(arr[end] - arr[start]) / (end_index - start_index)``, ``samples``
    documents the two range endpoints, and ``extra`` carries the raw
    deltas. ``iloc`` is set to ``end_index`` so the synthesizer can cite
    the slope at the end of the interval. Returns ``None`` when the column
    is missing, the range collapses to a single iloc, or either endpoint
    is non-finite.
    """
    a_idx = int(start_index)
    b_idx = int(end_index)
    if a_idx == b_idx:
        return None
    segment = df.loc[a_idx: b_idx]
    arr = _resolve_column(column, segment)
    if arr is None or len(arr) < 2:
        return None
    va, vb = arr[0], arr[-1]
    if not (np.isfinite(va) and np.isfinite(vb)):
        return None
    delta_v = float(vb - va)
    delta_i = float(b_idx - a_idx)
    slope = delta_v / delta_i
    return {
        "iloc": b_idx,
        "value": slope,
        "samples": [
            {"iloc": a_idx, "value": float(va)},
            {"iloc": b_idx, "value": float(vb)},
        ],
        "extra": {
            "slope": slope,
            "delta_value": delta_v,
            "delta_iloc": delta_i,
        },
    }


def _query_find_dips_on_main_slope(
    df: pd.DataFrame, start_index: int, end_index: int,
    column: str, smoothing_window: int, min_dip_depth: float,
) -> Optional[Dict[str, Any]]:
    """Find local dips along the linear-regression baseline of <column>.

    Fits a least-squares line to (smoothed) <column> over the zoom range,
    then identifies every local minimum of the residual (signal − baseline)
    where the residual is negative AND its magnitude ≥ <min_dip_depth>.

    Each surviving local min represents one "dip on the main slope" — a
    moment where the signal briefly fell behind the overall trend. The
    iloc reported per dip is the deepest point of that dip. Endpoints
    are excluded from local-min detection.

    Returns ``samples`` — one entry per dip, ``{iloc, value, depth}``,
    sorted by iloc ascending; ``value`` is the raw (unsmoothed) signal
    at the dip; ``depth`` is the absolute residual. Returns ``None`` when
    the column is missing, has fewer than 3 finite samples, or the smoothing
    window is < 1. When the regression succeeds but no dip meets the
    threshold, returns an empty samples list with extras populated.
    """
    try:
        window = int(smoothing_window)
    except (TypeError, ValueError):
        return None
    if window < 1:
        return None
    try:
        depth_thr = float(min_dip_depth)
    except (TypeError, ValueError):
        return None

    segment = df.loc[int(start_index): int(end_index)]
    arr_raw = _resolve_column(column, segment)
    if arr_raw is None or len(arr_raw) < 3:
        return None

    smoothed = (
        pd.Series(arr_raw)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    finite_mask = np.isfinite(smoothed)
    if int(finite_mask.sum()) < 3:
        return None

    x_local = np.arange(len(smoothed), dtype=float)
    slope, intercept = np.polyfit(x_local[finite_mask], smoothed[finite_mask], 1)
    baseline = intercept + slope * x_local
    residual = smoothed - baseline

    eps = 1e-9
    if slope > eps:
        slope_direction = "rising"
    elif slope < -eps:
        slope_direction = "falling"
    else:
        slope_direction = "flat"

    samples: List[Dict[str, Any]] = []
    for i in range(1, len(residual) - 1):
        ri, rp, rn = residual[i], residual[i - 1], residual[i + 1]
        if not (np.isfinite(ri) and np.isfinite(rp) and np.isfinite(rn)):
            continue
        if not (ri < rp and ri <= rn):
            continue
        if ri >= 0:
            continue
        depth = abs(float(ri))
        if depth < depth_thr:
            continue
        raw_val = arr_raw[i]
        if not np.isfinite(raw_val):
            continue
        samples.append({
            "iloc": int(start_index) + i,
            "value": float(raw_val),
            "depth": depth,
        })

    return {
        "iloc": None,
        "value": None,
        "samples": samples,
        "extra": {
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_direction": slope_direction,
            "n_dips": len(samples),
            "smoothing_window": window,
        },
    }


def _query_find_threshold_crossing(
    df: pd.DataFrame, start_index: int, end_index: int,
    columns: List[str], threshold: float, smoothing_window: int,
) -> Optional[Dict[str, Any]]:
    """Rank <columns> by which first crosses <threshold> on a denoised signal.

    Robust-statistics pre-processing: each column is passed through a
    centered rolling-median filter of width <smoothing_window>. The
    median filter provably suppresses any spike/dip narrower than
    ``floor(smoothing_window / 2)`` samples and — unlike a moving
    average or low-pass filter — does not blur the transition edge or
    shift the detected crossing in time. Edge samples use a shrinking
    one-sided window (``min_periods=1``) instead of dropping out.

    Direction is inferred per-column from the first valid cleaned
    sample's side of <threshold>: below → first rising crossing;
    above → first falling crossing; exactly on it → no crossing
    reported. The iloc is the first cleaned sample on the new side.

    Returns ``samples`` — one entry per column, shape
    ``{ranking, column, iloc}`` — with ``ranking`` 1=first, 2=second,
    …, ``None``=never crossed (and ``iloc`` also ``None``). Returns
    ``None`` when fewer than two columns are supplied, <threshold> is
    non-numeric, or <smoothing_window> is < 1.
    """
    if not isinstance(columns, list) or len(columns) < 2:
        return None
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        return None
    try:
        window = int(smoothing_window)
    except (TypeError, ValueError):
        return None
    if window < 1:
        return None
    segment = df.loc[int(start_index): int(end_index)]

    crossings: List[Dict[str, Any]] = []
    for column in columns:
        entry: Dict[str, Any] = {"column": column, "iloc": None}
        arr = _resolve_column(column, segment)
        if arr is None or len(arr) < 2:
            crossings.append(entry)
            continue
        clean = (
            pd.Series(arr)
            .rolling(window=window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        finite_mask = np.isfinite(clean)
        if not finite_mask.any():
            crossings.append(entry)
            continue
        first_local = int(np.argmax(finite_mask))
        first_value = float(clean[first_local])
        if first_value < thr:
            on_new_side = clean >= thr
        elif first_value > thr:
            on_new_side = clean <= thr
        else:
            crossings.append(entry)
            continue

        tail = on_new_side[first_local + 1:]
        hits = np.where(tail)[0]
        if len(hits) == 0:
            crossings.append(entry)
            continue
        found_local = int(hits[0]) + first_local + 1
        entry["iloc"] = int(start_index) + found_local
        crossings.append(entry)

    with_iloc = sorted(
        (c for c in crossings if c["iloc"] is not None),
        key=lambda c: c["iloc"],
    )
    without = [c for c in crossings if c["iloc"] is None]
    samples: List[Dict[str, Any]] = []
    for ranking, c in enumerate(with_iloc, start=1):
        samples.append({"ranking": ranking, "column": c["column"], "iloc": c["iloc"]})
    for c in without:
        samples.append({"ranking": None, "column": c["column"], "iloc": None})

    return {
        "iloc": None,
        "value": None,
        "samples": samples,
        "extra": {"smoothing_window": window},
    }


_RANGE_PARAM_DESC = (
    "[start_iloc, end_iloc] — inclusive parent-frame iloc bounds this query "
    "runs over. Must lie within the question's sub-range; pick a tight window "
    "around the feature you're trying to capture."
)

PIPELINE_QUERY_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "find_extremum",
        "label": "Global min/max",
        "description": (
            "iloc of the global min or max of <column> in <range>."
        ),
        "params_schema": {
            "range": _RANGE_PARAM_DESC,
            "column": "DataFrame column name",
            "kind": "min | max",
        },
        "callable": _query_find_extremum,
    },
    {
        "id": "find_first_match",
        "label": "First comparison match",
        "description": (
            "First iloc in <range> where <column> <operator> <value> holds."
        ),
        "params_schema": {
            "range": _RANGE_PARAM_DESC,
            "column": "DataFrame column name",
            "op": "equal | greater_than_or_equal | less_than_or_equal | greater_than | less_than",
            "value": "float",
        },
        "callable": _query_find_first_match,
    },
    {
        "id": "read_values_at_indices",
        "label": "Read values at specific ilocs",
        "description": (
            "Read <column> at each iloc in <indices>. Returns "
            "one entry per index; out-of-range (outside <range>) or NaN "
            "samples come back with value=null."
        ),
        "params_schema": {
            "range": _RANGE_PARAM_DESC,
            "column": "DataFrame column name",
            "indices": "list of parent-frame ilocs (int)",
        },
        "callable": _query_read_values_at_indices,
    },
    {
        "id": "compute_slope",
        "label": "Slope across the range",
        "description": (
            "Slope of <column> from the start to the end of <range> "
            "(value-delta / iloc-delta)."
        ),
        "params_schema": {
            "range": _RANGE_PARAM_DESC,
            "column": "DataFrame column name",
        },
        "callable": _query_compute_slope,
    },
    {
        "id": "find_dips_on_main_slope",
        "label": "Dips on linear-regression baseline",
        "description": (
            "Find local dips in <column> below its least-squares trend line "
            "over <range>. Returns `samples` — one `{iloc, value, depth}` "
            "per dip, where `iloc` is the deepest point of that dip."
        ),
        "params_schema": {
            "range": _RANGE_PARAM_DESC,
            "column": "DataFrame column name",
            "smoothing_window": "int ≥ 1 — rolling-median width (1=off, 5=light, 11=heavy)",
            "min_dip_depth": "float — minimum dip depth in signal units (e.g. ~0.05 for brake/throttle 0–1, ~5 for speed in km/h)",
        },
        "callable": _query_find_dips_on_main_slope,
    },
    {
        "id": "find_threshold_crossing",
        "label": "Threshold crossing (ranked)",
        "description": (
            "Rank <columns> by which first crosses <threshold> within <range>, "
            "with optional smoothing via <smoothing_window>. Returns ranking "
            "per column (1=first, null=never crossed). useful for comparing "
            "which curve first crossed the same threshold."
        ),
        "params_schema": {
            "range": _RANGE_PARAM_DESC,
            "columns": "list of 2+ DataFrame column names",
            "threshold": "float",
            "smoothing_window": (
                "int ≥ 1 — rolling-median width (use an odd value; "
                "5 cleans 2-sample spikes, 11 cleans 5-sample ones; "
                "raise for noisier signals, lower it to preserve fast "
                "transitions)"
            ),
        },
        "callable": _query_find_threshold_crossing,
    },
]


def get_pipeline_query(query_id: str) -> Optional[Dict[str, Any]]:
    """Lookup helper — returns the query definition or None."""
    return next(
        (q for q in PIPELINE_QUERY_DEFINITIONS if q["id"] == query_id),
        None,
    )


def render_query_catalog_for_prompt(columns: List[str]) -> str:
    """Render the query catalog + column menu as a markdown block.

    ``columns`` is the list of column names the VLM is allowed to pick from
    — supplied by the caller (zoom passes its graph-table columns so the
    menu is scoped to the data the parent agent constrained for this step).
    """
    lines: List[str] = ["**Available queries:**"]
    for q in PIPELINE_QUERY_DEFINITIONS:
        lines.append(f"- `{q['id']}` — {q['description']}")
        for pname, ptype in q["params_schema"].items():
            lines.append(f"    - `{pname}`: {ptype}")
    lines.append("")
    lines.append(
        "**Available columns:** "
        + (", ".join(f"`{c}`" for c in columns) if columns else "(none)")
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# graph_analysis skill — prompt rendering
# ---------------------------------------------------------------------------

_GRAPH_GUIDELINE_TRIGGERS: Dict[str, Dict[str, Any]] = {
    "brake_and_speed":            {"required": {"brake", "speed"},                     "any_of": []},
    "throttle_and_speed":         {"required": {"throttle", "speed"},                  "any_of": []},
    "time_delta_and_features":    {"required": {"time_delta"},                         "any_of": [
        {"brake", "throttle", "speed", "speed_delta", "push_limit", "trajectory_balance"},
    ]},
    "trajectory_and_features":    {"required": set(),                                  "any_of": [
        {"trajectory_detailed", "trajectory_gas_brake", "trajectory_offset"},
        {"throttle", "brake", "speed", "speed_delta", "push_limit", "trajectory_balance"},
    ]},
    "balance_and_push_limit":     {"required": {"trajectory_balance", "push_limit"},   "any_of": []},
    "brake_and_throttle_overlap": {"required": {"brake", "throttle"},                  "any_of": []},
}

_TRAJECTORY_IDS = {"trajectory_detailed", "trajectory_gas_brake", "trajectory_offset"}


def _render_graph_section(key: str, value: Any, indent: str = "  ") -> List[str]:
    if not value:
        return []
    out: List[str] = [key.replace("_", " ") + ":"]

    if isinstance(value, str):
        for ln in value.rstrip("\n").split("\n"):
            out.append(f"{indent}{ln}" if ln else "")
    elif isinstance(value, list):
        for item in value:
            out.append(f"{indent}- {item}")
    elif isinstance(value, dict):
        for k, v in value.items():
            v_str = "" if v is None else str(v).rstrip("\n")
            v_lines = v_str.split("\n")
            first = v_lines[0]
            out.append(f"{indent}- {k}: {first}" if first else f"{indent}- {k}:")
            cont_indent = indent + "    "
            for cont in v_lines[1:]:
                out.append(f"{cont_indent}{cont}" if cont else "")
    else:
        out.append(f"{indent}{value}")

    return out


def graph_analysis_prompt(graph_ids: List[str]) -> str:
    """Per-graph description block for VLM prompts that read graph images.

    Walks each graph's yaml record with a uniform formatter, appends the
    cross-graph guidelines whose triggers match this graph combination,
    and (when trajectory graphs are present) appends the trajectory shape
    vocabulary.
    """
    requested = list(graph_ids)
    paired: List[tuple] = []
    for gid in requested:
        entry = skills.get(f"graph_analysis.graphs.{gid}")
        if entry:
            paired.append((gid, entry))
    if not paired:
        return ""

    lines: List[str] = [
        "#### Graph Description Skill — How to Describe These Graphs",
        "",
    ]

    for gid, entry in paired:
        title = entry.get("title", gid)
        lines.append(f"##### {title} (id: {gid})")
        lines.append("")
        for key, value in entry.items():
            if key in ("title", "id"):
                continue
            section = _render_graph_section(key, value)
            if section:
                lines.extend(section)
                lines.append("")

    graph_id_set = set(requested)
    relevant: List[str] = []
    for gid, spec in _GRAPH_GUIDELINE_TRIGGERS.items():
        if not spec["required"].issubset(graph_id_set):
            continue
        if not all(any_set & graph_id_set for any_set in spec["any_of"]):
            continue
        text = skills.get(f"graph_analysis.cross_graph_guidelines.{gid}", "")
        if text:
            relevant.append(f"[{gid}] {str(text).strip()}")

    if relevant:
        lines.append("#### Cross-Graph Description Guidelines")
        for g in relevant:
            lines.append(g)
        lines.append("")

    if graph_id_set & _TRAJECTORY_IDS:
        traj_vocab = skills.get("graph_analysis.trajectory_shape_vocabulary", "")
        if traj_vocab:
            lines.append("#### Trajectory Shape Vocabulary")
            lines.append(str(traj_vocab).strip())
            lines.append("")

    return "\n".join(lines)


def _round_floats(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round floats in a query payload to ``ndigits``.

    Ints (ilocs, ranks, gear values) and non-finite floats pass through
    untouched. Applied at the dispatch boundary so every query exposes
    consistently formatted numbers regardless of how the callable produces them.
    """
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return obj
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_round_floats(v, ndigits) for v in obj)
    return obj


def run_pipeline_query(
    df: pd.DataFrame,
    query_id: str, params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Dispatch a query by id with the VLM-supplied params.

    The range is now a per-query param: ``params["range"]`` must be a
    ``[start_iloc, end_iloc]`` list. The dispatcher unpacks it and passes
    it positionally to the underlying callable (so the callable signatures
    stay unchanged).

    Returns ``(payload, error)`` where ``payload`` is a dict with any of:

      * ``iloc``    — primary parent-frame iloc (None when no match)
      * ``value``   — primary value (None when no match)
      * ``samples`` — list of ``{iloc, value, note?}`` for multi-point queries
      * ``extra``   — query-specific extras (e.g. slope deltas)

    ``error`` is a string when params are malformed or the query raises.
    Validates that all required params are present before invoking the
    callable. Missing keys default to None in the returned payload so the
    answer attachment always has a uniform shape.
    """
    base: Dict[str, Any] = {"iloc": None, "value": None, "samples": None, "extra": None}
    q = get_pipeline_query(query_id)
    if q is None:
        return base, f"unknown query '{query_id}'"
    accepted: Dict[str, Any] = {}
    for pname in q["params_schema"].keys():
        if pname not in params:
            return base, f"missing param '{pname}' for query '{query_id}'"
        accepted[pname] = params[pname]
    raw_range = accepted.pop("range")
    if (
        not isinstance(raw_range, (list, tuple))
        or len(raw_range) != 2
    ):
        return base, (
            f"param 'range' for query '{query_id}' must be a "
            f"[start_iloc, end_iloc] list — got {raw_range!r}"
        )
    try:
        start_index = int(raw_range[0])
        end_index = int(raw_range[1])
    except (TypeError, ValueError):
        return base, (
            f"param 'range' for query '{query_id}' must contain two ints "
            f"— got {raw_range!r}"
        )
    if end_index < start_index:
        return base, (
            f"param 'range' for query '{query_id}' has end < start "
            f"({start_index}, {end_index})"
        )
    try:
        raw = q["callable"](df, start_index, end_index, **accepted)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the caller
        return base, f"{type(exc).__name__}: {exc}"
    if raw is None:
        col = accepted.get("column")
        col_msg = ""
        if col is not None and col not in df.columns:
            col_msg = (
                f" — column '{col}' is not in the graph table; valid: "
                + ", ".join(df.columns)
            )
        return base, (
            f"query '{query_id}' returned no data (likely column mismatch, "
            f"out-of-range indices, or NaN values)" + col_msg
        )
    if not isinstance(raw, dict):
        return base, f"query '{query_id}' returned non-dict result: {type(raw).__name__}"
    return _round_floats({**base, **raw}), None


def _create_gas_brake_trajectory_plot(table: pd.DataFrame) -> Optional[Image.Image]:
    """Trajectory coloured by gas−brake balance (green = gas, red = brake).

    Consumes the parent-built table containing player/expert position
    columns + the pre-computed ``gas_brake_signal`` (= Physics_gas −
    Physics_brake).
    """
    track = _resolve_track_config(table)
    px_col = track.get("player_x")
    py_col = track.get("player_y")
    if not px_col or not py_col:
        return None
    if px_col not in table.columns or py_col not in table.columns:
        return None
    if len(table) < 2:
        return None

    x = table[px_col].values.astype(float)
    y = table[py_col].values.astype(float)

    if "gas_brake_signal" in table.columns:
        values = table["gas_brake_signal"].values.astype(float)
    else:
        values = np.ones(len(table))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Expert trajectory (reference line)
    ex_col = track.get("expert_x")
    ey_col = track.get("expert_y")
    if ex_col and ey_col and ex_col in table.columns and ey_col in table.columns:
        ax.plot(
            table[ex_col], table[ey_col],
            color="steelblue", linewidth=1.5, linestyle="--",
            label="Expert", zorder=1,
        )

    lc = _make_colored_line_collection(
        ax, x, y, values,
        cmap="RdYlGn", vmin=-1, vmax=1, linewidth=4,
    )
    plt.colorbar(lc, ax=ax, label="← Brake | Coast | Gas →")

    # Start / end markers
    ax.scatter(x[0], y[0], marker="x", color="black", s=80, zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], marker="o", color="black", s=80, zorder=5, label="End")

    ax.set_title("Gas/Brake Trajectory\nGreen = Throttle, Red = Brake")
    ax.invert_yaxis()
    ax.set_aspect("equal", "box")
    ax.axis("off")
    if ex_col:
        ax.legend(fontsize=8)
    ax.autoscale()

    return _plot_to_image(fig)


def _create_balance_line_plot(table: pd.DataFrame) -> Optional[Image.Image]:
    """Line plot of oversteer/understeer slip balance over segment index.

    Reads the pre-computed ``slip_balance`` column from the parent-built
    table (mean(|rear|) − mean(|front|), in radians). Positive → rear
    slipping more → oversteer (red shading above zero). Negative → front
    slipping more → understeer (blue shading below zero).
    """
    if "slip_balance" not in table.columns or len(table) < 2:
        return None

    balance = table["slip_balance"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 4))

    idx = table.index
    ax.plot(idx, balance, color="black", linewidth=1.2, label="Slip balance (rear − front)")
    ax.fill_between(
        idx, balance.values, 0.0,
        where=(balance.values > 0), interpolate=True,
        color="red", alpha=0.35, label="Oversteer (rear-slip dominant)",
    )
    ax.fill_between(
        idx, balance.values, 0.0,
        where=(balance.values < 0), interpolate=True,
        color="blue", alpha=0.35, label="Understeer (front-slip dominant)",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_title("Oversteer / Understeer Slip Balance")
    ax.set_xlabel("Index")
    ax.set_ylabel("Rear − Front mean |slip angle| (rad)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    return _plot_to_image(fig)


def _create_trajectory_plot(table: pd.DataFrame) -> Optional[Image.Image]:
    """Detailed trajectory plot — player + expert lines, start/end markers,
    plus expert-anchored entry / apex / exit markers per detected arc.

    Consumes the parent-built table with player/expert position columns;
    phase detection runs on the slice in front of us so the markers fit
    the rendered range.
    """
    track = _resolve_track_config(table)
    px_col = track.get("player_x")
    py_col = track.get("player_y")
    if not px_col or not py_col:
        return None
    if px_col not in table.columns or py_col not in table.columns:
        return None
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    # Player trajectory
    ax.plot(table[px_col], table[py_col], label="Player", color="green", linewidth=2)

    # Expert trajectory
    ex_col = track.get("expert_x")
    ey_col = track.get("expert_y")
    if ex_col and ey_col and ex_col in table.columns and ey_col in table.columns:
        ax.plot(
            table[ex_col], table[ey_col],
            label="Expert", color="blue", linewidth=1.5, linestyle="--",
        )

    # Mark start / end
    ax.scatter(
        table[px_col].iloc[0], table[py_col].iloc[0],
        marker="x", color="black", s=80, zorder=5, label="Start",
    )
    ax.scatter(
        table[px_col].iloc[-1], table[py_col].iloc[-1],
        marker="o", color="black", s=80, zorder=5, label="End",
    )

    # Phase markers (expert-anchored) — entry / apex / exit per detected arc.
    phases, _ = _detect_expert_phases(table)
    if phases and ex_col and ey_col and ex_col in table.columns and ey_col in table.columns:
        ex_arr = table[ex_col].to_numpy()
        ey_arr = table[ey_col].to_numpy()
        for k, ph in enumerate(phases):
            entry_i, apex_i, exit_i = ph["entry"], ph["apex"], ph["exit"]
            # Legend-label only the first arc's markers.
            entry_label = "Entry" if k == 0 else None
            apex_label = "Apex" if k == 0 else None
            exit_label = "Exit" if k == 0 else None
            ax.scatter(
                ex_arr[entry_i], ey_arr[entry_i],
                marker="o", color="yellow", s=90, zorder=6,
                edgecolor="black", linewidth=0.6, label=entry_label,
            )
            ax.scatter(
                ex_arr[apex_i], ey_arr[apex_i],
                marker="*", color="red", s=180, zorder=7,
                edgecolor="black", linewidth=0.6, label=apex_label,
            )
            ax.scatter(
                ex_arr[exit_i], ey_arr[exit_i],
                marker="^", color="limegreen", s=90, zorder=6,
                edgecolor="black", linewidth=0.6, label=exit_label,
            )
            if len(phases) > 1:
                ax.annotate(
                    f"#{k + 1}",
                    xy=(ex_arr[apex_i], ey_arr[apex_i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="red", zorder=8,
                )

    ax.set_title("Detailed Trajectory")
    ax.invert_yaxis()
    ax.legend()
    ax.set_aspect("equal", "box")
    ax.axis("off")
    return _plot_to_image(fig)


def _create_trajectory_offset_plot(table: pd.DataFrame) -> Optional[Image.Image]:
    """Trajectory offset plot — signed perpendicular distance between the
    player's position and the expert's path, plotted over segment index.

    Reads the pre-computed ``trajectory_offset`` column from the parent-built
    table. Phase markers (entry / apex / exit) come from
    ``_detect_expert_phases`` run on the table's expert position columns so
    they line up with the rendered range. Positive y = player wider than
    expert (toward outside of corner); negative y = tighter (toward inside).
    """
    if "trajectory_offset" not in table.columns or table.empty:
        return None

    offset = table["trajectory_offset"].to_numpy(dtype=float)

    # X-axis = table.index (matching the other feature plots so
    # brake/throttle/offset cross-reference cleanly).
    x_axis = table.index.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="Expert (zero offset)")
    ax.plot(x_axis, offset, color="green", linewidth=2, label="Player offset")

    phases, _ = _detect_expert_phases(table)
    for k, ph in enumerate(phases):
        entry_i, apex_i, exit_i = ph["entry"], ph["apex"], ph["exit"]
        entry_x, apex_x, exit_x = x_axis[entry_i], x_axis[apex_i], x_axis[exit_i]
        entry_label = "Entry" if k == 0 else None
        apex_label = "Apex" if k == 0 else None
        exit_label = "Exit" if k == 0 else None
        ax.scatter(
            entry_x, offset[entry_i],
            marker="o", color="yellow", s=90, zorder=6,
            edgecolor="black", linewidth=0.6, label=entry_label,
        )
        ax.scatter(
            apex_x, offset[apex_i],
            marker="*", color="red", s=180, zorder=7,
            edgecolor="black", linewidth=0.6, label=apex_label,
        )
        ax.scatter(
            exit_x, offset[exit_i],
            marker="^", color="limegreen", s=90, zorder=6,
            edgecolor="black", linewidth=0.6, label=exit_label,
        )
        if len(phases) > 1:
            ax.annotate(
                f"#{k + 1}",
                xy=(apex_x, offset[apex_i]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9, fontweight="bold", color="red", zorder=8,
            )

    ax.set_title("Trajectory Offset (signed)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Lateral offset (m) — wider > 0, tighter < 0")
    ax.grid(True)
    ax.legend(loc="best")
    return _plot_to_image(fig)


def _resolve_track_config(df: pd.DataFrame) -> Dict[str, str]:
    """Auto-detect trajectory column names from the DataFrame."""
    tc: Dict[str, str] = {}
    if "Graphics_player_pos_x" in df.columns:
        tc["player_x"] = "Graphics_player_pos_x"
        tc["player_y"] = "Graphics_player_pos_y"
    if "expert_optimal_player_pos_x" in df.columns:
        tc["expert_x"] = "expert_optimal_player_pos_x"
        tc["expert_y"] = "expert_optimal_player_pos_y"
    return tc


# ---------------------------------------------------------------------------
# Graph data builders — one per graph id. Each takes raw df and returns a
# DataFrame whose columns are exactly the data series that graph uses
# (drawn lines + per-row inputs renderers consume internally, e.g. position
# columns used for phase marker placement). The parent agent calls these to
# build the constrained ``graph_table`` it hands children; children query
# / render against the table only.
# ---------------------------------------------------------------------------


def _project_columns(df: pd.DataFrame, cols: List[str]) -> Optional[pd.DataFrame]:
    if any(c not in df.columns for c in cols):
        return None
    return df.loc[:, cols].copy()


def _build_throttle(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _project_columns(df, ["expert_optimal_throttle", "Physics_gas"])


def _build_brake(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _project_columns(df, ["expert_optimal_brake", "Physics_brake"])


def _build_time_delta(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _project_columns(df, ["expert_time_difference"])


def _build_speed_delta(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _project_columns(df, ["speed_difference"])


def _build_speed(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _project_columns(df, ["expert_optimal_speed", "Physics_speed_kmh"])


def _build_push_limit(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _project_columns(df, ["driver_push_to_limit"])


def _build_gear(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _project_columns(df, ["expert_optimal_gear", "Physics_gear"])


def _build_trajectory_balance(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute the single drawn series (mean(|rear|) − mean(|front|))."""
    raw = (
        "Physics_slip_angle_front_left",
        "Physics_slip_angle_front_right",
        "Physics_slip_angle_rear_left",
        "Physics_slip_angle_rear_right",
    )
    if any(c not in df.columns for c in raw):
        return None
    fl, fr, rl, rr = raw
    balance = (
        (df[rl].abs() + df[rr].abs()) / 2.0
        - (df[fl].abs() + df[fr].abs()) / 2.0
    ).astype(float)
    return pd.DataFrame({"slip_balance": balance}, index=df.index)


def _build_trajectory_detailed(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Player + expert position traces (renderer derives phase markers from them)."""
    track = _resolve_track_config(df)
    cols = [
        track[k] for k in ("player_x", "player_y", "expert_x", "expert_y")
        if track.get(k)
    ]
    if not cols:
        return None
    return _project_columns(df, cols)


def _build_trajectory_gas_brake(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Player + expert position traces plus the colouring signal (gas − brake)."""
    track = _resolve_track_config(df)
    pos_cols = [
        track[k] for k in ("player_x", "player_y", "expert_x", "expert_y")
        if track.get(k)
    ]
    if not pos_cols:
        return None
    if "Physics_gas" not in df.columns or "Physics_brake" not in df.columns:
        return None
    if any(c not in df.columns for c in pos_cols):
        return None
    out = df.loc[:, pos_cols].copy()
    out["gas_brake_signal"] = (df["Physics_gas"] - df["Physics_brake"]).astype(float)
    return out


def _build_trajectory_offset(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The signed offset line + expert positions (for phase marker placement)."""
    track = _resolve_track_config(df)
    if not all(track.get(k) for k in ("player_x", "player_y", "expert_x", "expert_y")):
        return None
    px_col, py_col = track["player_x"], track["player_y"]
    ex_col, ey_col = track["expert_x"], track["expert_y"]
    if any(c not in df.columns for c in (px_col, py_col, ex_col, ey_col)):
        return None

    kin = _smoothed_expert_kinematics(df)
    if kin is None:
        return None
    _x_s, _y_s, dx, dy, kappa, _w = kin

    tangent_norm = np.sqrt(dx * dx + dy * dy)
    tangent_norm = np.where(tangent_norm > 1e-9, tangent_norm, 1.0)
    tx = dx / tangent_norm
    ty = dy / tangent_norm

    px = df[px_col].to_numpy(dtype=float)
    py = df[py_col].to_numpy(dtype=float)
    ex = df[ex_col].to_numpy(dtype=float)
    ey = df[ey_col].to_numpy(dtype=float)
    ox = px - ex
    oy = py - ey

    cross = tx * oy - ty * ox
    sign_flip = -np.sign(kappa)
    sign_flip = np.where(sign_flip == 0, 1.0, sign_flip)
    offset = cross * sign_flip

    # Expert positions stay in the table so the renderer's phase detection
    # has the kinematic inputs it needs after the parent's projection.
    out = df.loc[:, [ex_col, ey_col]].copy()
    out["trajectory_offset"] = offset.astype(float)
    return out


_GRAPH_BUILDERS = {
    "throttle":             _build_throttle,
    "brake":                _build_brake,
    "time_delta":           _build_time_delta,
    "speed_delta":          _build_speed_delta,
    "speed":                _build_speed,
    "push_limit":           _build_push_limit,
    "gear":                 _build_gear,
    "trajectory_balance":   _build_trajectory_balance,
    "trajectory_detailed":  _build_trajectory_detailed,
    "trajectory_gas_brake": _build_trajectory_gas_brake,
    "trajectory_offset":    _build_trajectory_offset,
}


_GRAPH_RENDERERS = {
    "trajectory_detailed":  _create_trajectory_plot,
    "trajectory_gas_brake": _create_gas_brake_trajectory_plot,
    "trajectory_balance":   _create_balance_line_plot,
    "trajectory_offset":    _create_trajectory_offset_plot,
}


def build_graph(graph_id: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Build one graph's data table from raw df. Returns ``None`` when the
    raw inputs the graph needs aren't present."""
    builder = _GRAPH_BUILDERS.get(graph_id)
    if builder is None:
        return None
    return builder(df)


def _render_graph_table(graph_id: str, table: pd.DataFrame, title: str) -> Optional[Image.Image]:
    renderer = _GRAPH_RENDERERS.get(graph_id)
    if renderer is not None:
        return renderer(table)
    return _create_feature_plot(table, title)


def render_graph_builds(
    graph_builds: Dict[str, pd.DataFrame],
    start_index: int,
    end_index: int,
) -> List[Tuple[Image.Image, str]]:
    """Render the constrained-table form of ``generate_telemetry_graphs``.

    ``graph_builds`` maps graph id → its full-range data table (as produced
    by ``build_graph`` at the parent agent). Each table is sliced to
    ``[start_index, end_index)`` (Python-slice end-exclusive, matching
    ``generate_telemetry_graphs``) before being handed to its renderer.
    """
    if not graph_builds:
        return []
    desc_by_id = {d["id"]: (d["title"], d["description"]) for d in AGENT_GRAPH_DEFINITIONS}
    results: List[Tuple[Image.Image, str]] = []
    for gid, table in graph_builds.items():
        sliced = table.iloc[int(start_index): int(end_index)]
        if sliced.empty:
            continue
        title, desc = desc_by_id.get(gid, (gid, ""))
        img = _render_graph_table(gid, sliced, title)
        if img is not None:
            results.append((img, f"{title}: {desc}"))
    return results


def generate_telemetry_graphs(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    graph_ids: Optional[List[str]] = None,
) -> List[Tuple[Image.Image, str]]:
    """Build + render graphs straight from raw df.

    Convenience wrapper for callers (e.g. the describe_graphs planner) that
    have raw df in hand and want one-shot build+render. Returns
    ``(PIL.Image, description)`` pairs.
    """
    segment_df = df.iloc[int(start_index): int(end_index)]
    if segment_df.empty:
        return []

    defs = AGENT_GRAPH_DEFINITIONS
    if graph_ids:
        defs = [d for d in defs if d["id"] in graph_ids]

    results: List[Tuple[Image.Image, str]] = []
    for gdef in defs:
        gid = gdef["id"]
        table = build_graph(gid, segment_df)
        if table is None or table.empty:
            continue
        img = _render_graph_table(gid, table, gdef["title"])
        if img is not None:
            results.append((img, f"{gdef['title']}: {gdef['description']}"))
    return results
