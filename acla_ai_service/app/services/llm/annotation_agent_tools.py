"""
Tool definitions for the annotation agent pipeline.

Provides two tool categories that the agent nodes can invoke:

1. **Statistical analysis** — numerical summaries of telemetry columns
   (mean, min, max, std, deltas vs expert).
2. **Graph generation** — telemetry visualizations rendered as PIL Images
   (feature plots, trajectory plots) aligned with the GRAPH_DEFINITIONS
   used by the human annotation workflow.

These tools mirror what a human annotator sees and allow the vision-capable
LLM (VLM) to analyse the same visual evidence.
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

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph definitions — subset of the graphs used in the UI / Gemini workflow.
# Keep IDs in sync with ui/gemini_analyzer.GRAPH_DEFINITIONS.
# ---------------------------------------------------------------------------

AGENT_GRAPH_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "throttle",
        "title": "Throttle Application - ",
        "columns": ["expert_optimal_throttle", "Physics_gas"],
        "description": "Expert vs player throttle traces.",
    },
    {
        "id": "brake",
        "title": "Brake Application - ",
        "columns": ["expert_optimal_brake", "Physics_brake"],
        "description": "Expert vs player brake traces.",
    },
    {
        "id": "time_delta",
        "title": "Time Difference to Expert",
        "columns": ["expert_time_difference"],
        "description": "Instantaneous time delta vs expert.",
    },
    {
        "id": "speed_delta",
        "title": "Speed Difference (Expert - Player)",
        "columns": ["speed_difference"],
        "description": "Speed difference between expert and player.",
    },
    {
        "id": "speed",
        "title": "Speed Trace: Expert vs Player",
        "columns": ["expert_optimal_speed", "Physics_speed_kmh"],
        "description": "Expert vs player speed traces.",
    },
    {
        "id": "push_limit",
        "title": "Driver Push/Limit",
        "columns": ["driver_push_to_limit"],
        "description": "Driver push-to-limit metric.",
    },
    {
        "id": "trajectory_detailed",
        "title": "Detailed Trajectory",
        "columns": [],  # special: uses position columns
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
        "columns": [],  # special: uses position + Physics_gas / Physics_brake
        "description": (
            "Player trajectory coloured by throttle/brake balance "
            "(green = full gas, red = full brake, yellow = coasting). "
            "Mirrors the Gas/Brake colour mode in the human annotation track map."
        ),
    },
    {
        "id": "trajectory_balance",
        "title": "Oversteer/Understeer Slip Balance",
        "columns": [],  # special: uses slip angle columns directly
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
        "columns": [],  # special: uses position columns + smoothed kinematics
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
        "columns": ["expert_optimal_gear", "Physics_gear"],
        "description": "Expert vs player gear traces (integer steps).",
    },
]

# ---------------------------------------------------------------------------
# Telemetry column groups — map named group IDs to the DataFrame columns
# and player-vs-expert deltas the planner can selectively request.
# ---------------------------------------------------------------------------

TELEMETRY_COLUMN_GROUPS: Dict[str, Dict[str, Any]] = {
    "speed": {
        "columns": [ "Physics_speed_kmh", "expert_optimal_speed"],
        "deltas": [("Physics_speed_kmh", "expert_optimal_speed", "speed_delta")],
        "description": "Driver speed traces and expert comparison.",
    },
    "throttle": {
        "columns": ["Physics_gas", "expert_optimal_gas"],
        "deltas": [("Physics_gas", "expert_optimal_gas", "throttle_delta")],
        "description": "Driver throttle traces and expert comparison.",
    },
    "brake": {
        "columns": ["Physics_brake", "expert_optimal_brake"],
        "deltas": [("Physics_brake", "expert_optimal_brake", "brake_delta")],
        "description": "Driver brake traces and expert comparison.",
    },
    "steering": {
        "columns": ["steer_angle"],
        "deltas": [],
        "description": "Driver steering angle statistics.",
    },
    "push_limit": {
        "columns": ["driver_push_to_limit"],
        "deltas": [],
        "description": "Driver push-to-limit metric.",
    },
    "slip_angles": {
        "columns": [
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left",
            "Physics_slip_angle_rear_right",
        ],
        "deltas": [],
        "description": "Driver tyre slip angles (front/rear). Used to derive oversteer/understeer balance.",
    },
}

ALL_TELEMETRY_GROUP_IDS: List[str] = list(TELEMETRY_COLUMN_GROUPS.keys())

# ---------------------------------------------------------------------------
# Tool 1: Statistical summary
# ---------------------------------------------------------------------------


def get_telemetry_statistics(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    session_id: str = "unknown",
    stat_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a structured dict with numerical telemetry statistics.

    Parameters
    ----------
    stat_categories : list[str], optional
        Subset of category IDs from ``TELEMETRY_COLUMN_GROUPS`` to compute.
        ``None`` means compute all categories (backward-compatible).
    """
    segment_df = df.iloc[int(start_index): int(end_index)]

    track_name = "unknown"
    if "Static_track" in df.columns and not df.empty:
        track_name = str(df["Static_track"].iloc[0])

    # Resolve which categories to compute
    if stat_categories:
        selected = [c for c in stat_categories if c in TELEMETRY_COLUMN_GROUPS]
    else:
        selected = list(TELEMETRY_COLUMN_GROUPS.keys())

    # Collect columns and deltas from selected categories
    summary_cols: List[str] = []
    delta_pairs: List[Tuple[str, str, str]] = []
    for cat_id in selected:
        cat = TELEMETRY_COLUMN_GROUPS[cat_id]
        summary_cols.extend(cat["columns"])
        delta_pairs.extend(cat["deltas"])

    # Deduplicate while preserving order
    seen_cols: set = set()
    unique_cols: List[str] = []
    for c in summary_cols:
        if c not in seen_cols:
            seen_cols.add(c)
            unique_cols.append(c)

    telemetry_summary: Dict[str, Dict[str, Any]] = {}
    for col in unique_cols:
        if col in segment_df.columns:
            series = segment_df[col].dropna()
            if len(series) > 0:
                telemetry_summary[col] = {
                    "mean": round(float(series.mean()), 2),
                    "min": round(float(series.min()), 2),
                    "max": round(float(series.max()), 2),
                    "std": round(float(series.std()), 2),
                }

    feature_deltas: Dict[str, Any] = {}
    for player_col, expert_col, name in delta_pairs:
        if player_col in segment_df.columns and expert_col in segment_df.columns:
            delta = segment_df[player_col] - segment_df[expert_col]
            feature_deltas[name] = {
                "mean": round(float(delta.mean()), 2),
                "min": round(float(delta.min()), 2),
                "max": round(float(delta.max()), 2),
            }

    return {
        "session_id": session_id,
        "track_name": track_name,
        "start_index": start_index,
        "end_index": end_index,
        "segment_length": end_index - start_index,
        "telemetry_summary": telemetry_summary,
        "feature_deltas": feature_deltas,
    }


def format_statistics_as_text(stats: Dict[str, Any]) -> str:
    """Render statistics dict as human-readable text for prompt injection."""
    lines: list[str] = []
    lines.append(
        f"Session: {stats.get('session_id', '?')} | "
        f"Track: {stats.get('track_name', '?')}"
    )
    lines.append(
        f"Segment range: index {stats.get('start_index', '?')} → "
        f"{stats.get('end_index', '?')} "
        f"(length {stats.get('segment_length', '?')})"
    )

    telemetry = stats.get("telemetry_summary", {})
    if telemetry:
        lines.append("Key telemetry statistics:")
        for col, col_stats in telemetry.items():
            parts = [f"{k}={v}" for k, v in col_stats.items()]
            lines.append(f"  {col}: {', '.join(parts)}")

    features = stats.get("feature_deltas", {})
    if features:
        lines.append("Feature deltas (player vs expert):")
        for feat, val in features.items():
            lines.append(f"  {feat}: {val}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: Graph generation
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
    df: pd.DataFrame,
    columns: List[str],
    title: str,
) -> Optional[Image.Image]:
    """Line plot for one or more telemetry columns over the segment index."""
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols or df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in valid_cols:
        ax.plot(df.index, df[col], label=col)
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
    kappa = np.where(denom > 1e-9, (dx * ddy - dy * ddx) / denom, 0.0)
    kappa = _moving_average(kappa, window)

    return x_s, y_s, dx, dy, kappa, window


def _detect_expert_phases(
    segment: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int]:
    """Detect corner phases from the expert position trace alone.

    Returns ``(phases, smoothing_window)`` where ``phases`` is a list of
    one dict per detected arc (empty for non-corner segments). Each dict
    holds ilocs **relative to the segment start** plus auxiliary fields
    the VLM can use to reason about trail-braking and turn direction:

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

    All ilocs are relative to ``start_index``. ``phases`` is empty when
    no arc clears the curvature threshold (pure straight, or segment too
    short / missing position columns).
    """
    from .step_evaluator_agents import PipelineAttachment

    segment = df.iloc[int(start_index): int(end_index)]
    phases, window = _detect_expert_phases(segment)

    return PipelineAttachment(
        name="phase_indices",
        kind="structured",
        label="Phase Indices (expert-anchored)",
        content={"phases": phases, "smoothing_window": int(window)},
    )


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
]


def get_pipeline_tool(tool_id: str) -> Optional[Dict[str, Any]]:
    """Lookup helper — returns the tool definition or None."""
    return next(
        (t for t in PIPELINE_TOOL_DEFINITIONS if t["id"] == tool_id),
        None,
    )


def _create_gas_brake_trajectory_plot(
    df: pd.DataFrame,
    track_config: Dict[str, str],
) -> Optional[Image.Image]:
    """Trajectory coloured by gas−brake balance (green = gas, red = brake)."""
    px_col = track_config.get("player_x")
    py_col = track_config.get("player_y")
    if not px_col or not py_col:
        return None
    if px_col not in df.columns or py_col not in df.columns:
        return None
    if len(df) < 2:
        return None

    x = df[px_col].values.astype(float)
    y = df[py_col].values.astype(float)

    has_gas = "Physics_gas" in df.columns and "Physics_brake" in df.columns
    values = (
        (df["Physics_gas"] - df["Physics_brake"]).values.astype(float)
        if has_gas
        else np.ones(len(df))
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # Expert trajectory (reference line)
    ex_col = track_config.get("expert_x")
    ey_col = track_config.get("expert_y")
    if ex_col and ey_col and ex_col in df.columns and ey_col in df.columns:
        ax.plot(
            df[ex_col], df[ey_col],
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


def _create_balance_line_plot(df: pd.DataFrame) -> Optional[Image.Image]:
    """Line plot of oversteer/understeer slip balance over segment index.

    Balance = mean(|rear slip|) − mean(|front slip|), in radians.
    Positive → rear slipping more → oversteer (red shading above zero).
    Negative → front slipping more → understeer (blue shading below zero).
    No amplification — the y-axis carries the magnitude directly.
    """
    rear_l = "Physics_slip_angle_rear_left"
    rear_r = "Physics_slip_angle_rear_right"
    front_l = "Physics_slip_angle_front_left"
    front_r = "Physics_slip_angle_front_right"
    if not all(c in df.columns for c in (rear_l, rear_r, front_l, front_r)):
        return None
    if len(df) < 2:
        return None

    balance = (
        (df[rear_l].abs() + df[rear_r].abs()) / 2.0
        - (df[front_l].abs() + df[front_r].abs()) / 2.0
    ).astype(float)

    fig, ax = plt.subplots(figsize=(10, 4))

    idx = df.index
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


def _create_trajectory_plot(
    df: pd.DataFrame,
    track_config: Dict[str, str],
) -> Optional[Image.Image]:
    """Detailed trajectory plot — player + expert lines, start/end markers,
    plus expert-anchored entry / apex / exit markers per detected arc.

    Phase markers are drawn on the **expert** line — phases are expert-
    anchored per project convention (the player can stop or drive
    erratically mid-corner). Marker ilocs come from the same
    ``_detect_expert_phases`` helper that powers the ``compute_expert_phases``
    tool, so the image is consistent with the structured attachment.
    Chicanes / esses show numbered apex labels (#1, #2, …); single-arc
    corners show no number.
    """
    px_col = track_config.get("player_x")
    py_col = track_config.get("player_y")
    if not px_col or not py_col:
        return None
    if px_col not in df.columns or py_col not in df.columns:
        return None
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    # Player trajectory
    ax.plot(df[px_col], df[py_col], label="Player", color="green", linewidth=2)

    # Expert trajectory
    ex_col = track_config.get("expert_x")
    ey_col = track_config.get("expert_y")
    if ex_col and ey_col and ex_col in df.columns and ey_col in df.columns:
        ax.plot(
            df[ex_col], df[ey_col],
            label="Expert", color="blue", linewidth=1.5, linestyle="--",
        )

    # Mark start / end
    ax.scatter(
        df[px_col].iloc[0], df[py_col].iloc[0],
        marker="x", color="black", s=80, zorder=5, label="Start",
    )
    ax.scatter(
        df[px_col].iloc[-1], df[py_col].iloc[-1],
        marker="o", color="black", s=80, zorder=5, label="End",
    )

    # Phase markers (expert-anchored) — entry / apex / exit per detected arc.
    phases, _ = _detect_expert_phases(df)
    if phases and ex_col and ey_col and ex_col in df.columns and ey_col in df.columns:
        ex_arr = df[ex_col].to_numpy()
        ey_arr = df[ey_col].to_numpy()
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


def _create_trajectory_offset_plot(
    df: pd.DataFrame,
    track_config: Dict[str, str],
) -> Optional[Image.Image]:
    """Trajectory offset plot — signed perpendicular distance between the
    player's position and the expert's path, plotted over segment index.

    Y-axis sign: positive = player is wider than expert (toward outside of
    corner); negative = tighter (toward inside). Sign flip uses the local
    expert curvature κ — on left-handers (κ > 0) inside is to the left, on
    right-handers (κ < 0) inside is to the right; multiplying the raw
    signed cross-track by ``-sign(κ)`` makes positive consistently mean
    "outside / wider" regardless of corner handedness. On near-straight
    samples (κ ≈ 0) the multiplier collapses to ±1 keeping the raw
    cross-track sign (left = positive of expert direction-of-travel).

    Phase markers (entry / apex / exit) are placed on the offset trace at
    the same ilocs as the ``trajectory_detailed`` plot, using the same
    colours so the VLM transfers its existing marker vocabulary.
    """
    px_col = track_config.get("player_x")
    py_col = track_config.get("player_y")
    ex_col = track_config.get("expert_x")
    ey_col = track_config.get("expert_y")
    if not (px_col and py_col and ex_col and ey_col):
        return None
    if any(c not in df.columns for c in (px_col, py_col, ex_col, ey_col)):
        return None
    if df.empty:
        return None

    kin = _smoothed_expert_kinematics(df)
    if kin is None:
        return None
    _x_s, _y_s, dx, dy, kappa, _window = kin

    # Unit tangent of the smoothed expert path
    tangent_norm = np.sqrt(dx * dx + dy * dy)
    tangent_norm = np.where(tangent_norm > 1e-9, tangent_norm, 1.0)
    tx = dx / tangent_norm
    ty = dy / tangent_norm

    # Offset vector at each iloc — raw player and expert sample positions
    px = df[px_col].to_numpy(dtype=float)
    py = df[py_col].to_numpy(dtype=float)
    ex = df[ex_col].to_numpy(dtype=float)
    ey = df[ey_col].to_numpy(dtype=float)
    ox = px - ex
    oy = py - ey

    # Signed cross-track (z-component of T × O): + = player left of expert dir.
    cross = tx * oy - ty * ox

    # Flip so + = wider (outside of corner). On straights (κ ≈ 0) keep the
    # raw cross-track sign instead of zeroing the offset.
    sign_flip = -np.sign(kappa)
    sign_flip = np.where(sign_flip == 0, 1.0, sign_flip)
    offset = cross * sign_flip

    # X-axis = df.index (the dataframe's internal index, matching the
    # other feature plots so brake/throttle/offset cross-reference cleanly).
    # Phase ilocs from _detect_expert_phases are 0-based relative to the
    # segment, so convert via df.index[iloc] when placing markers.
    x_axis = df.index.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="Expert (zero offset)")
    ax.plot(x_axis, offset, color="green", linewidth=2, label="Player offset")

    phases, _ = _detect_expert_phases(df)
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


def generate_telemetry_graphs(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    graph_ids: Optional[List[str]] = None,
) -> List[Tuple[Image.Image, str]]:
    """Generate telemetry graphs for the given segment.

    Parameters
    ----------
    df : pd.DataFrame
        Full session telemetry.
    start_index, end_index : int
        Segment boundaries (iloc-based).
    graph_ids : list[str], optional
        Subset of graph IDs to generate. ``None`` means all available.

    Returns
    -------
    list of (PIL.Image, description)
        Each entry is a rendered graph image paired with its textual
        description (for use in the VLM prompt).
    """
    segment_df = df.iloc[int(start_index): int(end_index)]
    if segment_df.empty:
        return []

    defs = AGENT_GRAPH_DEFINITIONS
    if graph_ids:
        defs = [d for d in defs if d["id"] in graph_ids]

    track_config = _resolve_track_config(segment_df)
    results: List[Tuple[Image.Image, str]] = []

    for gdef in defs:
        gid = gdef["id"]
        title = gdef["title"]
        desc = gdef["description"]
        cols = gdef.get("columns", [])

        if gid == "trajectory_detailed":
            img = _create_trajectory_plot(segment_df, track_config)
        elif gid == "trajectory_gas_brake":
            img = _create_gas_brake_trajectory_plot(segment_df, track_config)
        elif gid == "trajectory_balance":
            img = _create_balance_line_plot(segment_df)
        elif gid == "trajectory_offset":
            img = _create_trajectory_offset_plot(segment_df, track_config)
        else:
            img = _create_feature_plot(segment_df, cols, title)

        if img is not None:
            results.append((img, f"{title}: {desc}"))

    return results
