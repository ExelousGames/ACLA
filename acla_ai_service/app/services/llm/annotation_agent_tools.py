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
            "Close-up trajectory with apex and min-speed annotations. "
            "Green=player, Blue=expert."
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
        "title": "Oversteer/Understeer Balance Trajectory",
        "columns": [],  # special: uses position + slip angle columns
        "description": (
            "Player trajectory coloured by rear-vs-front slip angle balance. "
            "Red = oversteer (rear slip dominant), Blue = understeer (front slip dominant). "
            "Mirrors the Balance colour mode in the human annotation track map."
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
# Expert-anchored corner phase detection
#
# Phase definitions (entry/apex/exit) are anchored on the EXPERT trajectory
# only. The player may stop mid-corner or drive erratically, so player-derived
# phases are unreliable. The expert defines *where* the corner phases live;
# the player is then described *at* those expert-defined indices. The same
# iloc identifies the same telemetry sample on both lines, so an expert-
# derived index is meaningful on the player curve too.
# ---------------------------------------------------------------------------


def _max_curvature_index(x: np.ndarray, y: np.ndarray) -> Optional[int]:
    """Return iloc of max curvature on the (x, y) line, or None if degenerate."""
    if len(x) < 6 or len(y) < 6:
        return None
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx ** 2 + dy ** 2, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.where(denominator > 1e-6, numerator / denominator, 0)
    curvature = np.nan_to_num(curvature)
    if not np.any(curvature > 0):
        return None
    return int(np.argmax(curvature))


def _expert_phase_indices(df: pd.DataFrame) -> Dict[str, Optional[int]]:
    """Detect entry/apex/exit phase ilocs from EXPERT telemetry.

    Returns iloc-based positions (0..len(df)-1) for brake_start, apex,
    brake_end, throttle_resume. A phase that cannot be detected is `None`.
    Never falls back to player data — see VLM-no-fallback / expert-anchor
    rules.
    """
    out: Dict[str, Optional[int]] = {
        "brake_start": None, "apex": None,
        "brake_end": None, "throttle_resume": None,
    }
    n = len(df)
    if n == 0:
        return out

    brake_col = "expert_optimal_brake"
    throttle_col = "expert_optimal_throttle"
    speed_col = "expert_optimal_speed"
    ex_col = "expert_optimal_player_pos_x"
    ey_col = "expert_optimal_player_pos_y"

    if brake_col in df.columns:
        brake_ilocs = np.where(df[brake_col].values > 0.1)[0]
        if len(brake_ilocs) > 0:
            out["brake_start"] = int(brake_ilocs[0])
            out["brake_end"] = int(brake_ilocs[-1])

    if ex_col in df.columns and ey_col in df.columns:
        out["apex"] = _max_curvature_index(
            df[ex_col].values.astype(float),
            df[ey_col].values.astype(float),
        )
    if out["apex"] is None and speed_col in df.columns:
        s = df[speed_col].dropna()
        if len(s) > 0:
            try:
                out["apex"] = int(df.index.get_loc(s.idxmin()))
            except Exception:
                out["apex"] = None

    if throttle_col in df.columns:
        search_start = out["apex"] if out["apex"] is not None else 0
        if 0 <= search_start < n:
            tail = df[throttle_col].values[search_start:]
            above = np.where(tail > 0.5)[0]
            if len(above) > 0:
                out["throttle_resume"] = int(search_start + above[0])
    if out["throttle_resume"] is None and out["brake_end"] is not None and out["brake_end"] + 1 < n:
        out["throttle_resume"] = out["brake_end"] + 1

    return out


# Per-phase label offsets are pushed into different quadrants so neighbouring
# markers (especially apex + throttle_resume, which often sit within 0-2
# samples of each other on tight corners) don't stack their labels on top of
# each other. The order in this dict is also the iloc-order we expect phases
# to appear in (brake_start → apex → throttle_resume).
_PHASE_STYLES: Dict[str, Dict[str, Any]] = {
    "brake_start":     {"marker": "s", "color": "darkorange", "size": 100, "label": "Brake",    "offset": (-46, 14)},
    "apex":            {"marker": "*", "color": "purple",     "size": 220, "label": "Apex",     "offset": (10, 16)},
    "throttle_resume": {"marker": "^", "color": "green",      "size": 120, "label": "Throttle", "offset": (10, -20)},
}

# Phases whose ilocs differ by ≤ this many samples get a single combined label
# (e.g. "Apex@100 / Throttle@101") rather than two overlapping annotations.
_PHASE_LABEL_MERGE_ILOCS = 3


def _annotate_phase_markers(
    ax,
    df: pd.DataFrame,
    phases: Dict[str, Optional[int]],
    x_col: str,
    y_col: str,
    *,
    hollow: bool = False,
) -> None:
    """Draw labelled phase markers at expert-anchored ilocs on a trajectory.

    `hollow=True` renders the expert-side markers with no fill so they
    visually distinguish from the filled player-side markers. Labels are
    only drawn on the filled (player) markers to avoid double-labelling
    the same iloc. When two filled markers fall within
    ``_PHASE_LABEL_MERGE_ILOCS`` samples of each other their labels are
    merged into one annotation with a leader line so the markers stay
    individually visible without overlapping label text.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return
    xs = df[x_col].values
    ys = df[y_col].values
    n = len(xs)

    entries: List[Tuple[int, str, Dict[str, Any]]] = []
    for phase_key, style in _PHASE_STYLES.items():
        iloc = phases.get(phase_key)
        if iloc is None or iloc < 0 or iloc >= n:
            continue
        if pd.isna(xs[iloc]) or pd.isna(ys[iloc]):
            continue
        entries.append((iloc, phase_key, style))
    if not entries:
        return
    entries.sort(key=lambda e: e[0])

    # Draw every marker first, regardless of label-merging.
    for iloc, _, style in entries:
        if hollow:
            ax.scatter(
                xs[iloc], ys[iloc],
                marker=style["marker"],
                facecolors="none",
                edgecolors=style["color"],
                linewidths=1.2,
                s=style["size"],
                zorder=6,
            )
        else:
            ax.scatter(
                xs[iloc], ys[iloc],
                marker=style["marker"],
                color=style["color"],
                s=style["size"],
                zorder=6,
                edgecolors="black",
                linewidths=0.5,
            )

    if hollow:
        return  # expert-side markers carry no labels

    # Cluster filled-marker entries that sit within the merge threshold.
    clusters: List[List[Tuple[int, str, Dict[str, Any]]]] = []
    for entry in entries:
        if clusters and entry[0] - clusters[-1][-1][0] <= _PHASE_LABEL_MERGE_ILOCS:
            clusters[-1].append(entry)
        else:
            clusters.append([entry])

    for cluster in clusters:
        anchor_iloc, _, anchor_style = cluster[0]
        labels = [f"{s['label']}@{df.index[i]}" for i, _, s in cluster]
        text = " / ".join(labels)
        color = anchor_style["color"] if len(cluster) == 1 else "black"
        kwargs: Dict[str, Any] = dict(
            xy=(xs[anchor_iloc], ys[anchor_iloc]),
            xytext=anchor_style["offset"],
            textcoords="offset points",
            fontsize=8,
            color=color,
            weight="bold",
        )
        if len(cluster) > 1:
            # Leader line so the merged label clearly belongs to the cluster.
            kwargs["arrowprops"] = dict(arrowstyle="-", color="grey", lw=0.5)
        ax.annotate(text, **kwargs)


def _create_gas_brake_trajectory_plot(
    df: pd.DataFrame,
    track_config: Dict[str, str],
    phases: Optional[Dict[str, Optional[int]]] = None,
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

    # Expert-anchored phase markers (entry / apex / exit). Filled markers on
    # the player curve, hollow markers on the expert curve at the same iloc.
    if phases is None:
        phases = _expert_phase_indices(df)
    _annotate_phase_markers(ax, df, phases, px_col, py_col, hollow=False)
    if ex_col and ey_col and ex_col in df.columns and ey_col in df.columns:
        _annotate_phase_markers(ax, df, phases, ex_col, ey_col, hollow=True)

    ax.set_title("Gas/Brake Trajectory\nGreen = Throttle, Red = Brake")
    ax.invert_yaxis()
    ax.set_aspect("equal", "box")
    ax.axis("off")
    if ex_col:
        ax.legend(fontsize=8)
    ax.autoscale()

    return _plot_to_image(fig)


def _create_balance_trajectory_plot(
    df: pd.DataFrame,
    track_config: Dict[str, str],
    phases: Optional[Dict[str, Optional[int]]] = None,
) -> Optional[Image.Image]:
    """Trajectory coloured by oversteer/understeer balance.

    Balance = mean(rear slip) − mean(front slip).
    Positive → rear slipping more → oversteer (red).
    Negative → front slipping more → understeer (amplified ×2, blue).
    """
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

    rear_l = "Physics_slip_angle_rear_left"
    rear_r = "Physics_slip_angle_rear_right"
    front_l = "Physics_slip_angle_front_left"
    front_r = "Physics_slip_angle_front_right"
    has_slip = all(c in df.columns for c in (rear_l, rear_r, front_l, front_r))

    if has_slip:
        understeer_amplifier = 2.0
        balance = (
            (df[rear_l].abs() + df[rear_r].abs()) / 2
            - (df[front_l].abs() + df[front_r].abs()) / 2
        )
        values = np.where(
            balance.values < 0,
            balance.values * understeer_amplifier,
            balance.values,
        ).astype(float)
    else:
        return None  # No slip angle data — skip rather than emit a blank graph

    fig, ax = plt.subplots(figsize=(8, 8))

    # Expert trajectory (reference line)
    ex_col = track_config.get("expert_x")
    ey_col = track_config.get("expert_y")
    if ex_col and ey_col and ex_col in df.columns and ey_col in df.columns:
        ax.plot(
            df[ex_col], df[ey_col],
            color="gray", linewidth=1.5, linestyle="--",
            label="Expert", zorder=1,
        )

    # RdBu_r: red = positive (oversteer), blue = negative (understeer)
    lc = _make_colored_line_collection(
        ax, x, y, values,
        cmap="RdBu_r", vmin=-0.1, vmax=0.1, linewidth=4,
    )
    plt.colorbar(lc, ax=ax, label="← Understeer | Neutral | Oversteer →")

    ax.scatter(x[0], y[0], marker="x", color="black", s=80, zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], marker="o", color="black", s=80, zorder=5, label="End")

    if phases is None:
        phases = _expert_phase_indices(df)
    _annotate_phase_markers(ax, df, phases, px_col, py_col, hollow=False)
    if ex_col and ey_col and ex_col in df.columns and ey_col in df.columns:
        _annotate_phase_markers(ax, df, phases, ex_col, ey_col, hollow=True)

    ax.set_title("Oversteer / Understeer Trajectory\nRed = Oversteer, Blue = Understeer")
    ax.invert_yaxis()
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.legend(fontsize=8)
    ax.autoscale()

    return _plot_to_image(fig)


def _create_trajectory_plot(
    df: pd.DataFrame,
    track_config: Dict[str, str],
    phases: Optional[Dict[str, Optional[int]]] = None,
) -> Optional[Image.Image]:
    """Detailed trajectory plot with expert-anchored phase + min-speed annotations."""
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

    # Expert-anchored phase markers (entry / apex / exit)
    if phases is None:
        phases = _expert_phase_indices(df)
    _annotate_phase_markers(ax, df, phases, px_col, py_col, hollow=False)
    if ex_col and ey_col and ex_col in df.columns and ey_col in df.columns:
        _annotate_phase_markers(ax, df, phases, ex_col, ey_col, hollow=True)

    # Min-speed annotation
    speed_col = None
    for candidate in ("speed_kmh", "Physics_speed_kmh"):
        if candidate in df.columns:
            speed_col = candidate
            break
    if speed_col is not None:
        min_idx = df[speed_col].idxmin()
        if pd.notna(min_idx) and min_idx in df.index:
            row = df.loc[min_idx]
            ax.annotate(
                f"Min speed: {row[speed_col]:.0f} km/h",
                xy=(row[px_col], row[py_col]),
                fontsize=8,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red"),
                xytext=(10, 10),
                textcoords="offset points",
            )

    ax.set_title("Detailed Trajectory")
    ax.invert_yaxis()
    ax.legend()
    ax.set_aspect("equal", "box")
    ax.axis("off")
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
    phases = _expert_phase_indices(segment_df)
    results: List[Tuple[Image.Image, str]] = []

    for gdef in defs:
        gid = gdef["id"]
        title = gdef["title"]
        desc = gdef["description"]
        cols = gdef.get("columns", [])

        if gid == "trajectory_detailed":
            img = _create_trajectory_plot(segment_df, track_config, phases=phases)
        elif gid == "trajectory_gas_brake":
            img = _create_gas_brake_trajectory_plot(segment_df, track_config, phases=phases)
        elif gid == "trajectory_balance":
            img = _create_balance_trajectory_plot(segment_df, track_config, phases=phases)
        else:
            img = _create_feature_plot(segment_df, cols, title)

        if img is not None:
            results.append((img, f"{title}: {desc}"))

    return results
