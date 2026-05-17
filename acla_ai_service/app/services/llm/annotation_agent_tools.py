"""
Tool definitions for the annotation agent pipeline.

Provides two tool categories that the agent nodes can invoke:

1. **Graph generation** — per-graph DataFrame builders + matplotlib
   renderers (feature plots, trajectory plots) aligned with the
   ``AGENT_GRAPH_DEFINITIONS`` used by the human annotation workflow.
2. **Deterministic queries** — the ``PIPELINE_QUERY_DEFINITIONS`` catalog
   of structured math operations (threshold crossings, extrema, slopes,
   onset ordering) the zoom executor runs against the graph tables to
   extract exact ilocs / values for the synthesizer to cite.

Together they let the vision-capable LLM see the same visual evidence a
human annotator sees while the executor produces verifiable numerical
readings off the underlying DataFrame.
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
    kappa = np.where(denom > 1e-9, (dx * ddy - dy * ddx) / denom, 0.0)
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
    from .step_evaluator_agents import PipelineAttachment

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


NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"


def split_lap_by_circuit_sections(
    df: pd.DataFrame, start_index: int, end_index: int,
    circuit_id: Optional[str] = None,
):
    """Tool — partition a lap-shaped range into per-`circuit_section` sub-ranges.

    Walks ``Graphics_normalized_car_position`` sample-by-sample across
    ``[start_index, end_index)`` and assigns every iloc to the
    ``circuit_section`` whose ``normalized_position_range`` contains its
    position fraction. Consecutive ilocs that land in the same section are
    grouped into one sub-range. Wrap-around sections (``range_end <
    range_start``) and lap roll-over (a sample where position resets
    1.0 → 0.0) are handled.

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
            "segments": [
                {
                    "start_index": int,
                    "end_index": int,
                    "circuit_section_id": str,
                    "circuit_section_name": str,
                    "normalized_position_range": [float, float],
                    "coverage_fraction": 0.0..1.0,
                },
                ...
            ],
            "unmatched_ilocs": int,   # samples that hit no defined section
        }

    Segments are ordered by ``start_index``. ``coverage_fraction`` is the
    share of the segment's iloc span that fell inside the matched section's
    range — informative when a section's range is narrow and the player
    crossed in/out at the boundary.
    """
    from .step_evaluator_agents import PipelineAttachment
    from app.models.label_catalog import get_label_catalog

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

    catalog = get_label_catalog()
    candidates: List[Dict[str, Any]] = []
    for entry in catalog.entries_by_type("circuit_section"):
        rng = entry.normalized_position_range
        if rng is None:
            continue
        if circuit_id is not None and entry.parent != circuit_id:
            continue
        candidates.append({
            "id": entry.id,
            "name": entry.name,
            "lo": float(rng[0]),
            "hi": float(rng[1]),
        })

    if not candidates:
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
            })
        cur_section = None
        cur_start_iloc = end_iloc_exclusive
        matched_in_run = 0

    for offset, p in enumerate(pos):
        iloc = s + offset
        section = _section_for(p)
        if section is None:
            unmatched += 1
            # Keep the run open — the player may have a noisy sample.
            if cur_section is None:
                cur_start_iloc = iloc + 1
            continue
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

    return _attach({
        "circuit_id": circuit_id,
        "range": [s, e],
        "segments": segments,
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
            "best_match": <first entry of top_matches, or None>,
        }

    ``top_matches`` is empty when the column is missing, all values are
    non-finite, or no circuit_section in the catalog has its range filled
    in yet. ``best_match`` is the agent's recommended label.
    """
    from .step_evaluator_agents import PipelineAttachment
    from app.models.label_catalog import get_label_catalog

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
            "best_match": None,
        })

    pos = df.iloc[s:e][NORMALIZED_POSITION_COLUMN].to_numpy(dtype=float)
    pos = pos[np.isfinite(pos)]
    if pos.size == 0:
        return _attach({
            "error": "no finite values in normalized position over the segment",
            "top_matches": [],
            "best_match": None,
        })

    seg_lo, seg_hi = float(pos.min()), float(pos.max())
    seg_span = max(seg_hi - seg_lo, 1e-6)

    catalog = get_label_catalog()
    matches: List[Dict[str, Any]] = []
    for entry in catalog.entries_by_type("circuit_section"):
        rng = entry.normalized_position_range
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
            "label_id": entry.id,
            "name": entry.name,
            "description": entry.description,
            "section_range": [r_lo, r_hi],
            "overlap_fraction": overlap / seg_span,
        })

    matches.sort(key=lambda m: m["overlap_fraction"], reverse=True)
    top = matches[:3]
    return _attach({
        "segment_position_range": [seg_lo, seg_hi],
        "top_matches": top,
        "best_match": top[0] if top else None,
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
            "(ranked by overlap fraction) and a 'best_match' suggestion. "
            "Use whenever you need to label which named corner / straight "
            "the segment is on — never guess from telemetry shape alone."
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
            "the sample. Produces a 'split_lap_sections' attachment with "
            "an ordered `segments` list (`start_index`, `end_index`, "
            "`circuit_section_id`, `coverage_fraction`). Used by the "
            "lap-to-segment excerpter to compute the rough split that "
            "feeds the per-section annotation agent."
        ),
        "callable": split_lap_by_circuit_sections,
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
