from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.shared.annotation_agent_tools import (
    _active_opponent_mask,
    _relative_position_frame,
    classify_opponent_interaction,
)


OUTCOME_LABELS = {
    "pass_completed": "Pass completed",
    "broken_defense": "Defense broken",
    "failed_attack": "Failed attack",
    "held_defense": "Held defense",
    "side_by_side": "Side by side",
    "incidental": "Incidental",
    "no_close_interaction": "No close interaction",
    "no_data": "No data",
}

ROLE_LABELS = {
    "attack": "Attack",
    "defense": "Defense",
    "side_by_side": "Side by side",
    "incidental": "Incidental",
    "none": "None",
    "unknown": "Unknown",
}

ROLE_COLORS = {
    "attack": "#ff7f0e",
    "defense": "#8a63d2",
    "side_by_side": "#17becf",
    "incidental": "#7f7f7f",
    "none": "#7f7f7f",
    "unknown": "#7f7f7f",
}


def _content_from_attachment(attachment: Any) -> dict[str, Any]:
    content = getattr(attachment, "content", None)
    return content if isinstance(content, dict) else {}


def _fmt_meters(value: Any) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "-"
        return f"{float(value):.1f} m"
    except (TypeError, ValueError):
        return "-"


def _fmt_confidence(value: Any, level: Any) -> str:
    try:
        return f"{float(value):.2f} ({level})"
    except (TypeError, ValueError):
        return str(level or "-")


def _candidate_label(candidate: dict[str, Any]) -> str:
    slot = candidate.get("slot", "?")
    role = ROLE_LABELS.get(str(candidate.get("role")), str(candidate.get("role", "unknown")))
    outcome = OUTCOME_LABELS.get(str(candidate.get("outcome")), str(candidate.get("outcome", "unknown")))
    min_dist = _fmt_meters(candidate.get("min_distance_m"))
    return f"Car {slot} - {role} - {outcome} - closest {min_dist}"


def compute_relative_trace(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    slot: int,
) -> pd.DataFrame:
    """Continuous player/opponent relative-position trace for one car slot."""
    s = max(0, int(start_idx))
    e = min(len(df), int(end_idx))
    if e <= s + 1:
        return pd.DataFrame()

    col_x = f"Car_{int(slot)}_pos_x"
    col_y = f"Car_{int(slot)}_pos_y"
    required = {"Graphics_player_pos_x", "Graphics_player_pos_y", col_x, col_y}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    seg = df.iloc[s:e]
    player_x = seg["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = seg["Graphics_player_pos_y"].to_numpy(dtype=float)
    opp_x = seg[col_x].to_numpy(dtype=float)
    opp_y = seg[col_y].to_numpy(dtype=float)
    active = _active_opponent_mask(seg, int(slot), opp_x, opp_y, player_x, player_y)

    signed_long, lateral, _player_s, _player_d, frame_name = _relative_position_frame(
        seg, player_x, player_y, opp_x, opp_y,
    )
    distance = np.sqrt((opp_x - player_x) ** 2 + (opp_y - player_y) ** 2)
    valid = active & np.isfinite(distance) & np.isfinite(signed_long) & np.isfinite(lateral)

    out = pd.DataFrame({
        "iloc": seg.index.to_numpy(dtype=int),
        "player_x": player_x,
        "player_y": player_y,
        "opponent_x": opp_x,
        "opponent_y": opp_y,
        "distance_m": np.where(valid, distance, np.nan),
        "signed_long_gap_m": np.where(valid, signed_long, np.nan),
        "lateral_offset_m": np.where(valid, lateral, np.nan),
        "active": valid,
    })
    out.attrs["coordinate_frame"] = frame_name

    player_z = seg.get("Graphics_player_pos_z")
    opp_z = seg.get(f"Car_{int(slot)}_pos_z")
    if player_z is not None:
        out["player_z"] = player_z.to_numpy(dtype=float)
    if opp_z is not None:
        out["opponent_z"] = opp_z.to_numpy(dtype=float)
    return out


def render_opponent_interaction_panel(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    key_prefix: str,
) -> Optional[dict[str, Any]]:
    """Render attack/defense evidence and return the focused opponent slot."""
    show = st.checkbox(
        "Show Racing Interaction",
        value=True,
        key=f"{key_prefix}_interaction_visible",
    )
    if not show:
        return None

    s = max(0, int(start_idx))
    e = min(len(df), int(end_idx))
    if e <= s + 1:
        st.info("Select at least two samples to analyze opponent interaction.")
        return None

    try:
        attachment = classify_opponent_interaction(df, s, e)
        content = _content_from_attachment(attachment)
    except Exception as exc:
        st.warning(f"Opponent interaction analysis failed: {exc}")
        return None

    if not content.get("data_available"):
        message = content.get("message") or "Opponent interaction data is not available for this range."
        st.info(message)
        return None

    candidates = [
        c for c in content.get("candidates", [])
        if isinstance(c, dict) and c.get("slot") is not None
    ]
    if not candidates:
        st.info(content.get("message") or "No active opponent close enough in this range.")
        return None

    primary_slot = content.get("primary_slot_for_role")
    default_idx = 0
    for idx, candidate in enumerate(candidates):
        if candidate.get("slot") == primary_slot:
            default_idx = idx
            break

    selected = st.selectbox(
        "Focus Car",
        options=candidates,
        index=default_idx,
        format_func=_candidate_label,
        key=f"{key_prefix}_interaction_focus_car",
    )

    role = ROLE_LABELS.get(str(selected.get("role")), str(selected.get("role", "unknown")))
    outcome = OUTCOME_LABELS.get(str(selected.get("outcome")), str(selected.get("outcome", "unknown")))
    recommended = selected.get("recommended_label") or "-"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Role", role)
    col2.metric("Outcome", outcome)
    col3.metric("Label Gate", recommended)
    col4.metric("Confidence", _fmt_confidence(selected.get("confidence"), selected.get("confidence_level")))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Entry Gap", _fmt_meters(selected.get("entry_signed_long_gap_m")))
    col6.metric("Exit Gap", _fmt_meters(selected.get("exit_signed_long_gap_m")))
    col7.metric("Closest", _fmt_meters(selected.get("min_distance_m")))
    col8.metric("Side-by-Side", str(selected.get("side_by_side_iloc_count", 0)))

    reason = selected.get("reason")
    if reason:
        st.caption(str(reason))

    return {
        "classification": content,
        "candidate": selected,
        "slot": int(selected["slot"]),
        "role": str(selected.get("role", content.get("role", "unknown"))),
        "outcome": str(selected.get("outcome", content.get("outcome", "unknown"))),
        "trace": compute_relative_trace(df, s, e, int(selected["slot"])),
    }


def add_interaction_overlay(
    fig: go.Figure,
    interaction: Optional[dict[str, Any]],
    *,
    use_3d: bool,
    has_z: bool,
) -> None:
    """Highlight the focused opponent trajectory and player-opponent links."""
    if not interaction:
        return
    trace = interaction.get("trace")
    if not isinstance(trace, pd.DataFrame) or trace.empty or "active" not in trace:
        return

    active = trace[trace["active"]].copy()
    if active.empty:
        return

    slot = interaction.get("slot", "?")
    role = str(interaction.get("role", "unknown"))
    role_label = ROLE_LABELS.get(role, role.title())
    color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
    line_name = f"Focus Car {slot} ({role_label})"

    if use_3d and has_z and {"opponent_z"}.issubset(active.columns):
        fig.add_trace(go.Scatter3d(
            x=active["opponent_x"],
            y=active["opponent_y"],
            z=active["opponent_z"],
            customdata=active[["iloc", "distance_m", "signed_long_gap_m", "lateral_offset_m"]],
            hovertemplate=(
                "Index: %{customdata[0]}<br>"
                "Distance: %{customdata[1]:.1f} m<br>"
                "Long gap: %{customdata[2]:.1f} m<br>"
                "Lateral: %{customdata[3]:.1f} m<extra></extra>"
            ),
            mode="lines",
            name=line_name,
            line=dict(color=color, width=9),
            opacity=1.0,
            showlegend=True,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=active["opponent_x"],
            y=active["opponent_y"],
            customdata=active[["iloc", "distance_m", "signed_long_gap_m", "lateral_offset_m"]],
            hovertemplate=(
                "Index: %{customdata[0]}<br>"
                "Distance: %{customdata[1]:.1f} m<br>"
                "Long gap: %{customdata[2]:.1f} m<br>"
                "Lateral: %{customdata[3]:.1f} m<extra></extra>"
            ),
            mode="lines",
            name=line_name,
            line=dict(color=color, width=4),
            opacity=1.0,
            showlegend=True,
        ))

    marker_rows = _interaction_marker_rows(active)
    for label, row in marker_rows:
        _add_player_opponent_link(fig, row, label, color=color, use_3d=use_3d, has_z=has_z)


def _interaction_marker_rows(active: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    rows: list[tuple[str, pd.Series]] = []
    rows.append(("Entry", active.iloc[0]))
    min_idx = active["distance_m"].astype(float).idxmin()
    rows.append(("Closest", active.loc[min_idx]))
    if int(active.iloc[-1]["iloc"]) != int(active.iloc[0]["iloc"]):
        rows.append(("Exit", active.iloc[-1]))

    deduped: list[tuple[str, pd.Series]] = []
    seen: set[int] = set()
    for label, row in rows:
        iloc = int(row["iloc"])
        if iloc in seen:
            continue
        seen.add(iloc)
        deduped.append((label, row))
    return deduped


def _add_player_opponent_link(
    fig: go.Figure,
    row: pd.Series,
    label: str,
    *,
    color: str,
    use_3d: bool,
    has_z: bool,
) -> None:
    hover = (
        f"{label}<br>"
        f"Index: {int(row['iloc'])}<br>"
        f"Distance: {float(row['distance_m']):.1f} m<br>"
        f"Long gap: {float(row['signed_long_gap_m']):.1f} m<br>"
        f"Lateral: {float(row['lateral_offset_m']):.1f} m"
    )
    if use_3d and has_z and "player_z" in row and "opponent_z" in row:
        fig.add_trace(go.Scatter3d(
            x=[row["player_x"], row["opponent_x"]],
            y=[row["player_y"], row["opponent_y"]],
            z=[row["player_z"], row["opponent_z"]],
            mode="lines+markers",
            name=f"{label} gap",
            hovertemplate=hover + "<extra></extra>",
            line=dict(color=color, width=5, dash="dash"),
            marker=dict(size=4, color=color),
            opacity=0.95,
            showlegend=True,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[row["player_x"], row["opponent_x"]],
            y=[row["player_y"], row["opponent_y"]],
            mode="lines+markers",
            name=f"{label} gap",
            hovertemplate=hover + "<extra></extra>",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=7, color=color),
            opacity=0.95,
            showlegend=True,
        ))

