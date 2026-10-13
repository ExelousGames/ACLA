from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from training.shared.circuit_sections import get_circuit_sections


NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"

SECTION_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#c026d3",
]


def track_sections_available(df: pd.DataFrame) -> bool:
    return bool(get_circuit_sections(df))


def add_track_section_trajectory(
    fig,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: Optional[str] = None,
    *,
    use_3d: bool = False,
) -> None:
    for run in _section_runs(df):
        run_df = run["df"]
        if len(run_df) < 2 or x_col not in run_df.columns or y_col not in run_df.columns:
            continue
        color = _section_color(run["section_id"])
        hover_text = _hover_texts(run_df, run)
        common = dict(
            mode="lines+markers",
            name=f"Section: {run['section_name']}",
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            opacity=0.78,
        )
        if use_3d and z_col and z_col in run_df.columns:
            fig.add_trace(go.Scatter3d(
                x=run_df[x_col],
                y=run_df[y_col],
                z=run_df[z_col],
                line=dict(color=color, width=8),
                marker=dict(color=color, size=3),
                **common,
            ))
        elif not use_3d:
            fig.add_trace(go.Scatter(
                x=run_df[x_col],
                y=run_df[y_col],
                line=dict(color=color, width=6),
                marker=dict(color=color, size=4),
                **common,
            ))


def add_track_section_bands(fig, df: pd.DataFrame, anchor_col: str) -> None:
    if anchor_col not in df.columns:
        return
    for run in _section_runs(df):
        run_df = run["df"]
        if run_df.empty:
            continue
        color = _section_color(run["section_id"])
        start = run["start_index"]
        end = run["end_index"]
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=_rgba(color, 0.10),
            line_width=1,
            line_color=_rgba(color, 0.45),
            layer="below",
        )
        fig.add_trace(go.Scatter(
            x=run_df.index,
            y=run_df[anchor_col],
            mode="lines+markers",
            line=dict(color=_rgba(color, 0.18), width=8),
            marker=dict(color=_rgba(color, 0.30), size=3),
            hoverinfo="text",
            hovertext=_hover_texts(run_df, run),
            showlegend=False,
        ))


def _section_for_position(
    position: float,
    candidates: list[tuple[str, tuple[float, float]]],
) -> Optional[str]:
    position = position % 1.0
    for section_id, (lo, hi) in candidates:
        if hi >= lo:
            if lo <= position <= hi:
                return section_id
        elif position >= lo or position <= hi:
            return section_id
    return None


def _section_runs(df: pd.DataFrame) -> list[dict]:
    if NORMALIZED_POSITION_COLUMN not in df.columns or df.empty:
        return []

    sections = {section["id"]: section for section in get_circuit_sections(df)}
    candidates = [
        (section["id"], section["normalized_position_range"])
        for section in sections.values()
    ]
    if not candidates:
        return []

    positions = pd.to_numeric(df[NORMALIZED_POSITION_COLUMN], errors="coerce")
    runs = []
    current_section: Optional[str] = None
    current_start = 0

    def close_run(end_iloc: int) -> None:
        if current_section is None or end_iloc <= current_start:
            return
        run_df = df.iloc[current_start:end_iloc]
        if run_df.empty:
            return
        section = sections[current_section]
        lo, hi = section["normalized_position_range"]
        runs.append({
            "section_id": current_section,
            "section_name": section["name"],
            "normalized_range": (lo, hi),
            "start_index": run_df.index[0],
            "end_index": run_df.index[-1],
            "df": run_df,
        })

    for iloc, value in enumerate(positions):
        section_id = None
        if pd.notna(value):
            section_id = _section_for_position(float(value), candidates)
        if section_id != current_section:
            close_run(iloc)
            current_section = section_id
            current_start = iloc

    close_run(len(df))
    return runs


def _section_color(section_id: str) -> str:
    color_idx = sum(ord(ch) for ch in section_id) % len(SECTION_COLORS)
    return SECTION_COLORS[color_idx]


def _hover_texts(run_df: pd.DataFrame, run: dict) -> list[str]:
    lo, hi = run["normalized_range"]
    texts = []
    positions = pd.to_numeric(
        run_df.get(NORMALIZED_POSITION_COLUMN, pd.Series(index=run_df.index)),
        errors="coerce",
    )
    for idx, position in positions.items():
        position_text = f"{position:.4f}" if pd.notna(position) else "N/A"
        texts.append(
            f"<b>{run['section_name']}</b>"
            f"<br>Section ID: {run['section_id']}"
            f"<br>Section range: {lo:.4f}-{hi:.4f}"
            f"<br>Index: {idx}"
            f"<br>Normalized position: {position_text}"
        )
    return texts


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"
