"""Dispatcher for the lap-to-segment excerpter (manual.py).

Renders both backends as independent expanders, followed by the shared
staged-review panel. Mirrors the layout of
``detailed_agent_annotation.py``.
"""

from __future__ import annotations

import streamlit as st

from ._lap_agent_shared import (
    render_lap_panel, render_lap_staged_review, track_name_to_circuit_id,
)
from .manual_lap_agent_claude import render_lap_agent_claude
from .manual_lap_agent_local import render_lap_agent_local


def render_manual_lap_agent(df, session_id, selected_annotation_key):
    """Render the lap-to-segment excerpter section."""
    st.markdown("---")
    st.subheader("Lap-to-Segment Excerpter (AI Agent)")
    st.caption(
        "Pick a lap range; the deterministic `split_lap_by_circuit_sections` "
        "tool rough-splits it into per-`circuit_section` sub-ranges. The "
        "agent then annotates **one section per click**, shrinking / "
        "extending the boundary when a rule fires."
    )

    track_name = (
        df["Static_track"].iloc[0]
        if "Static_track" in df.columns and not df.empty else None
    )
    circuit_id = track_name_to_circuit_id(track_name)

    # Lap range picker + rough split + current-section view render ONCE here
    # so the backend expanders share one widget key namespace.
    head = render_lap_panel(df, circuit_id)

    render_lap_agent_local(df, session_id, selected_annotation_key, circuit_id, head)
    render_lap_agent_claude(df, session_id, selected_annotation_key, circuit_id, head)

    render_lap_staged_review(session_id, selected_annotation_key, df=df)
