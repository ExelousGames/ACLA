"""Sidebar: pipeline picker + maintenance.

Picker chooses which pipeline is active. Creating a new pipeline just
takes a name; the user picks the source for each annotation component
individually in the main view.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from app.pipelines.training.config import TrainingPipelineConfig
from app.pipelines.manifest.models import Pipeline
from app.pipelines.manifest.registry import (
    create_pipeline,
    delete as delete_pipeline,
    list_pipelines,
    load as load_pipeline,
)


_ACTIVE_KEY = "active_pipeline_id"


def _query_param(name: str) -> Optional[str]:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _clear_route_query() -> None:
    for key in ("view", "node"):
        if key in st.query_params:
            del st.query_params[key]


def get_active_pipeline_id() -> Optional[str]:
    return st.session_state.get(_ACTIVE_KEY)


def set_active_pipeline_id(pid: Optional[str]) -> None:
    if pid is None:
        st.session_state.pop(_ACTIVE_KEY, None)
    else:
        st.session_state[_ACTIVE_KEY] = pid


def render_pipeline_sidebar(store: Any, cfg: TrainingPipelineConfig) -> Optional[Pipeline]:
    """Render the sidebar and return the active Pipeline (or None)."""
    st.header("Pipelines")

    pipelines = list_pipelines()
    query_pipeline = _query_param("pipeline")

    # ── Picker ──────────────────────────────────────────────────────────
    active_id = get_active_pipeline_id()
    if query_pipeline in pipelines and query_pipeline != active_id:
        set_active_pipeline_id(query_pipeline)
        st.session_state["pipeline_picker"] = query_pipeline
        active_id = query_pipeline
    if active_id and active_id not in pipelines:
        set_active_pipeline_id(None)
        active_id = None

    if pipelines:
        idx = pipelines.index(active_id) if active_id in pipelines else 0
        chosen = st.selectbox(
            "Active pipeline", pipelines, index=idx, key="pipeline_picker",
        )
        if chosen != active_id:
            set_active_pipeline_id(chosen)
            st.query_params["pipeline"] = chosen
            _clear_route_query()
            active_id = chosen
            st.rerun()
    else:
        st.caption("No pipelines yet — create one below.")

    # ── Create new ──────────────────────────────────────────────────────
    with st.expander("➕ New pipeline", expanded=not pipelines):
        new_name = st.text_input(
            "Name", placeholder="e.g. exp_2026_05_21_a", key="new_pipeline_name",
        )
        st.caption(
            "Pipeline starts empty. You'll pick a source dataset for each "
            "annotation component in the graph view."
        )
        if st.button("Create pipeline", type="primary",
                     width="stretch", disabled=not new_name):
            try:
                pipeline = create_pipeline(
                    name=new_name,
                    annotation_prefix=cfg.annotation_cache_key,
                )
                set_active_pipeline_id(pipeline.id)
                st.query_params["pipeline"] = pipeline.id
                _clear_route_query()
                st.success(f"Created pipeline `{new_name}`.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to create pipeline: {exc}")

    if active_id is None:
        return None

    pipeline = load_pipeline(active_id)
    if pipeline is None:
        st.error(f"Pipeline `{active_id}` no longer on disk.")
        set_active_pipeline_id(None)
        return None

    # ── Maintenance ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Maintenance"):
        if st.button("🗑️ Delete this pipeline", type="secondary",
                     width="stretch"):
            delete_pipeline(pipeline.id)
            set_active_pipeline_id(None)
            if "pipeline" in st.query_params:
                del st.query_params["pipeline"]
            _clear_route_query()
            st.success(f"Deleted pipeline `{pipeline.id}`.")
            st.rerun()

    return pipeline
