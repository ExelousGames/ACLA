"""Streamlit shell for the segment annotation + training pipeline.

The top-level navigation lives entirely in the Pipeline graph view: each
annotation node card has a *kind* dropdown and one Open button that
routes to the tab whose ``ui_route`` matches that kind. There is no
top-level view radio — opened pages render in place with a "← Back to
Pipeline" button to return.
"""

import torch

# Hack to fix Streamlit's file watcher crashing on torch.classes
try:
    if not hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = []
except Exception:
    pass

import streamlit as st
st.set_page_config(page_title="Segment Annotation Pipeline", layout="wide")
import os
import sys
import time
from pathlib import Path


# ── Path bootstrap ────────────────────────────────────────────────────────
def _ensure_app_module_on_path() -> None:
    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            scripts_path = (candidate / "scripts")
            if scripts_path.exists():
                scripts_path_str = scripts_path.as_posix()
                if scripts_path_str not in sys.path:
                    sys.path.insert(0, scripts_path_str)
            return
        candidate = candidate.parent


_ensure_app_module_on_path()


# ── Imports (after path bootstrap) ────────────────────────────────────────
from segment_tabs.shared import get_store, PipelineConfig, get_available_sessions
from segment_tabs.manual import render_manual_annotation
from segment_tabs.detailed import render_detailed_labeling
from segment_tabs.batch import (
    render_batch_bulk_label, render_batch_rule_based, render_batch_classifier,
    render_batch_subseg, render_batch_lap,
)
from segment_tabs.llm_pipeline import render_llm_pipeline
from segment_tabs.training import render_training
from segment_tabs.pipeline_view import render_pipeline_view
from segment_tabs.pipeline_sidebar import render_pipeline_sidebar


# ── Route → renderer map ──────────────────────────────────────────────────
# Each annotation tab takes (annotation_key, session_key, available_sessions).
_SESSION_GATED_ROUTES = {
    "lap":                render_manual_annotation,
    "detailed":           render_detailed_labeling,
    "batch_bulk_label":   render_batch_bulk_label,
    "batch_rule_based":   render_batch_rule_based,
    "batch_classifier":   render_batch_classifier,
    "batch_subseg":       render_batch_subseg,
    "batch_lap":          render_batch_lap,
}
# Routes that only need the annotation_key.
_ANNOTATION_ONLY_ROUTES = {
    "llm": render_llm_pipeline,
}
# Training routes — all three share the single training tab.
_TRAINING_ROUTES = {"classifier", "transformer", "llm_training"}

_ALL_ROUTES = (
    set(_SESSION_GATED_ROUTES) | set(_ANNOTATION_ONLY_ROUTES) | _TRAINING_ROUTES
)


def _go_back_to_pipeline() -> None:
    for k in ("active_view", "pipeline_routed_view"):
        st.session_state.pop(k, None)


def main() -> None:
    store = get_store()
    cfg = PipelineConfig()

    # ── Sidebar: pipeline picker + git-style snapshot view ────────────
    with st.sidebar:
        if st.button("Finish & Exit", type="primary",
                     help="Close the app"):
            st.success("Exiting...")
            time.sleep(0.5)
            os._exit(0)
        st.markdown("---")
        pipeline = render_pipeline_sidebar(store, cfg)

    st.title("Telemetry Annotation Pipeline")

    if pipeline is None:
        st.info("👈 Create a pipeline in the sidebar to begin. "
                "Then add annotation/training components in the graph view.")
        return

    # ── Resolve routing: a node may have asked us to switch tab ─────────
    routed = st.session_state.pop("pipeline_routed_view", None)
    if routed and routed in _ALL_ROUTES:
        st.session_state["active_view"] = routed

    active_view = st.session_state.get("active_view")
    if active_view not in _ALL_ROUTES:
        active_view = None

    # ── Default: pipeline graph view ────────────────────────────────────
    if active_view is None:
        render_pipeline_view(pipeline, store)
        return

    # ── Back navigation in place of the old top radio ───────────────────
    if st.button("← Back to Pipeline", key="back_to_pipeline"):
        _go_back_to_pipeline()
        st.rerun()

    # ── Resolve cache_keys to feed downstream tabs ──────────────────────
    annotation_key = st.session_state.get("pipeline_annotation_key")
    session_key = st.session_state.get("pipeline_session_key")

    # ── Training tab — all three training kinds land here ───────────────
    if active_view in _TRAINING_ROUTES:
        render_training(annotation_key)
        return

    if not annotation_key:
        st.warning("This view needs an annotation dataset — return to the Pipeline view "
                   "and open it from a node.")
        return

    st.info(f"Active annotation dataset: **{annotation_key}**")

    # ── Annotation-key-only tabs (no session data needed) ───────────────
    if active_view in _ANNOTATION_ONLY_ROUTES:
        _ANNOTATION_ONLY_ROUTES[active_view](annotation_key)
        return

    # ── Session-data-gated tabs ─────────────────────────────────────────
    if not session_key:
        st.error("Pipeline node has no input dataset — fork the source from the Pipeline view.")
        return
    if session_key not in store.list_cache_keys():
        st.error(f"Input dataset `{session_key}` not found in the store. "
                 "It may have been deleted — re-fork or re-pick the source.")
        return

    st.info(f"Annotating data from: `{session_key}`")

    available_sessions = get_available_sessions(session_key)
    if not available_sessions:
        st.warning("Input dataset has no sessions yet.")
        return

    renderer = _SESSION_GATED_ROUTES[active_view]
    renderer(annotation_key, session_key, available_sessions)


if __name__ == "__main__":
    main()
