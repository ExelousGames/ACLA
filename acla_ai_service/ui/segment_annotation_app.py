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
from segment_tabs.shared import (
    get_store, TrainingPipelineConfig, get_available_sessions, register_output_dir,
)
from segment_tabs.manual import render_manual_annotation
from segment_tabs.detailed import render_detailed_labeling
from segment_tabs.batch import (
    render_batch_bulk_label, render_batch_rule_based, render_batch_classifier,
    render_batch_subseg, render_batch_lap,
)
from segment_tabs.training import TRAINING_ROUTES, render_training
from segment_tabs.pipeline_view import render_pipeline_view
from segment_tabs.pipeline_sidebar import render_pipeline_sidebar
from segment_tabs.output_picker import needs_output_setup, render_output_picker
from app.pipelines.manifest.models import MODE_SOURCE
from app.pipelines.manifest.registry import list_pipelines, load as load_pipeline
from app.pipelines.manifest.source_copy import source_copy_key
from app.storage.lance.backup import (
    create_lance_backup,
    list_lance_backups,
    restore_lance_backup,
)


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
# Training routes share the single training tab.
_TRAINING_ROUTES = set(TRAINING_ROUTES)

_ALL_ROUTES = set(_SESSION_GATED_ROUTES) | _TRAINING_ROUTES


def _go_back_to_pipeline() -> None:
    for k in ("active_view", "pipeline_routed_view", "pipeline_active_node_id"):
        st.session_state.pop(k, None)


def _sync_pipeline_dir_map(pipeline) -> None:
    """Publish each annotation node's (output_key → output_dir) into
    session_state so the per-key store lookup picks up custom dirs
    even when reading from a sibling node's output."""
    for node in pipeline.annotations:
        if node.output_key:
            register_output_dir(node.output_key, node.output_dir)
            if node.mode == MODE_SOURCE:
                register_output_dir(source_copy_key(node.output_key), node.output_dir)


def _sync_all_pipeline_dir_maps() -> None:
    for pipeline_id in list_pipelines():
        pipeline = load_pipeline(pipeline_id)
        if pipeline is not None:
            _sync_pipeline_dir_map(pipeline)


def _format_backup_label(item: dict) -> str:
    size = item.get("size_mb")
    size_label = f"{size} MB" if size is not None else "unknown size"
    return f"{item.get('filename', 'unknown backup')} | {size_label}"


def _backup_preview_rows(store_info: dict) -> list[dict]:
    return [
        {
            "dataset": entry.get("cache_key"),
            "chunks": entry.get("chunk_count", 0),
            "records": entry.get("total_records", 0),
            "size_mb": entry.get("size_mb", 0),
            "strategy": entry.get("strategy"),
            "directory": entry.get("directory"),
            "updated_at": entry.get("updated_at"),
        }
        for entry in store_info.get("entries", [])
    ]


def _render_dataset_backup_section(store) -> None:
    with st.expander("Dataset backups", expanded=False):
        notice = st.session_state.pop("dataset_backup_notice", None)
        if notice:
            st.success(notice)

        try:
            store_info = store.get_cache_info()
            backup_info = list_lance_backups(store)
        except Exception as exc:
            st.error(f"Could not load backup status: {exc}")
            return

        st.caption(
            f"{store_info.get('entry_count', 0)} datasets | "
            f"{store_info.get('total_size_mb', 0)} MB"
        )
        st.caption(f"Source directory: `{store_info['store_directory']}`")
        st.caption(f"Backup directory: `{backup_info['backup_directory']}`")

        st.markdown("**This backup will save**")
        preview_rows = _backup_preview_rows(store_info)
        if preview_rows:
            st.dataframe(
                preview_rows,
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No datasets found in the Lance store.")

        if st.button("Create full dataset backup", width="stretch"):
            try:
                result = create_lance_backup(store)
                backup = result["backup"]
                manifest = result["manifest"]
                st.session_state["dataset_backup_notice"] = (
                    f"Created `{backup['filename']}` "
                    f"with {manifest['entry_count']} datasets."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Backup failed: {exc}")

        backups = backup_info["backups"]
        if not backups:
            st.info("No backups found yet.")
            return

        st.markdown("**Available backups**")
        st.dataframe(
            [
                {
                    "filename": backup["filename"],
                    "size_mb": backup["size_mb"],
                    "created_at": backup["created_at"],
                }
                for backup in backups[:10]
            ],
            hide_index=True,
            width="stretch",
        )

        selected = st.selectbox(
            "Restore backup",
            backups,
            format_func=_format_backup_label,
            key="dataset_backup_restore_choice",
        )
        confirmed = st.checkbox(
            "I understand restore replaces the current dataset store",
            key="dataset_backup_restore_confirmed",
        )
        if st.button(
            "Restore selected backup",
            type="secondary",
            width="stretch",
            disabled=not confirmed,
        ):
            try:
                result = restore_lance_backup(selected["filename"], store)
                notice = f"Restored `{result['restored_backup']['filename']}`."
                if result.get("safety_backup"):
                    notice += (
                        " Current data was saved first as "
                        f"`{result['safety_backup']['filename']}`."
                    )
                st.session_state["dataset_backup_notice"] = notice
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Restore failed: {exc}")


def main() -> None:
    store = get_store()
    cfg = TrainingPipelineConfig()

    # ── Sidebar: pipeline picker + git-style snapshot view ────────────
    with st.sidebar:
        if st.button("Finish & Exit", type="primary",
                     help="Close the app"):
            st.success("Exiting...")
            time.sleep(0.5)
            os._exit(0)
        st.markdown("---")
        pipeline = render_pipeline_sidebar(store, cfg)
        _sync_all_pipeline_dir_maps()
        st.markdown("---")
        _render_dataset_backup_section(store)

    st.title("Telemetry Annotation Pipeline")

    if pipeline is None:
        st.info("👈 Create a pipeline in the sidebar to begin. "
                "Then add annotation/training components in the graph view.")
        return

    # Publish per-node output dir mappings so the shared store helpers
    # route reads/writes to the right Lance store, even when one node
    # references another's output as a sibling source.
    _sync_pipeline_dir_map(pipeline)

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
    active_node_id = st.session_state.get("pipeline_active_node_id")

    # ── First-time output-location popup for own-output nodes ───────────
    # If we routed here from a node card and the node still has no
    # output_key, the popup blocks until the user picks a directory + name.
    active_node = None
    if active_node_id:
        try:
            active_node = pipeline.annotation(active_node_id)
        except KeyError:
            active_node = None
    if active_node is not None and needs_output_setup(active_node):
        render_output_picker(pipeline, active_node)
        return

    # ── Training tabs — one page per training kind ──────────────────────
    if active_view in _TRAINING_ROUTES:
        render_training(active_view, annotation_key)
        return

    # Reconcile keys from the active node after route changes or output setup.
    if active_node is not None:
        annotation_key = pipeline.effective_output_key(active_node)
        session_key = pipeline.effective_input_key(active_node)
        st.session_state["pipeline_annotation_key"] = annotation_key
        st.session_state["pipeline_session_key"] = session_key

    if not annotation_key:
        st.warning("This view needs an annotation dataset — return to the Pipeline view "
                   "and open it from a node.")
        return

    st.info(f"Active annotation dataset: **{annotation_key}**")

    # ── Session-data-gated tabs ─────────────────────────────────────────
    if not session_key:
        st.error("Pipeline node has no input dataset — pick a source from the Pipeline view.")
        return
    if session_key not in store.list_cache_keys():
        if active_node is not None and active_node.mode == MODE_SOURCE and active_node.output_key:
            st.warning(
                "This annotation reads from a private source copy, but that "
                "copy has not been created yet."
            )
            st.caption("Return to the Pipeline view and create the source copy there.")
            return
        st.error(f"Input dataset `{session_key}` not found in the store. "
                 "It may have been deleted — re-pick the source.")
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
