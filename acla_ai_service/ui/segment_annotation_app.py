"""
Streamlit UI for manually annotating telemetry segments with behavioral labels.
Refactored to use components.
"""

import torch 

# Hack to fix Streamlit's file watcher crashing on torch.classes
try:
    if not hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = [] 
except Exception:
    pass

import streamlit as st
st.set_page_config(page_title="Segment Annotation App", layout="wide")
import pandas as pd
import time
import os
from pathlib import Path
import sys

# Ensure app module is on path (redundant but safe entry point check)
def _ensure_app_module_on_path() -> None:
    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            
            # Also add scripts folder to path to allow importing migration scripts
            scripts_path = (candidate / "scripts")
            if scripts_path.exists():
                scripts_path_str = scripts_path.as_posix()
                if scripts_path_str not in sys.path:
                    sys.path.insert(0, scripts_path_str)
            return
        candidate = candidate.parent

_ensure_app_module_on_path()

# Imports from components
from segment_tabs.shared import (
    get_store, PipelineConfig, get_available_sessions, 
    load_annotations, save_annotations, SegmentUpdater
)
from segment_tabs.manual import render_manual_annotation
from segment_tabs.detailed import render_detailed_labeling

def main():
    
    store = get_store()

    pipeline_config = PipelineConfig()

    # Add sidebar controls
    with st.sidebar:
        st.header("App Controls")
        if st.button("Finish & Exit", type="primary", help="Close the app and return to the pipeline"):
            st.success("Exiting...")
            time.sleep(0.5)
            os._exit(0)
        
        st.markdown("---")
        st.header("Annotation Dataset")
        
        # Dataset Selection Logic
        default_ann_key = pipeline_config.annotation_cache_key
        
        # Get all keys and filter/sort
        all_keys = store.list_cache_keys()
        
        dataset_mode = st.radio("Dataset Mode", ["Select Existing", "Create New"], key="dataset_mode")
        
        selected_annotation_key = default_ann_key
        
        if dataset_mode == "Select Existing":
            # Filter keys to only those containing the annotation cache key
            options = sorted([k for k in all_keys if default_ann_key in k])
            
            # Put default at top if exists
            index = 0
            if default_ann_key in options:
                index = options.index(default_ann_key)
            elif options:
                index = 0
            
            selected_annotation_key = st.selectbox("Select Dataset", options, index=index, key="dataset_select")
        else:
            new_dataset_suffix = st.text_input("New Dataset Name Suffix", value="custom_v1")
            if new_dataset_suffix:
                # Auto-prepend the default key if not present
                if new_dataset_suffix.startswith(default_ann_key):
                    selected_annotation_key = new_dataset_suffix
                else:
                    # Add underscore if needed
                    sep = "_" if not default_ann_key.endswith("_") and not new_dataset_suffix.startswith("_") else ""
                    selected_annotation_key = f"{default_ann_key}{sep}{new_dataset_suffix}"
                
                st.info(f"Will create/use dataset: **{selected_annotation_key}**")
            else:
                st.warning("Please enter a dataset name suffix.")
                selected_annotation_key = None

        # --- Maintenance Section ---
        if selected_annotation_key:
            st.markdown("---")
            st.header("Maintenance")
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                run_update_features = st.button("Update Features (All Sessions)", help="Re-extract telemetry data for all annotations in this dataset to include new features from the source data.")
            with m_col2:
                run_update_labels = st.button("Migrate Legacy Labels", help="Update legacy labels (e.g. 28 -> MS) to new schema.")
            
            if run_update_labels:
                try:
                    import migrate_labels
                    with st.spinner("Migrating legacy labels..."):
                        stats = migrate_labels.update_legacy_labels(selected_annotation_key)
                    
                    st.success(f"Migration Complete! Updated {stats['labels_replaced']} labels in {stats['sessions_updated']} sessions.")
                    if stats['labels_replaced'] > 0:
                        time.sleep(2)
                        st.rerun()
                except ImportError:
                    st.error("Could not import migration script. Please check configuration.")
                except Exception as e:
                    st.error(f"Migration failed: {e}")

            if run_update_features:
                updater = SegmentUpdater()
                source_key = pipeline_config.enriched_sessions_cache_key
                
                # Get all sessions that have annotations
                annotated_sessions = get_available_sessions(selected_annotation_key)
                
                if not annotated_sessions:
                    st.warning("No annotated sessions found to update.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, sess_id in enumerate(annotated_sessions):
                        status_text.text(f"Updating session {sess_id} ({i+1}/{len(annotated_sessions)})...")
                        
                        # Load
                        anns = load_annotations(sess_id, selected_annotation_key)
                        if anns:
                            # Update
                            updated_anns = updater.update_segments(source_key, anns)
                            # Save directly to avoid UI spam
                            data_to_save = [a.to_dict() for a in updated_anns]
                            store.save_chunk(selected_annotation_key, sess_id, data_to_save)
                        
                        progress_bar.progress((i + 1) / len(annotated_sessions))
                    
                    status_text.text("Update complete!")
                    st.success(f"Updated features for {len(annotated_sessions)} sessions.")
                    
                    # Force reload of current session if it was updated
                    if "last_session_id" in st.session_state:
                        del st.session_state.last_session_id 
                        st.rerun()

    st.title("Telemetry Segment Annotation")

    if not selected_annotation_key:
        st.warning("Please select or create an annotation dataset in the sidebar.")
        return

    st.info(f"Using Annotation Dataset: **{selected_annotation_key}**")
    
    selected_session_key = pipeline_config.enriched_sessions_cache_key

    if selected_session_key not in store.list_cache_keys():
        st.error(f"Data key '{selected_session_key}' not found. Please run the data preparation pipeline first.")
        return
    
    st.info(f"Annotating data from: {selected_session_key}")
    
    if selected_session_key:
        # Check sessions availability
        available_sessions = get_available_sessions(selected_session_key)
        
        if not available_sessions:
             st.warning("Selected session key has no data.")
             return

        # --- Top Level Tabs (Implemented as Radio to avoid double rendering) ---
        # Using radio button as tabs to prevent loading both heavy views simultaneously
        selected_view = st.radio(
            "Select View", 
            ["Telemetry Segment Annotation", "Detailed Labeling"], 
            horizontal=True,
            label_visibility="collapsed"
        )

        if selected_view == "Telemetry Segment Annotation":
            render_manual_annotation(selected_annotation_key, selected_session_key, available_sessions)
        else:
            render_detailed_labeling(selected_annotation_key, selected_session_key, available_sessions)

if __name__ == "__main__":
    main()
