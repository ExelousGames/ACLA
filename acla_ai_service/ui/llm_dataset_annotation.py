"""
Streamlit UI for annotating LLM datasets (Label -> Sentence pairs).
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path
import time
from typing import List, Dict, Any

# Ensure app module is on path
def _ensure_app_module_on_path() -> None:
    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        candidate = candidate.parent

_ensure_app_module_on_path()

try:
    from segment_tabs.shared import get_store, PipelineConfig
    from app.models.segment_models import LABEL_MAPPING
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

def save_training_pair(labels: List[str], sentence: str, output_path: str):
    """Append a training pair to the JSONL file."""
    record = {
        "labels": labels,
        "sentence": sentence,
        "timestamp": time.time()
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_annotated_segments(store, cache_key) -> List[Dict[str, Any]]:
    """Load segments that have labels from the Zarr store."""
    segments_with_labels = []
    
    # This might be slow for large datasets, ideally we'd use an index
    # But for a simple tool, we iterate chunks
    try:
        chunks = store.get_cached_data_chunks(cache_key)
        for chunk in chunks:
            # Handle different chunk formats (list of items or dict with 'data'/'items')
            items = []
            if isinstance(chunk, list):
                items = chunk
            elif isinstance(chunk, dict):
                items = chunk.get("items", chunk.get("data", []))
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                labels = item.get("labels", [])
                if labels and len(labels) > 0:
                    segments_with_labels.append(item)
    except Exception as e:
        st.error(f"Error loading segments: {e}")
        
    return segments_with_labels

def main():
    st.set_page_config(page_title="LLM Dataset Annotation", layout="wide")
    
    st.title("LLM Description Annotation")
    st.markdown("Generate and refine human-readable sentences from segment labels.")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Configuration")
        
        # 1. Select Input Dataset (Zarr)
        store = get_store()
        pipeline_config = PipelineConfig()
        default_key = pipeline_config.annotation_cache_key
        
        all_keys = store.list_cache_keys()
        # Filter for keys that look like segment annotations
        annotated_keys = [k for k in all_keys if "annotation" in k.lower()]
        
        if not annotated_keys:
            annotated_keys = all_keys
            
        selected_key = st.selectbox(
            "Select Input Dataset (Segments)", 
            options=annotated_keys, 
            index=0 if annotated_keys else None
        )
        
        # 2. Select Output File
        output_dir = Path("logs/llm_datasets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = st.text_input("Output Filename", "telemetry_descriptions_v1.jsonl")
        output_path = output_dir / output_filename
        
        st.info(f"Saving to: `{output_path}`")
        
        # 3. Model Configuration
        st.subheader("Model Settings")
        st.info("Manual annotation mode enabled. No LLM suggestions will be generated.")
        
    if not selected_key:
        st.warning("Please select a dataset to begin.")
        return

    # --- Main Content ---
    
    # Session State Initialization
    if "segments" not in st.session_state or st.session_state.get("current_key") != selected_key:
        with st.spinner(f"Loading segments from {selected_key}..."):
            st.session_state.segments = load_annotated_segments(store, selected_key)
            st.session_state.current_key = selected_key
            st.session_state.current_index = 0
            
    segments = st.session_state.segments
    
    if not segments:
        st.warning(f"No annotated segments found in {selected_key}.")
        return
        
    total_segments = len(segments)
    
    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("Previous"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    with col_nav3:
        if st.button("Next"):
            st.session_state.current_index = min(total_segments - 1, st.session_state.current_index + 1)
            
    with col_nav2:
        st.markdown(f"**Segment {st.session_state.current_index + 1} of {total_segments}**")
        
    # Current Segment Data
    current_segment = segments[st.session_state.current_index]
    labels = current_segment.get("labels", [])
    
    # Display Labels
    st.subheader("Segment Labels")
    
    # Lookup full names
    full_labels = []
    for l in labels:
        full_name = LABEL_MAPPING.get(str(l), l)
        full_labels.append(f"{full_name} ({l})")
        
    st.write(full_labels)
    
    # Description Generation/Editing
    st.subheader("Description")
    
    # Key for the text area to reset when index changes
    text_key = f"desc_{st.session_state.current_index}"
    
    if text_key not in st.session_state:
        st.session_state[text_key] = ""
        
    description = st.text_area(
        "Edit the natural language description:", 
        value=st.session_state[text_key],
        height=150,
        key=f"input_{text_key}" # Unique key for widget
    )
    
    # Save Button
    if st.button("Save Pair", type="primary"):
        # internal labels -> human readable labels
        # Ensure we lookup using string keys as LABEL_MAPPING keys are strings
        human_readable_labels = [LABEL_MAPPING.get(str(l), l) for l in labels]
        save_training_pair(human_readable_labels, description, str(output_path))
        st.success("Saved!")
        time.sleep(0.5)
        # Advance to next
        st.session_state.current_index = min(total_segments - 1, st.session_state.current_index + 1)
        st.rerun()

    # Raw Data Expander
    with st.expander("View Raw Segment Data"):
        st.json(current_segment)

if __name__ == "__main__":
    main()
