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
    from segment_tabs.shared import get_store, PipelineConfig, get_available_sessions, load_annotations
    from app.models.segment_models import LABEL_MAPPING
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

def save_training_pair(segment_id: str, prompt: str, completion: str, labels: List[str], label_ids: List[str], output_path: str):
    """Append a training pair to the JSONL file."""
    record = {
        "segment_id": segment_id,
        "prompt": prompt,
        "completion": completion,
        "labels": labels,
        "label_ids": label_ids,
        "timestamp": time.time()
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_existing_annotations(output_path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing annotations to track changes and previous work."""
    records = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "segment_id" in record:
                            # Keep the latest annotation for each segment
                            records[record["segment_id"]] = record
                    except json.JSONDecodeError:
                        pass
    return records

def load_annotated_segments(session_id: str, cache_key: str) -> List[Dict[str, Any]]:
    """Load segments that have labels for a specific session."""
    segments_with_labels = []
    
    try:
        annotations = load_annotations(session_id, cache_key)
        
        all_segments = {}
        main_segments = []
        
        for item in annotations:
            # We work with dict representation for compatibility
            seg_dict = item.to_dict()
            seg_id = seg_dict.get("id")
            if seg_id:
                all_segments[seg_id] = seg_dict
            if not seg_dict.get("parent_id"):
                main_segments.append(seg_dict)
                
        # Second pass: for each main segment, aggregate labels from its sub-segments
        for main_seg in main_segments:
            labels = list(main_seg.get("labels", []))
            sub_segment_count = 0
            # Find sub-segments for this main segment
            for seg_id, seg in all_segments.items():
                if seg.get("parent_id") == main_seg.get("id"):
                    labels.extend(seg.get("labels", []))
                    sub_segment_count += 1
            
            # Remove duplicates while preserving order
            unique_labels = list(dict.fromkeys(labels))
            
            if unique_labels and len(unique_labels) > 0:
                main_seg_copy = dict(main_seg)
                main_seg_copy["labels"] = unique_labels
                main_seg_copy["sub_segment_count"] = sub_segment_count
                segments_with_labels.append(main_seg_copy)
                
    except Exception as e:
        import traceback
        st.error(f"Error loading segments: {e}\n{traceback.format_exc()}")
        
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
        
        available_sessions = []
        selected_session = None
        if selected_key:
            available_sessions = get_available_sessions(selected_key)
            if available_sessions:
                selected_session = st.selectbox("Select Session (Chunk)", options=available_sessions)
            else:
                st.warning("No sessions found in this dataset.")

        # 2. Select Output File
        output_dir = Path("logs/llm_datasets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = st.text_input("Output Filename", "telemetry_descriptions_v1.jsonl")
        output_path = output_dir / output_filename
        
        st.info(f"Saving to: `{output_path}`")
        
        # Load existing annotations to check for completed ones or changed labels
        existing_annotations = load_existing_annotations(str(output_path))
        st.write(f"Loaded {len(existing_annotations)} existing annotations.")
        
    if not selected_key or not selected_session:
        st.warning("Please select a dataset and a session chunk to begin.")
        return

    # --- Main Content ---
    
    state_key = f"{selected_key}_{selected_session}"
    
    # Session State Initialization
    if "segments" not in st.session_state or st.session_state.get("current_state_key") != state_key:
        with st.spinner(f"Loading segments from {selected_session}..."):
            st.session_state.segments = load_annotated_segments(selected_session, selected_key)
            st.session_state.current_state_key = state_key
            st.session_state.current_index = 0
            # Force rerun so sidebar can update with new segment labels
            st.rerun()
            
    segments = st.session_state.segments
    
    # --- Filter Configuration (After loading segments) ---
    with st.sidebar:
        st.header("Filter Settings")
        
        all_segment_labels = set()
        label_counts = {}
        
        if segments:
            for seg in segments:
                for l in seg.get("labels", []):
                     l_str = str(l)
                     all_segment_labels.add(l_str)
                     label_counts[l_str] = label_counts.get(l_str, 0) + 1
            
            # Create options list with counts
            filter_options = ["All"]
            # Sort by count desc
            sorted_label_ids = sorted(list(all_segment_labels), key=lambda x: label_counts[x], reverse=True)
            
            # Create mapping for the selectbox logic
            option_to_id = {"All": None}
            
            for l_id in sorted_label_ids:
                name = LABEL_MAPPING.get(l_id, l_id)
                count = label_counts[l_id]
                option_str = f"{name} ({count})"
                filter_options.append(option_str)
                option_to_id[option_str] = l_id
                
            selected_filter_options = st.multiselect("Filter by Labels", options=filter_options[1:])
            target_label_ids = [option_to_id[opt] for opt in selected_filter_options if opt in option_to_id]
        else:
            target_label_ids = []
            st.info("No labels found to filter.")

    
    # Apply Filter using the target_label_ids determined above
    filtered_segments = []
    if target_label_ids:
        # Filter: Keep segment if it has ALL of the selected labels (AND)
        filtered_segments = []
        for s in segments:
            segment_labels = [str(l) for l in s.get("labels", [])]
            # Check if all selected labels are present in the segment
            if all(t_id in segment_labels for t_id in target_label_ids):
                filtered_segments.append(s)
    else:
        filtered_segments = segments
    
    if not filtered_segments:
        if segments:
             # If filter is active but no results (shouldn't happen with valid logic but good for safety)
            st.warning(f"No segments match the filter.")
        else:
            st.warning(f"No annotated segments found in chunk {selected_session}.")
        # Determine if we should return or show something else. 
        # If no segments at all, return.
        if not segments:
            return
    
    # Check if filter changed to reset index
    # We use target_label_ids as the tracker
    if "last_filter_ids" not in st.session_state:
        st.session_state.last_filter_ids = target_label_ids
        
    # Check if list changed (by value)
    if st.session_state.last_filter_ids != target_label_ids:
        st.session_state.current_index = 0
        st.session_state.last_filter_ids = target_label_ids
        
    total_segments = len(filtered_segments)
    
    if total_segments == 0:
        st.warning("No segments match the current filter.")
        return

    # Ensure index is in bounds regarding the filtered list
    if st.session_state.current_index >= total_segments:
        st.session_state.current_index = 0
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("Previous"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    with col_nav3:
        if st.button("Next"):
            st.session_state.current_index = min(total_segments - 1, st.session_state.current_index + 1)
            
    with col_nav2:
        segment_options = [f"Segment {i+1} of {total_segments} (ID: {seg.get('id', 'Unknown')})" for i, seg in enumerate(filtered_segments)]
        
        def on_segment_change():
            selected_opt = st.session_state.segment_selector
            try:
                idx = segment_options.index(selected_opt)
                st.session_state.current_index = idx
            except ValueError:
                pass

        st.selectbox(
            "Select Segment",
            options=segment_options,
            index=st.session_state.current_index,
            key="segment_selector",
            on_change=on_segment_change,
            label_visibility="collapsed"
        )
        st.markdown(f"*(Filtered from {len(segments)} total)*")
        
    # Current Segment Data
    current_segment = filtered_segments[st.session_state.current_index]
    segment_id = str(current_segment.get("id"))
    labels = current_segment.get("labels", [])
    sub_segment_count = current_segment.get("sub_segment_count", 0)

    # Check annotation status and track changes
    segment_annotated = False
    labels_changed = False
    existing_record = None
    
    if segment_id in existing_annotations:
        existing_record = existing_annotations[segment_id]
        segment_annotated = True
        
        # Check if the labels we saved are different from current ones
        saved_label_ids = existing_record.get("label_ids", [])
        current_label_ids = [str(l) for l in labels]
        
        if sorted(saved_label_ids) != sorted(current_label_ids):
            labels_changed = True

    # Display Status Badge
    if labels_changed:
        st.warning(f"⚠️ **Labels Updated**: The labels for this segment have changed since it was last annotated.")
    elif segment_annotated:
        st.success(f"✅ **Already Annotated**: You have previously written a description for this segment.")
    else:
        st.info("🆕 **New Segment**: Requires description.")

    # Display Labels
    st.subheader("Segment Labels")
    
    if sub_segment_count > 0:
        st.info(f"ℹ️ **Contains {sub_segment_count} Sub-segment(s):** Labels from sub-segments have been included below automatically.")
    
    
    # Lookup full names
    full_labels = []
    human_readable_labels = []
    for l in labels:
        full_name = LABEL_MAPPING.get(str(l), l)
        full_labels.append(f"{full_name} ({l})")
        human_readable_labels.append(full_name)
        
    st.write(full_labels)
    
    # --- Content Tabs ---
    tab_manual, = st.tabs(["Manual Annotation"])
    
    with tab_manual:
        # --- Training Data Generation ---
        st.subheader("Training Data Pair")
        
        # Key for the text areas to reset when index changes
        prompt_key = f"prompt_{st.session_state.current_index}"
        desc_key = f"desc_{st.session_state.current_index}"
    
    with tab_manual:
        # Default prompt construction
        # Use full_labels if you want IDs, or just human_readable_labels
        default_prompt = f"Describe the driving behavior that corresponds to the following tags: {', '.join(human_readable_labels)}."
        
        # Initialize the widget state if not present (this handles navigation to new items)
        # Note: Streamlit widgets use their 'key' in session_state. 
        # We use a separate key variable just to check if we've initialized this index before.
        # Actually, let's just use the widget key directly for initialization check
        prompt_widget_key = f"input_{prompt_key}"
        desc_widget_key = f"input_{desc_key}"
        
        if prompt_widget_key not in st.session_state:
            # Pre-fill from existing annotation if present
            if existing_record:
                st.session_state[prompt_widget_key] = existing_record.get("prompt", default_prompt)
            else:
                st.session_state[prompt_widget_key] = default_prompt
    
        if desc_widget_key not in st.session_state:
            if existing_record:
                st.session_state[desc_widget_key] = existing_record.get("completion", "")
            else:
                st.session_state[desc_widget_key] = ""
    
        prompt_text = st.text_area(
            "Input Prompt (Instruction):",
            value=st.session_state[prompt_widget_key],
            height=100,
            key=prompt_widget_key
        )
    
        description = st.text_area(
            "Target Output (Description):", 
            value=st.session_state[desc_widget_key],
            height=150,
            key=desc_widget_key
        )
        
        # Save Button
        if st.button("Save Pair", type="primary"):
            current_label_ids = [str(l) for l in labels]
            save_training_pair(segment_id, prompt_text, description, human_readable_labels, current_label_ids, str(output_path))
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

