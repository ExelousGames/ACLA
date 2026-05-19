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
    from segment_tabs.shared import get_store, PipelineConfig, get_available_sessions, load_annotations, _run_async
    from app.domain.labels import LABEL_MAPPING
    from segment_tabs.components.llm_ai_annotation import render_ai_assisted_annotation_tab
    from segment_tabs.components.llm_dataset_editor import render_dataset_editor_tab
    from segment_tabs.components.llm_segment_labels import render_segment_labels_section
    from segment_tabs.components.llm_segment_navigation import render_segment_navigation
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

def load_existing_annotations(output_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load existing annotations to track changes and previous work."""
    records = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "segment_id" in record:
                            # Keep all annotations for each segment
                            sid = record["segment_id"]
                            if sid not in records:
                                records[sid] = []
                            records[sid].append(record)
                    except json.JSONDecodeError:
                        pass
    return records

def load_all_annotations_df(output_path: str) -> pd.DataFrame:
    """Load all annotations into a pandas DataFrame for editing."""
    records = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return pd.DataFrame(records)

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
        if selected_key:
            available_sessions = get_available_sessions(selected_key)

        # 2. Select Output File
        output_dir = Path("models/llm_datasets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = st.text_input("Output Filename", "telemetry_descriptions_v1.jsonl")
        output_path = output_dir / output_filename
        
        st.info(f"Saving to: `{output_path}`")
        
        # Load existing annotations to check for completed ones or changed labels
        existing_annotations = load_existing_annotations(str(output_path))
        st.write(f"Loaded {len(existing_annotations)} existing annotations.")

    # --- Main Content ---
    main_tab_single, main_tab_batch = st.tabs(["Single Segment Annotation", "Batch AI Annotation"])
    
    with main_tab_single:
        st.header("Single Segment Annotation")
        
        from segment_tabs.components.llm_segment_navigation import handle_segment_selection_and_filtering
        
        filtered_segments, selected_session, segments = handle_segment_selection_and_filtering(
            available_sessions, 
            selected_key, 
            load_annotated_segments, 
            LABEL_MAPPING
        )
        
        if not filtered_segments:
            return

        total_segments = len(filtered_segments)
            
        # Current Segment Data
        current_segment = filtered_segments[st.session_state.current_index]
        segment_id = str(current_segment.get("id"))
        labels = current_segment.get("labels", [])
        sub_segment_count = current_segment.get("sub_segment_count", 0)

        # Check annotation status and track changes
        segment_annotated = False
        labels_changed = False
        existing_records = []
        existing_record = None
        
        if segment_id in existing_annotations:
            existing_records = existing_annotations[segment_id]
            if existing_records:
                existing_record = existing_records[-1]
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
            st.success(f"✅ **Already Annotated**: You have previously written {len(existing_records)} description(s) for this segment.")
        else:
            st.info("🆕 **New Segment**: Requires description.")

        # Display Labels
        render_segment_labels_section(labels=labels, sub_segment_count=sub_segment_count)
        
        # --- Content Tabs ---
        tab_edit, tab_ai = st.tabs(["Dataset Editor", "AI Assisted Annotation"])
        
        prompt_key = f"prompt_{st.session_state.current_index}"
        desc_key = f"desc_{st.session_state.current_index}"
        prompt_widget_key = f"input_{prompt_key}"
        desc_widget_key = f"input_{desc_key}"

        with tab_edit:
            render_dataset_editor_tab(
                labels=labels,
                existing_record=existing_record,
                prompt_widget_key=prompt_widget_key,
                desc_widget_key=desc_widget_key,
                segment_id=segment_id,
                output_path=str(output_path),
                total_segments=total_segments,
                save_training_pair_fn=save_training_pair,
                load_all_annotations_df_fn=load_all_annotations_df
            )

        with tab_ai:
            render_ai_assisted_annotation_tab(
                labels=labels,
                existing_record=existing_record,
                prompt_widget_key=prompt_widget_key,
                desc_widget_key=desc_widget_key,
            )

        # Raw Data Expander
        with st.expander("View Raw Segment Data"):
            st.json(current_segment)

    with main_tab_batch:
        from segment_tabs.components.llm_batch_ai_annotation import render_batch_ai_annotation_tab

        batch_segments_list = []
        if selected_key and available_sessions:
            # Load all segments across all sessions for batch processing
            for session_id in available_sessions:
                loaded_segs = load_annotated_segments(session_id, selected_key)
                for s in loaded_segs:
                    s["session_id"] = session_id
                batch_segments_list.extend(loaded_segs)
            
        render_batch_ai_annotation_tab(
            segments=batch_segments_list,
            existing_annotations=existing_annotations,
            output_path=str(output_path),
            available_sessions=available_sessions,
            save_training_pair_fn=save_training_pair
        )




if __name__ == "__main__":
    main()

